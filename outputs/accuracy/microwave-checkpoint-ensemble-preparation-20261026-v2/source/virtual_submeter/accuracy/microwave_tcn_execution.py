"""Immutable execution stages for the registered full-source microwave experiment."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from ..data.common import DataError, inventory, read_json, sha256, utc, write_json
from ..models.arrays import aggregate_arrays, numeric, windows
from ..models.center_mask_tcn import load_state_model, predict_state_rows, predict_windows
from . import microwave_tcn as f
from .microwave_diagnosis import load_npz, metrics, save_csv
from ..models.center_mask_tcn import CenterMaskTCN, tensor_input, context_at_complete_centers
from .microwave_sampling import BatchStream, check_artifacts, rank
from .microwave_background import array_hash

PARTITION = "frozen_test_refit_17_0"


def registered(run):
    f.verify_preparation(run)
    baseline = read_json(run/"baseline_identity.json")
    for relative, hashes in baseline["artifacts"].items():
        check_artifacts(Path(baseline["reference_run"])/relative, hashes)
    if read_json(run/"runtime.json") != {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__}:
        raise DataError("Runtime differs from registration")
    p, matrix = read_json(run/"plan.json"), read_json(run/"matrix.json")
    if (not matrix["preparation_passed"] or matrix["fits"] != 6 or matrix["arms"] != f.ORDER
            or matrix["seeds"] != [42, 1019, 2026]):
        raise DataError("Invalid fixed six-fit matrix")
    return p, matrix


def authorize(run, path, phase):
    if path is None:
        raise DataError("Coordinator execution message required before "+phase)
    a = read_json(path)
    if (not a.get("coordinator_execution_message") or not a.get(phase)
            or a.get("maximum_fits") != 6 or a.get("registration_sha256") != sha256(run/"registration.json")):
        raise DataError("Explicit coordinator authorization must bind registration and six-fit limit")
    if phase == "evaluate" and a.get("freeze_sha256") != sha256(run/"freeze.json"):
        raise DataError("Evaluation authorization must bind the complete prediction freeze")
    prior = read_json(run/"plan.json").get("prior_failed_source_fit_starts", 0)
    if prior and a.get("cumulative_maximum_fit_starts") != prior+6:
        raise DataError("Explicit cumulative seven-start authorization required after failed source attempt")
    return a


def configure(seed, plan):
    torch.set_num_threads(plan["cpu_threads"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = CenterMaskTCN()
    return model, torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])


def model_dirs(run):
    m = read_json(run/"matrix.json")
    return [(a, s, run/f"models/{a}/seed_{s}") for a in m["arms"] for s in m["seeds"]]


def validation(run, arm):
    return {p.stem: load_npz(p) for p in sorted((run/"validation"/arm).glob("*.npz"))}


def source_row(seed, selected):
    rows = list(selected["block_metrics"].values())
    return {"seed": seed, "precision": float(np.mean([r["precision"] for r in rows])),
            "recall": float(np.mean([r["recall"] for r in rows])),
            "f1": float(np.mean([r["f1"] for r in rows])),
            "source_gate_passed": selected["source_gate_passed"],
            "degenerate": any(r["tp"]+r["fp"] in (0, r["n"]) for r in rows)}


def train(run, authorization=None):
    p, matrix = registered(run)
    authorize(run, authorization, "train")
    if (run/"training_started.json").exists() or (run/"models").exists():
        raise DataError("Training already started; no resume, overwrite or extra fits")
    shutil.copyfile(authorization, run/"execution_authorization.json")
    write_json(run/"training_started.json", {"registration_sha256": sha256(run/"registration.json"),
               "authorization_sha256": sha256(run/"execution_authorization.json"), "maximum_fits": matrix["fits"]})
    meta = read_json(run/"normalization.json")
    source_rows, complete, initializations = {}, [], {}
    for arm, seed, dest in model_dirs(run):
        dest.mkdir(parents=True, exist_ok=False)
        meta = {**meta, "window_samples": 31 if arm == "TCN31" else 127}
        blocks = validation(run, arm)
        data = load_npz(run/f"{arm}.npz")
        x = tensor_input(meta, data["x"], data["mask"])
        y = torch.from_numpy((data["y"] >= 200).astype(np.float32))
        started = time.perf_counter()
        model, optimizer = configure(seed, p)
        initial_digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            initial_digest.update(name.encode())
            initial_digest.update(value.numpy().tobytes())
        initializations[f"{arm}/{seed}"] = initial_digest.hexdigest()
        if initial_digest.hexdigest() != read_json(run/"initializations.json")[str(seed)]:
            raise DataError("Initialization differs from preparation")
        if arm != "TCN31" and initializations[f"TCN31/{seed}"] != initializations[f"{arm}/{seed}"]:
            raise DataError("Same-seed initialization differs")
        torch.save(model.state_dict(), dest/"initial.pt")
        sampler = iter(load_npz(run/f"sampling_{seed}.npz")["indices"])
        exposure = np.zeros(len(y), np.int64)
        history, batches, chosen, best = [], [], None, None
        for epoch in range(1, p["checkpoints"]+1):
            model.train()
            for step in range(1, p["steps_per_checkpoint"]+1):
                update = (epoch-1)*p["steps_per_checkpoint"]+step
                idx = next(sampler)
                batch_x = x[idx]
                optimizer.zero_grad(set_to_none=True)
                losses = torch.nn.functional.binary_cross_entropy_with_logits(model(batch_x), y[idx],
                         pos_weight=torch.tensor(p["pos_weight"]), reduction="none")
                loss = losses.mean()
                loss.backward()
                if not torch.isfinite(loss) or any(not torch.isfinite(v.grad).all() for v in model.parameters()):
                    raise DataError("Nonfinite training loss/gradient; preserve incomplete run")
                optimizer.step()
                np.add.at(exposure, idx, 1)
                pos = data["y"][idx] >= 200
                pl, nl = float(losses[pos].sum().detach()), float(losses[~pos].sum().detach())
                batches.append({"update": update, "loss": float(loss.detach()), "positive": int(pos.sum()),
                    "negative": int((~pos).sum()), "positive_loss_sum": pl, "negative_loss_sum": nl,
                    "positive_effective_loss_fraction": pl/(pl+nl) if pl+nl else 0.,
                    "batch_indices_sha256": array_hash(idx),
                    "unique_positive_events": len(set(zip(data["house"][idx[pos]].tolist(), data["event"][idx[pos]].tolist())))})
            scores = {k: predict_windows(meta, model, d["x"]) for k, d in blocks.items()}
            point = f.select_checkpoint(scores, blocks, p)
            torch.save(model.state_dict(), dest/f"epoch_{epoch}.pt")
            for key, value in scores.items():
                np.savez_compressed(dest/f"epoch_{epoch}_{key}.npz", scores=value, t=blocks[key]["t"])
            history.append({"epoch": epoch, "selection": point})
            if point is not None and (best is None or rank(point, epoch) < best):
                best, chosen = rank(point, epoch), {"epoch": epoch, **point, "source_gate_passed": True}
            write_json(dest/"history.json", {"rows": history})
            print(f"mask {arm} seed={seed} checkpoint={epoch}/12 source_eligible={point is not None}", flush=True)
        if chosen is None:
            chosen = {"epoch": p["checkpoints"], "threshold": .5, "source_gate_passed": False,
                      "block_metrics": {k: metrics(scores[k], blocks[k]["y"], .5) for k in matrix["supported_validation_blocks"]}}
        shutil.copyfile(dest/f"epoch_{chosen['epoch']}.pt", dest/"weights.pt")
        positive_seen = int(exposure[data["y"] >= 200].sum())
        negative_seen = int(exposure[data["y"] < 200].sum())
        info = {"kind": "microwave_center_mask_tcn_v1", "inference_support": "strict_complete_31", "truth_threshold_w": 200,
                "window_samples": 31 if arm == "TCN31" else 127, "features": False, **meta,
                "decision_threshold_probability": chosen["threshold"], "selected": chosen,
                "source_gate_passed": chosen["source_gate_passed"], "arm": arm, "seed": seed,
                "weights_sha256": sha256(dest/"weights.pt"), "initialization_sha256": initializations[f"{arm}/{seed}"],
                "updates": p["checkpoints"]*p["steps_per_checkpoint"], "seconds": time.perf_counter()-started,
                "positive_seen": positive_seen, "negative_seen": negative_seen,
                "positive_weight_mass_fraction": p["pos_weight"]*positive_seen/(p["pos_weight"]*positive_seen+negative_seen),
                "parameter_count": sum(v.numel() for v in model.parameters())}
        info.update({"registration_sha256": sha256(run/"registration.json"),
                     "training_array_sha256": sha256(run/f"{arm}.npz"),
                     "normalization_sha256": sha256(run/"normalization.json"),
                     "unique_windows_seen": int((exposure > 0).sum()),
                     "unique_positive_windows_seen": int(((exposure > 0) & (data["y"] >= 200)).sum()),
                     "repeated_exposures": int(exposure.sum()-(exposure > 0).sum())})
        write_json(dest/"model.json", info)
        save_csv(dest/"batches.csv", batches)
        np.savez_compressed(dest/"exposure.npz", count=exposure, t=data["t"], house=data["house"], event=data["event"], positive=data["y"] >= 200)
        grouped = {}
        for i, count in enumerate(exposure):
            key = int(data["house"][i]), int(data["t"][i]//86400), int(data["event"][i])
            row = grouped.setdefault(key, {"house": key[0], "utc_day": key[1], "event": key[2], "available_centers": 0, "unique_centers_seen": 0, "total_exposure": 0})
            row["available_centers"] += 1
            row["unique_centers_seen"] += int(count > 0)
            row["total_exposure"] += int(count)
        save_csv(dest/"date_event_exposure.csv", list(grouped.values()))
        source_rows.setdefault(arm, []).append(source_row(seed, chosen))
        write_json(dest/"complete.json", {"artifacts": inventory(dest)})
        complete.append(f"{arm}/{seed}")
    source_rows["partial_mask"] = read_json(run/"reference_source_rows.json")["partial_mask"]
    selection = f.select_source(source_rows, p)
    write_json(run/"source_rows.json", source_rows)
    write_json(run/"source_selection.json", selection)
    write_json(run/"training_complete.json", {"completed_models": complete, "fits": len(complete),
               "artifacts": inventory(run/"models"), "source_rows_sha256": sha256(run/"source_rows.json"),
               "source_selection_sha256": sha256(run/"source_selection.json"), "initializations": initializations})
    return {"status": "training_complete", "fits": len(complete), "source_selection": selection}


def target_input(run):
    baseline = read_json(run/"baseline_identity.json")
    part = baseline["development_partition"]
    artifact = part["artifacts"]["aggregate"]
    # The baseline run's original protocol, not copied relative paths, owns data.
    base = Path(baseline["reference_run"])/"protocol"
    path = base/artifact["path"]
    if sha256(path) != artifact["sha256"]:
        raise DataError("Development aggregate changed")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if {r["household_id"] for r in rows} != {"refit_17"}:
        raise DataError("Wrong development aggregate household")
    return rows


def reference_arrays(run, seed, group="A"):
    base = Path(read_json(run/"baseline_identity.json")["reference_run"])
    relative = f"reference/predictions/seed_{seed}" if group == "A" else f"predictions/W50/seed_{seed}"
    return load_npz(base/relative/f"{PARTITION}.npz")


def unchanged_view(arrays, thresholds):
    return {"t": arrays["stamps"], "kettle_power": arrays["raw"][:, 0],
            "kettle_state": arrays["raw"][:, 0] >= thresholds["kettle"],
            "washing_machine_power": arrays["raw"][:, 2],
            "washing_machine_state": arrays["raw"][:, 2] >= thresholds["washing_machine"],
            "microwave_power": arrays["raw"][:, 1]}


def freeze(run):
    p, matrix = registered(run)
    complete = read_json(run/"training_complete.json")
    expected = {f"{a}/{s}" for a in matrix["arms"] for s in p["seeds"]}
    if set(complete["completed_models"]) != expected or complete["fits"] != matrix["fits"]:
        raise DataError("Incomplete training matrix")
    check_artifacts(run/"models", complete["artifacts"])
    if complete["source_selection_sha256"] != sha256(run/"source_selection.json") or complete["source_rows_sha256"] != sha256(run/"source_rows.json"):
        raise DataError("Source selection changed after training")
    if f.select_source(read_json(run/"source_rows.json"), p) != read_json(run/"source_selection.json"):
        raise DataError("Source preselection does not reproduce")
    if (run/"predictions").exists() or (run/"freeze.json").exists():
        raise DataError("Freeze cannot overwrite an existing prediction attempt")
    upstream = Path(read_json(run/"upstream_identity.json")["run"])
    uf = read_json(upstream/"freeze.json")
    prior_arm = "partial_mask"
    if sha256(upstream/"freeze.json") != read_json(run/"reference_identity.json")["freeze_sha256"]:
        raise DataError("Old partial freeze changed")
    (run/"predictions").mkdir()
    rows = target_input(run)
    t, aggregate = aggregate_arrays(rows, 8)
    thresholds = read_json(run/"baseline_settings.json")["on_thresholds_w"]
    identities, checks = [], []
    for arm, seed, path in model_dirs(run):
        meta, model = load_state_model(path)
        x, centers = context_at_complete_centers(meta, t, aggregate)
        scores = np.full(len(t), np.nan, np.float32)
        scores[centers] = predict_windows(meta, model, x)
        ordinary_rows = predict_state_rows(meta, model, rows)
        ordinary = np.array([r["probability_on"] if r["probability_on"] is not None else np.nan for r in ordinary_rows], np.float32)
        np.testing.assert_array_equal(scores, ordinary)
        base, w50 = reference_arrays(run, seed), reference_arrays(run, seed, "W50")
        np.testing.assert_array_equal(t, base["stamps"])
        np.testing.assert_array_equal(aggregate, base["aggregate"])
        np.testing.assert_array_equal(np.isfinite(scores), np.isfinite(base["raw"][:, 1]))
        np.testing.assert_array_equal(t, w50["stamps"])
        np.testing.assert_array_equal(np.isfinite(scores), np.isfinite(w50["raw"][:, 1]))
        state = np.where(np.isfinite(scores), scores >= meta["decision_threshold_probability"], -1).astype(np.int8)
        ordinary_state = np.array([-1 if r["state"] is None else int(r["state"]) for r in ordinary_rows], np.int8)
        np.testing.assert_array_equal(state, ordinary_state)
        # Raw power and unchanged devices are copied from A exactly, without correction.
        candidate = {"stamps": t, "raw": base["raw"].copy()}
        f.aligned_unchanged(unchanged_view(base, thresholds), unchanged_view(candidate, thresholds))
        np.savez_compressed(run/f"predictions/{arm}_{seed}.npz", t=t, scores=scores, state=state,
                            raw=candidate["raw"], aggregate=aggregate)
        identities.append(f"{arm}/{seed}")
        checks.append({"arm": arm, "seed": seed, "valid_input_centers": len(centers),
                       "ordinary_scores_exact": True, "ordinary_state_exact": True, "raw_A_exact": True,
                       "input_center_t_sha256": array_hash(t[centers]), "weights_sha256": meta["weights_sha256"]})
    prior_identity = {"selected_arm": prior_arm, "freeze_sha256": sha256(upstream/"freeze.json"),
                      "source_selection_sha256": sha256(upstream/"source_selection.json")}
    if prior_arm is not None:
        for seed in p["seeds"]:
            name = f"predictions/{prior_arm}_{seed}.npz"
            if name not in uf["artifacts"]:
                raise DataError("Task4 preselected prediction is not bound")
            src = load_npz(upstream/name)
            if sha256(upstream/name) != uf["artifacts"][name]:
                raise DataError("Frozen reference prediction changed")
            from ..models.mask_state_classifier import load_state_model as load_old, predict_state_rows as predict_old
            old_path = upstream/f"models/partial_mask/seed_{seed}"
            bound = read_json(run/"reference_identity.json")["models"][str(seed)]
            if sha256(old_path/"model.json") != bound["model_json_sha256"]:
                raise DataError("Frozen old metadata changed")
            old_meta, old_model = load_old(old_path)
            old_rows = predict_old(old_meta, old_model, rows)
            old_scores = np.array([r["probability_on"] if r["probability_on"] is not None else np.nan for r in old_rows], np.float32)
            np.testing.assert_array_equal(old_scores, src["scores"])
            old_states = np.array([-1 if r["state"] is None else int(r["state"]) for r in old_rows], np.int8)
            np.testing.assert_array_equal(old_states, src["state"])
            own = load_npz(run/f"predictions/TCN31_{seed}.npz")
            np.testing.assert_array_equal(src["t"], own["t"])
            np.testing.assert_array_equal(np.isfinite(src["scores"]), np.isfinite(own["scores"]))
            np.testing.assert_array_equal(src["raw"], own["raw"])
            shutil.copyfile(upstream/name, run/f"predictions/partial_mask_{seed}.npz")
    write_json(run/"task4_source_comparison.json", prior_identity)
    write_json(run/"inference_checks.json", {"checks": checks, "house17_labels_opened": False})
    write_json(run/"freeze.json", {"predictions": identities, "registration_sha256": sha256(run/"registration.json"),
               "training_sha256": sha256(run/"training_complete.json"), "source_selection_sha256": sha256(run/"source_selection.json"),
               "artifacts": {**{"predictions/"+k: v for k, v in inventory(run/"predictions").items()},
                             "inference_checks.json": sha256(run/"inference_checks.json"),
                             "task4_source_comparison.json": sha256(run/"task4_source_comparison.json"),
                             "source_rows.json": sha256(run/"source_rows.json")},
               "ordinary_inference_matches": True, "common_support_and_other_devices_match": True,
               "house17_labels_opened": False})
    f.evaluation_guard(run)
    return {"status": "all_development_predictions_frozen", "predictions": len(identities), "house17_labels_opened": False}


def confusion(predicted, actual):
    tp, fp = int((predicted & actual).sum()), int((predicted & ~actual).sum())
    fn, tn = int((~predicted & actual).sum()), int((~predicted & ~actual).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(actual),
            "precision": tp/max(tp+fp, 1), "recall": tp/max(tp+fn, 1), "f1": 2*tp/max(2*tp+fp+fn, 1),
            "degenerate": int(predicted.sum()) in (0, len(actual)), "predicted_on": int(predicted.sum())}


def score_frozen(run, truth):
    p = read_json(run/"plan.json")
    rows, supports, power = [], {}, []
    for seed in p["seeds"]:
        base, w50 = reference_arrays(run, seed), reference_arrays(run, seed, "W50")
        mask = np.isfinite(truth) & np.isfinite(w50["raw"][:, 1])
        if int(mask.sum()) != 137523:
            raise DataError("Original W50 effective support must remain 137523")
        np.testing.assert_array_equal(mask, np.isfinite(truth) & np.isfinite(base["raw"][:, 1]))
        supports[str(seed)] = {"n": int(mask.sum()), "mask_sha256": array_hash(mask), "timestamps_sha256": array_hash(base["stamps"][mask])}
        for arm, array in (("A", base), ("W50", w50)):
            rows.append({"arm": arm, "seed": seed, **confusion(array["raw"][mask, 1] >= 200, truth[mask] >= 200)})
            pred = array["raw"][mask, 1]
            power.append({"arm": arm, "seed": seed, "mae_w": float(np.abs(pred-truth[mask]).mean()),
                          "energy_absolute_error_kwh": float(abs(pred.sum()-truth[mask].sum())*8/3600000)})
        arms = read_json(run/"matrix.json")["arms"]
        if read_json(run/"task4_source_comparison.json")["selected_arm"] is not None:
            arms = arms + ["partial_mask"]
        for arm in arms:
            a = load_npz(run/f"predictions/{arm}_{seed}.npz")
            np.testing.assert_array_equal(a["t"], base["stamps"])
            if not np.isfinite(a["scores"][mask]).all() or np.any(a["state"][mask] < 0):
                raise DataError("Candidate loses an original W50 effective evaluation timestamp")
            rows.append({"arm": arm, "seed": seed, **confusion(a["state"][mask].astype(bool), truth[mask] >= 200)})
            np.testing.assert_array_equal(a["raw"], base["raw"])
    return rows, supports, power


def summarize(run, rows, supports):
    p = read_json(run/"plan.json")
    selection = read_json(run/"source_selection.json")
    selected = selection["selected_arm"]
    groups = {a: [next(r for r in rows if r["arm"] == a and r["seed"] == s) for s in p["seeds"]] for a in sorted({r["arm"] for r in rows})}
    means = {a: {key: float(np.mean([r[key] for r in group])) for key in ("precision", "recall", "f1", "fp")} for a, group in groups.items()}
    passed = selected is not None and f.development_gate(groups["W50"], groups[selected], p)
    deltas = {} if selected is None else {a: {key: means[selected][key]-means[a][key] for key in ("precision", "recall", "f1", "fp")} for a in groups if a != selected}
    return {"selected_before_development": selected, "improvement_candidate": selection["improvement_candidate"],
            "long_context_selected": selected == "TCN127",
            "task4_source_preselected_comparison": read_json(run/"task4_source_comparison.json"),
            "development_acceptance_passed": bool(passed), "declared_improvement": bool(passed and selection["improvement_candidate"]),
            "means": means, "selected_deltas": deltas, "common_support": supports,
            "power_improvement_claim": False, "raw_power_and_other_devices_unchanged": True,
            "default_model_changed": False, "development_only_previously_inspected_house17": True,
            "final_houses_accessed": False, "additional_training_permitted": False}


def evaluate(run, authorization=None):
    registered(run)
    authorize(run, authorization, "evaluate")
    f.evaluation_guard(run)  # Before the first truth hash or parsing operation.
    verify_training(run)
    if (run/"evaluation").exists():
        raise DataError("Evaluation already attempted; completed or failed attempt is immutable")
    (run/"evaluation").mkdir()
    shutil.copyfile(authorization, run/"evaluation/authorization.json")
    write_json(run/"evaluation/started.json", {"freeze_sha256": sha256(run/"freeze.json"), "source_selection_sha256": sha256(run/"source_selection.json")})
    b = read_json(run/"baseline_identity.json")
    artifact = b["development_partition"]["artifacts"]["microwave"]
    path = Path(b["reference_run"])/"protocol"/artifact["path"]
    if sha256(path) != artifact["sha256"]:
        raise DataError("Development truth differs from original W50 protocol")
    with path.open(encoding="utf-8", newline="") as stream:
        mapping = {utc(r["timestamp"]): numeric(r["power_w"]) for r in csv.DictReader(stream)}
    t = reference_arrays(run, read_json(run/"plan.json")["seeds"][0])["stamps"]
    truth = np.array([mapping.get(int(s), np.nan) for s in t], np.float64)
    np.savez_compressed(run/"evaluation/truth_once.npz", t=t, y=truth)
    rows, supports, power = score_frozen(run, truth)
    summary = summarize(run, rows, supports)
    write_json(run/"evaluation/metrics.json", {"rows": rows, "support": supports, "power_baselines": power})
    write_json(run/"evaluation/summary.json", summary)
    save_csv(run/"evaluation/comparison.csv", rows)
    write_json(run/"evaluation_complete.json", {"freeze_sha256": sha256(run/"freeze.json"),
               "truth_original_sha256": artifact["sha256"], "source_selection_sha256": sha256(run/"source_selection.json"),
               "artifacts": inventory(run/"evaluation"), "truth_csv_parses": 1})
    return {"status": "development_evaluated_once", **summary}


def verify_training(run):
    p, matrix = registered(run)
    complete = read_json(run/"training_complete.json")
    expected = [f"{a}/{s}" for a in f.ORDER for s in p["seeds"]]
    if complete["completed_models"] != expected or complete["fits"] != 6:
        raise DataError("Incomplete trained matrix")
    check_artifacts(run/"models", complete["artifacts"])
    for key in ("source_rows", "source_selection"):
        if complete[key+"_sha256"] != sha256(run/(key+".json")):
            raise DataError("Training source selection binding mismatch")
    reproduced = {"partial_mask": read_json(run/"reference_source_rows.json")["partial_mask"]}
    dry = read_json(run/"dry_replay.json")["matrix"]
    for arm, seed, dest in model_dirs(run):
        blocks = validation(run, arm)
        d = load_npz(run/f"{arm}.npz")
        exposure = load_npz(dest/"exposure.npz")
        expected_exposure = load_npz(run/f"dry_exposure_{arm}_{seed}.npz")
        for key in ("count", "t", "house", "event"):
            np.testing.assert_array_equal(exposure[key], expected_exposure[key])
        with (dest/"batches.csv").open(encoding="utf-8", newline="") as stream:
            batches = list(csv.DictReader(stream))
        sampler = iter(load_npz(run/f"sampling_{seed}.npz")["indices"])
        digest = hashlib.sha256()
        if len(batches) != 3528:
            raise DataError("Unexpected update budget")
        for row in batches:
            idx = next(sampler)
            digest.update(idx.tobytes())
            if row["batch_indices_sha256"] != array_hash(idx) or int(row["positive"]) != 16 or int(row["negative"]) != 240:
                raise DataError("Training sampling replay differs")
        if digest.hexdigest() != dry[f"{arm}/{seed}"]["indices_sha256"]:
            raise DataError("Training exposure identity differs from registration")
        meta, model = load_state_model(dest)
        initial, _ = configure(seed, p)
        saved_initial = torch.load(dest/"initial.pt", map_location="cpu", weights_only=True)
        initial_digest = hashlib.sha256()
        for name, value in initial.state_dict().items():
            if not torch.equal(value, saved_initial[name]):
                raise DataError("Saved initialization differs from registered seed")
            initial_digest.update(name.encode())
            initial_digest.update(value.numpy().tobytes())
        if initial_digest.hexdigest() != complete["initializations"][f"{arm}/{seed}"]:
            raise DataError("Initialization hash mismatch")
        chosen, best = None, None
        for epoch in range(1, 13):
            scores = {}
            for key, block in blocks.items():
                a = load_npz(dest/f"epoch_{epoch}_{key}.npz")
                np.testing.assert_array_equal(a["t"], block["t"])
                scores[key] = a["scores"]
            point = f.select_checkpoint(scores, blocks, p)
            if point is not None and (best is None or rank(point, epoch) < best):
                best, chosen = rank(point, epoch), {"epoch": epoch, **point, "source_gate_passed": True}
        if chosen is None:
            chosen = {"epoch": 12, "threshold": .5, "source_gate_passed": False,
                      "block_metrics": {k: metrics(scores[k], blocks[k]["y"], .5) for k in matrix["supported_validation_blocks"]}}
        if chosen != meta["selected"] or sha256(dest/"weights.pt") != sha256(dest/f"epoch_{chosen['epoch']}.pt"):
            raise DataError("Chosen checkpoint/threshold fails source-only replay")
        for key, block in blocks.items():
            actual = predict_windows(meta, model, block["x"])
            np.testing.assert_array_equal(actual, load_npz(dest/f"epoch_{chosen['epoch']}_{key}.npz")["scores"])
        reproduced.setdefault(arm, []).append(source_row(seed, chosen))
    if reproduced != read_json(run/"source_rows.json") or f.select_source(reproduced, p) != read_json(run/"source_selection.json"):
        raise DataError("Source selection does not reproduce")
    for seed in p["seeds"]:
        if complete["initializations"][f"TCN31/{seed}"] != complete["initializations"][f"TCN127/{seed}"]:
            raise DataError("Paired network initializations differ")
    return True

