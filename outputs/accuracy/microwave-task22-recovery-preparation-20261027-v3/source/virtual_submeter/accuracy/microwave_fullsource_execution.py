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
from ..models.state_classifier import load_state_model, predict_state_rows, predict_windows
from . import microwave_fullsource as f
from .microwave_diagnosis import configure, load_npz, metrics, save_csv, tensor_input
from .microwave_sampling import BatchStream, check_artifacts, rank
from .microwave_background import BackgroundStream, array_hash

PARTITION = "frozen_test_refit_17_0"


def registered(run):
    f.verify_preparation(run)
    r = read_json(run/"registration.json")
    if r.get("execution_adapter_status") != "ready":
        raise DataError("Preparation-only run cannot execute; new authorized registration required")
    code = f.ROOT/("source" if (f.ROOT/"registration.json").exists() else "src")
    for name in ("accuracy/microwave_fullsource.py", "accuracy/microwave_fullsource_execution.py",
                 "accuracy/microwave_sampling.py", "accuracy/microwave_background.py",
                 "accuracy/microwave_diagnosis.py", "models/state_classifier.py", "models/state_selection.py"):
        path = "virtual_submeter/"+name
        if sha256(code/path) != sha256(run/"source"/path):
            raise DataError("Execution source differs; use sealed run entry point")
    if read_json(run/"runtime.json") != {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__}:
        raise DataError("Runtime changed after registration")
    matrix = read_json(run/"matrix.json")
    if not matrix["preparation_passed"] or matrix["fits"] > 9:
        raise DataError("Source support or fit budget invalid")
    return read_json(run/"plan.json"), matrix


def model_dirs(run):
    m = read_json(run/"matrix.json")
    return [(a, s, run/f"models/{a}/seed_{s}") for a in m["arms"] for s in m["seeds"]]


def validation(run):
    return {p.stem: load_npz(p) for p in sorted((run/"validation").glob("*.npz"))}


def source_row(seed, selected):
    rows = list(selected["block_metrics"].values())
    return {"seed": seed, "precision": float(np.mean([r["precision"] for r in rows])),
            "recall": float(np.mean([r["recall"] for r in rows])),
            "f1": float(np.mean([r["f1"] for r in rows])),
            "source_gate_passed": selected["source_gate_passed"],
            "degenerate": any(r["tp"]+r["fp"] in (0, r["n"]) for r in rows)}


def train(run):
    p, matrix = registered(run)
    if (run/"training_started.json").exists() or (run/"models").exists():
        raise DataError("Training already started; no resume, overwrite or extra fits")
    write_json(run/"training_started.json", {"registration_sha256": sha256(run/"registration.json"),
               "authorization_sha256": sha256(run/"execution_authorization.json"), "maximum_fits": matrix["fits"]})
    data, blocks = load_npz(run/"train.npz"), validation(run)
    meta, pools = read_json(run/"normalization.json"), load_npz(run/"pools.npz")
    x, y = tensor_input(meta, data["x"]), torch.from_numpy((data["y"] >= 200).astype(np.float32))
    source_rows, complete, initializations = {}, [], {}
    for arm, seed, dest in model_dirs(run):
        dest.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        model, optimizer = configure(seed, p)
        initial_digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            initial_digest.update(name.encode())
            initial_digest.update(value.numpy().tobytes())
        initializations[f"{arm}/{seed}"] = initial_digest.hexdigest()
        if arm != "natural" and initializations[f"natural/{seed}"] != initializations[f"{arm}/{seed}"]:
            raise DataError("Same-seed initialization differs")
        torch.save(model.state_dict(), dest/"initial.pt")
        sampler = BatchStream(data, "event" if arm == "event_background" else arm, seed, p)
        augment = BackgroundStream(data, pools, seed, p) if arm == "event_background" else None
        exposure = np.zeros(len(y), np.int64)
        history, batches, chosen, best = [], [], None, None
        augmentation_file = (dest/"augmentation.jsonl").open("w", encoding="utf-8") if augment else None
        try:
            for epoch in range(1, p["checkpoints"]+1):
                model.train()
                for step in range(1, p["steps_per_checkpoint"]+1):
                    update = (epoch-1)*p["steps_per_checkpoint"]+step
                    idx = sampler.next()
                    if augment:
                        augmented, records = augment.apply(idx, update)
                        for record in records:
                            augmentation_file.write(json.dumps(record, separators=(",", ":"))+"\n")
                        batch_x = tensor_input(meta, augmented)
                    else:
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
                print(f"fullsource {arm} seed={seed} checkpoint={epoch}/12 source_eligible={point is not None}", flush=True)
        finally:
            if augmentation_file:
                augmentation_file.close()
        if chosen is None:
            chosen = {"epoch": p["checkpoints"], "threshold": .5, "source_gate_passed": False,
                      "block_metrics": {k: metrics(scores[k], blocks[k]["y"], .5) for k in matrix["supported_validation_blocks"]}}
        shutil.copyfile(dest/f"epoch_{chosen['epoch']}.pt", dest/"weights.pt")
        positive_seen = int(exposure[data["y"] >= 200].sum())
        negative_seen = int(exposure[data["y"] < 200].sum())
        info = {"kind": "microwave_state_classifier_v1", "truth_threshold_w": 200,
                "window_samples": 31, "features": False, **meta,
                "decision_threshold_probability": chosen["threshold"], "selected": chosen,
                "source_gate_passed": chosen["source_gate_passed"], "arm": arm, "seed": seed,
                "weights_sha256": sha256(dest/"weights.pt"), "initialization_sha256": initializations[f"{arm}/{seed}"],
                "updates": p["checkpoints"]*p["steps_per_checkpoint"], "seconds": time.perf_counter()-started,
                "positive_seen": positive_seen, "negative_seen": negative_seen,
                "positive_weight_mass_fraction": p["pos_weight"]*positive_seen/(p["pos_weight"]*positive_seen+negative_seen),
                "parameter_count": sum(v.numel() for v in model.parameters())}
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
        if augment:
            write_json(dest/"augmentation_summary.json", augment.summary())
        source_rows.setdefault(arm, []).append(source_row(seed, chosen))
        write_json(dest/"complete.json", {"artifacts": inventory(dest)})
        complete.append(f"{arm}/{seed}")
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
    (run/"predictions").mkdir()
    rows = target_input(run)
    t, aggregate = aggregate_arrays(rows, 8)
    x, centers = windows(t, aggregate, 31, 8)
    thresholds = read_json(run/"baseline_settings.json")["on_thresholds_w"]
    identities, checks = [], []
    for arm, seed, path in model_dirs(run):
        meta, model = load_state_model(path)
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
    write_json(run/"inference_checks.json", {"checks": checks, "house17_labels_opened": False})
    write_json(run/"freeze.json", {"predictions": identities, "registration_sha256": sha256(run/"registration.json"),
               "training_sha256": sha256(run/"training_complete.json"), "source_selection_sha256": sha256(run/"source_selection.json"),
               "artifacts": {**{"predictions/"+k: v for k, v in inventory(run/"predictions").items()},
                             "inference_checks.json": sha256(run/"inference_checks.json"),
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
        np.testing.assert_array_equal(mask, np.isfinite(truth) & np.isfinite(base["raw"][:, 1]))
        supports[str(seed)] = {"n": int(mask.sum()), "mask_sha256": array_hash(mask), "timestamps_sha256": array_hash(base["stamps"][mask])}
        for arm, array in (("A", base), ("W50", w50)):
            rows.append({"arm": arm, "seed": seed, **confusion(array["raw"][mask, 1] >= 200, truth[mask] >= 200)})
            pred = array["raw"][mask, 1]
            power.append({"arm": arm, "seed": seed, "mae_w": float(np.abs(pred-truth[mask]).mean()),
                          "energy_absolute_error_kwh": float(abs(pred.sum()-truth[mask].sum())*8/3600000)})
        for arm in read_json(run/"matrix.json")["arms"]:
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
    groups = {a: [next(r for r in rows if r["arm"] == a and r["seed"] == s) for s in p["seeds"]] for a in ["A", "W50"]+read_json(run/"matrix.json")["arms"]}
    means = {a: {key: float(np.mean([r[key] for r in group])) for key in ("precision", "recall", "f1", "fp")} for a, group in groups.items()}
    passed = selected is not None and f.development_gate(groups["W50"], groups[selected], p)
    deltas = {} if selected is None else {a: {key: means[selected][key]-means[a][key] for key in ("precision", "recall", "f1", "fp")} for a in ("A", "W50", "natural")}
    return {"selected_before_development": selected, "improvement_candidate": selection["improvement_candidate"],
            "development_acceptance_passed": bool(passed), "declared_improvement": bool(passed and selection["improvement_candidate"]),
            "means": means, "selected_deltas": deltas, "common_support": supports,
            "power_improvement_claim": False, "raw_power_and_other_devices_unchanged": True,
            "default_model_changed": False, "development_only_previously_inspected_house17": True,
            "final_houses_accessed": False, "additional_training_permitted": False}


def evaluate(run):
    registered(run)
    f.evaluation_guard(run)  # Before the first truth hash or parsing operation.
    if (run/"evaluation").exists():
        raise DataError("Evaluation already attempted; completed or failed attempt is immutable")
    (run/"evaluation").mkdir()
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


def replay_training(run):
    p = read_json(run/"plan.json")
    data, pools = load_npz(run/"train.npz"), load_npz(run/"pools.npz")
    checks = []
    for arm, seed, dest in model_dirs(run):
        sampler = BatchStream(data, "event" if arm == "event_background" else arm, seed, p)
        aug = BackgroundStream(data, pools, seed, p) if arm == "event_background" else None
        counts = np.zeros(len(data["y"]), np.int64)
        stream = (dest/"augmentation.jsonl").open(encoding="utf-8") if aug else None
        try:
            with (dest/"batches.csv").open(encoding="utf-8", newline="") as batches:
                for update, row in enumerate(csv.DictReader(batches), 1):
                    idx = sampler.next()
                    if row["batch_indices_sha256"] != array_hash(idx) or int(row["update"]) != update:
                        raise DataError("Training sampler replay changed")
                    positives = int((data["y"][idx] >= 200).sum())
                    if int(row["positive"]) != positives or int(row["negative"]) != len(idx)-positives:
                        raise DataError("Actual class exposure evidence changed")
                    np.add.at(counts, idx, 1)
                    if aug:
                        _, records = aug.apply(idx, update)
                        for expected in records:
                            if json.loads(next(stream)) != expected:
                                raise DataError("Background record replay changed")
                if update != p["checkpoints"]*p["steps_per_checkpoint"]:
                    raise DataError("Incomplete optimizer update evidence")
            if stream and stream.read():
                raise DataError("Extra augmentation records")
        finally:
            if stream:
                stream.close()
        np.testing.assert_array_equal(counts, load_npz(dest/"exposure.npz")["count"])
        checks.append({"arm": arm, "seed": seed, "updates": update, "sampler_and_augmentation_exact": True})
    return checks


def verify_predictions(run):
    rows = target_input(run)
    checks = []
    thresholds = read_json(run/"baseline_settings.json")["on_thresholds_w"]
    for arm, seed, dest in model_dirs(run):
        meta, model = load_state_model(dest)
        ordinary = predict_state_rows(meta, model, rows)
        scores = np.array([np.nan if r["probability_on"] is None else r["probability_on"] for r in ordinary], np.float32)
        state = np.array([-1 if r["state"] is None else int(r["state"]) for r in ordinary], np.int8)
        frozen = load_npz(run/f"predictions/{arm}_{seed}.npz")
        np.testing.assert_array_equal(frozen["scores"], scores)
        np.testing.assert_array_equal(frozen["state"], state)
        base = reference_arrays(run, seed)
        f.aligned_unchanged(unchanged_view(base, thresholds), unchanged_view({"stamps": frozen["t"], "raw": frozen["raw"]}, thresholds))
        checks.append({"arm": arm, "seed": seed, "ordinary_inference_reloaded_exact": True})
    return checks


def verify(run):
    f.verify_preparation(run)
    if not (run/"freeze.json").exists():
        return {"status": "preparation_verified", "evaluated": False}
    f.evaluation_guard(run)
    p = read_json(run/"plan.json")
    if f.select_source(read_json(run/"source_rows.json"), p) != read_json(run/"source_selection.json"):
        raise DataError("Source arm selection replay differs")
    # Check each saved source score and checkpoint choice independently of labels on House17.
    blocks = validation(run)
    for arm, seed, dest in model_dirs(run):
        meta = read_json(dest/"model.json")
        feasible = []
        for epoch in range(1, p["checkpoints"]+1):
            scores = {k: load_npz(dest/f"epoch_{epoch}_{k}.npz")["scores"] for k in blocks}
            point = f.select_checkpoint(scores, blocks, p)
            if point is not None:
                feasible.append((rank(point, epoch), epoch, point))
        if feasible:
            _, epoch, point = min(feasible, key=lambda item: item[0])
            if meta["selected"] != {"epoch": epoch, **point, "source_gate_passed": True}:
                raise DataError("Selected checkpoint/common threshold does not reproduce")
        elif meta["source_gate_passed"]:
            raise DataError("Ineligible model marked source-deployable")
    replay = replay_training(run)
    result = {"status": "fullsource_frozen_verified", "models": len(model_dirs(run)), "training_replays": replay,
              "ordinary_inference_readbacks": verify_predictions(run),
              "target_truth_csv_reopened": False, "final_houses_accessed": False}
    if (run/"evaluation_complete.json").exists():
        receipt = read_json(run/"evaluation_complete.json")
        if receipt["freeze_sha256"] != sha256(run/"freeze.json") or receipt["source_selection_sha256"] != sha256(run/"source_selection.json"):
            raise DataError("Evaluation binding changed")
        check_artifacts(run/"evaluation", receipt["artifacts"])
        cached = load_npz(run/"evaluation/truth_once.npz")
        rows, supports, power = score_frozen(run, cached["y"])
        if {"rows": rows, "support": supports, "power_baselines": power} != read_json(run/"evaluation/metrics.json"):
            raise DataError("Development metrics do not reproduce from sealed truth")
        if summarize(run, rows, supports) != read_json(run/"evaluation/summary.json"):
            raise DataError("Development acceptance replay differs")
        result["status"] = "fullsource_evaluation_verified"
    return result


def report(run):
    f.verify_preparation(run)
    if not (run/"evaluation_complete.json").exists():
        return f.preparation_report(run)
    summary = read_json(run/"evaluation/summary.json")
    rows = read_json(run/"evaluation/metrics.json")["rows"]
    text = "# 微波炉全量来源开发验收\n\nHouse17已查看开发集，最终8/9/20未参与。\n\n"
    text += f"来源预选：{summary['selected_before_development']}；开发出口：{'通过' if summary['development_acceptance_passed'] else '未通过'}。\n\n"
    text += "所有臂仅用于透明诊断；只验收预选候选，不事后改选。水壶/洗衣机及全部原始功率保持原A，默认模型未切换。\n\n"
    text += "|方案|种子|Precision|Recall|F1|FP|FN|\n|---|---:|---:|---:|---:|---:|---:|\n"
    for r in rows:
        text += f"|{r['arm']}|{r['seed']}|{r['precision']:.6f}|{r['recall']:.6f}|{r['f1']:.6f}|{r['fp']}|{r['fn']}|\n"
    text += "\n```json\n"+json.dumps(summary, ensure_ascii=False, indent=2)+"\n```\n"
    path = run/"report.md"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise DataError("Completed report cannot be overwritten")
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return {"status": "report_ready", "path": str(path), "development_acceptance_passed": summary["development_acceptance_passed"]}
