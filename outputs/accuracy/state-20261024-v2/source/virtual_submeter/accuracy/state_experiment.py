"""Finite source-only state study with all-target prediction freeze before evaluation."""
import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil

import numpy as np

from ..data.common import DataError, read_json, safe_child, sha256, source_fingerprint, write_json, iso
from ..data.protocol import read_partition
from ..models.arrays import aggregate_arrays, windows, labels_at_centers
from ..models.training import dataset, train
from ..models.state_selection import pr_curve, constrained_point, checkpoint_rank
from .data_expansion import inventory, compact_metrics
from .precision import SOURCE_VALIDATION, DEVELOPMENT, DEVICES, SEEDS, validate_protocol, _write_csv

ROOT = Path(__file__).resolve().parents[3]
GROUPS = ["R_PR", "C31", "C_EVENT", "C_HARD", "C_FEATURES", "C61"]


def load_plan(path):
    plan = read_json(path)
    expected = {"schema_version": 1, "experiment_id": "microwave-state-20261024-v1",
                "parent_run": "outputs/accuracy/precision-20261024-v1", "seeds": SEEDS, "candidates": GROUPS,
                "classifier_epochs": 12, "classifier_steps_per_checkpoint": 294,
                "regression_steps_per_checkpoint": 587, "minimum_source_recall": .85,
                "minimum_precision": .08, "minimum_recall": .85, "minimum_f1_gain": .02,
                "maximum_seed_f1_drop": .02, "minimum_positive_seeds": 2,
                "minimum_fp_reduction": .30, "maximum_models": 18}
    if plan != expected:
        raise DataError("State plan differs from the finite registered experiment")
    return plan


def _inputs(run):
    return {name + "/" + p: h for name in ("protocol", "reference", "source", "entrypoints", "registration_inputs")
            for p, h in inventory(run / name).items()}


def registered(run, current=True):
    reg = read_json(run / "registration.json")
    if _inputs(run) != reg["inputs"] or (current and source_fingerprint() != reg["code_sha256"]):
        raise DataError("State registered input or source changed")
    return load_plan(run / "registration_inputs/plan.json")


def prepare(run, plan_path):
    plan = load_plan(plan_path)
    if run.exists():
        raise DataError("State destination exists")
    parent = safe_child(ROOT, plan["parent_run"])
    from .precision import verify
    verify(parent)  # verifies historical seals without parsing target labels
    protocol = read_json(parent / "protocol/protocol.json")
    validate_protocol(protocol)
    run.mkdir(parents=True)
    shutil.copytree(parent / "protocol", run / "protocol")
    protocol["protocol_id"] = plan["experiment_id"]
    protocol["parent_protocol_sha256"] = sha256(parent / "protocol/protocol.json")
    write_json(run / "protocol/protocol.json", protocol)
    shutil.copytree(parent / "reference", run / "reference")
    (run / "registration_inputs").mkdir()
    shutil.copyfile(plan_path, run / "registration_inputs/plan.json")
    shutil.copyfile(parent / "base_settings.json", run / "registration_inputs/base_settings.json")
    shutil.copyfile(ROOT / "docs/STATE_REGISTRATION_20261024.md", run / "registration_inputs/protocol.md")
    shutil.copytree(ROOT / "src/virtual_submeter", run / "source/virtual_submeter",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (run / "entrypoints").mkdir()
    shutil.copyfile(ROOT / "scripts/run_state_experiment.py", run / "entrypoints/run_state_experiment.py")
    write_json(run / "registration.json", {"registered_at": datetime.now(timezone.utc).isoformat(),
               "inputs": _inputs(run), "code_sha256": source_fingerprint(), "development_labels_parsed": False,
               "parent_freeze_sha256": sha256(parent / "freeze.json"), "final_households_accessed": False})
    write_json(run / "run.json", {"status": "prepared"})
    return {"status": "prepared", "maximum_models": 18}


def event_groups(keys, y):
    """Break at household/time/label discontinuities; negatives are at most 31 points."""
    groups = np.empty(len(y), np.int64)
    group, length = -1, 0
    for i, (house, stamp) in enumerate(keys):
        active = y[i] >= 200
        new = (i == 0 or house != keys[i-1][0] or stamp - keys[i-1][1] != 8
               or active != (y[i-1] >= 200) or (not active and length == 31))
        if new:
            group += 1
            length = 0
        groups[i] = group
        length += 1
    return groups


def sampling_probabilities(keys, y, candidate, scores=None):
    if not np.isfinite(y).all() or not (y >= 200).any() or not (y < 200).any():
        raise DataError("Sampling requires observed mixed source labels")
    probabilities = np.ones(len(y), np.float64)
    active = y >= 200
    audit = {"candidate": candidate, "source_only": True, "positive_windows": int(active.sum()),
             "windows": len(y), "households": {}}
    groups = event_groups(keys, y)
    audit["threshold_segments"] = len(np.unique(groups[active]))
    if candidate == "C_EVENT":
        houses = np.asarray([k[0] for k in keys])
        for label in (False, True):
            class_mass = float((active == label).mean())
            eligible = [h for h in np.unique(houses) if ((houses == h) & (active == label)).any()]
            for house in eligible:
                mask = (houses == house) & (active == label)
                ids, counts = np.unique(groups[mask], return_counts=True)
                for identity, count in zip(ids, counts):
                    probabilities[mask & (groups == identity)] = class_mass / len(eligible) / len(ids) / count
    elif candidate == "C_HARD":
        if scores is None or len(scores) != len(y) or not np.isfinite(scores).all():
            raise DataError("Hard coverage requires frozen C31 source scores")
        low, high = float(np.quantile(scores[active], .25)), float(np.quantile(scores[~active], .99))
        hard = (active & (scores <= low)) | (~active & (scores >= high))
        probabilities[hard] = 4
        # Keep class mass fixed; this changes within-class coverage only.
        for label in (False, True):
            mask = active == label
            probabilities[mask] *= mask.sum() / probabilities[mask].sum()
        audit.update(hard_positive=int((hard & active).sum()), hard_negative=int((hard & ~active).sum()),
                     positive_score_cutoff=low, negative_score_cutoff=high)
    probabilities /= probabilities.sum()
    for house in sorted({k[0] for k in keys}):
        mask = np.asarray([k[0] == house for k in keys])
        audit["households"][house] = {"windows": int(mask.sum()), "positive": int((mask & active).sum()),
                                      "events": len(np.unique(groups[mask & active])),
                                      "sampling_mass": float(probabilities[mask].sum())}
    audit["positive_sampling_mass"] = float(probabilities[active].sum())
    audit["source_identity_sha256"] = __import__("hashlib").sha256(json.dumps(keys).encode()).hexdigest()
    if scores is not None:
        audit["mining_scores_sha256"] = __import__("hashlib").sha256(scores.tobytes()).hexdigest()
    return probabilities, audit


def _classifier_fit(run, candidate, seed, cached):
    import torch
    from ..models.state_classifier import StateClassifier, predict_windows, load_state_model
    target = run / f"models/{candidate}/seed_{seed}"
    if target.exists():
        # Interrupted individual fits are never silently accepted.
        if not (target / "complete.json").exists():
            raise DataError("Incomplete model directory; use a new run rather than overwrite")
        for p, h in read_json(target / "complete.json")["artifacts"].items():
            if sha256(safe_child(target, p)) != h:
                raise DataError("Resumed classifier changed")
        load_state_model(target)
        return
    width = 61 if candidate == "C61" else 31
    x, y, keys, vx, vy, counts, vcounts = cached[width]
    protocol = read_json(run / "protocol/protocol.json")
    controls = protocol["expansion_training"]
    settings = read_json(run / "registration_inputs/base_settings.json")
    scores = None
    if candidate == "C_HARD":
        base_meta, base_model = load_state_model(run / f"models/C31/seed_{seed}")
        scores = predict_windows(base_meta, base_model, x)
    probabilities, audit = sampling_probabilities(keys, y, candidate, scores)
    target.mkdir(parents=True)
    write_json(target / "sampling.json", audit)
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = StateClassifier(width, candidate == "C_FEATURES")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    tx = torch.from_numpy((np.log1p(x) - controls["input_log_mean"]) / controls["input_log_std"])
    ty = torch.from_numpy((y >= 200).astype(np.float32))
    meta = {"schema_version": 1, "kind": "microwave_state_classifier_v1", "candidate": candidate, "seed": seed,
            "window_samples": width, "features": candidate == "C_FEATURES", "batch_size": 256,
            "cpu_threads": settings["cpu_threads"], "input_log_mean": controls["input_log_mean"],
            "input_log_std": controls["input_log_std"], "truth_threshold_w": 200,
            "protocol_sha256": sha256(run / "protocol/protocol.json"), "code_sha256": source_fingerprint(),
            "source_train_windows": counts, "source_validation_windows": vcounts,
            "parameter_count": sum(p.numel() for p in model.parameters()), "optimizer_steps": 3528,
            "target_labels_read": False, "confirmation_budget": 0}
    generator = torch.Generator().manual_seed(seed)
    def batches():
        while True:
            if candidate in ("C_EVENT", "C_HARD"):
                indices = torch.multinomial(torch.from_numpy(probabilities), len(tx), True, generator=generator)
            else:
                indices = torch.randperm(len(tx), generator=generator)
            yield from indices.split(256)
    stream = batches()
    history, best = [], None
    for epoch in range(1, 13):
        model.train()
        loss_sum = 0.
        for _ in range(294):
            indices = next(stream)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx[indices]), ty[indices], pos_weight=torch.tensor(50.))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        score = predict_windows(meta, model, vx)
        curve = pr_curve(score, vy >= 200)
        point = constrained_point(curve)
        checkpoint = target / f"epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint)
        pr_path = target / f"epoch_{epoch}_pr.npz"
        np.savez_compressed(pr_path, **curve)
        fixed = state_metrics(score, vy, .5)
        history.append({"epoch": epoch, "bce": loss_sum / 294, "fixed_decision": fixed,
                        "constrained": point, "weights_sha256": sha256(checkpoint), "pr_sha256": sha256(pr_path)})
        rank = checkpoint_rank(point, epoch)
        if best is None or rank < best:
            best = rank
            meta.update(selected_epoch=epoch, decision_threshold_probability=point["threshold"], source_selection=point)
        print(f"{candidate} seed={seed} checkpoint={epoch}/12 source_P={point['precision']:.5f} R={point['recall']:.5f}", flush=True)
    shutil.copyfile(target / f"epoch_{meta['selected_epoch']}.pt", target / "weights.pt")
    meta["weights_sha256"] = sha256(target / "weights.pt")
    write_json(target / "model.json", meta)
    write_json(target / "history.json", {"epochs": history})
    write_json(target / "complete.json", {"artifacts": inventory(target)})


def state_metrics(scores, truth_w, threshold):
    valid = np.isfinite(scores) & np.isfinite(truth_w)
    predicted, actual = scores[valid] >= threshold, truth_w[valid] >= 200
    tp, fp, fn = int((predicted & actual).sum()), int((predicted & ~actual).sum()), int((~predicted & actual).sum())
    return {"valid_samples": int(valid.sum()), "actual_on_samples": int(actual.sum()),
            "predicted_on_samples": int(predicted.sum()), "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "state_pattern": "unavailable" if not valid.any() else "all_off" if not predicted.any() else "all_on" if predicted.all() else "mixed",
            "support_sha256": __import__("hashlib").sha256(valid.tobytes()).hexdigest()}


def _regression_fit(run, seed):
    import torch
    from ..models.artifacts import load_model
    target = run / f"models/R_PR/seed_{seed}"
    settings_path = run / f"settings/R_PR_{seed}.json"
    settings = {**read_json(run / "registration_inputs/base_settings.json"), "seed": seed}
    write_json(settings_path, settings)
    if not target.exists():
        train(run / "protocol/protocol.json", settings_path, target, appliances=["microwave"],
              checkpoint_metric="recall_precision", checkpoint_evidence=True)
    meta, _ = load_model(target)
    if meta["code_sha256"] != source_fingerprint() or meta["protocol_sha256"] != sha256(run / "protocol/protocol.json"):
        raise DataError("Regression resume changed")
    history = read_json(target / "history.json")["epochs"]
    mae_epoch = min(history, key=lambda r: (r["validation_normalized_mae"], r["epoch"]))["epoch"]
    _, original = load_model(run / f"reference/models/seed_{seed}")
    actual = torch.load(target / f"checkpoints/epoch_{mae_epoch}.pt", weights_only=True)
    exact = all(torch.equal(v, actual[k]) for k, v in original.state_dict().items())
    if not exact:
        raise DataError("Old MAE baseline no longer reproduces exactly")
    write_json(target / "mae_reference_check.json", {"passed": exact, "mae_epoch": mae_epoch,
               "precision_epoch": meta["selected_epoch"], "maximum_parameter_difference": 0})


def predict_case(run, candidate, seed, partition):
    from ..models.state_classifier import load_state_model, predict_windows
    from ..models.artifacts import load_model
    from .precision import _predict_values
    data = read_partition(run / "protocol/protocol.json", partition, "inference")
    stamps, aggregate = aggregate_arrays(data["aggregate"], 8)
    if candidate.startswith("C"):
        meta, model = load_state_model(run / f"models/{candidate}/seed_{seed}")
        width, threshold = meta["window_samples"], meta["decision_threshold_probability"]
    else:
        meta, model = load_model(run / f"models/{candidate}/seed_{seed}")
        width, threshold = 31, meta["source_state_selection"]["point"]["threshold"]
    x, centers = windows(stamps, aggregate, width, 8)
    score = np.full(len(stamps), np.nan, np.float32)
    score[centers] = predict_windows(meta, model, x) if candidate.startswith("C") else _predict_values(meta, model, x)[:, 0]
    with np.load(run / f"reference/predictions/seed_{seed}/{partition}.npz", allow_pickle=False) as reference:
        if not np.array_equal(stamps, reference["stamps"]) or not np.array_equal(aggregate, reference["aggregate"], equal_nan=True):
            raise DataError("State inference timeline differs from A")
        power = reference["raw"].copy()
    if candidate == "R_PR":
        power[:, 1] = score
    path = run / f"predictions/{candidate}/seed_{seed}/{partition}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, stamps=stamps, aggregate=aggregate, scores=score, power=power)
    # Check independent public state inference including all missing locations.
    if candidate.startswith("C") and partition == SOURCE_VALIDATION:
        from ..models.state_classifier import predict_state_rows
        rows = data["aggregate"][:2048]
        ordinary = predict_state_rows(meta, model, rows)
        values = np.array([r["probability_on"] if r["probability_on"] is not None else np.nan for r in ordinary])
        end = len(rows) - width // 2
        if (not np.array_equal(np.isfinite(values[:end]), np.isfinite(score[:end]))
                or not np.allclose(values[:end], score[:end], rtol=1e-6, atol=1e-8, equal_nan=True)
                or not np.array_equal(values[:end] >= threshold, score[:end] >= threshold)):
            raise DataError("Independent state inference readback differs")
    return {"candidate": candidate, "seed": seed, "partition": partition, "threshold": threshold,
            "path": path.relative_to(run).as_posix(), "sha256": sha256(path)}


def _truth(run, partition, stamps, consumer):
    data = read_partition(run / "protocol/protocol.json", partition, consumer, include_labels=True)
    return labels_at_centers(data, DEVICES, stamps, np.arange(len(stamps)))


def _common(run, partition):
    with np.load(run / f"predictions/C61/seed_42/{partition}.npz", allow_pickle=False) as a:
        return np.isfinite(a["scores"])


def choose_source(rows):
    if len(rows) != 18 or len({r["support_sha256"] for r in rows}) != 1:
        raise DataError("Source selection requires identical common support")
    results = []
    for group in GROUPS:
        rs = [r for r in rows if r["candidate"] == group]
        if len(rs) != 3 or {r["seed"] for r in rs} != set(SEEDS):
            raise DataError("Source choice needs complete seed matrix")
        eligible = all(r["recall"] is not None and r["recall"] >= .85 and r["state_pattern"] == "mixed" for r in rs)
        results.append({"candidate": group, "eligible": eligible,
                        "mean_precision": float(np.mean([r["precision"] or 0 for r in rs])),
                        "mean_f1": float(np.mean([r["f1"] or 0 for r in rs]))})
    valid = [r for r in results if r["eligible"]]
    selected = max(valid, key=lambda r: (r["mean_precision"], r["mean_f1"], -GROUPS.index(r["candidate"]))) if valid else None
    return {"selected": selected["candidate"] if selected else "C31", "source_gate_passed": bool(valid),
            "selection_data": "source_validation_common_support_only", "candidates": results}


def train_and_freeze(run, resume=False):
    registered(run)
    status = read_json(run / "run.json")["status"]
    if status != "prepared" and not (resume and status in ("training", "failed_training")):
        raise DataError("Completed state runs cannot be retrained")
    write_json(run / "run.json", {"status": "training"})
    manifest = run / "protocol/protocol.json"
    protocol = read_json(manifest)
    validate_protocol(protocol)
    cached = {}
    for width in (31, 61):
        settings = {**read_json(run / "registration_inputs/base_settings.json"), "window_samples": width}
        x, y, counts, keys = dataset(manifest, [p for p in protocol["partitions"] if p["role"] == "source_train"], ["microwave"], settings, 8, True)
        vx, vy, vcounts = dataset(manifest, [p for p in protocol["partitions"] if p["role"] == "source_validation"], ["microwave"], settings, 8)
        cached[width] = (x, y[:, 0], keys, vx, vy[:, 0], counts, vcounts)
    try:
        cases = []
        for candidate in GROUPS:
            for seed in SEEDS:
                print(f"Fitting {candidate} seed={seed}", flush=True)
                if candidate == "R_PR":
                    _regression_fit(run, seed)
                else:
                    _classifier_fit(run, candidate, seed, cached)
                cases.append(predict_case(run, candidate, seed, SOURCE_VALIDATION))
        source_rows = []
        common = _common(run, SOURCE_VALIDATION)
        with np.load(run / cases[0]["path"], allow_pickle=False) as a:
            truth = _truth(run, SOURCE_VALIDATION, a["stamps"], "training")[:, 1]
        for c in cases:
            with np.load(run / c["path"], allow_pickle=False) as a:
                score = a["scores"].copy()
            score[~common] = np.nan
            source_rows.append({"candidate": c["candidate"], "seed": c["seed"], **state_metrics(score, truth, c["threshold"])})
        write_json(run / "source_metrics.json", {"rows": source_rows})
        calibration = {}
        for seed in SEEDS:
            with np.load(run / f"reference/predictions/seed_{seed}/{SOURCE_VALIDATION}.npz", allow_pickle=False) as a:
                calibration[str(seed)] = constrained_point(pr_curve(a["raw"][:, 1], np.where(np.isfinite(truth), truth >= 200, np.nan)))
        write_json(run / "baseline_source_calibration.json", calibration)
        selection = choose_source(source_rows)
        write_json(run / "selection.json", selection)
        cases += [predict_case(run, g, s, DEVELOPMENT) for g in GROUPS for s in SEEDS]
        artifacts = {name + "/" + p: h for name in ("models", "settings", "predictions") for p, h in inventory(run / name).items()}
        artifacts.update({p: sha256(run / p) for p in ("selection.json", "source_metrics.json", "registration.json", "baseline_source_calibration.json")})
        write_json(run / "freeze.json", {"cases": cases, "artifacts": artifacts,
                   "development_labels_parsed": False, "models": 18, "registered_inputs": _inputs(run)})
        write_json(run / "run.json", {"status": "frozen", "freeze_sha256": sha256(run / "freeze.json")})
        return {"status": "frozen", "models": 18, "predictions": 36, "selected": selection["selected"]}
    except Exception as error:
        write_json(run / "run.json", {"status": "failed_training", "error": str(error)})
        raise


def validate_freeze(run):
    registered(run, False)
    state = read_json(run / "run.json")
    frozen = read_json(run / "freeze.json")
    expected = {(g, s, p) for g in GROUPS for s in SEEDS for p in (SOURCE_VALIDATION, DEVELOPMENT)}
    if (sha256(run / "freeze.json") != state["freeze_sha256"] or frozen["development_labels_parsed"] is not False
            or {(c["candidate"], c["seed"], c["partition"]) for c in frozen["cases"]} != expected
            or len(frozen["cases"]) != 36 or frozen["models"] != 18 or frozen["registered_inputs"] != _inputs(run)):
        raise DataError("Incomplete or changed state freeze")
    for p, h in frozen["artifacts"].items():
        if sha256(safe_child(run, p)) != h:
            raise DataError("Frozen state artifact changed: " + p)
    for c in frozen["cases"]:
        if sha256(safe_child(run, c["path"])) != c["sha256"]:
            raise DataError("Prediction receipt changed")
    if choose_source(read_json(run / "source_metrics.json")["rows"]) != read_json(run / "selection.json"):
        raise DataError("Source selection does not reproduce")
    verify_checkpoints(run)
    return frozen


def verify_checkpoints(run):
    """Reconstruct source-only checkpoint and threshold decisions without labels."""
    for candidate in GROUPS:
        for seed in SEEDS:
            target = run / f"models/{candidate}/seed_{seed}"
            meta = read_json(target / "model.json")
            history = read_json(target / "history.json")["epochs"]
            if len(history) != 12 or [r["epoch"] for r in history] != list(range(1, 13)):
                raise DataError("Checkpoint matrix is incomplete")
            points = []
            for row in history:
                epoch = row["epoch"]
                checkpoint = target / (f"checkpoints/epoch_{epoch}.pt" if candidate == "R_PR" else f"epoch_{epoch}.pt")
                pr_path = target / (f"checkpoints/epoch_{epoch}_microwave_pr.npz" if candidate == "R_PR" else f"epoch_{epoch}_pr.npz")
                expected = row["state_evidence"][0]["constrained"] if candidate == "R_PR" else row["constrained"]
                expected_hash = row["checkpoint_sha256"] if candidate == "R_PR" else row["weights_sha256"]
                if sha256(checkpoint) != expected_hash:
                    raise DataError("Checkpoint history hash changed")
                with np.load(pr_path, allow_pickle=False) as curve:
                    point = constrained_point(curve)
                if point != expected:
                    raise DataError("Source PR constrained point does not reproduce")
                points.append((checkpoint_rank(point, epoch), epoch, point))
            _, epoch, point = min(points)
            actual_point = meta["source_state_selection"]["point"] if candidate == "R_PR" else meta["source_selection"]
            if epoch != meta["selected_epoch"] or point != actual_point:
                raise DataError("Selected source checkpoint does not reproduce")
            if candidate != "R_PR" and point["threshold"] != meta["decision_threshold_probability"]:
                raise DataError("State probability decision changed")


def gate(rows, candidate, support):
    from .precision import development_gate
    rs = [{**r, "view": "raw"} for r in rows if r["support"] == support and r["candidate"] in ("W50", candidate)]
    plan = {"minimum_precision": .08, "minimum_recall": .85, "minimum_f1_gain": .02,
            "maximum_seed_f1_drop": .02, "minimum_positive_seeds": 2, "minimum_fp_reduction": .30}
    return development_gate(rs, candidate, plan)


def evaluate(run):
    registered(run)
    state = read_json(run / "run.json")
    if state["status"] != "frozen" or (run / "metrics.json").exists():
        raise DataError("Evaluation requires complete unexamined state freeze")
    frozen = validate_freeze(run)
    rows, segments, curves, support_rows = [], [], [], []
    # This is the first development label access, after the full matrix hash check.
    for partition in (SOURCE_VALIDATION, DEVELOPMENT):
        common = _common(run, partition)
        with np.load(run / frozen["cases"][0]["path"].replace(SOURCE_VALIDATION, partition), allow_pickle=False) as a:
            stamps = a["stamps"].copy()
            truth = _truth(run, partition, stamps, "evaluation")
        for seed in SEEDS:
            with np.load(run / f"reference/predictions/seed_{seed}/{partition}.npz", allow_pickle=False) as b:
                baseline_power, aggregate = b["raw"].copy(), b["aggregate"].copy()
            for candidate in ["W50", "W50_THR", "R_FIXED"] + GROUPS:
                if candidate in ("W50", "W50_THR"):
                    power, score, threshold = baseline_power.copy(), baseline_power[:, 1].copy(), 200
                    if candidate == "W50_THR":
                        threshold = read_json(run / "baseline_source_calibration.json")[str(seed)]["threshold"]
                else:
                    c = next(c for c in frozen["cases"] if (c["candidate"], c["seed"], c["partition"]) == ("R_PR" if candidate == "R_FIXED" else candidate, seed, partition))
                    with np.load(run / c["path"], allow_pickle=False) as a:
                        power, score = a["power"].copy(), a["scores"].copy()
                    threshold = c["threshold"]
                    if candidate == "R_FIXED":
                        threshold = 200
                if not np.array_equal(power[:, [0, 2]], baseline_power[:, [0, 2]], equal_nan=True):
                    raise DataError("Other-device raw power changed")
                for support in ("native", "common61"):
                    allowed = np.isfinite(score) & (common if support == "common61" else True)
                    support_rows.append({"partition": partition, "seed": seed, "candidate": candidate,
                                         "support": support, "aggregate_valid_centers": int(allowed.sum()),
                                         "observed_microwave_centers": int((allowed & np.isfinite(truth[:, 1])).sum()),
                                         "positive_centers": int((allowed & (truth[:, 1] >= 200)).sum()),
                                         "threshold_segments": int(np.sum((allowed & (truth[:, 1] >= 200)) &
                                             ~np.r_[False, (allowed & (truth[:, 1] >= 200))[:-1] & (np.diff(stamps) == 8)]))})
                    for i, appliance in enumerate(DEVICES):
                        p = power[:, i].copy()
                        p[~allowed] = np.nan
                        metrics = compact_metrics(p, truth[:, i], {"kettle": 2000, "microwave": 200, "washing_machine": 20}[appliance])
                        if appliance == "microwave":
                            s = score.copy()
                            s[~allowed] = np.nan
                            metrics.update(state_metrics(s, truth[:, i], threshold))
                        rows.append({"candidate": candidate, "seed": seed, "partition": partition, "appliance": appliance,
                                     "support": support, "decision_threshold": threshold if i == 1 else {0: 2000, 2: 20}[i],
                                     "power_source": "selected_regression" if candidate in ("R_PR", "R_FIXED") and i == 1 else "independent_A",
                                     **metrics})
                if partition == DEVELOPMENT:
                    _errors(segments, curves, candidate, seed, stamps, aggregate, truth[:, 1], score, threshold, baseline_power[:, 1], common)
    write_json(run / "metrics.json", {"freeze_sha256": sha256(run / "freeze.json"), "rows": rows})
    _write_csv(run / "comparison.csv", rows)
    _write_csv(run / "support.csv", support_rows)
    _write_csv(run / "error_segments.csv", segments)
    _write_csv(run / "error_curves.csv", curves)
    selected = read_json(run / "selection.json")["selected"]
    summary = {"selected": selected, "common_support_gates": {g: gate(rows, g, "common61") for g in GROUPS},
               "native_support_gates": {g: gate(rows, g, "native") for g in GROUPS if g != "C61"},
               "selection_diagnostics": {g: gate(rows, g, "common61") for g in ("W50_THR", "R_FIXED")},
               "other_devices_raw_unchanged": True, "final_households_accessed": False, "default_model_changed": False,
               "new_development_household_check": "not_completed_no_additional_designated_household",
               "formal_evaluation": False, "confirmation_budget": 0}
    write_json(run / "summary.json", summary)
    table = "".join("<tr>" + "".join(f"<td>{escape(str(r[k]))}</td>" for k in
                    ("candidate", "seed", "precision", "recall", "f1", "false_positive", "false_negative", "mae_w", "active_mae_w")) + "</tr>"
                    for r in rows if r["partition"] == DEVELOPMENT and r["support"] == "common61" and r["appliance"] == "microwave")
    (run / "report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:32px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>微波炉状态分类受控实验</h1>'
        + '<p>House17已查看开发集；来源选择先于统一评价。真值≥200 W；分类决策为来源选定概率阈值，功率独立输出。最终8/9/20未访问。</p><p>来源预选：' + selected
        + '</p><table><tr><th>候选</th><th>种子</th><th>P</th><th>R</th><th>F1</th><th>FP</th><th>FN</th><th>MAE W</th><th>开机MAE W</th></tr>' + table
        + '</table><p><a href="comparison.csv">逐设备/种子/支持指标</a> · <a href="support.csv">正例与连续段支持</a> · <a href="error_segments.csv">错误段</a> · <a href="error_curves.csv">对照曲线</a></p><pre>'
        + escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
    outputs = ("metrics.json", "comparison.csv", "support.csv", "error_segments.csv", "error_curves.csv", "summary.json", "report.html")
    write_json(run / "evidence.json", {"artifacts": {p: sha256(run / p) for p in outputs}})
    write_json(run / "run.json", {"status": "evaluated", "freeze_sha256": sha256(run / "freeze.json"), "evidence_sha256": sha256(run / "evidence.json")})
    return summary


def _errors(segments, curves, candidate, seed, stamps, aggregate, truth, score, threshold, baseline, common):
    valid = common & np.isfinite(truth) & np.isfinite(score)
    active, predicted = truth >= 200, score >= threshold
    for name, mask in (("FP", valid & predicted & ~active), ("FN", valid & ~predicted & active), ("TP", valid & predicted & active)):
        indices = np.flatnonzero(mask)
        runs = np.split(indices, np.flatnonzero((np.diff(indices) != 1) | (np.diff(stamps[indices]) != 8)) + 1)
        for rank, group in enumerate(sorted((r for r in runs if len(r)), key=lambda r: (-len(r), int(r[0])))[:3]):
            identity = f"{candidate}_{seed}_{name}_{rank}"
            segments.append({"id": identity, "candidate": candidate, "seed": seed, "type": name,
                             "start": iso(int(stamps[group[0]])), "end": iso(int(stamps[group[-1]] + 8)),
                             "samples": len(group), "selection": "longest_threshold_runs_post_evaluation"})
            center = group[len(group)//2]
            for i in range(max(0, center-30), min(len(stamps), center+31)):
                curves.append({"id": identity, "timestamp": iso(int(stamps[i])),
                               **{k: float(v) if np.isfinite(v) else "" for k, v in
                                  {"aggregate_w": aggregate[i], "truth_w": truth[i], "candidate_score": score[i],
                                   "decision_threshold": threshold, "baseline_w": baseline[i]}.items()}})


def verify(run):
    frozen = validate_freeze(run)
    state = read_json(run / "run.json")
    if state["status"] == "evaluated":
        if sha256(run / "evidence.json") != state["evidence_sha256"]:
            raise DataError("State evidence changed")
        for p, h in read_json(run / "evidence.json")["artifacts"].items():
            if sha256(safe_child(run, p)) != h:
                raise DataError("State evaluation artifact changed")
        metrics = read_json(run / "metrics.json")
        if metrics["freeze_sha256"] != sha256(run / "freeze.json"):
            raise DataError("Metrics belong to another freeze")
        summary = read_json(run / "summary.json")
        for g in GROUPS:
            if summary["common_support_gates"][g] != gate(metrics["rows"], g, "common61"):
                raise DataError("State gate does not reproduce")
    elif state["status"] != "frozen":
        raise DataError("State run incomplete")
    return {"status": "state_verified", "models": frozen["models"], "predictions": len(frozen["cases"]),
            "target_labels_reopened": False, "final_households_accessed": False,
            "execution_source_matches_current": source_fingerprint() == read_json(run / "registration.json")["code_sha256"]}
