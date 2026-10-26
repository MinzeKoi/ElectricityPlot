"""Registered microwave loss/negative-weight experiments on inspected development data.

Source selection is sealed before development labels are opened. Historical A
data, weights, and other-device predictions are copied, never edited in place.
"""

import csv
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import shutil

import numpy as np

from ..data.common import DataError, iso, read_json, safe_child, sha256, source_fingerprint, write_json
from ..data.protocol import read_partition
from ..models.arrays import aggregate_arrays, labels_at_centers, windows
from ..models.artifacts import load_model
from ..models.training import dataset, train
from .data_expansion import compact_metrics, inventory

ROOT = Path(__file__).resolve().parents[3]
SEEDS = [42, 1019, 2026]
DEVICES = ["kettle", "microwave", "washing_machine"]
SOURCE_VALIDATION = "source_validation_refit_5_0"
DEVELOPMENT = "frozen_test_refit_17_0"


def load_plan(path):
    plan = read_json(path)
    fixed = {"schema_version": 1, "source_group": "A", "seeds": SEEDS,
             "loss_weights": [50, 10, 1], "hard_negative_factor": 4,
             "source_recall_drop_limit": .05, "minimum_precision": .08,
             "minimum_recall": .85, "minimum_f1_gain": .02,
             "maximum_seed_f1_drop": .02, "minimum_positive_seeds": 2,
             "minimum_fp_reduction": .30, "maximum_models": 12}
    if (set(plan) != set(fixed) | {"experiment_id", "parent_run"}
            or any(plan.get(k) != v for k, v in fixed.items())
            or not isinstance(plan["experiment_id"], str) or not plan["experiment_id"]
            or not isinstance(plan["parent_run"], str)):
        raise DataError("Precision plan differs from the registered finite experiment")
    return plan


def validate_protocol(protocol):
    expected = {"source_train_refit_2_0": ("source_train", "refit_2"),
                "source_train_refit_19_0": ("source_train", "refit_19"),
                SOURCE_VALIDATION: ("source_validation", "refit_5"),
                DEVELOPMENT: ("frozen_test", "refit_17")}
    actual = {p["partition_id"]: (p["role"], p["household_id"]) for p in protocol["partitions"]}
    if (actual != expected or len(protocol["partitions"]) != len(expected)
            or protocol.get("purpose") != "engineering_pilot_only"
            or protocol.get("accuracy_usage") != "development_only"
            or protocol.get("sample_interval_seconds") != 8
            or protocol.get("target_appliances") != DEVICES
            or protocol.get("split_gap_seconds") != 3600
            or protocol.get("expansion_training", {}).get("steps_per_checkpoint") != 587):
        raise DataError("Precision experiment must retain A source/validation/development roles and support")


def _immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise DataError("Saved experiment artifact changed: " + path.name)
    else:
        write_json(path, value)


def _inputs(run):
    result = {"plan.json": sha256(run / "plan.json"), "base_settings.json": sha256(run / "base_settings.json")}
    for name in ("protocol", "reference", "source", "entrypoints"):
        result.update({name + "/" + p: digest for p, digest in inventory(run / name).items()})
    return result


def _registered(run, current=True):
    registration = read_json(run / "registration.json")
    if current and registration["code_sha256"] != source_fingerprint():
        raise DataError("Precision source changed; use its saved source or create a new experiment")
    if _inputs(run) != registration["inputs"]:
        raise DataError("Registered precision data, references or configuration changed")
    return load_plan(run / "plan.json")


def prepare(run, plan_path):
    plan = load_plan(plan_path)
    if run.exists():
        raise DataError("Precision output exists; choose a new directory")
    parent = safe_child(ROOT, plan["parent_run"])
    protocol = read_json(parent / "protocols/A/protocol.json")
    validate_protocol(protocol)  # Reject role changes before opening any data artifact.
    settings = read_json(parent / "settings.json")
    fixed = {"window_samples": 31, "channels": 8, "hidden_units": 32, "epochs": 12,
             "batch_size": 256, "learning_rate": .001, "active_loss_weight": 50.0,
             "on_thresholds_w": {"kettle": 2000.0, "microwave": 200.0, "washing_machine": 20.0}}
    if any(settings.get(k) != value for k, value in fixed.items()):
        raise DataError("A training recipe differs from the registered precision baseline")
    parent_seal = read_json(parent / "evidence_refit.json")["artifacts"]
    frozen = read_json(parent / "freeze_refit.json")
    required = [p.relative_to(parent).as_posix() for p in (parent / "protocols/A").rglob("*") if p.is_file()]
    required += [f"models/A/seed_{seed}/microwave/{name}" for seed in SEEDS
                 for name in ("model.json", "weights.pt")]
    for relative in required:
        if relative not in parent_seal or sha256(safe_child(parent, relative)) != parent_seal[relative]:
            raise DataError("Historical A evidence changed: " + relative)
    run.mkdir(parents=True)
    try:
        shutil.copyfile(plan_path, run / "plan.json")
        write_json(run / "base_settings.json", settings)
        shutil.copytree(parent / "protocols/A", run / "protocol")
        protocol["parent_protocol_sha256"] = sha256(parent / "protocols/A/protocol.json")
        protocol["protocol_id"] = plan["experiment_id"]
        protocol["data_kind"] = "real_REFIT_microwave_precision_development"
        write_json(run / "protocol/protocol.json", protocol)
        for seed in SEEDS:
            shutil.copytree(parent / f"models/A/seed_{seed}/microwave", run / f"reference/models/seed_{seed}")
            for partition in (SOURCE_VALIDATION, DEVELOPMENT):
                receipt = next(r for r in frozen["predictions"] if
                               r["group"] == "A" and r["seed"] == seed and r["partition_id"] == partition)
                path = safe_child(parent, receipt["path"])
                if sha256(path) != receipt["sha256"]:
                    raise DataError("Historical A prediction changed")
                dest = run / f"reference/predictions/seed_{seed}/{partition}.npz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
        shutil.copytree(ROOT / "src/virtual_submeter", run / "source/virtual_submeter",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (run / "entrypoints").mkdir()
        shutil.copyfile(ROOT / "scripts/run_precision.py", run / "entrypoints/run_precision.py")
        write_json(run / "registration.json", {"experiment_id": plan["experiment_id"],
                   "registered_at": datetime.now(timezone.utc).isoformat(), "code_sha256": source_fingerprint(),
                   "parent_evidence_sha256": sha256(parent / "evidence_refit.json"), "inputs": _inputs(run),
                   "final_households_accessed": False, "development_labels_parsed": False})
        write_json(run / "run.json", {"status": "prepared", "formal_evaluation": False})
    except Exception as error:
        write_json(run / "run.json", {"status": "failed_preparation", "error": str(error)})
        raise
    return {"status": "prepared", "run": str(run)}


def choose_source(rows, plan, groups):
    """Relative recall guards are per seed; absolute 85% is a development gate."""
    keyed = {(r["candidate"], r["seed"]): r for r in rows}
    if len(keyed) != len(rows) or any((g, s) not in keyed for g in groups for s in SEEDS):
        raise DataError("Source selection needs the complete three-seed matrix")
    baseline = [keyed["W50", s] for s in SEEDS]
    results = []
    for group in groups:
        current = [keyed[group, s] for s in SEEDS]
        defined = all(r.get(k) is not None for r in current + baseline for k in ("precision", "recall", "f1"))
        complete = defined and all(r["state_pattern"] == "mixed" and r["support_sha256"] == b["support_sha256"]
                                   for r, b in zip(current, baseline))
        mean_p = float(np.mean([r["precision"] for r in current])) if defined else None
        mean_f1 = float(np.mean([r["f1"] for r in current])) if defined else None
        passed = (complete and mean_f1 >= np.mean([b["f1"] for b in baseline])
                  and all(r["recall"] >= b["recall"] - plan["source_recall_drop_limit"] for r, b in zip(current, baseline))
                  and sum(r["precision"] > b["precision"] for r, b in zip(current, baseline)) >= plan["minimum_positive_seeds"])
        results.append({"candidate": group, "eligible": bool(passed), "mean_precision": mean_p, "mean_f1": mean_f1})
    # Explicit stable order: parent before its hard-negative variant, W50/W10/W1.
    priority = [g for w in (50, 10, 1) for g in (f"W{w}", f"W{w}_HN")]
    eligible = [r for r in results if r["eligible"]]
    selected = max(eligible, key=lambda r: (r["mean_precision"], r["mean_f1"], -priority.index(r["candidate"]))) if eligible else None
    return {"selected": selected["candidate"] if selected else "W50", "candidate_passed_source_gate": bool(eligible),
            "selection_data": "source_validation_only", "candidates": results}


def development_gate(rows, selected, plan):
    micro = [r for r in rows if r["partition"] == DEVELOPMENT and r["appliance"] == "microwave" and r["view"] == "raw"]
    keyed = {(r["candidate"], r["seed"]): r for r in micro}
    if len(keyed) != len(micro) or any((g, s) not in keyed for g in ("W50", selected) for s in SEEDS):
        raise DataError("Development gate needs every paired seed")
    baseline, current = [keyed["W50", s] for s in SEEDS], [keyed[selected, s] for s in SEEDS]
    defined = all(r.get(k) is not None for r in baseline + current for k in ("precision", "recall", "f1"))
    checks = {"defined_metrics": defined,
              "same_support": all(r["support_sha256"] == b["support_sha256"] for r, b in zip(current, baseline)),
              "nondegenerate": all(r["state_pattern"] == "mixed" for r in current)}
    checks.update(mean_precision=bool(defined and np.mean([r["precision"] for r in current]) >= plan["minimum_precision"]),
                  every_seed_recall=bool(defined and all(r["recall"] >= plan["minimum_recall"] for r in current)),
                  mean_f1_gain=bool(defined and np.mean([r["f1"] - b["f1"] for r, b in zip(current, baseline)]) >= plan["minimum_f1_gain"]),
                  paired_positive_seeds=bool(defined and sum(r["precision"] > b["precision"] and r["f1"] > b["f1"]
                                                           for r, b in zip(current, baseline)) >= plan["minimum_positive_seeds"]),
                  bounded_seed_f1_drop=bool(defined and all(r["f1"] >= b["f1"] - plan["maximum_seed_f1_drop"] for r, b in zip(current, baseline))),
                  false_positive_reduction=bool(np.mean([r["false_positive"] for r in current]) <=
                                                (1 - plan["minimum_fp_reduction"]) * np.mean([b["false_positive"] for b in baseline])))
    means = {k: float(np.mean([r[k] for r in current])) if all(r.get(k) is not None for r in current) else None
             for k in ("precision", "recall", "f1", "false_positive", "false_negative")}
    return {"selected_before_development_evaluation": selected, "precision_exit_passed": all(checks.values()),
            "checks": checks, "selected_means": means,
            "baseline_means": {k: float(np.mean([r[k] for r in baseline])) if defined else None for k in means},
            "formal_evaluation": False, "final_households_accessed": False, "default_model_changed": False}


def _predict_values(metadata, model, x):
    import torch
    torch.set_num_threads(metadata["settings"]["cpu_threads"])
    tx = torch.from_numpy((np.log1p(x) - metadata["input_log_mean"]) / metadata["input_log_std"])
    with torch.inference_mode():
        values = np.concatenate([model(batch).numpy() for batch in tx.split(metadata["settings"]["batch_size"])])
    return (values * np.asarray(metadata["power_scales_w"])).astype(np.float32)


def _predict(run, group, seed, partition):
    manifest = run / "protocol/protocol.json"
    metadata, model = load_model(run / f"models/{group}/seed_{seed}")
    data = read_partition(manifest, partition, "inference")  # aggregate-only
    stamps, aggregate = aggregate_arrays(data["aggregate"], 8)
    x, centers = windows(stamps, aggregate, 31, 8)
    micro = np.full(len(stamps), np.nan)
    micro[centers] = _predict_values(metadata, model, x)[:, 0]
    with np.load(run / f"reference/predictions/seed_{seed}/{partition}.npz", allow_pickle=False) as a:
        if not np.array_equal(stamps, a["stamps"]) or not np.array_equal(aggregate, a["aggregate"], equal_nan=True):
            raise DataError("Other-device reference uses different input support")
        raw = a["raw"].copy()
        if not np.array_equal(np.isfinite(raw[:, 1]), np.isfinite(micro)):
            raise DataError("Microwave prediction changed valid input support")
        if group == "W50" and (not np.allclose(raw[:, 1], micro, rtol=1e-6, atol=2e-4, equal_nan=True)
                                or not np.array_equal(raw[:, 1] >= 200, micro >= 200)):
            raise DataError("Reproduced W50 prediction differs from historical A")
    raw[:, 1] = micro
    total = raw.sum(axis=1)
    factor = np.minimum(1, np.divide(aggregate, total, out=np.ones_like(total), where=total > 0))
    target = run / f"predictions/{group}/seed_{seed}/{partition}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if any(not np.array_equal(saved[k], v, equal_nan=True) for k, v in
                   {"stamps": stamps, "aggregate": aggregate, "raw": raw, "corrected": raw * factor[:, None]}.items()):
                raise DataError("Resumed prediction differs")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, stamps=stamps, aggregate=aggregate, raw=raw, corrected=raw * factor[:, None])
    return {"candidate": group, "seed": seed, "partition": partition,
            "path": target.relative_to(run).as_posix(), "sha256": sha256(target),
            "model_sha256": sha256(run / f"models/{group}/seed_{seed}/model.json")}


def _truth(run, partition, arrays, consumer, cache):
    if partition not in cache:
        data = read_partition(run / "protocol/protocol.json", partition, consumer, include_labels=True)
        stamps = arrays["stamps"]
        cache[partition] = labels_at_centers(data, DEVICES, stamps, np.arange(len(stamps)))
    return cache[partition]


def _metrics(run, case, consumer, cache):
    with np.load(safe_child(run, case["path"]), allow_pickle=False) as arrays:
        y = _truth(run, case["partition"], arrays, consumer, cache)
        thresholds = read_json(run / "base_settings.json")["on_thresholds_w"]
        return [{"candidate": case["candidate"], "seed": case["seed"], "partition": case["partition"],
                 "appliance": appliance, "view": view,
                 **compact_metrics(arrays[view][:, i], y[:, i], thresholds[appliance])}
                for i, appliance in enumerate(DEVICES) for view in ("raw", "corrected")]


def _fit(run, group, seed, weight, negative=None):
    settings = {**read_json(run / "base_settings.json"), "active_loss_weight": float(weight), "seed": seed}
    settings_path = run / f"settings/{group}_{seed}.json"
    _immutable_json(settings_path, settings)
    target = run / f"models/{group}/seed_{seed}"
    if not target.exists():
        train(run / "protocol/protocol.json", settings_path, target, appliances=["microwave"], negative_weights_path=negative)
    meta, model = load_model(target)
    if (meta["protocol_sha256"] != sha256(run / "protocol/protocol.json")
            or meta["settings_sha256"] != sha256(settings_path) or meta["code_sha256"] != source_fingerprint()
            or meta.get("optimizer_steps") != 7044 or meta["appliances"] != ["microwave"]
            or meta.get("negative_sample_weighting", {}).get("plan_sha256") != (sha256(negative) if negative else None)):
        raise DataError("Resumed precision model differs from the registered recipe")
    if meta.get("state_threshold_view") != "raw_power_w":
        from .baseline import align_state
        align_state(target)
        meta = read_json(target / "model.json")
    if group == "W50":
        import torch
        original, old = load_model(run / f"reference/models/seed_{seed}")
        for name, value in old.state_dict().items():
            if not torch.equal(value, model.state_dict()[name]):
                raise DataError("W50 baseline failed exact parameter reproduction")
        for key in ("input_log_mean", "input_log_std", "power_scales_w", "selected_epoch", "source_train_windows"):
            if meta[key] != original[key]:
                raise DataError("W50 baseline metadata differs: " + key)
        _immutable_json(run / f"reference_checks/seed_{seed}.json", {"passed": True, "maximum_weight_difference": 0,
                        "historical_model_sha256": sha256(run / f"reference/models/seed_{seed}/model.json")})


def _mine(run, group, seed):
    manifest = run / "protocol/protocol.json"
    protocol = read_json(manifest)
    metadata, model = load_model(run / f"models/{group}/seed_{seed}")
    parts = [p for p in protocol["partitions"] if p["role"] == "source_train"]
    x, y, _, keys = dataset(manifest, parts, ["microwave"], metadata["settings"], 8, return_keys=True)
    predicted = _predict_values(metadata, model, x)[:, 0]
    indices = np.flatnonzero(np.isfinite(y[:, 0]) & (y[:, 0] < 200) & (predicted >= 200))
    selected = [{"household_id": keys[i][0], "timestamp": iso(keys[i][1])} for i in indices]
    plan = {"schema_version": 1, "appliance": "microwave", "factor": 4,
            "protocol_sha256": sha256(manifest), "source_model_sha256": sha256(run / f"models/{group}/seed_{seed}/model.json"),
            "source_candidate": group, "seed": seed, "mining_labels": "source_train_only",
            "windows": selected, "counts_by_household": {p["household_id"]: sum(w["household_id"] == p["household_id"] for w in selected) for p in parts},
            "empty_pool_keeps_ordinary_weights": not bool(selected)}
    target = run / f"negative_windows/seed_{seed}.json"
    _immutable_json(target, plan)
    return target


def _readback(run, group, seed, case):
    from ..models.prediction import predict_rows
    metadata, model = load_model(run / f"models/{group}/seed_{seed}")
    data = read_partition(run / "protocol/protocol.json", SOURCE_VALIDATION, "inference")
    rows = data["aggregate"][:2048]
    ordinary = predict_rows(metadata, model, rows)
    actual = np.asarray([r["appliances"]["microwave"]["raw_power_w"] if r["appliances"]["microwave"]["raw_power_w"] is not None else np.nan for r in ordinary])
    with np.load(safe_child(run, case["path"]), allow_pickle=False) as arrays:
        expected = arrays["raw"][:len(rows), 1]
        actual, expected = actual[:-15], expected[:-15]  # truncated input has no right context at its end
        if not np.array_equal(np.isfinite(actual), np.isfinite(expected)) or not np.allclose(actual, expected, rtol=1e-6, atol=2e-4, equal_nan=True):
            raise DataError("Ordinary microwave inference differs from compact experiment prediction")
        if not np.array_equal(actual >= 200, expected >= 200):
            raise DataError("Ordinary inference state differs")
        difference = float(np.nanmax(np.abs(actual - expected)))
    return {"candidate": group, "seed": seed, "rows": len(actual), "maximum_raw_power_difference_w": difference, "passed": True}


def train_and_freeze(run, resume=False):
    plan = _registered(run)
    state = read_json(run / "run.json")
    if state["status"] not in ("prepared", "training", "failed_training") or (state["status"] != "prepared" and not resume):
        raise DataError("Use resume only for an incomplete unchanged precision run; completed runs are immutable")
    write_json(run / "run.json", {"status": "training", "formal_evaluation": False})
    try:
        source_rows, source_cases, cache = [], [], {}
        # All three W50 references must reproduce before changing a loss weight.
        for weight in plan["loss_weights"]:
            group = f"W{weight}"
            for seed in SEEDS:
                _fit(run, group, seed, weight)
                case = _predict(run, group, seed, SOURCE_VALIDATION)
                source_cases.append(case)
                source_rows.extend(r for r in _metrics(run, case, "training", cache) if r["appliance"] == "microwave" and r["view"] == "raw")
        stage_one = choose_source(source_rows, plan, ["W50", "W10", "W1"])
        _immutable_json(run / "stage_one_selection.json", stage_one)
        parent = stage_one["selected"]
        hard_group = parent + "_HN"
        for seed in SEEDS:
            negative = _mine(run, parent, seed)
            _fit(run, hard_group, seed, int(parent[1:]), negative)
            case = _predict(run, hard_group, seed, SOURCE_VALIDATION)
            source_cases.append(case)
            source_rows.extend(r for r in _metrics(run, case, "training", cache) if r["appliance"] == "microwave" and r["view"] == "raw")
        groups = ["W50", "W10", "W1", hard_group]
        selection = choose_source(source_rows, plan, groups)
        _immutable_json(run / "source_metrics.json", {"rows": source_rows})
        _immutable_json(run / "selection.json", selection)
        readbacks = [_readback(run, c["candidate"], c["seed"], c) for c in source_cases]
        _immutable_json(run / "readback.json", {"checks": readbacks, "all_passed": all(r["passed"] for r in readbacks)})
        cases = source_cases + [_predict(run, group, seed, DEVELOPMENT) for group in groups for seed in SEEDS]
        frozen = {"groups": groups, "cases": cases, "model_count": len(groups) * len(SEEDS),
                  "selection_sha256": sha256(run / "selection.json"), "development_labels_parsed": False,
                  "artifacts": {p.relative_to(run).as_posix(): sha256(p) for p in sorted(run.rglob("*"))
                                if p.is_file() and p.relative_to(run).parts[0] not in ("run.json", "freeze.json")}}
        _immutable_json(run / "freeze.json", frozen)
        write_json(run / "run.json", {"status": "frozen", "freeze_sha256": sha256(run / "freeze.json"), "formal_evaluation": False})
    except Exception as error:
        write_json(run / "run.json", {"status": "failed_training", "error": str(error), "formal_evaluation": False})
        raise
    return {"status": "frozen", "models": 12, "selected": selection["selected"], "development_labels_parsed": False}


def validate_frozen(run):
    frozen = read_json(run / "freeze.json")
    selection = read_json(run / "selection.json")
    first = read_json(run / "stage_one_selection.json")
    groups = ["W50", "W10", "W1", first["selected"] + "_HN"]
    expected = {(g, s, p) for g in groups for s in SEEDS for p in (SOURCE_VALIDATION, DEVELOPMENT)}
    actual = {(c["candidate"], c["seed"], c["partition"]) for c in frozen["cases"]}
    if (frozen["groups"] != groups or frozen["model_count"] != 12 or actual != expected
            or len(frozen["cases"]) != len(expected) or frozen["development_labels_parsed"] is not False
            or sha256(run / "selection.json") != frozen["selection_sha256"] or selection["selected"] not in groups):
        raise DataError("Incomplete precision freeze matrix or changed source selection")
    for relative, digest in frozen["artifacts"].items():
        if sha256(safe_child(run, relative)) != digest:
            raise DataError("Frozen precision artifact changed: " + relative)
    for case in frozen["cases"]:
        if sha256(safe_child(run, case["path"])) != case["sha256"] or sha256(run / f"models/{case['candidate']}/seed_{case['seed']}/model.json") != case["model_sha256"]:
            raise DataError("Frozen prediction or model receipt changed")
    return frozen


def _write_csv(path, rows):
    if not rows:
        raise DataError("Cannot publish an empty precision table")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _check_other_devices(run, cases):
    for case in cases:
        with np.load(safe_child(run, case["path"]), allow_pickle=False) as a, np.load(
                run / f"reference/predictions/seed_{case['seed']}/{case['partition']}.npz", allow_pickle=False) as b:
            if not np.array_equal(a["stamps"], b["stamps"]):
                raise DataError("Reference timeline changed")
            for column in (0, 2):
                if not np.array_equal(a["raw"][:, column], b["raw"][:, column], equal_nan=True):
                    raise DataError("Other-device raw predictions changed")
    return True


def evaluate(run):
    plan = _registered(run)
    state = read_json(run / "run.json")
    if state["status"] != "frozen" or (run / "metrics.json").exists():
        raise DataError("Precision evaluation needs a complete unexamined freeze; completed results are immutable")
    if sha256(run / "freeze.json") != state["freeze_sha256"]:
        raise DataError("Precision freeze changed")
    frozen = validate_frozen(run)  # Complete matrix and hashes BEFORE reading development labels.
    others = _check_other_devices(run, frozen["cases"])
    write_json(run / "run.json", {"status": "evaluating", "freeze_sha256": sha256(run / "freeze.json"), "formal_evaluation": False})
    try:
        return _evaluate_complete(run, plan, frozen, others)
    except Exception as error:
        write_json(run / "run.json", {"status": "failed_evaluation", "freeze_sha256": sha256(run / "freeze.json"),
                   "error": str(error), "formal_evaluation": False})
        raise


def _evaluate_complete(run, plan, frozen, others):
    rows, cache = [], {}
    for case in frozen["cases"]:
        rows.extend(_metrics(run, case, "evaluation", cache))
    summary = development_gate(rows, read_json(run / "selection.json")["selected"], plan)
    summary["other_devices_raw_unchanged"] = others
    summary["precision_exit_passed"] &= others
    write_json(run / "metrics.json", {"freeze_sha256": sha256(run / "freeze.json"), "rows": rows})
    write_json(run / "summary.json", summary)
    _write_csv(run / "comparison.csv", rows)
    _error_examples(run, frozen, cache[DEVELOPMENT])
    report(run)
    names = ["metrics.json", "summary.json", "comparison.csv", "error_segments.csv", "error_curves.csv", "report.html"]
    write_json(run / "evidence.json", {"artifacts": {name: sha256(run / name) for name in names}})
    write_json(run / "run.json", {"status": "evaluated", "freeze_sha256": sha256(run / "freeze.json"),
               "evidence_sha256": sha256(run / "evidence.json"), "formal_evaluation": False})
    return {"status": "evaluated", "metric_rows": len(rows), **summary}


def _error_examples(run, frozen, truth):
    selected = read_json(run / "selection.json")["selected"]
    segments, curves = [], []
    for seed in SEEDS:
        paths = {g: next(c["path"] for c in frozen["cases"] if c["candidate"] == g and c["seed"] == seed and c["partition"] == DEVELOPMENT)
                 for g in {"W50", selected}}
        loaded = {}
        for group, path in paths.items():
            with np.load(safe_child(run, path), allow_pickle=False) as a:
                loaded[group] = {k: a[k].copy() for k in a.files}
        base, chosen = loaded["W50"], loaded[selected]
        stamps, y = base["stamps"], truth[:, 1]
        for group, arrays in loaded.items():
            p = arrays["raw"][:, 1]
            valid = np.isfinite(p) & np.isfinite(y)
            masks = {"false_positive": valid & (p >= 200) & (y < 200),
                     "true_positive": valid & (p >= 200) & (y >= 200),
                     "false_negative": valid & (p < 200) & (y >= 200)}
            for kind, mask in masks.items():
                indices = np.flatnonzero(mask)
                breaks = np.flatnonzero((np.diff(indices) != 1) | (np.diff(stamps[indices]) != 8)) + 1
                runs = np.split(indices, breaks) if len(indices) else []
                top = sorted(runs, key=lambda r: (-len(r), int(stamps[r[0]])))[:3]
                for rank, indices in enumerate(top, 1):
                    identity = f"{group}-{seed}-{kind}-{rank}"
                    segments.append({"segment_id": identity, "candidate": group, "seed": seed, "error_type": kind,
                                     "start": iso(int(stamps[indices[0]])), "end": iso(int(stamps[indices[-1]]) + 8),
                                     "samples": len(indices), "selection": "longest_contiguous_runs_post_evaluation_only"})
                    center = int(indices[len(indices) // 2])
                    for i in range(max(0, center - 60), min(len(stamps), center + 61)):
                        values = {"truth_w": y[i], "baseline_w": base["raw"][i, 1], "selected_w": chosen["raw"][i, 1], "aggregate_w": base["aggregate"][i]}
                        curves.append({"segment_id": identity, "timestamp": iso(int(stamps[i])),
                                       **{k: float(v) if np.isfinite(v) else "" for k, v in values.items()}})
    # Rare empty cases remain explicitly represented rather than fabricating events.
    if not segments:
        segments = [{"segment_id": "none", "candidate": selected, "seed": "", "error_type": "no_supported_events", "start": "", "end": "", "samples": 0, "selection": "post_evaluation_only"}]
        curves = [{"segment_id": "none", "timestamp": "", "truth_w": "", "baseline_w": "", "selected_w": "", "aggregate_w": ""}]
    _write_csv(run / "error_segments.csv", segments)
    _write_csv(run / "error_curves.csv", curves)


def report(run):
    rows = read_json(run / "metrics.json")["rows"]
    summary = read_json(run / "summary.json")
    body = "<h1>微波炉精确率实验</h1><p>House 17 已查看开发区间；来源选择先于开发评价。最终 8/9/20 未参与。固定 200 W 状态阈值。</p>"
    body += "<p>预定候选：" + escape(summary["selected_before_development_evaluation"]) + "；开发出口：" + ("通过" if summary["precision_exit_passed"] else "未通过，保留原模型") + "。</p>"
    body += '<table><tr><th>候选</th><th>种子</th><th>Precision</th><th>Recall</th><th>F1</th><th>FP</th><th>FN</th><th>运行 MAE W</th><th>电量误差 kWh</th></tr>'
    for r in rows:
        if r["partition"] == DEVELOPMENT and r["appliance"] == "microwave" and r["view"] == "raw":
            values = [r[k] for k in ("candidate", "seed", "precision", "recall", "f1", "false_positive", "false_negative", "active_mae_w", "energy_absolute_error_kwh")]
            body += "<tr>" + "".join("<td>" + escape(f"{v:.6f}" if isinstance(v, float) else str(v)) + "</td>" for v in values) + "</tr>"
    body += '</table><p><a href="comparison.csv">完整设备与原始/修正功率指标</a> · <a href="error_segments.csv">误报/检出/漏报片段</a> · <a href="error_curves.csv">基准与候选曲线数据</a></p>'
    body += "<p>水壶/洗衣机原始预测逐点不变；组合功率修正可能改变。功率、电量与状态效果分别报告。候选单设备权重可通过普通预测接口加载；界面默认模型不替换。</p>"
    body += "<pre>" + escape(__import__("json").dumps(summary, ensure_ascii=False, indent=2)) + "</pre>"
    text = '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:40px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>' + body
    target = run / "report.html"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise DataError("Completed precision report is immutable")
    if not target.exists():
        target.write_text(text, encoding="utf-8")
    return {"status": "report_ready", "path": str(target)}


def verify(run):
    plan = _registered(run, current=False)
    frozen = validate_frozen(run)
    state = read_json(run / "run.json")
    source_rows = read_json(run / "source_metrics.json")["rows"]
    first_rows = [r for r in source_rows if not r["candidate"].endswith("_HN")]
    if choose_source(first_rows, plan, ["W50", "W10", "W1"]) != read_json(run / "stage_one_selection.json"):
        raise DataError("First-stage selection does not reproduce")
    selection = choose_source(source_rows, plan, frozen["groups"])
    if selection != read_json(run / "selection.json") or not _check_other_devices(run, frozen["cases"]):
        raise DataError("Frozen precision selection or other devices changed")
    if state["status"] == "evaluated":
        if sha256(run / "evidence.json") != state["evidence_sha256"]:
            raise DataError("Precision evidence changed")
        for path, digest in read_json(run / "evidence.json")["artifacts"].items():
            if sha256(safe_child(run, path)) != digest:
                raise DataError("Precision evaluation artifact changed")
        metrics = read_json(run / "metrics.json")
        if metrics["freeze_sha256"] != sha256(run / "freeze.json"):
            raise DataError("Evaluation belongs to a different freeze")
        actual = development_gate(metrics["rows"], selection["selected"], plan)
        actual["other_devices_raw_unchanged"] = True
        if actual != read_json(run / "summary.json"):
            raise DataError("Precision summary does not reproduce")
    elif state["status"] != "frozen":
        raise DataError("Precision run is not complete")
    return {"status": "precision_verified", "models": frozen["model_count"], "cases": len(frozen["cases"]),
            "target_labels_reopened": False, "final_households_accessed": False,
            "execution_source_matches_current": read_json(run / "registration.json")["code_sha256"] == source_fingerprint()}
