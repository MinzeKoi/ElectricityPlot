"""Registered, compute-matched data expansion; final REFIT homes are inaccessible here."""

import csv
from datetime import datetime, timezone
from html import escape
import math
from pathlib import Path
import shutil

import numpy as np

from ..data.common import DataError, inventory, iso, read_json, safe_child, sha256, source_fingerprint, utc, write_json
from ..data.protocol import read_partition
from ..models.arrays import aggregate_arrays, labels_at_centers, windows
from ..models.settings import load_settings
from ..models.training import dataset, train
from .diagnosis import event_support
from .r1 import merge_single_models


ROOT = Path(__file__).resolve().parents[3]
DEVICES = ("kettle", "microwave", "washing_machine")
TRAIN_HOUSES = {"2", "19", "10", "12", "15", "16", "18"}


def prepare_additional(output, plan_path):
    """Slice exact source CSV fields before calling the existing strict resampler.

    The predecessor and closing row retain the original observation age. Full
    originals and acquisition receipts remain the authoritative provenance.
    """
    import pandas as pd
    from ..data.prepare import prepare as prepare_refit
    plan = load_plan(plan_path)
    if output.exists():
        raise DataError("Additional data destination exists")
    catalog = ROOT / "configs/refit_channels_complete_20261022.json"
    mapping = read_json(catalog)["households"]
    audit = read_json(ROOT / "docs/R0_RESULTS.json")
    dates = {h["household_id"]: h for h in audit["households"] if h["household_id"] in plan["additional_sources"]}
    slices = ROOT / "data/raw" / (output.name + "_slices")
    slices.mkdir(parents=True, exist_ok=True)
    inputs, provenance = {}, {}
    for house in plan["additional_sources"]:
        info = dates[house]
        chosen = info["all_aligned_28_day_windows"]["first_pass"]
        start, end = utc(chosen["start"]), utc(chosen["end"])
        original = ROOT / f"data/raw/refit_complete_20261022/households/CLEAN_House{house}.csv"
        original_hash = sha256(original)
        receipt = read_json(original.with_suffix(".receipt.json"))
        if original_hash != info["source_sha256"] or receipt["sha256"] != original_hash:
            raise DataError("Original REFIT file differs from R0 audit/receipt")
        target = slices / f"CLEAN_House{house}.csv"
        slice_receipt = target.with_suffix(".slice.json")
        if slice_receipt.exists():
            saved = read_json(slice_receipt)
            if saved["original_sha256"] != original_hash or sha256(target) != saved["slice_sha256"]:
                raise DataError("Existing REFIT slice changed")
        else:
            print(f"Slice exact REFIT house {house} on registered dates", flush=True)
            cols = ["Unix", "Aggregate", "Issues", *dict.fromkeys(c for group in mapping[house].values() for c in group)]
            pieces, previous = [], None
            for chunk in pd.read_csv(original, usecols=cols, dtype=str, chunksize=200000, keep_default_na=False):
                stamps = chunk["Unix"].astype("int64")
                before = chunk.loc[stamps < start]
                if len(before):
                    previous = before.tail(1)
                selected = chunk.loc[(stamps >= start) & (stamps < end)]
                if len(selected):
                    if not pieces and previous is not None:
                        pieces.append(previous)
                    pieces.append(selected)
                closing = chunk.loc[stamps >= end]
                if len(closing):
                    pieces.append(closing.head(1))
                    break
            if not pieces:
                raise DataError("Selected REFIT range unavailable")
            pd.concat(pieces).to_csv(target, index=False)
            write_json(slice_receipt, {"original_sha256": original_hash, "slice_sha256": sha256(target), "start": iso(start), "end": iso(end)})
        inputs[house] = target
        provenance[house] = {"original": str(original), "source_sha256": original_hash, "receipt": receipt,
                             "slice_sha256": sha256(target), "selection": chosen}
    selection = slices / "selection.json"
    write_json(selection, {"schema_version": 1, "protocol_id": "expansion-r0-exact-slices-v1", "catalog_sha256": sha256(catalog),
               "households": {h: {"start": p["selection"]["start"], "end": p["selection"]["end"],
                                    "source_sha256": p["slice_sha256"]} for h, p in provenance.items()}})
    prepare_refit(inputs, output, catalog, selection=selection)
    manifest = read_json(output / "manifest.json")
    for house, p in provenance.items():
        manifest["households"][house].update(source_file=Path(p["original"]).name, source_sha256=p["source_sha256"],
                provenance=p["receipt"], exact_slice_sha256=p["slice_sha256"], full_record_audit_sha256=sha256(ROOT / "docs/R0_RESULTS.json"))
    write_json(output / "manifest.json", manifest)
    return {"status": "additional_sources_prepared", "households": list(inputs)}


def load_plan(path):
    plan = read_json(path)
    if (plan.get("source_train") != ["2", "19"]
            or set(plan.get("additional_sources", [])) != TRAIN_HOUSES - {"2", "19"}
            or plan.get("source_validation") != ["5"] or plan.get("development") != ["17"]
            or plan.get("final_reserved") != ["8", "9", "20"]
            or plan.get("ukdale_train") != ["1", "2"] or plan.get("ukdale_external_test") != ["5"]
            or plan.get("appliances") != list(DEVICES) or plan.get("maximum_days") != 28
            or plan.get("seeds") != [42, 1019, 2026]):
        raise DataError("Expansion must preserve registered source, validation, development and reserved roles")
    if (plan.get("minimum_coverage") != .9 or plan.get("minimum_observed_days") != 7
            or plan.get("minimum_complete_events") != 2 or plan.get("epochs") != 12
            or plan.get("active_loss_weight") != 50.0):
        raise DataError("Expansion quality gates and training recipe are fixed in this version")
    return plan


def check_registered(run):
    registration = read_json(run / "registration.json")
    if registration["code_sha256"] != source_fingerprint():
        raise DataError("Expansion source changed after registration; use a new run")
    for name, digest in registration["inputs"].items():
        if sha256(run / name) != digest:
            raise DataError("Registered expansion input changed")
    return registration


def admission(rows, aggregate_rows, threshold, plan, audit_events=True):
    stats = event_support(rows, threshold, 8) if audit_events else {
        "valid_samples": sum(r["power_w"] != "" for r in rows),
        "observed_days": sum(r["power_w"] != "" for r in rows) / 10800,
        "complete_threshold_events": None}
    coverage = stats["valid_samples"] / len(rows) if rows else 0
    aggregate_coverage = sum(r["aggregate_power_w"] != "" for r in aggregate_rows) / len(aggregate_rows) if aggregate_rows else 0
    reasons = []
    if min(coverage, aggregate_coverage) < plan["minimum_coverage"]:
        reasons.append("coverage_below_minimum")
    if min(stats["observed_days"], aggregate_coverage * len(aggregate_rows) / 10800) < plan["minimum_observed_days"]:
        reasons.append("insufficient_observed_days")
    if audit_events and stats["complete_threshold_events"] < plan["minimum_complete_events"]:
        reasons.append("insufficient_complete_events")
    return {**stats, "coverage": coverage, "aggregate_coverage": aggregate_coverage,
            "included": not reasons, "reasons": reasons}


def partition(prepared, house, role, start, end, pid, destination, settings, plan, dataset_name="refit"):
    """Copy only the explicit allowlisted household and half-open segment."""
    if dataset_name == "refit" and house not in TRAIN_HOUSES | {"5", "17"}:
        raise DataError("Reserved or quarantined REFIT household")
    if dataset_name == "ukdale" and house not in {"1", "2", "5"}:
        raise DataError("Unregistered UK-DALE household")
    audit = read_json(prepared / "manifest.json")
    info = audit["households"][house]
    identity = f"{dataset_name}_{house}"
    result = {"partition_id": pid, "household_id": identity, "role": role,
              "start": iso(start), "end": iso(end), "artifacts": {},
              "origin": {"dataset": dataset_name, "household": house,
                         "source_sha256": info.get("source_sha256"), "license": audit.get("license"),
                         "provenance": info.get("provenance"), "channel_mapping": info.get("channel_mapping")}}
    selected, checks = {}, {}
    for channel in ("aggregate", *DEVICES):
        if channel not in info["artifacts"]:
            checks[channel] = {"included": False, "reasons": ["unmonitored_channel"]}
            continue
        artifact = info["artifacts"][channel]
        source = safe_child(prepared, artifact["path"])
        if sha256(source) != artifact["sha256"]:
            raise DataError("Prepared input hash mismatch")
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [{**r, "household_id": identity} for r in reader if start <= utc(r["timestamp"]) < end]
        if len(rows) != (end - start) // 8:
            raise DataError("Selected segment does not retain every grid bin")
        selected[channel] = rows
        if channel != "aggregate":
            checks[channel] = admission(rows, selected["aggregate"], settings["on_thresholds_w"][channel], plan,
                                        audit_events=not (dataset_name == "ukdale" and role == "frozen_test"))
            if role == "source_train" and not checks[channel]["included"]:
                continue
        target = destination / f"data/{pid}/{channel}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        result["artifacts"][channel] = {"path": target.relative_to(destination).as_posix(), "sha256": sha256(target)}
    return result, {"partition_id": pid, "household_id": identity, "role": role, "start": iso(start), "end": iso(end), "devices": checks}


def base_protocol(group, plan):
    return {"schema_version": 1, "protocol_id": f"{plan['experiment_id']}-{group}",
            "purpose": "engineering_pilot_only", "accuracy_usage": "development_only",
            "data_kind": "real_data_expansion", "sample_interval_seconds": 8,
            "target_appliances": list(DEVICES), "split_gap_seconds": 3600,
            "max_model_context_seconds": 3600, "partial_source_labels": True,
            "upstream_limitations": read_json(ROOT / "configs/refit_channels_complete_20261022.json")["upstream_limitations"],
            "partitions": []}


def prepare(run, additional, plan_path):
    if run.exists():
        raise DataError("Expansion output exists; select a fresh version")
    plan = load_plan(plan_path)
    run.mkdir(parents=True)
    shutil.copyfile(plan_path, run / "plan.json")
    settings = load_settings(ROOT / "configs/m2_baseline.json")
    settings.update(epochs=plan["epochs"], active_loss_weight=plan["active_loss_weight"])
    write_json(run / "settings.json", settings)
    r0 = read_json(ROOT / "docs/R0_RESULTS.json")
    dates = {h["household_id"]: h["all_aligned_28_day_windows"]["first_pass"] for h in r0["households"]
             if h["household_id"] in TRAIN_HOUSES | {"5", "17"}}
    write_json(run / "dates.json", dates)
    registration = {"status": "registered_before_admission_and_training", "created_at": datetime.now(timezone.utc).isoformat(),
                    "code_sha256": source_fingerprint(), "r0_results_sha256": sha256(ROOT / "docs/R0_RESULTS.json"),
                    "inputs": {p: sha256(run / p) for p in ("plan.json", "settings.json", "dates.json")},
                    "final_households_accessed": False}
    write_json(run / "registration.json", registration)
    shutil.copytree(ROOT / "src/virtual_submeter", run / "source/virtual_submeter", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = ROOT / "data/processed/refit_r1_sources_20261022"
    development = ROOT / "data/processed/refit_r1_development_20261022"
    admissions = {}
    for group in ("A", "B", "C"):
        dest = run / "protocols" / group
        protocol, checks = base_protocol(group, plan), []
        houses = plan["source_train"] + (plan["additional_sources"] if group == "C" else [])
        for house in houses + ["5", "17"]:
            info = dates[house]
            start, last = utc(info["start"]), utc(info["end"])
            middle = start + 14 * 86400
            if house == "5":
                spans = [("source_validation", start, middle)]
            elif house == "17":
                spans = [("frozen_test", middle + 3600, last)]
            else:
                spans = [("source_train", start, middle)]
                if group != "A":
                    spans.append(("source_train", middle + 3600, last))
            prepared = development if house == "17" else additional if house in plan["additional_sources"] else source
            for index, (role, first, end) in enumerate(spans):
                pid = f"{role}_refit_{house}_{index}"
                part, check = partition(prepared, house, role, first, end, pid, dest, settings, plan)
                if group == "A" and role == "source_train" and any(not c["included"] for c in check["devices"].values()):
                    raise DataError("Original A source fails admission; cannot silently change the reference data")
                protocol["partitions"].append(part)
                checks.append(check)
        write_json(dest / "protocol.json", protocol)
        admissions[group] = checks
    write_json(run / "admission.json", admissions)
    # Fit a shared normalization to the union of admitted SOURCE windows, once per group.
    for group in ("A", "B", "C"):
        path = run / f"protocols/{group}/protocol.json"
        protocol = read_json(path)
        parts = [p for p in protocol["partitions"] if p["role"] == "source_train"]
        x, _, counts = dataset(path, parts, list(DEVICES), settings, 8)
        if group == "A":
            reference_windows = sum(counts.values())
            if reference_windows != 150069:
                raise DataError(f"Original source support changed: {reference_windows} != 150069")
        logged = np.log1p(x)
        protocol["expansion_training"] = {"input_log_mean": float(logged.mean()), "input_log_std": max(float(logged.std()), 1e-6),
                                           "normalization_source_windows": counts,
                                           "steps_per_checkpoint": math.ceil(reference_windows / settings["batch_size"]),
                                           "normalization_scope": "source_train_union_only"}
        write_json(path, protocol)
    write_json(run / "prepared.json", {"status": "prepared", "artifacts": inventory(run / "protocols"),
                                       "admission_sha256": sha256(run / "admission.json")})
    return {"status": "prepared", "run": str(run)}


def predict_compact(model_dir, manifest, partition_id, output):
    """Ordinary network inference, with lossless compact arrays instead of repeated JSON curves."""
    import torch
    from ..models.artifacts import load_model
    meta, model = load_model(model_dir)
    data = read_partition(manifest, partition_id, "inference")
    stamps, aggregate = aggregate_arrays(data["aggregate"], 8)
    x, centers = windows(stamps, aggregate, meta["settings"]["window_samples"], 8)
    torch.set_num_threads(meta["settings"]["cpu_threads"])
    tx = torch.from_numpy((np.log1p(x) - meta["input_log_mean"]) / meta["input_log_std"])
    with torch.inference_mode():
        values = np.concatenate([model(b).numpy() for b in tx.split(meta["settings"]["batch_size"])])
    raw = np.full((len(stamps), len(DEVICES)), np.nan, dtype=np.float64)
    raw[centers] = values * np.asarray(meta["power_scales_w"])
    total = raw.sum(axis=1)
    factor = np.minimum(1, np.divide(aggregate, total, out=np.ones_like(total), where=total > 0))
    corrected = raw * factor[:, None]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, stamps=stamps, aggregate=aggregate, raw=raw, corrected=corrected)
    return {"path": output.as_posix(), "sha256": sha256(output), "model_sha256": sha256(model_dir / "model.json"),
            "partition_id": partition_id, "protocol_sha256": sha256(manifest)}


def compact_metrics(prediction, truth, threshold):
    valid = np.isfinite(prediction) & np.isfinite(truth)
    p, y = prediction[valid], truth[valid]
    ps, ys = p >= threshold, y >= threshold
    tp, fp, fn = int((ps & ys).sum()), int((ps & ~ys).sum()), int((~ps & ys).sum())
    n, on = len(p), int(ps.sum())
    return {"valid_samples": n, "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "predicted_on_samples": on, "actual_on_samples": int(ys.sum()),
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "mae_w": float(np.abs(p - y).mean()) if n else None,
            "active_mae_w": float(np.abs(p[ys] - y[ys]).mean()) if ys.any() else None,
            "energy_absolute_error_kwh": float(abs(p.sum() - y.sum()) * 8 / 3600000) if n else None,
            "state_pattern": "unavailable" if not n else "all_off" if not on else "all_on" if on == n else "mixed",
            "support_sha256": __import__("hashlib").sha256(valid.tobytes()).hexdigest()}


def train_groups(run, groups=("A", "B", "C")):
    check_registered(run)
    prepared = read_json(run / "prepared.json")
    current = inventory(run / "protocols")
    if any(current.get(k) != v for k, v in prepared["artifacts"].items()):
        raise DataError("Prepared expansion data changed")
    plan, settings = load_plan(run / "plan.json"), read_json(run / "settings.json")
    frozen = []
    for group in groups:
        manifest = run / f"protocols/{group}/protocol.json"
        for seed in plan["seeds"]:
            base = run / f"models/{group}/seed_{seed}"
            setting_path = run / f"settings/{group}_{seed}.json"
            write_json(setting_path, {**settings, "seed": seed})
            components = []
            for appliance in DEVICES:
                target = base / appliance
                if not target.exists():
                    print(f"Expansion {group} seed {seed} {appliance}", flush=True)
                    train(manifest, setting_path, target, appliances=[appliance])
                else:
                    from ..models.artifacts import load_model
                    meta, _ = load_model(target)
                    if meta["protocol_sha256"] != sha256(manifest) or meta["settings_sha256"] != sha256(setting_path):
                        raise DataError("Existing component differs from registered inputs")
                components.append(target)
            packed = base / "packed"
            if not packed.exists():
                merge_single_models(components, packed)
            if group == "A":
                reference_check(run, seed, packed)
            protocol = read_json(manifest)
            for part in protocol["partitions"]:
                if part["role"] not in ("frozen_test", "source_validation"):
                    continue
                target = run / f"predictions/{group}/seed_{seed}/{part['partition_id']}.npz"
                receipt_path = target.with_suffix(".json")
                if not receipt_path.exists():
                    receipt = predict_compact(packed, manifest, part["partition_id"], target)
                    receipt["path"] = target.relative_to(run).as_posix()
                    write_json(receipt_path, receipt)
                receipt = read_json(receipt_path)
                if sha256(target) != receipt["sha256"] or receipt["model_sha256"] != sha256(packed / "model.json"):
                    raise DataError("Frozen prediction/model mismatch")
                frozen.append({"group": group, "seed": seed, "protocol_group": group, **receipt})
    if groups == ("D",):
        manifest = run / "protocols/D/protocol.json"
        external = [p for p in read_json(manifest)["partitions"] if p["household_id"] == "ukdale_5"]
        if not external:
            raise DataError("External evaluation household missing")
        # C remains the frozen REFIT model. Generate its zero-shot external
        # predictions before EITHER model is allowed to see external labels.
        for seed in plan["seeds"]:
            for part in external:
                model_dir = run / f"models/C/seed_{seed}/packed"
                target = run / f"predictions/C_external/seed_{seed}/{part['partition_id']}.npz"
                receipt = predict_compact(model_dir, manifest, part["partition_id"], target)
                receipt["path"] = target.relative_to(run).as_posix()
                frozen.append({"group": "C", "seed": seed, "protocol_group": "D", **receipt})
    write_json(run / ("freeze_external.json" if groups == ("D",) else "freeze_refit.json"),
               {"status": "all_models_and_predictions_frozen", "predictions": frozen, "code_sha256": source_fingerprint()})
    return {"status": "frozen", "groups": list(groups)}


def reference_check(run, seed, packed):
    import torch
    from ..models.artifacts import load_model
    previous = ROOT / f"outputs/accuracy/r1-20261022/selected/seed_{seed}"
    original, model = load_model(previous)
    current, reproduced = load_model(packed)
    if original["input_log_mean"] != current["input_log_mean"] or original["input_log_std"] != current["input_log_std"]:
        raise DataError("A normalization differs from R1")
    delta = max(float((model.state_dict()[k] - reproduced.state_dict()[k]).abs().max()) for k in model.state_dict())
    probe = torch.linspace(-2, 2, 31 * 100).reshape(100, 31)
    with torch.inference_mode():
        torch.testing.assert_close(model(probe), reproduced(probe), atol=1e-6, rtol=1e-5)
    if delta > 1e-6:
        raise DataError("A reproduction differs from historical R1 weights")
    write_json(run / f"reference_checks/seed_{seed}.json", {"passed": True, "maximum_weight_difference": delta,
               "historical_model_sha256": sha256(previous / "model.json"), "reproduced_model_sha256": sha256(packed / "model.json")})


def evaluate(run, external=False):
    check_registered(run)
    target = run / ("metrics_external.json" if external else "metrics_refit.json")
    if target.exists():
        raise DataError("Evaluation already completed; saved results are immutable")
    frozen_path = run / ("freeze_external.json" if external else "freeze_refit.json")
    frozen = read_json(frozen_path)
    expected = {"C", "D"} if external else {"A", "B", "C"}
    cases = frozen["predictions"]
    if {p["group"] for p in cases} != expected:
        raise DataError("Incomplete model/prediction freeze barrier")
    plan, settings = load_plan(run / "plan.json"), read_json(run / "settings.json")
    expected_cases = set()
    for group in expected:
        protocol_group = "D" if external else group
        for part in read_json(run / f"protocols/{protocol_group}/protocol.json")["partitions"]:
            if part["role"] not in ("source_validation", "frozen_test"):
                continue
            if external and group == "C" and part["household_id"] != "ukdale_5":
                continue
            expected_cases.update((group, seed, part["partition_id"]) for seed in plan["seeds"])
    if len(cases) != len(expected_cases) or {(p["group"], p["seed"], p["partition_id"]) for p in cases} != expected_cases:
        raise DataError("Freeze must contain every seed and evaluation partition")
    for p in cases:
        if sha256(safe_child(run, p["path"])) != p["sha256"]:
            raise DataError("Frozen predictions changed before evaluation")
        from ..models.artifacts import load_model
        model = run / f"models/{p['group']}/seed_{p['seed']}/packed"
        if sha256(model / "model.json") != p["model_sha256"]:
            raise DataError("Frozen model changed before evaluation")
        load_model(model)
    metrics = []
    for case in cases:
        manifest = run / f"protocols/{case.get('protocol_group', case['group'])}/protocol.json"
        if sha256(manifest) != case["protocol_sha256"]:
            raise DataError("Frozen protocol changed")
        data = read_partition(manifest, case["partition_id"], "evaluation", include_labels=True)
        with np.load(safe_child(run, case["path"]), allow_pickle=False) as arrays:
            stamps = arrays["stamps"]
            for i, appliance in enumerate(DEVICES):
                if appliance not in data:
                    continue
                y = labels_at_centers(data, [appliance], stamps, np.arange(len(stamps)))[:, 0]
                for view in ("raw", "corrected"):
                    metrics.append({"group": case["group"], "seed": case["seed"], "partition": case["partition_id"],
                                    "appliance": appliance, "view": view,
                                    **compact_metrics(arrays[view][:, i], y, settings["on_thresholds_w"][appliance])})
    write_json(target, {"freeze_sha256": sha256(frozen_path), "metrics": metrics})
    report(run)
    sealed = {p.relative_to(run).as_posix(): sha256(p) for p in run.rglob("*") if p.is_file()
              and (p.relative_to(run).parts[0] in ("protocols", "models", "predictions", "source", "reference_checks", "settings")
                   or p.name in ("registration.json", "plan.json", "settings.json", "dates.json", "admission.json", "prepared.json", frozen_path.name, target.name))}
    write_json(run / ("evidence_external.json" if external else "evidence_refit.json"), {"artifacts": sealed})
    return {"status": "evaluated", "metric_rows": len(metrics)}


def paired_gains(metrics, before, after, plan):
    selected = [m for m in metrics if m["view"] == "raw" and m["partition"].startswith("frozen_test_refit_17")]
    keyed = {(m["group"], m["seed"], m["appliance"]): m for m in selected}
    differences, device_deltas, new_degenerate = [], {}, 0
    for seed in plan["seeds"]:
        changes = []
        for appliance in DEVICES:
            left, right = keyed[before, seed, appliance], keyed[after, seed, appliance]
            if left["support_sha256"] != right["support_sha256"]:
                raise DataError("Expansion comparison must use identical evaluation support")
            delta = right["f1"] - left["f1"] if right["f1"] is not None and left["f1"] is not None else None
            device_deltas.setdefault(appliance, []).append(delta)
            new_degenerate += int(right["state_pattern"] in ("all_off", "all_on") and left["state_pattern"] == "mixed")
            if appliance in plan["primary_appliances"]:
                changes.append(delta)
        differences.append(float(np.mean(changes)) if all(d is not None for d in changes) else None)
    complete = all(d is not None for d in differences)
    mean = float(np.mean(differences)) if complete else None
    means = {a: float(np.mean(ds)) if all(d is not None for d in ds) else None for a, ds in device_deltas.items()}
    positives = sum(d is not None and d > 0 for d in differences)
    passed = (complete and mean >= plan["minimum_mean_f1_gain"] and positives >= plan["minimum_positive_seeds"]
              and all(means[a] is not None and means[a] >= -plan["maximum_device_f1_drop"] for a in plan["primary_appliances"])
              and not new_degenerate)
    return {"comparison": f"{after}-{before}", "paired_seed_f1_deltas": differences, "mean_primary_f1_delta": mean,
            "positive_seeds": positives, "mean_device_f1_deltas": means, "new_degenerate_cases": new_degenerate,
            "internal_target_met": bool(passed), "final_generalization_accepted": False}


def report(run):
    metrics = []
    for name in ("metrics_refit.json", "metrics_external.json"):
        if (run / name).exists():
            metrics.extend(read_json(run / name)["metrics"])
    plan = load_plan(run / "plan.json")
    gains = [paired_gains(metrics, a, b, plan) for a, b in (("A", "B"), ("B", "C"))]
    if (run / "metrics_external.json").exists():
        gains.append(paired_gains(metrics, "C", "D", plan))
    write_json(run / "summary.json", {"status": "development_results", "comparisons": gains,
               "final_households_accessed": False, "ui_default_changed": False,
               "external_status": read_json(run / "ukdale_status.json") if (run / "ukdale_status.json").exists() else {"status": "pending"}})
    fields = list(metrics[0]) if metrics else []
    with (run / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    body = "<h1>数据集扩充对照</h1><p>开发实验；最终家庭 8/9/20 未参与。固定模型与更新步数；未修改界面默认模型。</p>"
    body += "<h2>配对收益与内部目标</h2><pre>" + escape(__import__("json").dumps(gains, ensure_ascii=False, indent=2)) + "</pre>"
    body += "<h2>House 17：三种子均值</h2><table><tr><th>组</th><th>设备</th><th>Precision</th><th>Recall</th><th>F1</th><th>运行 MAE W</th><th>电量绝对误差 kWh</th></tr>"
    for group in sorted({m["group"] for m in metrics}):
        for appliance in DEVICES:
            rows = [m for m in metrics if m["group"] == group and m["appliance"] == appliance and m["view"] == "raw" and m["partition"].startswith("frozen_test_refit_17")]
            body += f"<tr><td>{group}</td><td>{appliance}</td>"
            for key in ("precision", "recall", "f1", "active_mae_w", "energy_absolute_error_kwh"):
                values = [r[key] for r in rows if r[key] is not None]
                body += f"<td>{float(np.mean(values)):.6f}</td>" if values else "<td>不可用</td>"
            body += "</tr>"
    body += "</table><p>原始及一致性修正视图、逐种子指标与有效支持见 comparison.csv。开机片段是固定阈值事件，不等于物理周期。功率与电量结论单独判断。</p>"
    (run / "report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:40px;max-width:1200px}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}</style>' + body, encoding="utf-8")


def verify(run):
    check_registered(run)
    if not (run / "evidence_refit.json").exists():
        raise DataError("No completed REFIT experiment evidence")
    for evidence in ("evidence_refit.json", "evidence_external.json"):
        if (run / evidence).exists():
            for name, digest in read_json(run / evidence)["artifacts"].items():
                if sha256(safe_child(run, name)) != digest:
                    raise DataError("Sealed experiment artifact changed: " + name)
    for name in ("freeze_refit.json", "freeze_external.json"):
        if not (run / name).exists():
            continue
        for receipt in read_json(run / name)["predictions"]:
            if sha256(safe_child(run, receipt["path"])) != receipt["sha256"]:
                raise DataError("Prediction evidence hash mismatch")
            model = run / f"models/{receipt['group']}/seed_{receipt['seed']}/packed/model.json"
            if sha256(model) != receipt["model_sha256"]:
                raise DataError("Model evidence mismatch")
            from ..models.artifacts import load_model
            load_model(model.parent)
    return {"status": "expansion_verified", "final_labels_reopened": False}
