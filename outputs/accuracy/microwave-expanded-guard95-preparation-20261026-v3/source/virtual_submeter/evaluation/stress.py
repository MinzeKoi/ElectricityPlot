"""Paired, explicitly synthetic background injection; targets stay unchanged."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

from ..data.common import DataError, publish_directory, read_json, sha256, source_fingerprint, utc, write_json
from ..data.protocol import read_partition
from ..models.prediction import predict, write_prediction_csv
from ..reporting.html import render_report
from ..trust.calibration import load_settings
from ..trust.output import quality_summary, validate_output
from .metrics import appliance_metrics, attribution_metrics, consistency_metrics, interval_metrics

NOTICE = "Synthetic additional background only; not a collected unknown appliance or real fault. Target submeter labels are unchanged. Unexplained power is not unknown-device ground truth."


def inject_background(rows: list[dict], interval: int, settings: dict) -> tuple[list[dict], list[dict]]:
    origin = utc(rows[0]["timestamp"])
    modified, schedule = [], []
    for row in rows:
        index = (utc(row["timestamp"]) - origin) // interval
        active = (index - settings["offset_samples"]) % settings["period_samples"] < settings["duration_samples"]
        available = row["aggregate_power_w"] != ""
        added = settings["power_w"] if active and available else 0.0 if available else None
        item = dict(row)
        if available:
            item["aggregate_power_w"] = str(float(row["aggregate_power_w"]) + added)
        flags = json.loads(item.get("quality_flags") or "[]")
        item["quality_flags"] = json.dumps(sorted(set(flags) | {"synthetic_background_injection"}))
        modified.append(item)
        schedule.append({"timestamp": row["timestamp"], "scheduled_active": active, "injected_power_w": added})
    return modified, schedule


def score(rows: list[dict], data: dict, payload: dict) -> dict:
    appliances, interval = payload["appliances"], payload["sample_interval_seconds"]
    return {"metrics": {a: {version: appliance_metrics(rows, data[a], a, payload["model_settings"]["on_thresholds_w"][a], interval, key)
                             for version, key in (("raw", "raw_power_w"), ("corrected", "power_w"))} for a in appliances},
            "consistency": consistency_metrics(rows, interval),
            "intervals": {a: interval_metrics(rows, data[a], a) for a in appliances},
            "attribution": attribution_metrics(rows, data, appliances, interval)}


def paired_metrics(base: dict, stressed: dict, data: dict, schedule: list[dict]) -> dict:
    groups = {}
    for name, active in (("background_present", True), ("background_absent", False)):
        selected = [i for i, entry in enumerate(schedule) if entry["scheduled_active"] == active]
        base_rows = [base["rows"][i] for i in selected]
        stress_rows = [stressed["rows"][i] for i in selected]
        groups[name] = {"baseline": score(base_rows, data, base), "injected": score(stress_rows, data, stressed)}
        false_activations = {}
        for appliance in base["appliances"]:
            truth = {(r["household_id"], utc(r["timestamp"])): float(r["power_w"]) for r in data[appliance] if r["power_w"] != ""}
            off_pairs = []
            for b, s in zip(base_rows, stress_rows):
                actual = truth.get((b["household_id"], utc(b["timestamp"])))
                bv, sv = b["appliances"][appliance], s["appliances"][appliance]
                if actual is not None and actual < base["model_settings"]["on_thresholds_w"][appliance] and bv["state"] is not None and sv["state"] is not None:
                    off_pairs.append((bv["state"], sv["state"]))
            false_activations[appliance] = {"actual_off_samples": len(off_pairs),
                "baseline_false_activation_rate": sum(b for b, _ in off_pairs) / len(off_pairs) if off_pairs else None,
                "injected_false_activation_rate": sum(s for _, s in off_pairs) / len(off_pairs) if off_pairs else None,
                "new_false_activation_samples": sum(s and not b for b, s in off_pairs)}
        groups[name]["false_activations"] = false_activations
    valid = [(b, s, e) for b, s, e in zip(base["rows"], stressed["rows"], schedule) if b["raw_excess_w"] is not None and s["raw_excess_w"] is not None and e["injected_power_w"] is not None]
    injected = sum(e["injected_power_w"] for _, _, e in valid)
    incremental = sum(sum(s["appliances"][a]["power_w"] - b["appliances"][a]["power_w"] for a in base["appliances"]) for b, s, _ in valid)
    return {"groups": groups, "common_valid_samples": len(valid),
            "injected_energy_kwh_on_common_support": injected * base["sample_interval_seconds"] / 3600000 if valid else None,
            "incremental_known_assignment_fraction": incremental / injected if injected > 0 else None,
            "notice": "Groups use scheduled injection at the center bin. Offline context effects can extend beyond pulses. Incremental assignment is a sensitivity measure, not a measured unknown-device identification score."}


def stress_test(model_dir: Path, manifest_path: Path, partition_id: str, settings_path: Path, output: Path, calibration_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    if output.exists():
        raise DataError("Stress destination exists; choose a new run directory")
    settings = load_settings(settings_path)
    protocol = read_json(manifest_path)
    partition = next((p for p in protocol["partitions"] if p["partition_id"] == partition_id), None)
    if partition is None or partition["role"] != "frozen_test":
        raise DataError("Background stress requires a frozen test partition")
    # All forecasts and settings are persisted before opening evaluation labels.
    aggregate = read_partition(manifest_path, partition_id, "inference")["aggregate"]
    modified, schedule = inject_background(aggregate, protocol["sample_interval_seconds"], settings["stress"])
    if not any(e["injected_power_w"] and e["injected_power_w"] > 0 for e in schedule):
        raise DataError("No observable injected background; use a longer input")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stress-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        (staging / "settings.json").write_bytes(settings_path.read_bytes())
        (staging / "protocol.json").write_bytes(manifest_path.read_bytes())
        source = staging / "injected_aggregate.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(modified[0]))
            writer.writeheader()
            writer.writerows(modified)
        write_json(staging / "injection.json", {"notice": NOTICE, "schedule": schedule})
        predict(model_dir, staging / "baseline", manifest_path, partition_id, calibration_dir=calibration_dir)
        predict(model_dir, staging / "injected", input_path=source, calibration_dir=calibration_dir)
        base = read_json(staging / "baseline/predictions.json")
        stressed = read_json(staging / "injected/predictions.json")
        stressed.update({"purpose": "synthetic_background_stress", "partition_id": partition_id,
                         "data_kind": protocol.get("data_kind", "dataset_measurements"),
                         "protocol_sha256": sha256(manifest_path),
                         "source_input_sha256": base["input_sha256"],
                         "stress_notice": NOTICE, "stress_settings": settings["stress"]})
        validate_output(stressed)
        stressed["quality_summary"] = quality_summary(stressed)
        write_json(staging / "injected/predictions.json", stressed)
        write_prediction_csv(staging / "injected/predictions.csv", stressed)
        write_json(staging / "frozen_predictions.json", {"baseline_sha256": sha256(staging / "baseline/predictions.json"),
                   "injected_sha256": sha256(staging / "injected/predictions.json"), "settings_sha256": sha256(staging / "settings.json"),
                   "injected_input_sha256": sha256(source), "injection_sha256": sha256(staging / "injection.json"),
                   "predictions_frozen_before_test_labels": True})
        data = read_partition(manifest_path, partition_id, "evaluation", include_labels=True)
        evaluations = {}
        for name, payload in (("baseline", base), ("injected", stressed)):
            evaluation = {"status": "evaluated", "model_id": payload["model_id"], "purpose": payload["purpose"],
                          "partition_id": partition_id, "household_id": partition["household_id"],
                          "prediction_sha256": sha256(staging / name / "predictions.json"),
                          "calibration": payload.get("calibration"), **score(payload["rows"], data, payload)}
            write_json(staging / name / "metrics.json", evaluation)
            render_report(payload, staging / name / "report.html", evaluation)
            from ..app.results import write_result_manifest
            write_result_manifest(staging / name, payload)
            evaluations[name] = evaluation
        result = {"schema_version": 1, "status": "stress_evaluated", "created_at": datetime.now(timezone.utc).isoformat(),
                  "notice": NOTICE, "model_id": base["model_id"], "model_manifest_sha256": base["model_manifest_sha256"],
                  "code_sha256": source_fingerprint(), "protocol_sha256": sha256(manifest_path),
                  "settings_sha256": sha256(settings_path), "source_purpose": protocol["purpose"],
                  "data_kind": protocol.get("data_kind", "dataset_measurements"),
                  "partition_id": partition_id, "household_id": partition["household_id"],
                  "predictions_frozen_before_test_labels": True,
                  "baseline": evaluations["baseline"], "injected": evaluations["injected"],
                  "paired": paired_metrics(base, stressed, data, schedule), "elapsed_seconds": time.perf_counter() - started}
        from ..reporting.trust import render_stress_report
        render_stress_report(result, staging / "report.html")
        result["artifacts_sha256"] = {p.relative_to(staging).as_posix(): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        write_json(staging / "stress.json", result)
        publish_directory(staging, output)
    return {"status": "stress_complete", "metrics": str(output / "stress.json"), "report": str(output / "report.html")}
