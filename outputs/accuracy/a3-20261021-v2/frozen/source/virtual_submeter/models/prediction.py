"""Aggregate-only prediction and traceable nonnegative consistency outputs."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from ..data.common import DataError, iso, read_json, sha256, source_fingerprint, write_json
from ..data.protocol import read_partition
from .arrays import aggregate_arrays, windows
from .artifacts import load_model
from ..trust.calibration import apply_intervals, load_calibration, METHOD
from ..trust.output import reconcile, validate_output, quality_summary


def predict_rows(metadata: dict, model, rows: list[dict], calibration: dict | None = None) -> list[dict]:
    if metadata.get("adaptation_household_id") and any(row["household_id"] != metadata["adaptation_household_id"] for row in rows):
        raise DataError("Adapted state head belongs to a different household")
    interval = metadata["sample_interval_seconds"]
    stamps, aggregate = aggregate_arrays(rows, interval)
    settings = metadata["settings"]
    x, centers = windows(stamps, aggregate, settings["window_samples"], interval)
    if not len(centers):
        raise DataError("No complete model windows; check missing data and input duration")
    appliances = metadata["appliances"]
    powers = np.full((len(rows), len(appliances)), np.nan, np.float32)
    if metadata["kind"] in ("constant_mean", "zero"):
        powers[centers] = model
    else:
        import torch
        torch.set_num_threads(settings["cpu_threads"])
        standardized = (np.log1p(x) - metadata["input_log_mean"]) / metadata["input_log_std"]
        with torch.inference_mode():
            predicted = np.concatenate([model(batch).numpy() for batch in torch.from_numpy(standardized).split(settings["batch_size"])])
        powers[centers] = predicted * np.asarray(metadata["power_scales_w"])
    if not np.isfinite(powers[centers]).all() or (powers[centers] < 0).any():
        raise DataError("Model produced invalid power estimates")
    result = []
    for index, row in enumerate(rows):
        valid = np.isfinite(powers[index]).all()
        total = float(aggregate[index]) if np.isfinite(aggregate[index]) else None
        flags = set(json.loads(row.get("quality_flags") or "[]"))
        if not valid:
            flags.add("context_missing_or_boundary")
        if total is None:
            flags.add("aggregate_missing")
        raw_sum = sum(powers[index].astype(float).tolist()) if valid else None
        _, factor = reconcile(total, powers[index].astype(float).tolist()) if valid else ([], 1.0)
        if factor < 1:
            flags.add("sum_exceeded_aggregate_scaled_down")
        item = {
            "household_id": row["household_id"], "timestamp": iso(int(stamps[index])),
            "aggregate_power_w": total, "appliances": {},
            "unexplained_power_w": max(0.0, total - raw_sum * factor) if valid else None,
            "raw_excess_w": max(0.0, raw_sum - total) if valid else None,
            "correction_method": "scale_down_only" if valid and factor < 1 else "none" if valid else None,
            "quality_flags": sorted(flags),
        }
        for column, appliance in enumerate(appliances):
            raw = float(powers[index, column]) if valid else None
            corrected = raw * factor if valid else None
            threshold = settings["on_thresholds_w"][appliance]
            item["appliances"][appliance] = {
                "raw_power_w": raw, "power_w": corrected,
                "raw_state": raw >= threshold if valid else None,
                "state": (raw if metadata.get("state_threshold_view") == "raw_power_w" else corrected) >= threshold if valid else None,
                "correction_w": corrected - raw if valid else None,
                "confidence": None,
            }
        result.append(item)
    if metadata.get("state_adapter"):
        from ..adaptation.state import apply_head
        apply_head(result, metadata)
    if metadata.get("accuracy_state_head"):
        from ..accuracy.learners import apply_adapter, temporal_features
        temporal = metadata["accuracy_state_head"]["variant"].startswith("L2")
        hidden = temporal_features(metadata, model, rows) if temporal else None
        apply_adapter(result, metadata, hidden)
    apply_intervals(result, calibration)
    return result


def write_prediction_csv(path: Path, payload: dict) -> None:
    fields = ["household_id", "timestamp", "aggregate_power_w"]
    for appliance in payload["appliances"]:
        fields.extend(f"{appliance}_{key}" for key in ("raw_power_w", "power_w", "raw_state", "state", "correction_w", "confidence", "raw_lower_w", "raw_upper_w", "lower_w", "upper_w"))
        if payload.get("adaptation") in ("weak_window_state_only", "a2_fixed_query_state_only"):
            fields.append(f"{appliance}_state_score")
    fields.extend(["unexplained_power_w", "raw_excess_w", "correction_method", "quality_flags"])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            flat = {key: row[key] for key in fields if key in row}
            flat["quality_flags"] = json.dumps(row["quality_flags"])
            for appliance, values in row["appliances"].items():
                flat.update({f"{appliance}_{key}": value for key, value in values.items()})
            writer.writerow(flat)


def predict(model_dir: Path, output: Path, manifest_path: Path | None = None, partition_id: str | None = None, input_path: Path | None = None, calibration_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    if output.exists():
        raise DataError("Prediction destination exists; choose a new run directory")
    if (input_path is not None) == (manifest_path is not None):
        raise DataError("Choose a normalized input CSV or a protocol partition, exclusively")
    metadata, model = load_model(model_dir)
    calibration = load_calibration(calibration_dir, metadata) if calibration_dir is not None else None
    protocol = None
    if manifest_path is not None:
        protocol = read_json(manifest_path)
        if sha256(manifest_path) != metadata["protocol_sha256"]:
            raise DataError("Prediction protocol differs from the trained model protocol")
        partition = next((p for p in protocol["partitions"] if p["partition_id"] == partition_id), None)
        if partition is None:
            raise DataError("Unknown prediction partition")
        consumer = "selection" if partition["role"] == "target_adaptation" else "inference"
        rows = read_partition(manifest_path, partition_id, consumer)["aggregate"]
        input_hash = partition["artifacts"]["aggregate"]["sha256"]
    else:
        with input_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        input_hash = sha256(input_path)
    predictions = predict_rows(metadata, model, rows, calibration)
    payload = {
        "schema_version": 1, "status": "complete", "run_id": output.name, "project": "ElectricityPlot",
        "created_at": datetime.now(timezone.utc).isoformat(), "model_id": metadata["model_id"],
        "model_manifest_sha256": sha256(model_dir / "model.json"), "model_kind": metadata["kind"],
        "model_settings": metadata["settings"], "code_sha256": source_fingerprint(),
        "protocol_sha256": sha256(manifest_path) if manifest_path else None,
        "partition_id": partition_id if manifest_path else None,
        "purpose": protocol["purpose"] if protocol else "user_import_no_ground_truth",
        "data_kind": protocol.get("data_kind", "dataset_measurements") if protocol else "user_import",
        "model_data_kind": metadata.get("data_kind", "dataset_measurements"),
        "input_sha256": input_hash, "sample_interval_seconds": metadata["sample_interval_seconds"],
        "appliances": metadata["appliances"], "upstream_limitations": metadata["upstream_limitations"],
        "inference_seconds": time.perf_counter() - started, "confidence_method": METHOD if calibration else None,
        "calibration": {"sha256": sha256(calibration_dir / "calibration.json"),
                        "interval_level": calibration["interval_level"], "notice": calibration["notice"],
                        "validation_partitions": calibration["validation_partitions"],
                        "coverage_guarantee": False} if calibration else None,
        "adaptation": metadata.get("adaptation", "none"), "state_definition": metadata["state_definition"],
        "query_summary": metadata.get("query_summary", {"budget_used": 0}),
        "base_model_id": metadata.get("base_model_id"),
        "rows": predictions,
    }
    validate_output(payload)
    payload["quality_summary"] = quality_summary(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".predict-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        write_json(staging / "predictions.json", payload)
        if calibration_dir is not None:
            (staging / "calibration.json").write_bytes((calibration_dir / "calibration.json").read_bytes())
        write_prediction_csv(staging / "predictions.csv", payload)
        from ..reporting.html import render_report
        render_report(payload, staging / "report.html")
        from ..app.results import write_result_manifest
        write_result_manifest(staging, payload)
        from ..data.common import publish_directory
        publish_directory(staging, output)
    return {"status": "prediction_complete", "run_id": output.name, "rows": len(predictions), "model_id": metadata["model_id"], "report": str(output / "report.html")}
