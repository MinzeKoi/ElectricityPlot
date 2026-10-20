"""Evaluate saved predictions after training is frozen; never feed answers to fitting."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile

from ..data.common import DataError, read_json, sha256, utc, write_json
from ..data.protocol import read_partition
from .metrics import appliance_metrics, consistency_metrics


def evaluate(prediction_dir: Path, manifest_path: Path, output: Path) -> dict:
    if output.exists():
        raise DataError("Evaluation destination exists; never replace a previous result")
    prediction_path = prediction_dir / "predictions.json"
    payload = read_json(prediction_path)
    protocol = read_json(manifest_path)
    if payload.get("status") != "complete" or payload["protocol_sha256"] != sha256(manifest_path):
        raise DataError("Predictions do not belong to this frozen protocol")
    partition = next((p for p in protocol["partitions"] if p["partition_id"] == payload["partition_id"]), None)
    if partition is None or partition["role"] not in ("source_validation", "frozen_test"):
        raise DataError("Evaluation requires source validation or frozen test, never adaptation answers")
    if any(row["household_id"] != partition["household_id"] or not utc(partition["start"]) <= utc(row["timestamp"]) < utc(partition["end"]) for row in payload["rows"]):
        raise DataError("Prediction rows escape the declared evaluation partition")
    data = read_partition(manifest_path, payload["partition_id"], "evaluation", include_labels=True)
    expected = [(row["household_id"], row["timestamp"]) for row in data["aggregate"]]
    actual = [(row["household_id"], row["timestamp"]) for row in payload["rows"]]
    if actual != expected or payload["input_sha256"] != partition["artifacts"]["aggregate"]["sha256"]:
        raise DataError("Evaluation must include every input bin, retaining unavailable predictions as null")
    interval = payload["sample_interval_seconds"]
    if interval != protocol["sample_interval_seconds"]:
        raise DataError("Evaluation sampling interval changed")
    metrics = {}
    for appliance in payload["appliances"]:
        threshold = payload["model_settings"]["on_thresholds_w"][appliance]
        metrics[appliance] = {"raw": appliance_metrics(payload["rows"], data[appliance], appliance, threshold, interval, "raw_power_w"),
                              "corrected": appliance_metrics(payload["rows"], data[appliance], appliance, threshold, interval)}
    result = {"schema_version": 1, "status": "evaluated", "created_at": datetime.now(timezone.utc).isoformat(),
              "model_id": payload["model_id"], "model_kind": payload["model_kind"], "purpose": payload["purpose"],
              "partition_id": payload["partition_id"], "role": partition["role"],
              "prediction_sha256": sha256(prediction_path), "protocol_sha256": sha256(manifest_path),
              "metrics": metrics, "consistency": consistency_metrics(payload["rows"], interval),
              "notice": "Engineering pilot on real file prefixes; not final multi-house performance." if payload["purpose"] == "engineering_pilot_only" else "Frozen evaluation; report all coverage limitations."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluate-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        write_json(staging / "metrics.json", result)
        from ..reporting.html import render_report
        render_report(payload, staging / "report.html", result)
        staging.rename(output)
    return {"status": "evaluation_complete", "model_id": payload["model_id"], "metrics": str(output / "metrics.json"), "report": str(output / "report.html")}
