"""Empirical residual intervals fitted only on source validation households.

Validation was also used for checkpoint selection. These are descriptive source
quantiles, NOT split-conformal coverage guarantees, especially across households.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

from ..data.common import DataError, publish_directory, read_json, sha256, source_fingerprint, utc, write_json
from ..data.protocol import read_partition

METHOD = "source_validation_absolute_residual_quantiles"
NOTICE = "Source validation reused for model selection; empirical marginal intervals only. No target-household or simultaneous coverage guarantee. State confidence remains unavailable."


def power_fingerprint(metadata: dict) -> str:
    # Weak state adaptation changes no power parameters, so it may reuse these
    # intervals. A changed backbone, scaling, target list or protocol may not.
    keys = ("kind", "appliances", "settings", "sample_interval_seconds", "protocol_sha256",
            "input_log_mean", "input_log_std", "power_scales_w", "constant_power_w", "weights")
    body = {key: metadata.get(key) for key in keys}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_settings(path: Path) -> dict:
    settings = read_json(path)
    if not isinstance(settings, dict) or set(settings) != {"schema_version", "interval_level", "minimum_samples", "stress"} or type(settings["schema_version"]) is not int or settings["schema_version"] != 1:
        raise DataError("Incomplete or unknown M4 settings")
    level = settings["interval_level"]
    if type(level) not in (float, int) or not math.isfinite(level) or not 0 < level < 1:
        raise DataError("interval_level must lie strictly between zero and one")
    if type(settings["minimum_samples"]) is not int or settings["minimum_samples"] < 2:
        raise DataError("minimum_samples must be at least two")
    stress = settings["stress"]
    if not isinstance(stress, dict) or set(stress) != {"power_w", "period_samples", "duration_samples", "offset_samples"}:
        raise DataError("Incomplete stress settings")
    if type(stress["power_w"]) not in (int, float) or not math.isfinite(stress["power_w"]) or stress["power_w"] <= 0:
        raise DataError("Stress power must be finite and positive")
    if any(type(stress[k]) is not int for k in ("period_samples", "duration_samples", "offset_samples")) or not 0 <= stress["offset_samples"] < stress["period_samples"] or not 0 < stress["duration_samples"] < stress["period_samples"]:
        raise DataError("Stress pulses require 0 < duration < period and 0 <= offset < period")
    return settings


def calibrate(model_dir: Path, manifest_path: Path, settings_path: Path, output: Path) -> dict:
    from ..models.artifacts import load_model
    from ..models.prediction import predict_rows
    if output.exists():
        raise DataError("Calibration destination exists; previous calibrations are immutable")
    settings = load_settings(settings_path)
    metadata, model = load_model(model_dir)
    protocol = read_json(manifest_path)
    if metadata["protocol_sha256"] != sha256(manifest_path):
        raise DataError("Calibration protocol differs from the trained model")
    if metadata.get("adaptation_household_id"):
        raise DataError("Calibrate the frozen power backbone before target state adaptation")
    partitions = [p for p in protocol["partitions"] if p["role"] == "source_validation"]
    forbidden = set(metadata["training_households"]) | {p["household_id"] for p in protocol["partitions"] if p["role"] in ("target_adaptation", "frozen_test")}
    if not partitions or any(p["household_id"] in forbidden or p["household_id"] not in metadata["validation_households"] for p in partitions):
        raise DataError("Calibration requires isolated source validation households")
    residuals = {a: {"raw": [], "corrected": []} for a in metadata["appliances"]}
    counts, totals = {}, []
    for partition in partitions:
        data = read_partition(manifest_path, partition["partition_id"], "calibration", include_labels=True)
        rows = predict_rows(metadata, model, data["aggregate"])
        counts[partition["partition_id"]] = {}
        totals.extend(row["aggregate_power_w"] for row in rows if row["raw_excess_w"] is not None)
        for appliance in metadata["appliances"]:
            truth = {(r["household_id"], utc(r["timestamp"])): float(r["power_w"]) for r in data[appliance] if r["power_w"] != ""}
            count = 0
            for row in rows:
                actual = truth.get((row["household_id"], utc(row["timestamp"])))
                values = row["appliances"][appliance]
                if actual is None or values["power_w"] is None:
                    continue
                if not math.isfinite(actual) or actual < 0:
                    raise DataError("Invalid calibration ground truth")
                for version, key in (("raw", "raw_power_w"), ("corrected", "power_w")):
                    residuals[appliance][version].append(abs(values[key] - actual))
                count += 1
            counts[partition["partition_id"]][appliance] = count
    quantiles = {}
    for appliance, versions in residuals.items():
        quantiles[appliance] = {}
        for version, errors in versions.items():
            if len(errors) < settings["minimum_samples"]:
                raise DataError(f"Insufficient source validation samples for {appliance}")
            ordered = sorted(errors)
            q = ordered[math.ceil(len(ordered) * settings["interval_level"]) - 1]
            quantiles[appliance][version] = {"radius_w": q, "samples": len(errors),
                                           "source_empirical_coverage": sum(e <= q for e in errors) / len(errors)}
    result = {"schema_version": 1, "status": "calibrated", "method": METHOD,
              "created_at": datetime.now(timezone.utc).isoformat(), "model_id": metadata["model_id"],
              "power_fingerprint": power_fingerprint(metadata), "protocol_sha256": sha256(manifest_path),
              "model_manifest_sha256": sha256(model_dir / "model.json"), "code_sha256": source_fingerprint(),
              "settings_sha256": sha256(settings_path), "interval_level": settings["interval_level"],
              "sample_interval_seconds": metadata["sample_interval_seconds"], "appliances": metadata["appliances"],
              "data_kind": protocol.get("data_kind", "dataset_measurements"),
              "validation_partitions": counts, "target_labels_read": False,
              "source_aggregate_range_w": [min(totals), max(totals)], "quantiles": quantiles,
              "validation_reused_for_model_selection": True, "coverage_guarantee": False, "notice": NOTICE}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".calibrate-", dir=output.parent) as temporary:
        staging = Path(temporary) / "calibration"
        staging.mkdir()
        write_json(staging / "calibration.json", result)
        (staging / "settings.json").write_bytes(settings_path.read_bytes())
        (staging / "protocol.json").write_bytes(manifest_path.read_bytes())
        publish_directory(staging, output)
    return {"status": "calibration_complete", "calibration": str(output / "calibration.json"), "notice": NOTICE}


def load_calibration(directory: Path, metadata: dict) -> dict:
    value = read_json(directory / "calibration.json")
    if value.get("schema_version") != 1 or value.get("status") != "calibrated" or value.get("method") != METHOD or value.get("power_fingerprint") != power_fingerprint(metadata):
        raise DataError("Calibration is invalid or belongs to a different power model")
    if value["protocol_sha256"] != metadata["protocol_sha256"] or value["appliances"] != metadata["appliances"] or not 0 < value["interval_level"] < 1:
        raise DataError("Calibration targets/protocol/level mismatch")
    if sha256(directory / "settings.json") != value["settings_sha256"] or sha256(directory / "protocol.json") != value["protocol_sha256"]:
        raise DataError("Calibration provenance hash mismatch")
    settings = load_settings(directory / "settings.json")
    if value["interval_level"] != settings["interval_level"] or value.get("sample_interval_seconds") != metadata["sample_interval_seconds"]:
        raise DataError("Calibration level or sampling interval differs from its frozen settings")
    bounds = value.get("source_aggregate_range_w")
    if not isinstance(bounds, list) or len(bounds) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds) or not 0 <= bounds[0] <= bounds[1]:
        raise DataError("Invalid source calibration aggregate range")
    if value.get("target_labels_read") is not False or value.get("coverage_guarantee") is not False:
        raise DataError("Calibration has unsupported label access or coverage claims")
    for appliance in value["appliances"]:
        for version in ("raw", "corrected"):
            q = value["quantiles"][appliance][version]
            if type(q["radius_w"]) not in (float, int) or not math.isfinite(q["radius_w"]) or q["radius_w"] < 0 or type(q["samples"]) is not int or q["samples"] < settings["minimum_samples"]:
                raise DataError("Invalid calibration radius/support")
            coverage = q.get("source_empirical_coverage")
            if type(coverage) not in (int, float) or not math.isfinite(coverage) or not value["interval_level"] <= coverage <= 1:
                raise DataError("Invalid source empirical coverage")
    return value


def apply_intervals(rows: list[dict], calibration: dict | None) -> None:
    for row in rows:
        flags = set(row["quality_flags"])
        flags.add("source_validation_intervals_no_target_guarantee" if calibration else "power_uncalibrated")
        if row["unexplained_power_w"] is not None and row["unexplained_power_w"] > 0:
            flags.add("unexplained_load_present")
        if calibration and row["aggregate_power_w"] is not None:
            low, high = calibration["source_aggregate_range_w"]
            if not low <= row["aggregate_power_w"] <= high:
                flags.add("aggregate_outside_calibration_range")
        row["quality_flags"] = sorted(flags)
        for appliance, values in row["appliances"].items():
            for version, key, lower, upper in (("raw", "raw_power_w", "raw_lower_w", "raw_upper_w"), ("corrected", "power_w", "lower_w", "upper_w")):
                center = values[key]
                radius = calibration["quantiles"][appliance][version]["radius_w"] if calibration and center is not None else None
                values[lower] = max(0.0, center - radius) if radius is not None else None
                values[upper] = center + radius if radius is not None else None
