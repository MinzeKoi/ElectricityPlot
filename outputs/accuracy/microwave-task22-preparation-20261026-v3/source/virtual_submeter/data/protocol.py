"""Freeze household/time assignments and materialize separate label artifacts."""

import csv
from dataclasses import asdict
import math
from pathlib import Path
import tempfile

from ..config import ProjectConfig
from .common import DataError, iso, read_json, safe_child, sha256, source_fingerprint, utc, write_json

ROLES = ("source_train", "source_validation", "target_adaptation", "frozen_test")


def freeze(prepared: Path, plan_path: Path, output: Path, config: ProjectConfig) -> dict:
    manifest_path = prepared / "manifest.json"
    audit = read_json(manifest_path)
    plan = read_json(plan_path)
    required = {"schema_version", "protocol_id", "purpose", "source_train", "source_validation", "target_households", "adaptation_fraction", "max_model_context_seconds", "minimum_aggregate_coverage", "minimum_label_coverage"}
    if set(plan) != required or plan["schema_version"] != 1:
        raise DataError("Unknown or incomplete split plan")
    if not isinstance(plan["protocol_id"], str) or not plan["protocol_id"].strip():
        raise DataError("protocol_id is required")
    if plan["purpose"] not in ("engineering_pilot_only", "final_evaluation"):
        raise DataError("Split purpose must be explicit")
    groups = [plan[key] for key in ("source_train", "source_validation", "target_households")]
    if any(not isinstance(group, list) or not group or any(not isinstance(h, str) or not h.isdecimal() for h in group) for group in groups):
        raise DataError("Each household group must be a non-empty list of numeric strings")
    households = [house for group in groups for house in group]
    if len(set(households)) != len(households):
        raise DataError("Source train, source validation and target households must be disjoint")
    if any(house not in audit["households"] for house in households):
        raise DataError("A planned household has not been audited")
    context = plan["max_model_context_seconds"]
    if type(context) is not int or context <= 0 or config.split_gap_seconds < max(context, config.query_window_seconds):
        raise DataError("Isolation gap is shorter than model context or confirmation window")
    fraction = plan["adaptation_fraction"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise DataError("adaptation_fraction must be between zero and one")
    for field in ("minimum_aggregate_coverage", "minimum_label_coverage"):
        if type(plan[field]) not in (int, float) or not math.isfinite(plan[field]) or not 0 < plan[field] <= 1:
            raise DataError(f"{field} must be in (0, 1]")
    if audit["sample_interval_seconds"] != config.sample_interval_seconds:
        raise DataError("Sampling interval differs from project config")
    if output.exists():
        raise DataError("Protocol destination already exists; frozen protocols are immutable")
    assignments = []
    interval = config.sample_interval_seconds
    for house in households:
        stats = audit["households"][house]
        if any(appliance not in stats["artifacts"] for appliance in config.target_appliances):
            raise DataError(f"House {house}: missing target channel; absent channels are not off labels")
        if plan["purpose"] == "final_evaluation" and (not stats["provenance"] or stats["provenance"].get("kind") != "real_complete"):
            raise DataError("A partial or unverified file cannot freeze a final-evaluation protocol")
        start, end = utc(stats["start"]), utc(stats["end"])
        if house in plan["target_households"]:
            cut = start + int((end - start) * fraction / interval) * interval
            test_start = cut + config.split_gap_seconds
            if cut - start < config.query_window_seconds or end - test_start < max(interval, context):
                raise DataError(f"House {house}: insufficient duration for adaptation, isolation and test")
            assignments.extend([(house, "target_adaptation", start, cut), (house, "frozen_test", test_start, end)])
        else:
            role = "source_train" if house in plan["source_train"] else "source_validation"
            assignments.append((house, role, start, end))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        result = {
            "schema_version": 1, "project": "ElectricityPlot", "protocol_id": plan["protocol_id"], "purpose": plan["purpose"],
            "audit_sha256": sha256(manifest_path), "plan_sha256": sha256(plan_path),
            "code_sha256": source_fingerprint(), "config_snapshot": asdict(config),
            "sample_interval_seconds": interval, "target_appliances": list(config.target_appliances),
            "split_gap_seconds": config.split_gap_seconds, "max_model_context_seconds": context,
            "query_window_seconds": config.query_window_seconds, "query_budgets": list(config.query_budgets),
            "random_seeds": list(config.random_seeds), "upstream_limitations": audit["upstream_limitations"], "partitions": [],
        }
        for house, role, start, end in assignments:
            partition_id = f"{role}_house_{house}"
            partition = {"partition_id": partition_id, "household_id": house, "role": role,
                         "start": iso(start), "end": iso(end), "artifacts": {}}
            for channel in ("aggregate", *config.target_appliances):
                artifact = audit["households"][house]["artifacts"][channel]
                source = safe_child(prepared, artifact["path"])
                if sha256(source) != artifact["sha256"]:
                    raise DataError(f"Audit artifact hash mismatch: {source.name}")
                # Test and oracle labels live outside aggregate input directories.
                area = "inputs" if channel == "aggregate" else "evaluation_labels" if role == "frozen_test" else "oracle_labels" if role == "target_adaptation" else "source_labels"
                relative = f"{area}/{partition_id}/{channel}.csv"
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                rows = valid = 0
                with source.open(encoding="utf-8", newline="") as incoming, target.open("w", encoding="utf-8", newline="") as outgoing:
                    reader = csv.DictReader(incoming)
                    writer = csv.DictWriter(outgoing, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    power_column = "aggregate_power_w" if channel == "aggregate" else "power_w"
                    for row in reader:
                        stamp = utc(row["timestamp"])
                        if start <= stamp and stamp + interval <= end:
                            writer.writerow(row)
                            rows += 1
                            valid += int(bool(row[power_column]))
                threshold = plan["minimum_aggregate_coverage" if channel == "aggregate" else "minimum_label_coverage"]
                if rows == 0 or valid / rows < threshold:
                    raise DataError(f"{partition_id}/{channel}: coverage below declared minimum")
                partition["artifacts"][channel] = {"path": relative, "sha256": sha256(target), "rows": rows, "valid_rows": valid}
            result["partitions"].append(partition)
        write_json(staging / "protocol.json", result)
        # Keep the exact source bytes: the stored fingerprint includes whitespace.
        (staging / "plan.json").write_bytes(plan_path.read_bytes())
        from .common import publish_directory
        publish_directory(staging, output)
    return {"status": "protocol_frozen", "purpose": plan["purpose"], "manifest": str(output / "protocol.json"), "partitions": len(assignments)}


def read_partition(manifest_path: Path, partition_id: str, consumer: str, include_labels: bool = False) -> dict:
    """Policy boundary for cooperating code, not an OS-level security sandbox."""
    manifest = read_json(manifest_path)
    partitions = [item for item in manifest["partitions"] if item["partition_id"] == partition_id]
    if len(partitions) != 1:
        raise DataError("Unknown partition")
    partition = partitions[0]
    if consumer == "development_diagnosis" and (
            manifest.get("purpose") != "engineering_pilot_only"
            or manifest.get("accuracy_usage") != "development_only"):
        raise DataError("Diagnosis labels require an explicitly marked development protocol")
    allowed = {
        "training": {"source_train", "source_validation"},
        "calibration": {"source_validation"},
        "selection": {"target_adaptation"},
        "adaptation": {"target_adaptation"},
        "inference": {"source_validation", "frozen_test"},
        "evaluation": {"source_validation", "frozen_test"},
        "development_diagnosis": set(ROLES),
    }
    if partition["role"] not in allowed.get(consumer, set()):
        raise DataError(f"{consumer} cannot access {partition['role']}")
    if include_labels and consumer not in ("training", "calibration", "evaluation", "development_diagnosis"):
        raise DataError("Selection, adaptation and inference cannot request submeter labels")
    result = {}
    for channel, artifact in partition["artifacts"].items():
        if channel != "aggregate" and not include_labels:
            continue
        path = safe_child(manifest_path.parent, artifact["path"])
        if sha256(path) != artifact["sha256"]:
            raise DataError("Partition artifact hash mismatch")
        with path.open(encoding="utf-8", newline="") as stream:
            result[channel] = list(csv.DictReader(stream))
    return result


def validate_confirmation(record: dict, manifest: dict) -> None:
    """Validate a single budgeted weak response without accepting power labels."""
    fields = {"query_id", "household_id", "window_start", "window_end", "appliance_type", "response", "label_source", "selection_strategy", "model_version", "budget_before", "budget_after"}
    if set(record) != fields:
        raise DataError("Confirmation has missing/unknown fields; pointwise labels are forbidden")
    for key in fields - {"budget_before", "budget_after"}:
        if not isinstance(record[key], str) or not record[key].strip():
            raise DataError(f"{key} must be non-empty text")
    if record["response"] not in ("used", "not_used", "unsure", "skipped"):
        raise DataError("Unknown weak response")
    if record["label_source"] not in ("simulated_from_submeter", "human_confirmed"):
        raise DataError("Unknown label source")
    before, after = record["budget_before"], record["budget_after"]
    if type(before) is not int or type(after) is not int or before < 0 or after != before + 1 or after > max(manifest["query_budgets"]):
        raise DataError("Every answer consumes one confirmation, including unsure/skipped")
    if record["appliance_type"] not in manifest["target_appliances"]:
        raise DataError("Unknown target appliance")
    start, end = utc(record["window_start"]), utc(record["window_end"])
    interval = manifest["sample_interval_seconds"]
    if end - start != manifest["query_window_seconds"] or start % interval or end % interval:
        raise DataError("Confirmation must use one complete aligned query window")
    if not any(item["role"] == "target_adaptation" and item["household_id"] == record["household_id"] and utc(item["start"]) <= start < end <= utc(item["end"]) for item in manifest["partitions"]):
        raise DataError("Confirmation lies outside the household adaptation pool")


def validate_confirmation_history(records: list[dict], manifest: dict) -> None:
    seen = set()
    budgets = {}
    for record in records:
        validate_confirmation(record, manifest)
        if record["query_id"] in seen:
            raise DataError("Duplicate query_id")
        seen.add(record["query_id"])
        house = record["household_id"]
        if record["budget_before"] != budgets.get(house, 0):
            raise DataError("Budget history is not continuous per household")
        budgets[house] = record["budget_after"]
