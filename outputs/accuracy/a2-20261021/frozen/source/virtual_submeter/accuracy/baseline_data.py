"""A1 source/development protocol and quality-only household admission."""

import csv
import re
from pathlib import Path

from ..data.common import DataError, iso, read_json, safe_child, sha256, source_fingerprint, utc, write_json
from .diagnosis import event_support


def load_plan(path):
    plan = read_json(path)
    if (plan.get("schema_version") != 1 or plan.get("purpose") != "development_only"
            or plan.get("selection_rule_id") != "primary_f1_active_mae_energy_v1"
            or plan.get("state_threshold_view") != "raw_power_w"):
        raise DataError("Unsupported A1 plan or selection rule")
    groups = [plan[k] for k in ("source_train_required", "source_train_optional", "source_validation", "development_households")]
    households = [h for group in groups for h in group]
    if any(not isinstance(group, list) for group in groups) or any(not isinstance(h, str) or not h.isdecimal() for h in households) or len(set(households)) != len(households):
        raise DataError("A1 household roles must be disjoint lists of numeric identifiers")
    if any(not plan[k] for k in ("source_train_required", "source_validation", "development_households")):
        raise DataError("A1 needs source train, validation and development households")
    if (not 1 <= len(plan["candidates"]) <= 3 or not plan["seeds"]
            or len(set(plan["seeds"])) != len(plan["seeds"])
            or any(type(s) is not int or s < 0 for s in plan["seeds"])):
        raise DataError("A1 allows at most three candidates and unique nonnegative training seeds")
    ids = [c["id"] for c in plan["candidates"]]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or re.fullmatch(r"[a-z][a-z0-9_]*", i) is None for i in ids):
        raise DataError("Invalid A1 candidate identifiers")
    for candidate in plan["candidates"]:
        if set(candidate) != {"id", "epochs", "active_loss_weight"}:
            raise DataError("A1 candidates change only fixed epochs and active loss weight")
    if type(plan["maximum_days_per_household"]) is not int or plan["maximum_days_per_household"] <= 0:
        raise DataError("A1 maximum duration must be positive whole days")
    if type(plan.get("development_duration_days")) is not int or not 0 < plan["development_duration_days"] <= plan["maximum_days_per_household"]:
        raise DataError("A1 development duration must fit within the household maximum")
    if not 0 < plan["minimum_coverage"] <= 1 or plan["minimum_source_observed_days"] <= 0 or type(plan["minimum_complete_events"]) is not int or plan["minimum_complete_events"] < 1:
        raise DataError("Invalid A1 source-quality thresholds")
    targets = plan["primary_appliances"] + plan["extension_appliances"]
    if not plan["primary_appliances"] or len(targets) != len(set(targets)) or "kettle" not in targets:
        raise DataError("A1 requires unique primary/extension devices including kettle")
    return plan


def copy_partition(prepared, info, house, role, start, end, destination, targets, thresholds):
    interval = 8
    pid = f"{role}_house_{house}"
    partition = {"partition_id": pid, "household_id": house, "role": role,
                 "start": iso(start), "end": iso(end), "artifacts": {}}
    support = {}
    for channel in ("aggregate", *targets):
        if channel not in info["artifacts"]:
            raise DataError(f"House {house}: missing {channel}; absent channels are not zero")
        artifact = info["artifacts"][channel]
        source = safe_child(prepared, artifact["path"])
        if sha256(source) != artifact["sha256"]:
            raise DataError("A1 input audit artifact hash mismatch")
        area = "inputs" if channel == "aggregate" else "oracle_labels" if role == "target_adaptation" else "evaluation_labels" if role == "frozen_test" else "source_labels"
        relative = f"{area}/{pid}/{channel}.csv"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        selected = []
        with source.open(encoding="utf-8", newline="") as incoming, target.open("w", encoding="utf-8", newline="") as outgoing:
            reader = csv.DictReader(incoming)
            writer = csv.DictWriter(outgoing, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                stamp = utc(row["timestamp"])
                if start <= stamp and stamp + interval <= end:
                    writer.writerow(row)
                    selected.append(row)
        power = "aggregate_power_w" if channel == "aggregate" else "power_w"
        valid = sum(r[power] != "" for r in selected)
        partition["artifacts"][channel] = {"path": relative, "sha256": sha256(target), "rows": len(selected), "valid_rows": valid}
        support[channel] = {"bins": len(selected), "valid_bins": valid,
                            "coverage": valid / len(selected) if selected else 0,
                            "observed_days": valid * interval / 86400}
        if channel != "aggregate":
            support[channel].update(event_support(selected, thresholds[channel], interval))
            support[channel]["off_samples"] = valid - support[channel]["active_samples"]
            support[channel]["off_fraction"] = support[channel]["off_samples"] / valid if valid else None
    return partition, support


def build_protocol(sources, development, output, plan, settings):
    if output.exists():
        raise DataError("A1 protocol output already exists")
    output.mkdir(parents=True)
    source_audit, dev_audit = read_json(sources / "manifest.json"), read_json(development / "manifest.json")
    if any(a["sample_interval_seconds"] != 8 for a in (source_audit, dev_audit)):
        raise DataError("A1 requires the existing 8-second grid")
    targets = plan["primary_appliances"] + plan["extension_appliances"]
    measured = all((info.get("provenance") or {}).get("kind") in ("real_prefix", "real_complete")
                   for audit in (source_audit, dev_audit) for info in audit["households"].values())
    protocol = {"schema_version": 1, "project": "ElectricityPlot", "protocol_id": "refit-a1-development-v1",
                "purpose": "engineering_pilot_only", "accuracy_usage": "development_only",
                "data_kind": "real_REFIT_prefix" if measured else "synthetic_contract_fixture",
                "sample_interval_seconds": 8, "target_appliances": targets,
                "split_gap_seconds": 3600, "max_model_context_seconds": 3600, "query_window_seconds": 1200,
                "query_budgets": [0, 5, 10, 20], "random_seeds": plan["seeds"],
                "upstream_limitations": source_audit["upstream_limitations"], "code_sha256": source_fingerprint(),
                "source_audit_sha256": sha256(sources / "manifest.json"), "development_audit_sha256": sha256(development / "manifest.json"),
                "notice": "Inspected development household; no final evaluation or confirmation labels used in source training.", "partitions": []}
    admission, support_rows = [], []
    assignments = [(h, "source_train", sources, source_audit, h in plan["source_train_optional"]) for h in plan["source_train_required"] + plan["source_train_optional"]]
    assignments += [(h, "source_validation", sources, source_audit, False) for h in plan["source_validation"]]
    assignments += [(h, "target_adaptation", development, dev_audit, False) for h in plan["development_households"]]
    for house, role, prepared, audit, optional in assignments:
        info = audit["households"].get(house)
        if info is None:
            if optional:
                admission.append({"household_id": house, "included": False, "reasons": ["file_unavailable"]})
                continue
            raise DataError(f"A1 required house {house} unavailable")
        start = utc(info["start"])
        days = plan["development_duration_days"] if role == "target_adaptation" else plan["maximum_days_per_household"]
        end = min(utc(info["end"]), start + days * 86400)
        spans = [(role, start, end)]
        if role == "target_adaptation":
            middle = start + ((end - start) // 16) * 8
            spans = [("target_adaptation", start, middle), ("frozen_test", middle + 3600, end)]
        pending, house_support = [], []
        for part_role, first, last in spans:
            if last - first < 3600:
                raise DataError("A1 intervals are too short for contexts and isolation")
            partition, support = copy_partition(prepared, info, house, part_role, first, last, output, targets, settings["on_thresholds_w"])
            pending.append(partition)
            house_support.append({"household_id": house, "role": part_role, "partition_id": partition["partition_id"], "channels": support})
        reasons = []
        for part in house_support:
            if any(v["coverage"] < plan["minimum_coverage"] for v in part["channels"].values()):
                reasons.append("coverage_below_minimum")
            for appliance in plan["primary_appliances"]:
                stats = part["channels"][appliance]
                if role != "target_adaptation" and stats["observed_days"] < plan["minimum_source_observed_days"]:
                    reasons.append(f"{appliance}_insufficient_observed_days")
                if stats["complete_threshold_events"] < plan["minimum_complete_events"]:
                    reasons.append(f"{part['role']}_{appliance}_insufficient_complete_events")
        included = not reasons
        admission.append({"household_id": house, "role": role, "optional": optional, "included": included,
                          "start": iso(start), "end": iso(end), "kind": (info.get("provenance") or {}).get("kind"), "reasons": sorted(set(reasons))})
        support_rows.extend(house_support)
        if not included and not optional:
            write_json(output / "admission.json", {"households": admission, "partition_support": support_rows})
            raise DataError(f"A1 required house {house} fails predeclared data-quality gates: {reasons}")
        if included:
            protocol["partitions"].extend(pending)
    write_json(output / "admission.json", {"households": admission, "partition_support": support_rows})
    write_json(output / "protocol.json", protocol)
    return output / "protocol.json"
