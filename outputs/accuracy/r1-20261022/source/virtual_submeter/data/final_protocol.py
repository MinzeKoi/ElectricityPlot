"""A5 data-side partitioning; admission uses coverage, never model scores.

Creating label files is a data preparation operation. No event statistics or
target power values are returned to selection/training or used for admission.
"""

import copy
import csv
from pathlib import Path
import shutil

from .common import DataError, iso, read_json, safe_child, sha256, utc, write_json


def prepare(parent_protocol, prepared, plan, output, declaration=None):
    from ..accuracy.final_evaluation import validate_plan
    validate_plan(plan, require_targets=True)
    if output.exists():
        raise DataError("A5 protocol destination exists")
    original, audit = read_json(parent_protocol), read_json(prepared / "manifest.json")
    targets = plan["target_households"]
    if audit["sample_interval_seconds"] != original["sample_interval_seconds"]:
        raise DataError("A5 target sample interval differs from frozen sources")
    used = {p["household_id"] for p in original["partitions"]} | set(plan["excluded_households"])
    if set(targets) & used:
        raise DataError("A5 targets overlap source, validation or inspected development households")
    statement = read_json(declaration) if declaration else None
    if plan["mode"] == "final":
        if (not statement or statement.get("schema_version") != 1 or statement.get("future_labels_unseen") is not True
                or statement.get("target_households") != targets or not str(statement.get("declared_by", "")).strip()):
            raise DataError("Final A5 needs an explicit named, dated unseen-label declaration")
        utc(statement.get("declared_at", ""))
    admission = []
    for house in targets:
        info = audit["households"].get(house)
        if info is None:
            raise DataError(f"A5 target house {house} is unavailable")
        kind = (info.get("provenance") or {}).get("kind")
        if plan["mode"] == "final" and kind != "real_complete":
            raise DataError("Final A5 rejects prefix or synthetic target files")
        if any(a not in info["artifacts"] for a in ("aggregate", *original["target_appliances"])):
            raise DataError(f"A5 target house {house} is missing a frozen appliance channel")
        admission.append({"household_id": house, "provenance": info.get("provenance"),
                          "start": info["start"], "end": info["end"]})
    output.mkdir(parents=True)
    protocol = copy.deepcopy(original)
    protocol.update(protocol_id="a5-final-v1" if plan["mode"] == "final" else "a5-engineering-v1",
                    purpose="final_evaluation" if plan["mode"] == "final" else "engineering_pilot_only",
                    accuracy_usage="final_evaluation" if plan["mode"] == "final" else "engineering_only",
                    data_kind="real_REFIT_complete" if plan["mode"] == "final" else "engineering_fixture",
                    a5_parent_protocol_sha256=sha256(parent_protocol), a5_plan=plan,
                    a5_admission=admission, a5_unseen_declaration=statement,
                    a5_target_audit_sha256=sha256(prepared / "manifest.json"), partitions=[])
    if "date_selection" in audit:
        protocol["protocol_id"] = "a5-final-date-selected-v1" if plan["mode"] == "final" else "a5-engineering-date-selected-v1"
        protocol["a5_date_selection"] = copy.deepcopy(audit["date_selection"])
        protocol["a5_date_selection_sha256"] = audit["date_selection_sha256"]
    try:
        for partition in original["partitions"]:
            if partition["role"] not in ("source_train", "source_validation"):
                continue
            for artifact in partition["artifacts"].values():
                source = safe_child(parent_protocol.parent, artifact["path"])
                if sha256(source) != artifact["sha256"]:
                    raise DataError("Frozen source artifact changed")
                destination = safe_child(output, artifact["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            protocol["partitions"].append(copy.deepcopy(partition))
        interval = protocol["sample_interval_seconds"]
        for house in targets:
            info = audit["households"][house]
            start = utc(info["start"])
            end = min(utc(info["end"]), start + plan["maximum_days"] * 86400)
            cut = start + int((end - start) * plan["adaptation_fraction"] / interval) * interval
            for role, first, last in (("target_adaptation", start, cut), ("frozen_test", cut + protocol["split_gap_seconds"], end)):
                pid = f"{role}_house_{house}"
                partition = {"partition_id": pid, "household_id": house, "role": role,
                             "start": iso(first), "end": iso(last), "artifacts": {}}
                for channel in ("aggregate", *protocol["target_appliances"]):
                    artifact = info["artifacts"][channel]
                    source = safe_child(prepared, artifact["path"])
                    if sha256(source) != artifact["sha256"]:
                        raise DataError("Target audit artifact changed")
                    area = "inputs" if channel == "aggregate" else "oracle_labels" if role == "target_adaptation" else "evaluation_labels"
                    relative = f"{area}/{pid}/{channel}.csv"
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    count = valid = 0
                    power = "aggregate_power_w" if channel == "aggregate" else "power_w"
                    with source.open(encoding="utf-8", newline="") as incoming, destination.open("w", encoding="utf-8", newline="") as outgoing:
                        reader = csv.DictReader(incoming)
                        writer = csv.DictWriter(outgoing, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        for row in reader:
                            stamp = utc(row["timestamp"])
                            if first <= stamp and stamp + interval <= last:
                                writer.writerow(row)
                                count += 1
                                valid += int(bool(row[power]))
                    if (not count or valid / count < plan["minimum_coverage"]
                            or valid * interval / 86400 < plan["minimum_observed_days"] / 2
                            or last - first < protocol["query_window_seconds"]):
                        raise DataError(f"A5 {pid}/{channel} fails predeclared duration/coverage")
                    partition["artifacts"][channel] = {"path": relative, "sha256": sha256(destination), "rows": count, "valid_rows": valid}
                protocol["partitions"].append(partition)
        write_json(output / "protocol.json", protocol)
        write_json(output / "preparation.json", {"status": "prepared", "protocol_sha256": sha256(output / "protocol.json"),
                   "label_values_used_for_admission": False, "admission_basis": "frozen channels, duration, nonmissing coverage; no model metrics"})
        return {"status": "a5_protocol_prepared", "manifest": str(output / "protocol.json"), "formal_evaluation": plan["mode"] == "final"}
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "preparation.json", {"status": "failed", "error": str(error)})
        raise
