"""Simulated user: expose just one weak answer for a persisted pending query.

Only this data-side adapter reads oracle_labels. Selection and fitting never
receive these samples, power values, exact event times or unqueried answers.
"""

import csv
import math
from pathlib import Path

from .common import DataError, read_json, safe_child, sha256, utc


def answer_pending(session_dir: Path) -> dict:
    # Import lazily so the ordinary data import does not require NumPy.
    from ..adaptation.session import read_session
    session = read_session(session_dir)
    query = session["pending"]
    if query is None or len(session["records"]) >= session["budget"]:
        raise DataError("Oracle requires a committed, budgeted pending query")
    manifest_path = Path(session["manifest_path"])
    manifest = read_json(manifest_path)
    partition = next(p for p in manifest["partitions"] if p["partition_id"] == session["partition_id"])
    if partition["role"] != "target_adaptation":
        raise DataError("Oracle cannot access frozen test answers")
    artifact = partition["artifacts"][query["appliance_type"]]
    path = safe_child(manifest_path.parent, artifact["path"])
    if not path.relative_to(manifest_path.parent.resolve()).parts[0] == "oracle_labels":
        raise DataError("Oracle label must live in oracle_labels")
    if sha256(path) != artifact["sha256"]:
        raise DataError("Oracle artifact hash mismatch")
    start, end = utc(query["window_start"]), utc(query["window_end"])
    interval = manifest["sample_interval_seconds"]
    expected = set(range(start, end, interval))
    observed, values = set(), []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            stamp = utc(row["timestamp"])
            if not start <= stamp < end:
                continue
            if stamp in observed or row["household_id"] != query["household_id"]:
                raise DataError("Invalid oracle window alignment")
            observed.add(stamp)
            value = float(row["power_w"]) if row["power_w"] else None
            coverage = float(row.get("coverage", 1))
            if value is not None and (not math.isfinite(value) or value < 0):
                raise DataError("Invalid oracle power")
            values.append(value if coverage == 1 else None)
    threshold = read_json(Path(session["model_path"]) / "model.json")["settings"]["on_thresholds_w"][query["appliance_type"]]
    # Missing support never becomes a negative/off label, even in simulation.
    response = "unsure" if observed != expected or any(v is None for v in values) else "used" if any(v >= threshold for v in values) else "not_used"
    return {"query_id": query["query_id"], "response": response, "label_source": "simulated_from_submeter"}
