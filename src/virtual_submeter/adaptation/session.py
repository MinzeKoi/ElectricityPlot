"""Persistent confirmation budget with a single committed pending question."""

from contextlib import contextmanager
import copy
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import tempfile

from ..data.common import DataError, read_json, sha256, source_fingerprint, write_json
from ..data.protocol import validate_confirmation_history
from ..models.artifacts import load_model
from ..models.prediction import predict
from .selection import STRATEGIES, candidates, choose, load_settings


@lru_cache(maxsize=8)
def _candidate_cache(prediction_path, prediction_digest, partition_json, protocol_json, settings_json):
    path = Path(prediction_path)
    if sha256(path) != prediction_digest:
        raise DataError("Session predictions changed")
    return candidates(read_json(path), json.loads(partition_json), json.loads(protocol_json), json.loads(settings_json))


def session_candidates(directory, session, protocol):
    """Reuse only immutable visible candidates; recheck bytes on every access."""
    path = directory / "prediction/predictions.json"
    if sha256(path) != session["prediction_sha256"]:
        raise DataError("Session predictions changed")
    partition = next(p for p in protocol["partitions"] if p["partition_id"] == session["partition_id"])
    visible_partition = {k: partition[k] for k in ("role", "partition_id", "household_id", "start", "end")}
    visible_protocol = {k: protocol[k] for k in ("sample_interval_seconds", "query_window_seconds")}
    encoded = [json.dumps(v, sort_keys=True) for v in (visible_partition, visible_protocol, session["settings"])]
    return copy.deepcopy(_candidate_cache(str(path.resolve()), session["prediction_sha256"], *encoded))


@contextmanager
def locked(directory):
    lock = directory / ".session.lock"
    try:
        stream = lock.open("x")
    except FileExistsError as error:
        raise DataError("Session is busy; retry after the current operation") from error
    try:
        stream.close()
        yield
    finally:
        lock.unlink(missing_ok=True)


def save(directory, session):
    # Replace one complete JSON document; failed writes keep the prior budget.
    temporary = directory / ".session.json.tmp"
    write_json(temporary, session)
    temporary.replace(directory / "session.json")


def read_session(directory):
    session = read_json(directory / "session.json")
    manifest = Path(session["manifest_path"])
    model = Path(session["model_path"])
    if session.get("schema_version") != 1 or session.get("status") != "active":
        raise DataError("Invalid confirmation session")
    if sha256(manifest) != session["protocol_sha256"] or sha256(model / "model.json") != session["model_manifest_sha256"]:
        raise DataError("Session protocol or model changed; create a new session")
    if sha256(directory / "prediction/predictions.json") != session["prediction_sha256"]:
        raise DataError("Session predictions changed")
    protocol = read_json(manifest)
    validate_confirmation_history(session["records"], protocol)
    if (session["budget"] not in protocol["query_budgets"] or len(session["records"]) > session["budget"]
            or any(r["household_id"] != session["household_id"] or r["model_version"] != session["model_id"]
                   or r["selection_strategy"] != session["strategy"] for r in session["records"])):
        raise DataError("Session budget or confirmation identity is invalid")
    eligible = session_candidates(directory, session, protocol)
    indexed = {item["query_id"]: item for item in eligible}
    for index, record in enumerate(session["records"]):
        expected = indexed.get(record["query_id"])
        if expected is None or any(record[k] != expected[k] for k in ("household_id", "appliance_type", "window_start", "window_end")):
            raise DataError("Confirmation does not match a selected complete window")
        selected = choose(eligible, session["records"][:index], session["strategy"], session["seed"])
        if selected is None or selected["query_id"] != record["query_id"]:
            raise DataError("Confirmation history does not replay the selection policy")
    expected = choose(eligible, session["records"], session["strategy"], session["seed"])
    if session["pending"] is not None and (len(session["records"]) >= session["budget"] or session["pending"] != expected):
        raise DataError("Pending question does not match the selection policy")
    return session


def create_session(model_dir: Path, manifest_path: Path, partition_id: str, output: Path,
                   settings_path: Path, budget: int, strategy="proposed", seed=42):
    if output.exists():
        raise DataError("Session destination exists; resume it or choose a new directory")
    settings = load_settings(settings_path)
    protocol = read_json(manifest_path)
    metadata, _ = load_model(model_dir)
    if metadata.get("adaptation") != "none":
        raise DataError("M3 comparisons start from one unadapted base model")
    if type(budget) is not int or budget not in protocol["query_budgets"]:
        raise DataError("Budget must be one of the frozen protocol budgets")
    if type(seed) is not int or seed not in protocol["random_seeds"] or strategy not in STRATEGIES:
        raise DataError("Strategy/seed must follow the frozen comparison protocol")
    partition = next((p for p in protocol["partitions"] if p["partition_id"] == partition_id), None)
    if partition is None or partition["role"] != "target_adaptation":
        raise DataError("Confirmation sessions require target adaptation, never test")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".session-", dir=output.parent) as temporary:
        staging = Path(temporary) / "session"
        staging.mkdir()
        predict(model_dir, staging / "prediction", manifest_path, partition_id)
        session = {"schema_version": 1, "status": "active", "created_at": datetime.now(timezone.utc).isoformat(),
                   "manifest_path": str(manifest_path.resolve()), "model_path": str(model_dir.resolve()),
                   "protocol_sha256": sha256(manifest_path), "model_manifest_sha256": sha256(model_dir / "model.json"),
                   "prediction_sha256": sha256(staging / "prediction/predictions.json"),
                   "model_id": metadata["model_id"], "partition_id": partition_id, "household_id": partition["household_id"],
                   "purpose": protocol["purpose"], "budget": budget, "strategy": strategy, "seed": seed,
                   "settings": settings, "code_sha256": source_fingerprint(), "records": [], "pending": None}
        write_json(staging / "session.json", session)
        (staging / "protocol.json").write_bytes(manifest_path.read_bytes())
        from ..data.common import publish_directory
        publish_directory(staging, output)
    return {"status": "session_created", "session": str(output), "budget": budget, "model_id": metadata["model_id"]}


def select_next(directory: Path):
    with locked(directory):
        session = read_session(directory)
        if session["pending"] is not None:
            return session["pending"]
        if len(session["records"]) >= session["budget"]:
            return None
        protocol = read_json(Path(session["manifest_path"]))
        session["pending"] = choose(session_candidates(directory, session, protocol), session["records"], session["strategy"], session["seed"])
        save(directory, session)
        return session["pending"]


def confirm(directory: Path, query_id: str, response: str, label_source="human_confirmed"):
    with locked(directory):
        session = read_session(directory)
        pending = session["pending"]
        if pending is None or query_id != pending["query_id"]:
            raise DataError("Answer must match the persisted pending query; duplicate/stale answer rejected")
        count = len(session["records"])
        record = {k: pending[k] for k in ("query_id", "household_id", "window_start", "window_end", "appliance_type")}
        record.update({"response": response, "label_source": label_source, "selection_strategy": session["strategy"],
                       "model_version": session["model_id"], "budget_before": count, "budget_after": count + 1})
        records = session["records"] + [record]
        validate_confirmation_history(records, read_json(Path(session["manifest_path"])))
        if len(records) > session["budget"]:
            raise DataError("Confirmation budget exhausted")
        session.update(records=records, pending=None)
        save(directory, session)
        return record
