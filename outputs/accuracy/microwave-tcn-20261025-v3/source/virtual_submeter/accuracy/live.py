"""Human confirmation service using the same fits and round replay as A3/R2.

Only aggregates are copied into a session. Submeter answers are never opened.
This is a development/demo interface, not admission of a final method.
"""

import csv
import shutil

import numpy as np

from ..adaptation.session import locked
from ..data.common import DataError, inventory, iso, read_json, safe_child, sha256, utc, write_json
from ..data.protocol import read_partition
from ..models.artifacts import load_model
from ..models.prediction import predict
from . import dynamic_session as sessions
from .dynamic_queries import _export
from .fixed_queries import load_plan as learner_plan
from .learners import fit_adapter, score_adapter, temporal_features


def bind_import(model, parent_manifest, aggregate, output):
    """Bind unchanged source weights to a user aggregate with no hidden labels."""
    if output.exists():
        raise DataError("Imported context destination exists")
    protocol = read_json(parent_manifest)
    if protocol.get("purpose") == "final_evaluation" or protocol.get("accuracy_usage") == "final_evaluation":
        raise DataError("Cannot use a final protocol for an import demonstration")
    metadata, _ = load_model(model)
    if metadata["protocol_sha256"] != sha256(parent_manifest) or metadata.get("adaptation_household_id"):
        raise DataError("Import confirmations need a matching frozen source model")
    with aggregate.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["household_id"] for r in rows}) != 1:
        raise DataError("Import needs one nonempty household aggregate")
    house, interval = rows[0]["household_id"], protocol["sample_interval_seconds"]
    if any(utc(b["timestamp"]) - utc(a["timestamp"]) != interval for a, b in zip(rows, rows[1:])):
        raise DataError("Imported aggregate must use a complete normalized timeline")
    output.mkdir(parents=True)
    destination = output / "protocol/inputs/aggregate.csv"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(aggregate, destination)
    partition_id = "target_adaptation_import"
    protocol.update(protocol_id="user-import-" + sha256(aggregate)[:16], purpose="user_import_no_ground_truth",
                    accuracy_usage="development_only", data_kind="user_import", partitions=[{
                        "partition_id": partition_id, "household_id": house, "role": "target_adaptation",
                        "start": rows[0]["timestamp"], "end": iso(utc(rows[-1]["timestamp"]) + interval),
                        "artifacts": {"aggregate": {"path": "inputs/aggregate.csv", "sha256": sha256(destination),
                        "rows": len(rows), "valid_rows": sum(r["aggregate_power_w"] != "" for r in rows)}}}])
    manifest = output / "protocol/protocol.json"
    write_json(manifest, protocol)
    shutil.copytree(model, output / "model")
    metadata.update(imported_without_training=True, training_protocol_sha256=metadata["protocol_sha256"],
                    protocol_sha256=sha256(manifest), model_id=metadata["model_id"] + "-import-" + sha256(aggregate)[:12])
    write_json(output / "model/model.json", metadata)
    return {"model": str(output / "model"), "manifest": str(manifest), "partition_id": partition_id}


def create(directory, model, manifest, partition_id, learner, query, legacy,
           strategy, seed, budget, source_head=None):
    """Snapshot the exact source/config and visible aggregate, then fit round 0."""
    if directory.exists():
        raise DataError("Live session destination exists; load it or choose a new directory")
    plan = sessions.load_plan(query)
    learner_plan(learner)
    protocol = read_json(manifest)
    part = next((p for p in protocol["partitions"] if p["partition_id"] == partition_id), None)
    if not part or part["role"] != "target_adaptation":
        raise DataError("Confirmations require an adaptation partition")
    if protocol.get("accuracy_usage") == "final_evaluation" or protocol.get("purpose") == "final_evaluation":
        raise DataError("Live demonstrations cannot open a reserved final protocol")
    metadata, _ = load_model(model)
    if metadata["protocol_sha256"] != sha256(manifest):
        raise DataError("Live source model and protocol differ")
    if metadata.get("adaptation_household_id") or metadata.get("accuracy_power_head"):
        raise DataError("Start from a frozen source model, without target or power adaptation")
    if plan["learner"] == "L2_max" and source_head is None:
        raise DataError("L2_max requires its frozen source head and replay")
    directory.mkdir(parents=True)
    inputs = directory / "inputs"
    shutil.copytree(model, inputs / "model", ignore=shutil.ignore_patterns("__pycache__"))
    for path, name in ((manifest, "protocol/protocol.json"), (learner, "learner.json"),
                       (query, "query.json"), (legacy, "legacy.json")):
        target = inputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    artifact = part["artifacts"]["aggregate"]
    original = safe_child(manifest.parent, artifact["path"])
    if sha256(original) != artifact["sha256"]:
        raise DataError("Adaptation aggregate changed")
    target = safe_child(inputs / "protocol", artifact["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, target)
    if plan["learner"] == "L2_max":
        head = read_json(source_head / "source_head.json")
        if (head["base_model_sha256"] != sha256(model / "model.json")
                or head["weights_sha256"] != metadata["weights"]["sha256"]
                or head["protocol_sha256"] != sha256(manifest)):
            raise DataError("L2 source head does not belong to this source model")
        (inputs / "source_head").mkdir()
        for name in ("source_head.json", "source_replay.npz"):
            shutil.copyfile(source_head / name, inputs / "source_head" / name)
    local_manifest = inputs / "protocol/protocol.json"
    predict(inputs / "model", directory / "prediction", local_manifest, partition_id)
    payload = read_json(directory / "prediction/predictions.json")
    pool = sessions.build_pool(payload, part, protocol, read_json(inputs / "legacy.json"))
    sessions.create_session(directory / "session", local_manifest, inputs / "model", partition_id,
                            pool, plan, strategy, seed, budget)
    write_json(directory / "service.json", {
        "schema_version": 1, "usage": "development_demo_only", "learner": plan["learner"],
        "inputs_sha256": inventory(inputs), "prediction_sha256": sha256(directory / "prediction/predictions.json"),
        "partition_id": partition_id, "strategy": strategy, "seed": seed, "budget": budget,
        "formal_method_approved": False, "target_pointwise_labels_read": False})
    update(directory)
    return status(directory)


def context(directory):
    service = read_json(directory / "service.json")
    inputs = directory / "inputs"
    if (service.get("schema_version") != 1 or service.get("usage") != "development_demo_only"
            or inventory(inputs) != service["inputs_sha256"]
            or sha256(directory / "prediction/predictions.json") != service["prediction_sha256"]):
        raise DataError("Live source/config/prediction changed; restore inputs or start a new session")
    session = sessions.read_session(directory / "session")
    if (session["model_path"] != str((inputs / "model").resolve())
            or session["manifest_path"] != str((inputs / "protocol/protocol.json").resolve())
            or any(session[k] != service[k] for k in ("partition_id", "strategy", "seed", "budget"))
            or service["learner"] != session["plan"]["learner"]
            or session["plan"] != sessions.load_plan(inputs / "query.json")):
        raise DataError("Live session identity changed")
    return service, session


def update(directory):
    """Idempotently recover a fit interrupted after answer 5/10/15/20."""
    with locked(directory):
        _, session = context(directory)
        count = len(session["records"])
        expected = len(session["rounds"]) * session["plan"]["update_every"]
        if count != expected:
            return
        inputs = directory / "inputs"
        metadata, model = load_model(inputs / "model")
        payload = read_json(directory / "prediction/predictions.json")
        settings = learner_plan(inputs / "learner.json")
        head, replay, hidden, head_hash = None, None, None, None
        if session["plan"]["learner"] == "L2_max":
            head_path = inputs / "source_head/source_head.json"
            head, head_hash = read_json(head_path), sha256(head_path)
            with np.load(inputs / "source_head/source_replay.npz", allow_pickle=False) as saved:
                replay = {a: (saved[a + "_x"].copy(), saved[a + "_y"].copy()) for a in metadata["appliances"]}
            rows = read_partition(inputs / "protocol/protocol.json", session["partition_id"], "selection")["aggregate"]
            hidden = temporal_features(metadata, model, rows)
        adapter, history = fit_adapter(payload["rows"], session["records"], metadata, settings,
                                       session["plan"]["learner"], head, hidden, replay,
                                       read_json(inputs / "legacy.json"))
        destination = directory / "session/models" / f"budget_{count}"
        _export(inputs / "model", metadata, adapter, session["records"],
                {"learner_settings": settings, "query_plan": session["plan"], "strategy": session["strategy"]},
                history, destination, head_hash, session["strategy"])
        if session["plan"]["schema_version"] in (2, 3):
            from .r2 import selection_scores
            values = selection_scores(payload["rows"], metadata, adapter)
        else:
            values = score_adapter(payload["rows"], metadata, adapter, hidden)
        sessions.commit_round(directory / "session", destination, values)


def status(directory):
    update(directory)
    service, session = context(directory)
    count = len(session["records"])
    fitted = (len(session["rounds"]) - 1) * session["plan"]["update_every"]
    return {**service, "records": session["records"], "budget_used": count, "fitted_budget": fitted,
            "model": str(directory / "session/models" / f"budget_{fitted}"),
            "pending": sessions.select_next(directory / "session")}


def answer(directory, query_id, response, label_source="human_confirmed"):
    context(directory)
    record = sessions.confirm(directory / "session", query_id, response, label_source)
    # A saved answer remains committed if fitting fails. Loading retries the fit.
    update(directory)
    return record
