"""Version 2 confirmation sessions: replay each choice at its visible round."""

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from ..adaptation.selection import candidates
from ..adaptation.session import locked, save
from ..data.common import DataError, read_json, sha256, utc, write_json, safe_child
from ..data.protocol import validate_confirmation_history

STRATEGIES = ("random", "uncertainty", "proposed", "uncertainty_static", "proposed_static",
              "proposed_no_diversity", "proposed_no_coverage")
IDENTITY = ("query_id", "household_id", "appliance_type", "window_start", "window_end")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_plan(path):
    plan = read_json(path)
    if (set(plan) != {"schema_version", "learner", "update_every", "strategies", "uncertainty_weight",
                     "diversity_weight", "coverage_weight", "exploration_fraction", "minimum_random_f1_gain"}
            or plan["schema_version"] != 1 or plan["learner"] != "L2_max" or type(plan["update_every"]) is not int
            or plan["update_every"] != 5):
        raise DataError("Invalid A3 plan or unsupported learner/update frequency")
    strategies = plan["strategies"]
    if (not isinstance(strategies, list) or len(strategies) != len(set(strategies))
            or not {"random", "uncertainty", "proposed"} <= set(strategies) <= set(STRATEGIES)):
        raise DataError("A3 requires random, uncertainty and proposed comparisons")
    for key in ("uncertainty_weight", "diversity_weight", "coverage_weight", "exploration_fraction", "minimum_random_f1_gain"):
        value = plan[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise DataError("Invalid A3 weight/fraction")
    if not math.isclose(sum(plan[k] for k in ("uncertainty_weight", "diversity_weight", "coverage_weight")), 1):
        raise DataError("A3 score weights must sum to one")
    return plan


def build_pool(payload, partition, protocol, legacy):
    """Reuse M3 complete disjoint grid; shape descriptors see aggregate only."""
    items = candidates(payload, partition, protocol, legacy)
    stamps = {r["timestamp"]: i for i, r in enumerate(payload["rows"])}
    width = protocol["query_window_seconds"] // protocol["sample_interval_seconds"]
    result = []
    for item in items:
        start = stamps[item["window_start"]]
        x = np.asarray([r["aggregate_power_w"] for r in payload["rows"][start:start + width]], dtype=float)
        descriptor = np.log1p([x.mean(), x.std(), np.ptp(x), np.abs(np.diff(x)).mean() if len(x) > 1 else 0])
        result.append({**{k: item[k] for k in IDENTITY}, "start": start, "end": start + width,
                       "shape": descriptor.tolist(), "column": payload["appliances"].index(item["appliance_type"])})
    if result:
        shapes = np.asarray([r["shape"] for r in result])
        shapes = (shapes - shapes.mean(axis=0)) / np.maximum(shapes.std(axis=0), 1e-6)
        for item, shape in zip(result, shapes):
            item["shape"] = shape.tolist()
    return result


def window_scores(pool, values):
    result = {}
    for item in pool:
        segment = values[item["start"]:item["end"], item["column"]]
        if len(segment) != item["end"] - item["start"] or not np.isfinite(segment).all() or (segment < 0).any() or (segment > 1).any():
            raise DataError("A3 round changed complete candidate support")
        result[item["query_id"]] = float(segment.max())
    return result


def choose(pool, records, scores, strategy, seed, plan):
    asked = {r["query_id"] for r in records}
    available = sorted((p for p in pool if p["query_id"] not in asked), key=lambda p: p["query_id"])
    rng = random.Random(f"a3:{seed}:{len(records)}")
    rng.shuffle(available)
    if not available:
        return None
    if strategy == "random" or (strategy.startswith("proposed") and rng.random() < plan["exploration_fraction"]):
        return available[0]
    previous = [p for p in pool if p["query_id"] in asked]
    counts = {p["appliance_type"]: sum(r["appliance_type"] == p["appliance_type"] for r in records) for p in available}

    def rank(item):
        uncertainty = 1 - 2 * abs(scores[item["query_id"]] - 0.5)
        if strategy.startswith("uncertainty"):
            return uncertainty
        diversity = min((1 - math.exp(-float(np.linalg.norm(np.asarray(item["shape"]) - p["shape"])) / 2)
                         for p in previous), default=1.0)
        return (plan["uncertainty_weight"] * uncertainty
                + (0 if strategy == "proposed_no_diversity" else plan["diversity_weight"] * diversity)
                + (0 if strategy == "proposed_no_coverage" else plan["coverage_weight"] / (1 + counts[item["appliance_type"]])))

    return max(available, key=rank)


def create_session(directory, manifest, model, partition_id, pool, plan, strategy, seed, budget):
    protocol = read_json(manifest)
    partition = next((p for p in protocol["partitions"] if p["partition_id"] == partition_id), None)
    if (strategy not in plan["strategies"] or seed not in protocol["random_seeds"] or budget not in protocol["query_budgets"]
            or not partition or partition["role"] != "target_adaptation" or len(pool) < budget):
        raise DataError("Invalid A3 session inputs or insufficient complete candidates")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "pool.json", {"items": pool})
    write_json(directory / "query_plan.json", plan)
    session = {"schema_version": 2, "status": "active", "manifest_path": str(manifest.resolve()),
               "model_path": str(model.resolve()), "protocol_sha256": sha256(manifest),
               "model_manifest_sha256": sha256(model / "model.json"), "pool_sha256": sha256(directory / "pool.json"),
               "partition_id": partition_id, "household_id": partition["household_id"], "strategy": strategy,
               "seed": seed, "budget": budget, "plan": plan, "plan_sha256": sha256(directory / "query_plan.json"),
               "rounds": [], "records": [], "pending": None}
    save(directory, session)


def _round_for(session, count):
    return 0 if session["strategy"].endswith("_static") else count // session["plan"]["update_every"] * session["plan"]["update_every"]


def read_session(directory):
    session = read_json(directory / "session.json")
    if session.get("schema_version") != 2 or session.get("status") != "active":
        raise DataError("Invalid A3 session")
    if (sha256(Path(session["manifest_path"])) != session["protocol_sha256"]
            or sha256(Path(session["model_path"]) / "model.json") != session["model_manifest_sha256"]
            or sha256(directory / "pool.json") != session["pool_sha256"]
            or sha256(directory / "query_plan.json") != session["plan_sha256"]
            or load_plan(directory / "query_plan.json") != session["plan"]):
        raise DataError("A3 session input changed")
    protocol = read_json(Path(session["manifest_path"]))
    validate_confirmation_history(session["records"], protocol)
    if (session["strategy"] not in session["plan"]["strategies"] or session["seed"] not in protocol["random_seeds"]
            or session["budget"] not in protocol["query_budgets"] or len(session["records"]) > session["budget"]):
        raise DataError("A3 session budget/policy changed")
    pool = read_json(directory / "pool.json")["items"]
    rounds = {}
    for entry in session["rounds"]:
        path = safe_child(directory, entry["path"])
        if sha256(path) != entry["sha256"]:
            raise DataError("A3 scoring round changed")
        snapshot = read_json(path)
        count = snapshot["visible_count"]
        model_path = safe_child(directory, snapshot["model_path"])
        if (count in rounds or count != len(rounds) * session["plan"]["update_every"] or count > len(session["records"])
                or snapshot["prefix_sha256"] != digest(session["records"][:count])
                or sha256(model_path / "model.json") != snapshot["model_sha256"]):
            raise DataError("A3 scoring round has an invalid model or visible prefix")
        metadata = read_json(model_path / "model.json")
        queries = read_json(model_path / "queries.json")
        if (queries["records"] != session["records"][:count] or metadata["queries_sha256"] != sha256(model_path / "queries.json")
                or metadata["model_id"] != snapshot["model_id"] or set(snapshot["scores"]) != {p["query_id"] for p in pool}
                or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in snapshot["scores"].values())):
            raise DataError("A3 score/model prefix identity differs")
        rounds[count] = snapshot
    for index, record in enumerate(session["records"]):
        snapshot = rounds.get(_round_for(session, index))
        if snapshot is None:
            raise DataError("A3 history missing scoring round")
        item = choose(pool, session["records"][:index], snapshot["scores"], session["strategy"], session["seed"], session["plan"])
        if (item is None or any(record[k] != item[k] for k in IDENTITY)
                or record["selection_strategy"] != session["strategy"] or record["model_version"] != snapshot["model_id"]):
            raise DataError("A3 history does not replay its visible scoring model")
    if session["pending"] is not None:
        expected = _pending(session, pool, rounds)
        if session["pending"] != expected or expected is None:
            raise DataError("A3 pending question changed")
    return session


def _pending(session, pool, rounds):
    count = len(session["records"])
    if count >= session["budget"]:
        return None
    snapshot = rounds.get(_round_for(session, count))
    if snapshot is None:
        raise DataError("A3 needs a committed scoring update before selecting")
    item = choose(pool, session["records"], snapshot["scores"], session["strategy"], session["seed"], session["plan"])
    return {**{k: item[k] for k in IDENTITY}, "model_version": snapshot["model_id"]} if item else None


def commit_round(directory, model, values):
    with locked(directory):
        session = read_session(directory)
        count = len(session["records"])
        if session["pending"] is not None or count != len(session["rounds"]) * session["plan"]["update_every"]:
            raise DataError("A3 scoring update must follow exactly five committed answers")
        metadata = read_json(model / "model.json")
        queries = read_json(model / "queries.json")
        if queries["records"] != session["records"]:
            raise DataError("A3 scoring model uses a different visible prefix")
        import os
        snapshot = {"visible_count": count, "prefix_sha256": digest(session["records"]),
                    "model_path": Path(os.path.relpath(model, directory)).as_posix(), "model_sha256": sha256(model / "model.json"),
                    "model_id": metadata["model_id"], "scores": window_scores(read_json(directory / "pool.json")["items"], values)}
        # Models are kept under the session, so safe_child can validate them.
        safe_child(directory, snapshot["model_path"])
        path = directory / "rounds" / f"budget_{count}.json"
        write_json(path, snapshot)
        session["rounds"].append({"path": path.relative_to(directory).as_posix(), "sha256": sha256(path)})
        save(directory, session)


def select_next(directory):
    with locked(directory):
        session = read_session(directory)
        if session["pending"] is None:
            rounds = {read_json(directory / e["path"])["visible_count"]: read_json(directory / e["path"]) for e in session["rounds"]}
            session["pending"] = _pending(session, read_json(directory / "pool.json")["items"], rounds)
            save(directory, session)
        return session["pending"]


def confirm(directory, query_id, response, label_source="human_confirmed"):
    with locked(directory):
        session = read_session(directory)
        pending = session["pending"]
        if pending is None or pending["query_id"] != query_id:
            raise DataError("A3 answer must match committed pending query; duplicate/stale answer rejected")
        count = len(session["records"])
        record = {k: pending[k] for k in IDENTITY}
        record.update(response=response, label_source=label_source, selection_strategy=session["strategy"],
                      model_version=pending["model_version"], budget_before=count, budget_after=count + 1)
        records = session["records"] + [record]
        validate_confirmation_history(records, read_json(Path(session["manifest_path"])))
        session.update(records=records, pending=None)
        save(directory, session)
        return record
