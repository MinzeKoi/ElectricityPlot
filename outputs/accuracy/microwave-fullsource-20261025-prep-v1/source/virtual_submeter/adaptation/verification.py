"""Read saved M3 evidence without reopening submeter answers."""

from pathlib import Path

from ..data.common import DataError, read_json, sha256
from ..models.artifacts import load_model
from .session import read_session


def verify(run: Path):
    record = read_json(run / "run.json")
    if record["status"] not in ("all_models_frozen", "m3_pilot_evaluated"):
        raise DataError("M3 run is incomplete or failed")
    if sha256(run / "protocol.json") != record["protocol_sha256"]:
        raise DataError("M3 protocol snapshot changed")
    verified, groups = [], {}
    for case in record["cases"]:
        from ..data.common import safe_child
        directory = safe_child(run, case["path"])
        session = read_session(directory / "session")
        metadata, _ = load_model(directory / "model")
        if (metadata["model_id"] != case["model_id"] or metadata["protocol_sha256"] != record["protocol_sha256"]
                or metadata["base_model_manifest_sha256"] != record["base_model_manifest_sha256"]
                or metadata.get("target_pointwise_labels_read") is not False or not metadata["power_output_frozen"]):
            raise DataError("M3 model identity or supervision boundary is invalid")
        queries = read_json(directory / "model/queries.json")
        if queries["records"] != session["records"] or queries["budget_used"] != case["budget"] or session["budget"] != case["budget"]:
            raise DataError("M3 case did not use its declared equal budget")
        if session["seed"] != case["seed"] or session["strategy"] != case["strategy"]:
            raise DataError("M3 seed/strategy differs from its evidence")
        if record["status"] == "m3_pilot_evaluated":
            prediction = read_json(directory / "prediction/predictions.json")
            evaluation = read_json(directory / "evaluation/metrics.json")
            if (prediction["model_manifest_sha256"] != sha256(directory / "model/model.json")
                    or evaluation["prediction_sha256"] != sha256(directory / "prediction/predictions.json")
                    or prediction["model_id"] != metadata["model_id"] or evaluation["model_id"] != metadata["model_id"]
                    or prediction["partition_id"] != case["test_partition"]):
                raise DataError("M3 prediction/evaluation evidence changed")
            powers = [[row["timestamp"], row["aggregate_power_w"], row["unexplained_power_w"],
                       [[a["raw_power_w"], a["power_w"]] for a in row["appliances"].values()]] for row in prediction["rows"]]
            group = case["household_id"]
            if group in groups and powers != groups[group]:
                raise DataError("State-only adapter changed power outputs across cases")
            groups[group] = powers
        verified.append({key: case[key] for key in ("household_id", "strategy", "seed", "budget", "model_id")})
    expected = len({c["household_id"] for c in verified}) * len(record["strategies"]) * len(record["seeds"]) * len(record["budgets"])
    identities = {(c["household_id"], c["strategy"], c["seed"], c["budget"]) for c in verified}
    if not verified or len(identities) != expected or len(verified) != expected:
        raise DataError("M3 run has missing or duplicate comparison cases")
    return {"status": "m3_artifacts_verified", "cases": len(verified), "purpose": record["purpose"],
            "power_output_frozen": True, "test_labels_reopened": False}
