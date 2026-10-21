"""Read-only A3 evidence checks plus ordinary prediction/model round trips."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from virtual_submeter.accuracy.dynamic_queries import verify
from virtual_submeter.accuracy.dynamic_session import window_scores, read_session
from virtual_submeter.data.common import DataError, read_json, safe_child, sha256, write_json
from virtual_submeter.data.protocol import read_partition
from virtual_submeter.models.artifacts import load_model
from virtual_submeter.models.prediction import predict_rows


def inspect(run, a2):
    original = Path.open

    def deny_labels(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if ("evaluation_labels" in path.parts or "oracle_labels" in path.parts) and "r" in mode and "b" not in mode:
            raise DataError("A3 read-only verification attempted to parse target labels")
        return original(path, *args, **kwargs)

    with patch.object(Path, "open", deny_labels):
        result = verify(run)
        frozen = run / "frozen"
        state = read_json(run / "run.json")
        learner = read_json(frozen / "inputs/plan.json")
        weight_checks = no_answer_checks = single_class_checks = 0
        for path in sorted((frozen / "trajectories").rglob("model.json")):
            metadata = read_json(path)
            records = read_json(path.parent / "queries.json")["records"]
            seed = metadata["settings"]["seed"]
            base = frozen / "inputs/bases" / f"seed_{seed}"
            source = read_json(frozen / "inputs/source_heads" / f"seed_{seed}" / "source_head.json")
            if (sha256(path.parent / "weights.pt") != read_json(base / "model.json")["weights"]["sha256"]
                    or metadata["base_model_manifest_sha256"] != sha256(base / "model.json")):
                raise DataError("A3 checkpoint changed its frozen power weights/base identity")
            weight_checks += 1
            for appliance in metadata["appliances"]:
                prior = np.asarray(source["coefficients"][appliance])
                coefficient = np.asarray(metadata["accuracy_state_head"]["coefficients"][appliance])
                responses = {r["response"] for r in records if r["appliance_type"] == appliance and r["response"] in ("used", "not_used")}
                if not responses:
                    if not np.array_equal(prior, coefficient):
                        raise DataError("A3 device without effective answers changed source parameters")
                    no_answer_checks += 1
                elif len(responses) == 1:
                    if not np.array_equal(prior[1:], coefficient[1:]) or abs(prior[0] - coefficient[0]) > learner["single_class_bias_bound"] + 1e-12:
                        raise DataError("A3 single-class update exceeded its frozen conservative rule")
                    single_class_checks += 1
                elif np.linalg.norm(prior - coefficient) > learner["parameter_radius"] + 1e-12:
                    raise DataError("A3 mixed-class update exceeded its frozen parameter radius")
        parent_cases = read_json(a2 / "run.json")["cases"]
        parent_zero = {(c["source_seed"], c["query_seed"], c["household_id"]): c for c in parent_cases
                       if c["variant"] == "L2_max" and c["budget"] == 0}
        zero_checked = 0
        for case in state["cases"]:
            if case["budget"] != 0:
                continue
            parent = parent_zero[(case["source_seed"], case["query_seed"], case["household_id"])]
            with np.load(safe_child(frozen, case["path"]) / "states.npz", allow_pickle=False) as current:
                with np.load(safe_child(a2 / "frozen", parent["path"]) / "states.npz", allow_pickle=False) as before:
                    if not np.array_equal(current["scores"], before["scores"], equal_nan=True):
                        raise DataError("A3 zero-confirmation scores differ from sealed A2 learner")
            zero_checked += 1
        query_seed = min(c["query_seed"] for c in state["cases"])
        checked = []
        for case in state["cases"]:
            if case["variant"] not in ("uncertainty", "proposed") or case["budget"] != 10 or case["query_seed"] != query_seed:
                continue
            metadata, model = load_model(safe_child(frozen, case["model_path"]))
            aggregate = read_partition(frozen / "inputs/protocol/protocol.json", case["future_partition"], "inference")["aggregate"]
            rows = predict_rows(metadata, model, aggregate)
            base = read_json(safe_child(frozen, case["base_prediction"]))
            with np.load(safe_child(frozen, case["path"]) / "states.npz", allow_pickle=False) as saved:
                scores = saved["scores"]
                for index, row in enumerate(rows):
                    for column, appliance in enumerate(metadata["appliances"]):
                        item, value = row["appliances"][appliance], scores[index, column]
                        expected = bool(value >= 0.5) if np.isfinite(value) else None
                        if item["state"] != expected or item["raw_state"] != expected:
                            raise DataError("Exported A3 model does not reproduce sealed states")
                        if item["state_score"] is not None and not np.isclose(item["state_score"], value, atol=1e-10, rtol=1e-10):
                            raise DataError("Exported A3 model does not reproduce sealed scores")
                        for key in ("power_w", "raw_power_w"):
                            if item[key] != base["rows"][index]["appliances"][appliance][key]:
                                raise DataError("A3 changed exported power estimates")
                    if row["unexplained_power_w"] != base["rows"][index]["unexplained_power_w"]:
                        raise DataError("A3 changed unexplained power")
            checked.append({k: case[k] for k in ("source_seed", "query_seed", "household_id", "variant", "budget")})
        first = next(c for c in state["cases"] if c["variant"] == "proposed")
        directory = safe_child(frozen, first["trajectory"])
        session = read_session(directory)
        pool = read_json(directory / "pool.json")["items"]
        aggregate = read_partition(frozen / "inputs/protocol/protocol.json", session["partition_id"], "selection")["aggregate"]
        rounds_checked = []
        for entry in session["rounds"]:
            snapshot = read_json(directory / entry["path"])
            metadata, model = load_model(safe_child(directory, snapshot["model_path"]))
            rows = predict_rows(metadata, model, aggregate)
            scores = np.asarray([[r["appliances"][a]["state_score"] if r["appliances"][a]["state_score"] is not None else np.nan
                                  for a in metadata["appliances"]] for r in rows])
            actual = window_scores(pool, scores)
            if any(not np.isclose(actual[k], v, atol=1e-10, rtol=1e-10) for k, v in snapshot["scores"].items()):
                raise DataError("A3 query scoring does not reproduce from its recorded round model")
            rounds_checked.append(snapshot["visible_count"])
    return {**result, "zero_baselines_match_a2": zero_checked, "exported_model_roundtrips": checked,
            "power_weight_checkpoints_unchanged": weight_checks, "no_answer_exact_source_devices": no_answer_checks,
            "single_class_bias_only_devices": single_class_checks,
            "scoring_rounds_reproduced": rounds_checked, "target_labels_parsed_during_verification": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--a2-run", type=Path, default=ROOT / "outputs/accuracy/a2-20261021")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = inspect(args.run, args.a2_run)
    except (DataError, OSError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "a3_verification_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
