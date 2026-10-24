"""Verify all real R2 L1 checkpoints against their frozen source and bounds."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.dynamic_queries import verify_frozen
from virtual_submeter.accuracy.learners import validate_adapter
from virtual_submeter.data.common import DataError, read_json, safe_child, sha256, write_json


def check(run):
    state, query_plan, learner = verify_frozen(run)
    if query_plan["schema_version"] not in (2, 3):
        raise DataError("Use an R2 matrix")
    frozen = run / "frozen"
    models = {}
    for case in state["cases"]:
        parent = safe_child(frozen, case["model_path"]).parent
        for count in range(0, max(learner["budgets"]) + 1, query_plan["update_every"]):
            models[parent / f"budget_{count}"] = case["source_seed"]
    result = {"status": "passed", "cases": len(state["cases"]), "checkpoints": 0,
              "power_weights_unchanged": 0, "devices_checked": 0,
              "no_answer_exact_source_devices": 0, "bounded_devices": 0,
              "single_class_bias_only_devices": 0}
    for directory, seed in models.items():
        metadata = read_json(directory / "model.json")
        adapter = metadata["accuracy_state_head"]
        validate_adapter(adapter, metadata)
        if adapter["variant"] != query_plan["learner"] or adapter["normalizer"] is not None:
            raise DataError("R2 checkpoint is not the frozen L1 learner")
        source = read_json(frozen / f"inputs/bases/seed_{seed}/model.json")
        if sha256(directory / "weights.pt") != source["weights"]["sha256"]:
            raise DataError("Confirmation changed the source power weights")
        result["power_weights_unchanged"] += 1
        records = read_json(directory / "queries.json")["records"]
        for appliance, coefficients in adapter["coefficients"].items():
            delta = np.asarray(coefficients) - [0., 1.]
            answers = [r for r in records if r["appliance_type"] == appliance
                       and r["response"] in ("used", "not_used")]
            if (appliance in adapter["active_appliances"]) != bool(answers):
                raise DataError("L1 active devices differ from usable confirmations")
            np.testing.assert_array_equal(delta[1:], [0.])
            if not answers:
                np.testing.assert_array_equal(delta, [0., 0.])
                result["no_answer_exact_source_devices"] += 1
            else:
                single = len({r["response"] for r in answers}) == 1
                bound = learner["single_class_bias_bound"] if single or query_plan["schema_version"] == 3 else learner["bias_bound"]
                if abs(delta[0]) > bound + 1e-12:
                    raise DataError("L1 bias exceeded its predeclared bound")
                result["bounded_devices"] += 1
                result["single_class_bias_only_devices"] += int(single)
            result["devices_checked"] += 1
        result["checkpoints"] += 1
    result.update(code_sha256=state["code_sha256"], target_pointwise_labels_parsed=False,
                  future_labels_opened_at_check=state["development_future_labels_opened"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.run.resolve()):
        raise DataError("Choose a new evidence path outside the immutable run")
    original_open = Path.open

    def deny_target_labels(path, *values, **options):
        mode = values[0] if values else options.get("mode", "r")
        if any(p in path.parts for p in ("oracle_labels", "evaluation_labels")) and "r" in mode and "b" not in mode:
            raise DataError("Boundary verification may not parse target pointwise labels")
        return original_open(path, *values, **options)

    with patch.object(Path, "open", deny_target_labels):
        result = check(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
