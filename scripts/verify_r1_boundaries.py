"""Check real R1 power identity, update bounds and model reload without labels."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.data.common import DataError, read_json, sha256, write_json
from virtual_submeter.data.protocol import read_partition
from virtual_submeter.models.artifacts import load_model
from virtual_submeter.models.prediction import predict_rows
from virtual_submeter.accuracy.fixed_queries import verify_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise DataError("Boundary evidence destination exists")
    state, plan = verify_frozen(args.run)
    frozen = args.run / "frozen"
    checks = dict(cases=0, power_weights_unchanged=0, no_answer_exact_source_devices=0,
                  bounded_devices=0, single_class_slope_preserved_devices=0)
    for case in state["cases"]:
        directory = frozen / case["path"] / "model"
        metadata = read_json(directory / "model.json")
        base = read_json(frozen / f"bases/seed_{case['source_seed']}/model.json")
        source = read_json(frozen / f"source_heads/seed_{case['source_seed']}/source_head.json")
        if sha256(directory / "weights.pt") != base["weights"]["sha256"]:
            raise DataError("Target confirmation changed power weights")
        checks["power_weights_unchanged"] += 1
        records = read_json(directory / "queries.json")["records"]
        for appliance, coefficients in metadata["accuracy_state_head"]["coefficients"].items():
            prior = np.array(source["coefficients"][appliance] if case["variant"].startswith("L2") else [0, 1])
            answers = [r for r in records if r["appliance_type"] == appliance and r["response"] in ("used", "not_used")]
            delta = np.array(coefficients) - prior
            if not answers:
                np.testing.assert_array_equal(delta, np.zeros(len(prior)))
                checks["no_answer_exact_source_devices"] += 1
                continue
            classes = {r["response"] for r in answers}
            if len(classes) == 1 or case["variant"] == "L1":
                np.testing.assert_array_equal(delta[1:], np.zeros(len(prior) - 1))
                bound = plan["single_class_bias_bound"] if len(classes) == 1 else plan["bias_bound"]
                if abs(delta[0]) > bound + 1e-12:
                    raise DataError("Bias bound exceeded")
                checks["single_class_slope_preserved_devices"] += int(len(classes) == 1)
            elif np.linalg.norm(delta) > plan["parameter_radius"] + 1e-12:
                raise DataError("Parameter radius exceeded")
            checks["bounded_devices"] += 1
        checks["cases"] += 1
    case = next(c for c in state["cases"] if c["source_seed"] == 1019 and c["query_seed"] == 42 and c["variant"] == "L2_max" and c["budget"] == 10)
    metadata, model = load_model(frozen / case["path"] / "model")
    inputs = read_partition(frozen / "protocol/protocol.json", case["future_partition"], "inference")["aggregate"]
    rows = predict_rows(metadata, model, inputs)
    actual = np.array([[r["appliances"][a]["state_score"] if r["appliances"][a]["state_score"] is not None else np.nan
                        for a in metadata["appliances"]] for r in rows])
    with np.load(frozen / case["path"] / "states.npz", allow_pickle=False) as stored:
        np.testing.assert_allclose(actual, stored["scores"], rtol=0, atol=1e-12, equal_nan=True)
    checks.update(status="passed", real_model_reload_score_parity=True, future_pointwise_labels_read=False,
                  inference_rows=len(rows), code_sha256=state["code_sha256"])
    write_json(args.output, checks)
    print(checks)


if __name__ == "__main__":
    main()
