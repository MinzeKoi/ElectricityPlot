"""Independently replay A4 outputs and source calibration without target labels."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from virtual_submeter.accuracy.power_experiment import verify, materialize
from virtual_submeter.data.common import DataError, read_json, safe_child, write_json, sha256
from virtual_submeter.data.protocol import read_partition
from virtual_submeter.models.artifacts import load_model
from virtual_submeter.models.prediction import predict_rows, predict
from virtual_submeter.trust.calibration import calibrate, load_calibration


def inspect(run, output):
    if output.exists():
        raise DataError("Validation output exists; choose a new directory")
    original = Path.open
    def deny_target_labels(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if any(p in path.parts for p in ("oracle_labels", "evaluation_labels")) and "r" in mode and "b" not in mode:
            raise DataError("A4 independent verification attempted to parse target labels")
        return original(path, *args, **kwargs)
    with patch.object(Path, "open", deny_target_labels):
        result = verify(run)
        frozen = run / "frozen"
        state = read_json(run / "run.json")
        learner = read_json(frozen / "learner_plan.json")
        cases = state["cases"]
        pairs, no_answer, zero_count = 0, 0, 0
        for case in cases:
            if case["mode"] != "coupled":
                continue
            directory = safe_child(frozen, case["path"])
            zero = directory.parent.parent / "budget_0/coupled"
            metadata, _ = load_model(directory / "model")
            source = read_json(zero / "model/model.json")
            records = read_json(directory / "model/queries.json")["records"]
            if metadata["accuracy_power_head"] != source["accuracy_power_head"]:
                raise DataError("A4 amplitude changed with target confirmation budget")
            with np.load(directory / "prediction.npz", allow_pickle=False) as current, np.load(zero / "prediction.npz", allow_pickle=False) as before:
                control = directory.parent / "state_only/prediction.npz"
                with np.load(control, allow_pickle=False) as other:
                    if not np.array_equal(current["scores"], other["scores"], equal_nan=True):
                        raise DataError("A4 coupled/control state scores differ")
                for column, appliance in enumerate(metadata["appliances"]):
                    responses = [r for r in records if r["appliance_type"] == appliance and r["response"] in ("used", "not_used")]
                    if not responses:
                        if (metadata["accuracy_state_head"]["coefficients"][appliance] != source["accuracy_state_head"]["coefficients"][appliance]
                                or not np.array_equal(current["raw"][:, column], before["raw"][:, column], equal_nan=True)):
                            raise DataError("A4 no-answer appliance changed source power")
                        no_answer += 1
                if case["budget"] == 0:
                    zero_count += 1
            pairs += 1
        aggregates, payloads = {}, {}
        roundtrips = []
        query_seed = min(learner["query_seeds"])
        for case in cases:
            if case["mode"] != "coupled" or case["budget"] != learner["main_budget"] or case["query_seed"] != query_seed:
                continue
            directory = safe_child(frozen, case["path"])
            pid = case["future_partition"]
            if pid not in aggregates:
                aggregates[pid] = read_partition(frozen / "protocol/protocol.json", pid, "inference")["aggregate"]
            metadata, model = load_model(directory / "model")
            calibration = load_calibration(directory / "calibration", metadata)
            actual = predict_rows(metadata, model, aggregates[pid], calibration)
            if case["base_prediction"] not in payloads:
                payloads[case["base_prediction"]] = read_json(safe_child(frozen, case["base_prediction"]))
            expected = materialize(frozen, case, payloads[case["base_prediction"]])
            for appliance in metadata["appliances"]:
                for key in ("raw_power_w", "power_w", "state_score", "raw_lower_w", "raw_upper_w", "lower_w", "upper_w"):
                    left = [r["appliances"][appliance][key] if r["appliances"][appliance][key] is not None else np.nan for r in actual]
                    right = [r["appliances"][appliance][key] if r["appliances"][appliance][key] is not None else np.nan for r in expected]
                    np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-4, equal_nan=True)
            roundtrips.append({"source_seed": case["source_seed"], "variant": case["variant"], "rows": len(actual)})
        representative = next(c for c in cases if c["mode"] == "coupled" and c["budget"] == learner["main_budget"])
        directory = safe_child(frozen, representative["path"])
        # This ordinary entry point opens source validation labels only. Its
        # residuals must agree with the compact vector calibration used in A4.
        calibrate(directory / "model", frozen / "protocol/protocol.json", directory / "calibration/settings.json", output / "source-calibration")
        original_calibration = read_json(directory / "calibration/calibration.json")
        normal_calibration = read_json(output / "source-calibration/calibration.json")
        maximum_radius_difference = 0.0
        for appliance in original_calibration["appliances"]:
            for view in ("raw", "corrected"):
                left, right = original_calibration["quantiles"][appliance][view], normal_calibration["quantiles"][appliance][view]
                if left["samples"] != right["samples"] or not np.isclose(left["radius_w"], right["radius_w"], rtol=1e-6, atol=1e-4):
                    raise DataError("A4 compact source residual calibration disagrees with ordinary calibration")
                maximum_radius_difference = max(maximum_radius_difference, abs(left["radius_w"] - right["radius_w"]))
        predict(directory / "model", output / "ordinary-prediction", frozen / "protocol/protocol.json", representative["future_partition"],
                calibration_dir=directory / "calibration")
        result.update(paired_models_checked=pairs, no_effective_answer_devices_exact=no_answer, zero_cases_checked=zero_count,
                      generic_roundtrips=roundtrips, source_calibration_maximum_radius_difference_w=maximum_radius_difference,
                      source_calibration_fingerprint_matches=original_calibration["power_fingerprint"] == normal_calibration["power_fingerprint"],
                      target_labels_parsed=False, a4_evidence_sha256=sha256(run / "evidence.json"))
        write_json(output / "validation.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = inspect(args.run, args.output)
    except (DataError, OSError, KeyError, RuntimeError, AssertionError) as error:
        print(json.dumps({"status": "a4_validation_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
