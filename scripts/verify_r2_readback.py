"""Reload one sealed R2 model through ordinary inference without target labels."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.dynamic_queries import verify_frozen
from virtual_submeter.data.common import DataError, read_json, safe_child, sha256, write_json
from virtual_submeter.models.prediction import predict


def check(run, output):
    state, plan, _ = verify_frozen(run)
    if plan["schema_version"] not in (2, 3):
        raise DataError("An R2 frozen matrix is required")
    case = next(c for c in state["cases"] if (c["source_seed"], c["query_seed"], c["variant"], c["budget"]) == (42, 2026, "proposed", 10))
    frozen = run / "frozen"
    model = safe_child(frozen, case["model_path"])
    predict(model, output, frozen / "inputs/protocol/protocol.json", case["future_partition"])
    payload = read_json(output / "predictions.json")
    base = read_json(safe_child(frozen, case["base_prediction"]))
    with np.load(safe_child(frozen, case["path"]) / "states.npz", allow_pickle=False) as saved:
        values = saved["scores"]
        if len(payload["rows"]) != len(base["rows"]) or values.shape != (len(payload["rows"]), len(payload["appliances"])):
            raise DataError("Ordinary inference changed R2 prediction support")
        for i, (actual, original) in enumerate(zip(payload["rows"], base["rows"])):
            if actual["timestamp"] != original["timestamp"] or actual["unexplained_power_w"] != original["unexplained_power_w"]:
                raise DataError("Ordinary inference changed timestamps or unexplained power")
            for j, appliance in enumerate(payload["appliances"]):
                item, prior = actual["appliances"][appliance], original["appliances"][appliance]
                expected = bool(values[i, j] >= .5) if np.isfinite(values[i, j]) else None
                if item["raw_state"] != expected or item["state"] != expected:
                    raise DataError("Ordinary inference differs from the frozen R2 state cache")
                if any(item[k] != prior[k] for k in ("raw_power_w", "power_w")):
                    raise DataError("Ordinary inference changed frozen source power")
    return {"status": "passed", "experiment_id": plan["experiment_id"],
            "case": {k: case[k] for k in ("source_seed", "query_seed", "variant", "budget", "household_id")},
            "rows": len(payload["rows"]), "state_cache_matches": True, "power_and_unexplained_unchanged": True,
            "target_pointwise_labels_parsed": False, "code_sha256": state["code_sha256"],
            "model_sha256": sha256(model / "model.json"), "predictions_sha256": sha256(output / "predictions.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists() or args.output.exists() or any(p.resolve().is_relative_to(args.run.resolve()) for p in (args.report, args.output)):
        raise DataError("Choose new readback evidence paths outside the immutable run")
    original_open = Path.open

    def deny_labels(path, *values, **options):
        mode = values[0] if values else options.get("mode", "r")
        if any(area in path.parts for area in ("oracle_labels", "evaluation_labels")) and "r" in mode and "b" not in mode:
            raise DataError("Readback may not parse target pointwise labels")
        return original_open(path, *values, **options)

    with patch.object(Path, "open", deny_labels):
        result = check(args.run, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, result)
    print(result)


if __name__ == "__main__":
    main()
