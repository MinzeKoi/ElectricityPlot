"""Full source-validation public-interface readback; never opens development labels."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.accuracy.state_experiment import verify, GROUPS, SEEDS, SOURCE_VALIDATION
from virtual_submeter.models.state_classifier import load_state_model, predict_state_rows
from virtual_submeter.models.artifacts import load_model
from virtual_submeter.models.prediction import predict_rows
from virtual_submeter.data.common import DataError, write_json, sha256
from virtual_submeter.data.protocol import read_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/accuracy/state-20261024-v2")
    args = parser.parse_args()
    run = args.run
    verify(run)
    rows = read_partition(run / "protocol/protocol.json", SOURCE_VALIDATION, "inference")["aggregate"]
    checks = []
    for g in GROUPS:
        for s in SEEDS:
            path = run / f"models/{g}/seed_{s}"
            if g == "R_PR":
                meta, model = load_model(path)
                ordinary = predict_rows(meta, model, rows)
                actual = np.array([r["appliances"]["microwave"]["raw_power_w"] if r["appliances"]["microwave"]["raw_power_w"] is not None else np.nan for r in ordinary])
                threshold = meta["source_state_selection"]["point"]["threshold"]
            else:
                meta, model = load_state_model(path)
                ordinary = predict_state_rows(meta, model, rows)
                actual = np.array([r["probability_on"] if r["probability_on"] is not None else np.nan for r in ordinary])
                threshold = meta["decision_threshold_probability"]
            with np.load(run / f"predictions/{g}/seed_{s}/{SOURCE_VALIDATION}.npz", allow_pickle=False) as a:
                expected = a["scores"].copy()
            missing = np.array_equal(np.isfinite(actual), np.isfinite(expected))
            states = np.array_equal(actual >= threshold, expected >= threshold)
            if not missing or not states or not np.allclose(actual, expected, rtol=1e-6, atol=1e-8, equal_nan=True):
                raise DataError(f"Full public inference differs: {g} {s}")
            check = {"candidate": g, "seed": s, "rows": len(rows), "maximum_score_difference": float(np.nanmax(abs(actual-expected))),
                     "missing_equal": missing, "states_equal": states, "decision_threshold": threshold,
                     "ordinary_output": "raw_power_w_with_explicit_source_threshold" if g == "R_PR" else "probability_on_and_state",
                     "model_sha256": sha256(path / "model.json")}
            checks.append(check)
            print(f"readback {g} seed={s}: passed", flush=True)
    write_json(ROOT / "tmp/state-full-readback-20261024.json", {"checks": checks, "all_passed": True,
               "freeze_sha256": sha256(run / "freeze.json"), "target_labels_reopened": False})


if __name__ == "__main__":
    main()
