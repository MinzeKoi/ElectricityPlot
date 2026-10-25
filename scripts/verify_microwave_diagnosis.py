"""Additional readback: independent confusion/AP arithmetic, selection and isolation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.data.common import read_json, sha256, utc, write_json
from virtual_submeter.models.state_classifier import StateClassifier, predict_windows
from virtual_submeter.accuracy.microwave_diagnosis import registered, check_freeze, load_npz


def independent(scores, y, threshold):
    actual, predicted = y >= 200, scores >= threshold
    tp, fp, fn, tn = [int(m.sum()) for m in [actual & predicted, ~actual & predicted, actual & ~predicted, ~actual & ~predicted]]
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(scores)-1]
    ctp = actual[order].cumsum()[ends]
    precision, recall = ctp/(ends+1), ctp/actual.sum()
    ap = float((np.diff(np.r_[0., recall])*precision).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ap": ap}, precision, recall, ordered[ends]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/accuracy/microwave-diagnosis-20261025-v2")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/MICROWAVE_DIAGNOSIS_VALIDATION_20261025.json")
    args = parser.parse_args()
    run = args.run
    plan = registered(run)
    check_freeze(run)
    results = read_json(run / "evaluation/results.json")["rows"]
    stats = read_json(run / "support.json")
    models, checkpoints, outer_predictions, small_checks = 0, 0, 0, []
    for house in plan["households"]:
        sets = {r: load_npz(run / f"arrays/{house}/{r}.npz") for r in ["train", "validation", "outer"]}
        # Actual full-context endpoints and event identity sets, not just center times.
        for left, right in [("train", "validation"), ("validation", "outer"), ("train", "outer")]:
            assert int(sets[left]["t"].max())+120 < int(sets[right]["t"].min())-120
            assert not (set(sets[left]["event"]) & set(sets[right]["event"])) - {-1}
        for role, data in sets.items():
            block = stats[house]["blocks"][role]
            assert int(data["t"].min())-120 >= utc(block["start"])
            assert int(data["t"].max())+128 <= utc(block["end"])
            assert np.isfinite(data["x"]).all()
        train = sets["train"]
        mean, std = float(np.log1p(train["x"]).mean()), float(np.log1p(train["x"]).std())
        for seed in plan["seeds"]:
            directory = run / f"models/{house}/seed_{seed}"
            info = read_json(directory / "model.json")
            meta = info["normalization"]
            assert mean == meta["input_log_mean"] and std == meta["input_log_std"]
            ranks = []
            for epoch in range(1, plan["checkpoints"]+1):
                data = load_npz(directory / f"epoch_{epoch}_validation.npz")
                _, p, r, thresholds = independent(data["scores"], sets["validation"]["y"], .5)
                f1 = 2*p*r/(p+r+1e-300)
                possible = np.flatnonzero(r >= .85)
                i = max(possible, key=lambda i: (p[i], f1[i], thresholds[i]))
                np.testing.assert_allclose(data["precision"], p, rtol=0, atol=0)
                np.testing.assert_allclose(data["recall"], r, rtol=0, atol=0)
                ranks.append((-float(p[i]), -float(data["f1"][i]), epoch, float(thresholds[i])))
                checkpoints += 1
            best = min(ranks)
            assert info["selected"]["epoch"] == best[2] and info["selected"]["threshold"] == best[3]
            assert info["steps"] == 3528
            model = StateClassifier(31, False)
            model.load_state_dict(torch.load(directory / "weights.pt", weights_only=True))
            for target in plan["households"]:
                outer = load_npz(run / f"arrays/{target}/outer.npz")
                saved = load_npz(run / f"predictions/train_{house}_seed_{seed}_target_{target}.npz")
                scores = predict_windows(meta, model, outer["x"])
                np.testing.assert_allclose(scores, saved["scores"], rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(scores >= best[3], saved["scores"] >= best[3])
                y = load_npz(run / f"outer_labels/{target}.npz")["y"]
                m, _, _, _ = independent(saved["scores"], y, best[3])
                row = next(r for r in results if r["train_house"] == house and r["target_house"] == target and r["seed"] == seed and r["role"].endswith("outer"))
                assert all(row[k] == v for k, v in m.items())
                outer_predictions += 1
            models += 1
    small = load_npz(run / "arrays/small.npz")
    for seed in plan["seeds"]:
        directory = run / f"small/seed_{seed}"
        info = read_json(directory / "result.json")
        first_update = read_json(directory / "first_update.json")
        assert first_update["gradient_l2"] > 0 and first_update["parameter_delta_l2"] > 0 and first_update["finite"]
        model = StateClassifier(31, False)
        model.load_state_dict(torch.load(directory / "weights.pt", weights_only=True))
        scores = predict_windows(info["normalization"], model, small["x"])
        saved = load_npz(directory / "final_scores.npz")["scores"]
        np.testing.assert_allclose(scores, saved, rtol=1e-6, atol=1e-7)
        history = read_json(directory / "history.json")["rows"]
        last = history[-2:]
        success = all(r["precision"] >= .98 and r["recall"] >= .98 and r["ap"] >= .99 and r["bce"] <= .1 for r in last)
        assert success == info["success"] and info["updates"] <= 2000
        small_checks.append({"seed": seed, "success_recomputed": success, "gradient_and_update_nonzero": True})
    receipt = {"status": "passed", "model_readbacks": models, "checkpoint_pr_and_selection_checks": checkpoints,
               "outer_prediction_readbacks": outer_predictions, "independent_metric_checks": outer_predictions,
               "small_checks": small_checks, "normalization_train_only_exact": True,
               "context_and_event_isolation_pass": True, "final_households_accessed": False,
               "scope": "local automated verification; not a second investigator or machine", "registration_sha256": sha256(run / "registration.json"),
               "freeze_sha256": sha256(run / "freeze.json"), "verifier_sha256": sha256(Path(__file__))}
    write_json(args.out, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
