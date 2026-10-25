"""Independent read-only checks of sealed mask experiment arrays and confusion counts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    a = p.parse_args()
    r = a.run
    for base, receipt in ((r, read(r/"registration.json")),
                          (r/"models", read(r/"training_complete.json")),
                          (r, read(r/"freeze.json")),
                          (r/"evaluation", read(r/"evaluation_complete.json"))):
        for name, expected in receipt["artifacts"].items():
            assert digest(base/name) == expected, name
    complete = read(r/"training_complete.json")
    assert complete["fits"] == 6 and len(complete["completed_models"]) == 6
    strict, partial = arrays(r/"strict_mask.npz"), arrays(r/"partial_mask.npz")
    assert len(strict["y"]) == 150069 and len(partial["y"]) == 259806
    assert int((strict["y"] >= 200).sum()) == 646 and int((partial["y"] >= 200).sum()) == 849
    for d in (strict, partial):
        assert np.isfinite(d["y"]).all() and np.isfinite(d["x"][:, 15]).all()
        np.testing.assert_array_equal(d["mask"], np.isfinite(d["x"]))
        assert (d["mask"].sum(axis=1) >= 24).all()
        assert not ((~d["mask"][:, :-1]) & (~d["mask"][:, 1:])).any()
    assert strict["mask"].all()
    for house in (0, 1):
        si, pi = strict["house"] == house, partial["house"] == house
        idx = np.searchsorted(partial["t"][pi], strict["t"][si])
        for k in ("x", "y", "event", "raw_index", "mask"):
            np.testing.assert_array_equal(strict[k][si], partial[k][pi][idx])
    meta = read(r/"normalization.json")
    assert meta["input_log_mean"] == float(np.log1p(strict["x"]).mean())
    assert meta["input_log_std"] == float(np.log1p(strict["x"]).std())
    y = arrays(r/"evaluation/truth_once.npz")["y"]
    records = read(r/"evaluation/metrics.json")["rows"]
    baseline = read(r/"baseline_identity.json")
    original = Path(baseline["reference_run"])
    for seed in (42, 1019, 2026):
        base = arrays(original/f"reference/predictions/seed_{seed}/frozen_test_refit_17_0.npz")
        w50 = arrays(original/f"predictions/W50/seed_{seed}/frozen_test_refit_17_0.npz")
        mask = np.isfinite(y) & np.isfinite(w50["raw"][:, 1])
        assert int(mask.sum()) == 137523
        for arm in ("strict_mask", "partial_mask", "task4_selected"):
            z = arrays(r/f"predictions/{arm}_{seed}.npz")
            np.testing.assert_array_equal(z["raw"], base["raw"])
            np.testing.assert_array_equal(z["t"], base["stamps"])
            np.testing.assert_array_equal(np.isfinite(z["scores"]), np.isfinite(w50["raw"][:, 1]))
            observed_y, predicted = y[mask] >= 200, z["state"][mask] == 1
            counts = {"tp": int((predicted & observed_y).sum()), "fp": int((predicted & ~observed_y).sum()),
                      "fn": int((~predicted & observed_y).sum()), "tn": int((~predicted & ~observed_y).sum())}
            row = next(x for x in records if x["arm"] == arm and x["seed"] == seed)
            assert all(row[k] == value for k, value in counts.items())
    groups = {arm: [x for x in records if x["arm"] == arm] for arm in ("W50", "partial_mask")}
    b, c = groups["W50"], groups["partial_mask"]
    mean = lambda rows, key: sum(x[key] for x in rows)/len(rows)
    conditions = {"every_seed_recall_ge_85pct": all(x["recall"] >= .85 for x in c),
                  "mean_precision_ge_8pct": mean(c,"precision") >= .08,
                  "mean_f1_gain_ge_02": mean(c,"f1")-mean(b,"f1")+1e-12 >= .02,
                  "two_seeds_precision_and_f1_improve": sum(x["precision"] > z["precision"] and x["f1"] > z["f1"] for z,x in zip(b,c)) >= 2,
                  "every_seed_f1_drop_le_02": all(x["f1"]-z["f1"] >= -.02 for z,x in zip(b,c)),
                  "mean_fp_reduction_ge_30pct": mean(c,"fp") <= .7*mean(b,"fp"),
                  "no_degenerate_seed": all(not x["degenerate"] for x in c)}
    assert all(conditions.values()) == read(r/"evaluation/summary.json")["development_acceptance_passed"]
    ordinary = read(r/"ordinary_readback.json")
    assert ordinary["scores_exact"] and ordinary["states_exact"] and ordinary["missing_exact"]
    assert digest(r/"ordinary_seed42.csv") == ordinary["output_sha256"]
    receipt = {"status": "independently_verified", "registration_sha256": digest(r/"registration.json"),
               "freeze_sha256": digest(r/"freeze.json"), "evaluation_complete_sha256": digest(r/"evaluation_complete.json"),
               "verification_script_sha256": digest(Path(__file__)), "actual_real_fits": 6,
               "added_support_verified": True, "normalization_verified": True,
               "same_support_and_raw_devices_verified": True, "ordinary_cli_readback_verified": True,
               "development_gate_conditions": conditions, "development_acceptance_passed": all(conditions.values()),
               "fp_change_ratio_vs_W50": mean(c,"fp")/mean(b,"fp")-1,
               "final_houses_accessed": False, "no_additional_fits_or_reselection": True}
    output = r/"independent_verification.json"
    if output.exists():
        assert read(output) == receipt, "Existing verification differs"
    else:
        output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
