"""Read-only diagnosis of already evaluated REFIT development predictions.

This is a separate diagnostic, not a new official score or production setting.
Decision cutoffs are chosen on source validation before opening development labels.
Ground-truth on/off definitions stay fixed. No final household is accessed.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.accuracy.data_expansion import compact_metrics
from virtual_submeter.data.common import read_json, sha256
from virtual_submeter.data.protocol import read_partition
from virtual_submeter.models.arrays import labels_at_centers

RUN = ROOT / "outputs/accuracy/data-expansion-20261024-v1"
OUTPUT = ROOT / "tmp/f1-diagnosis-20261024.json"
DEVICES = ["kettle", "microwave", "washing_machine"]
SEEDS = [42, 1019, 2026]
SETTINGS = read_json(RUN / "settings.json")
FROZEN = read_json(RUN / "freeze_refit.json")
SAVED = read_json(RUN / "metrics_refit.json")["metrics"]
GRID = np.unique(np.r_[np.geomspace(0.25, 16, 65), 1.0])


def scores(predicted, truth, cutoff, truth_threshold):
    valid = np.isfinite(predicted) & np.isfinite(truth)
    p, y = predicted[valid] >= cutoff, truth[valid] >= truth_threshold
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "valid": int(valid.sum())}


def ranking_diagnosis(predicted, truth, truth_threshold):
    """Post-hoc upper bound only; never supplies a deployable threshold."""
    valid = np.isfinite(predicted) & np.isfinite(truth)
    p, y = predicted[valid], truth[valid] >= truth_threshold
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]
    tp, predicted_on = np.cumsum(y)[ends], ends + 1
    precision, recall = tp / predicted_on, tp / y.sum()
    f1 = 2 * tp / (predicted_on + y.sum())
    return {"post_hoc_best_possible_scalar_cutoff_f1": float(f1.max()),
            "average_precision": float(np.sum(np.diff(np.r_[0, recall]) * precision)),
            "deployment_use": "forbidden_post_hoc_development_diagnostic_only"}


def load_predictions(group, partition):
    result = []
    for seed in SEEDS:
        receipt = next(r for r in FROZEN["predictions"] if
                       r["group"] == group and r["seed"] == seed and r["partition_id"] == partition)
        path = RUN / receipt["path"]
        assert sha256(path) == receipt["sha256"], "Frozen prediction changed"
        with np.load(path, allow_pickle=False) as a:
            result.append({k: a[k].copy() for k in ["stamps", "raw", "aggregate"]})
    assert all(np.array_equal(a["stamps"], result[0]["stamps"]) for a in result)
    return result


def labels(partition, arrays):
    manifest = RUN / "protocols/A/protocol.json"
    protocol = read_json(manifest)
    assert protocol["accuracy_usage"] == "development_only"
    part = next(p for p in protocol["partitions"] if p["partition_id"] == partition)
    assert part["household_id"] in ["refit_5", "refit_17"]
    data = read_partition(manifest, partition, "evaluation", include_labels=True)
    return labels_at_centers(data, DEVICES, arrays[0]["stamps"], np.arange(len(arrays[0]["stamps"])))


report = {"purpose": "separate_read_only_development_diagnosis", "final_households_accessed": False,
          "production_model_changed": False, "cutoff_selection": "source_validation_only_mean_three_seed_f1",
          "prediction_freeze_sha256": sha256(RUN / "freeze_refit.json"),
          "grid_multipliers": GRID.tolist(), "groups": {}}
for group in "ABC":
    source = load_predictions(group, "source_validation_refit_5_0")
    source_y = labels("source_validation_refit_5_0", source)
    selected = {}
    for col, device in enumerate(DEVICES):
        threshold = SETTINGS["on_thresholds_w"][device]
        candidates = []
        for multiplier in GRID:
            cutoff = float(threshold * multiplier)
            f1 = float(np.mean([scores(a["raw"][:, col], source_y[:, col], cutoff, threshold)["f1"] for a in source]))
            candidates.append((f1, -abs(np.log(multiplier)), cutoff))
        best = max(candidates)
        selected[device] = {"cutoff_w": best[2], "source_validation_f1": best[0]}
    # All diagnostic cutoffs for this group are fixed before development labels.
    development = load_predictions(group, "frozen_test_refit_17_0")
    dev_y = labels("frozen_test_refit_17_0", development)
    result = {}
    for col, device in enumerate(DEVICES):
        threshold = SETTINGS["on_thresholds_w"][device]
        current, alternative, distributions = [], [], []
        for seed, a in zip(SEEDS, development):
            y, p = dev_y[:, col], a["raw"][:, col]
            actual = compact_metrics(p, y, threshold)
            saved = next(r for r in SAVED if r["group"] == group and r["seed"] == seed
                         and r["partition"] == "frozen_test_refit_17_0" and r["appliance"] == device and r["view"] == "raw")
            for key in ["valid_samples", "true_positive", "false_positive", "false_negative"]:
                assert actual[key] == saved[key], (group, seed, device, key)
            assert abs(actual["f1"] - saved["f1"]) < 1e-12
            current.append(scores(p, y, threshold, threshold))
            alternative.append(scores(p, y, selected[device]["cutoff_w"], threshold))
            valid = np.isfinite(p) & np.isfinite(y)
            distributions.append({"seed": seed,
                **ranking_diagnosis(p, y, threshold),
                "on_prediction_quantiles_10_50_90": np.quantile(p[valid & (y >= threshold)], [.1, .5, .9]).tolist(),
                "off_prediction_quantiles_50_90_99_999": np.quantile(p[valid & (y < threshold)], [.5, .9, .99, .999]).tolist()})
        result[device] = {"ground_truth_threshold_w": threshold, **selected[device],
                          "current": current, "diagnostic_cutoff": alternative,
                          "prediction_distributions": distributions}
        before = float(np.mean([r["f1"] for r in current]))
        after = float(np.mean([r["f1"] for r in alternative]))
        print(group, device, f"fixed cutoff={threshold:g} F1={before:.6f}; source-selected cutoff={selected[device]['cutoff_w']:.2f} F1={after:.6f}", flush=True)
    report["groups"][group] = result
OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
baseline = report["groups"]["A"]["microwave"]["current"]
low_precision = all(r["precision"] < .1 and r["fp"] > 10 * r["tp"] for r in baseline)
print("Severe microwave false positives reproduced:", low_precision)
print("27 device/seed original metrics independently reproduced; final households untouched.")
sys.exit(1 if low_precision else 0)
