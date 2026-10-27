"""A4 source-only conditional power branch with a fixed state/power relation."""

import copy
import math

import numpy as np

from ..data.common import DataError, read_json
from ..trust.output import reconcile


def load_plan(path):
    plan = read_json(path)
    expected = {"schema_version", "strategies", "main_strategy", "ridge", "upper_quantile",
                "minimum_active_samples", "minimum_relative_improvement", "maximum_relative_degradation",
                "near_zero_mae_w", "near_zero_energy_kwh", "interval_level", "minimum_calibration_samples"}
    if set(plan) != expected or type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise DataError("Invalid A4 plan schema")
    strategies = plan["strategies"]
    if (not isinstance(strategies, list) or not strategies or len(set(strategies)) != len(strategies)
            or not set(strategies) <= {"random", "uncertainty", "proposed", "uncertainty_static"}
            or plan["main_strategy"] not in strategies):
        raise DataError("Invalid A4 policies")
    for key in expected - {"schema_version", "strategies", "main_strategy"}:
        value = plan[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise DataError("A4 numeric settings must be finite and positive")
    for key in ("upper_quantile", "interval_level", "minimum_relative_improvement", "maximum_relative_degradation"):
        if plan[key] >= 1:
            raise DataError("A4 fraction must be below one")
    for key in ("minimum_active_samples", "minimum_calibration_samples"):
        if type(plan[key]) is not int or plan[key] < 2:
            raise DataError("A4 support must be an integer of at least two")
    return plan


def amplitude_design(hidden, normalizer):
    features = np.clip((hidden - normalizer["mean"]) / normalizer["std"], -8, 8)
    return np.column_stack((np.ones(len(features)), features))


def fit_branch(matrices, labels, metadata, normalizer, plan):
    """Only source training arrays enter fitting; no target label argument."""
    head = {"schema_version": 1, "method": "source_on_power_ridge_times_state_score",
            "normalizer": copy.deepcopy(normalizer), "coefficients": {}, "bounds_w": {},
            "training_support": {}, "target_power_labels_read": False}
    for column, appliance in enumerate(metadata["appliances"]):
        # A2 matrices are [intercept, raw-power logit, temporal features].
        x = np.delete(matrices[appliance], 1, axis=1)
        y = labels[:, column]
        threshold = metadata["settings"]["on_thresholds_w"][appliance]
        mask = np.isfinite(y) & (y >= threshold) & np.isfinite(x).all(axis=1)
        if int(mask.sum()) < plan["minimum_active_samples"]:
            raise DataError(f"Insufficient source active support for A4 {appliance}")
        xx, yy = x[mask], y[mask]
        penalty = np.eye(xx.shape[1]) * plan["ridge"]
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(xx.T @ xx + penalty, xx.T @ yy)
        head["coefficients"][appliance] = coefficients.tolist()
        head["bounds_w"][appliance] = [float(threshold), float(np.quantile(yy, plan["upper_quantile"]))]
        head["training_support"][appliance] = int(mask.sum())
    validate_branch(head, metadata)
    return head


def validate_branch(head, metadata):
    if (head.get("schema_version") != 1 or head.get("method") != "source_on_power_ridge_times_state_score"
            or head.get("target_power_labels_read") is not False
            or metadata.get("accuracy_state_head", {}).get("variant") != "L2_max"):
        raise DataError("Invalid A4 power branch or missing frozen temporal state head")
    dimension = metadata["settings"]["hidden_units"]
    norm = head.get("normalizer", {})
    if norm != metadata["accuracy_state_head"]["normalizer"]:
        raise DataError("A4 power/state feature normalization differs")
    for key in ("mean", "std"):
        value = np.asarray(norm.get(key), dtype=float)
        if value.shape != (dimension,) or not np.isfinite(value).all() or (key == "std" and (value <= 0).any()):
            raise DataError("Invalid A4 normalization")
    appliances = metadata["appliances"]
    if any(set(head.get(key, {})) != set(appliances) for key in ("coefficients", "bounds_w", "training_support")):
        raise DataError("A4 appliance mismatch")
    for appliance in appliances:
        coefficients = np.asarray(head["coefficients"][appliance], dtype=float)
        bounds = np.asarray(head["bounds_w"][appliance], dtype=float)
        support = head["training_support"][appliance]
        if (coefficients.shape != (dimension + 1,) or not np.isfinite(coefficients).all()
                or bounds.shape != (2,) or not np.isfinite(bounds).all()
                or not 0 < bounds[0] <= bounds[1] or type(support) is not int or support < 2):
            raise DataError("Invalid A4 source amplitude parameters")


def conditional_power(head, appliances, hidden):
    x = amplitude_design(hidden, head["normalizer"])
    return np.column_stack([np.clip(x @ np.asarray(head["coefficients"][a]), *head["bounds_w"][a]) for a in appliances])


def replace_power(rows, appliances, raw):
    """Reconcile again after coupling; remove obsolete intervals and flags."""
    if raw.shape != (len(rows), len(appliances)):
        raise DataError("A4 power shape mismatch")
    for index, row in enumerate(rows):
        valid = row["raw_excess_w"] is not None
        if np.isfinite(raw[index]).all() != valid or (valid and (raw[index] < 0).any()):
            raise DataError("A4 cannot change prediction support or produce negative power")
        flags = set(row["quality_flags"]) - {"sum_exceeded_aggregate_scaled_down", "unexplained_load_present",
                "source_validation_intervals_no_target_guarantee", "aggregate_outside_calibration_range", "power_uncalibrated"}
        if valid:
            corrected, factor = reconcile(row["aggregate_power_w"], raw[index].tolist())
            row.update(raw_excess_w=max(0.0, float(raw[index].sum()) - row["aggregate_power_w"]),
                       unexplained_power_w=max(0.0, row["aggregate_power_w"] - sum(corrected)),
                       correction_method="scale_down_only" if factor < 1 else "none")
            if factor < 1:
                flags.add("sum_exceeded_aggregate_scaled_down")
        for column, appliance in enumerate(appliances):
            value = row["appliances"][appliance]
            if valid:
                value.update(raw_power_w=float(raw[index, column]), power_w=corrected[column],
                             correction_w=corrected[column] - float(raw[index, column]))
            for key in ("raw_lower_w", "raw_upper_w", "lower_w", "upper_w"):
                value[key] = None
        row["quality_flags"] = sorted(flags)


def apply_power(rows, metadata, hidden):
    head = metadata["accuracy_power_head"]
    validate_branch(head, metadata)
    scores = np.asarray([[r["appliances"][a]["state_score"] if r["appliances"][a]["state_score"] is not None else np.nan
                          for a in metadata["appliances"]] for r in rows])
    replace_power(rows, metadata["appliances"], scores * conditional_power(head, metadata["appliances"], hidden))
