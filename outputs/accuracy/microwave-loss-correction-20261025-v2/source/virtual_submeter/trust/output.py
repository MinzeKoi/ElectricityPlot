"""Output invariants shared by inference, evaluation, reports and recovery."""

from collections import Counter
import math

from ..data.common import DataError, utc


def reconcile(total: float, raw: list[float]) -> tuple[list[float], float]:
    if any(not math.isfinite(v) or v < 0 for v in [total, *raw]):
        raise DataError("Consistency requires finite nonnegative active power")
    factor = min(1.0, total / sum(raw)) if sum(raw) else 1.0
    return [v * factor for v in raw], factor


def validate_output(payload: dict) -> None:
    """Reject corrupt/stale artifacts before showing curves or computing metrics."""
    if payload.get("status") != "complete" or not payload.get("rows"):
        raise DataError("Prediction is incomplete or empty; run analysis again")
    interval = payload.get("sample_interval_seconds")
    if type(interval) is not int or interval <= 0:
        raise DataError("Prediction sampling interval must be a positive integer")
    appliances = payload["appliances"]
    if not appliances or len(set(appliances)) != len(appliances):
        raise DataError("Invalid prediction appliance list")
    previous = None
    household = payload["rows"][0]["household_id"]
    for row in payload["rows"]:
        stamp = utc(row["timestamp"])
        if stamp % interval:
            raise DataError("Prediction timestamps must align with the sampling grid")
        if row["household_id"] != household or (previous is not None and stamp <= previous):
            raise DataError("Prediction household/timestamp order changed")
        previous = stamp
        if set(row["appliances"]) != set(appliances):
            raise DataError("Prediction appliance fields changed")
        if not isinstance(row["quality_flags"], list) or any(not isinstance(v, str) for v in row["quality_flags"]):
            raise DataError("Invalid prediction quality flags")
        total = row["aggregate_power_w"]
        if total is not None and (not math.isfinite(total) or total < 0):
            raise DataError("Invalid aggregate power")
        values = [row["appliances"][a] for a in appliances]
        raw = [v["raw_power_w"] for v in values]
        if any(v is None for v in raw):
            keys = ("raw_power_w", "power_w", "raw_state", "state", "correction_w", "confidence", "state_score", "raw_lower_w", "raw_upper_w", "lower_w", "upper_w")
            if any(v.get(key) is not None for v in values for key in keys) or any(row[key] is not None for key in ("unexplained_power_w", "raw_excess_w")):
                raise DataError("Unavailable predictions must remain null, including state and residual")
            if row.get("correction_method") is not None:
                raise DataError("Unavailable predictions cannot have a correction method")
            continue
        if total is None:
            raise DataError("A power estimate requires a valid aggregate")
        corrected, factor = reconcile(total, raw)
        for v, expected in zip(values, corrected):
            for key, number in (("power_w", expected), ("correction_w", expected - v["raw_power_w"])):
                actual = v[key]
                if actual is None or not math.isfinite(actual) or not math.isclose(actual, number, rel_tol=1e-6, abs_tol=1e-5):
                    raise DataError("Prediction consistency/correction trace is corrupt")
            if type(v["state"]) is not bool or type(v["raw_state"]) is not bool:
                raise DataError("A valid estimate requires Boolean states")
            if v.get("confidence") is not None:
                raise DataError("M4 residual intervals are not calibrated state probabilities")
            score = v.get("state_score")
            if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
                raise DataError("Invalid uncalibrated state score")
            for lower, upper, center in (("raw_lower_w", "raw_upper_w", "raw_power_w"), ("lower_w", "upper_w", "power_w")):
                lo, hi = v.get(lower), v.get(upper)
                if not payload.get("calibration") and (lo is not None or hi is not None):
                    raise DataError("Power intervals require calibration provenance")
                if payload.get("calibration") and (lo is None or hi is None):
                    raise DataError("Calibrated predictions require power intervals")
                if (lo is None) != (hi is None) or (lo is not None and (not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo <= v[center] <= hi)):
                    raise DataError("Invalid power interval")
        for key, expected in (("unexplained_power_w", max(0.0, total - sum(corrected))), ("raw_excess_w", max(0.0, sum(raw) - total))):
            actual = row[key]
            if actual is None or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-5):
                raise DataError("Residual/exceedance trace is corrupt")
        if row["correction_method"] != ("scale_down_only" if factor < 1 else "none"):
            raise DataError("Unknown or inconsistent correction method")


def quality_summary(payload: dict) -> dict:
    from ..evaluation.metrics import consistency_metrics
    flags = Counter(flag for row in payload["rows"] for flag in set(row["quality_flags"]))
    return {"quality_flag_samples": dict(sorted(flags.items())),
            "confidence_method": payload.get("confidence_method"),
            "calibration": payload.get("calibration"),
            **consistency_metrics(payload["rows"], payload["sample_interval_seconds"])}
