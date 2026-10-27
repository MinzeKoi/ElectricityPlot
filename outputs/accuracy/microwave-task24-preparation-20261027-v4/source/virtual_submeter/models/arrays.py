"""Aligned arrays and complete temporal windows, with no gap imputation."""

import json
import numpy as np

from ..data.common import DataError, utc


def numeric(value: str) -> float:
    if value == "":
        return float("nan")
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise DataError("Normalized power must be finite nonnegative Watts or null")
    return number


def aggregate_arrays(rows: list[dict], interval: int):
    if not rows:
        raise DataError("Empty aggregate input")
    stamps = np.array([utc(row["timestamp"]) for row in rows], dtype=np.int64)
    if len({row["household_id"] for row in rows}) != 1:
        raise DataError("One household is required per series")
    if np.any(np.diff(stamps) <= 0) or np.any(stamps % interval):
        raise DataError("Input timestamps must be increasing and aligned to the frozen grid")
    values = np.array([numeric(row["aggregate_power_w"]) for row in rows], dtype=np.float32)
    for row in rows:
        flags = json.loads(row.get("quality_flags") or "[]")
        if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
            raise DataError("quality_flags must be a string array")
        coverage = float(row.get("coverage", 1))
        if not np.isfinite(coverage) or not 0 <= coverage <= 1:
            raise DataError("Invalid coverage")
        if row["aggregate_power_w"] and coverage != 1:
            raise DataError("Partial bins must have null normalized power")
    return stamps, values


def windows(stamps, aggregate, width: int, interval: int):
    if len(aggregate) < width:
        return np.empty((0, width), np.float32), np.empty(0, np.int64)
    view = np.lib.stride_tricks.sliding_window_view(aggregate, width)
    time_view = np.lib.stride_tricks.sliding_window_view(stamps, width)
    valid = np.isfinite(view).all(axis=1) & (np.diff(time_view, axis=1) == interval).all(axis=1)
    indices = np.arange(width // 2, len(aggregate) - width // 2)[valid]
    return np.array(view[valid], dtype=np.float32), indices


def labels_at_centers(data: dict, appliances: list[str], stamps, centers):
    labels = []
    for appliance in appliances:
        if appliance not in data:
            raise DataError(f"Missing source labels: {appliance}")
        rows = data[appliance]
        mapping = {utc(row["timestamp"]): numeric(row["power_w"]) for row in rows}
        if len(mapping) != len(rows):
            raise DataError("Duplicate label timestamps")
        labels.append([mapping.get(int(stamps[index]), float("nan")) for index in centers])
    return np.array(labels, dtype=np.float32).T
