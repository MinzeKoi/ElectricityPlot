"""Strict, frozen baseline settings, independent of target household answers."""

import math
from pathlib import Path

from ..data.common import DataError, read_json


def load_settings(path: Path) -> dict:
    settings = read_json(path)
    fields = {"schema_version", "window_samples", "channels", "hidden_units", "epochs", "batch_size", "learning_rate", "active_loss_weight", "seed", "cpu_threads", "on_thresholds_w"}
    if set(settings) != fields or settings["schema_version"] != 1:
        raise DataError("Incomplete or unknown M2 settings")
    for key in ("window_samples", "channels", "hidden_units", "epochs", "batch_size", "cpu_threads"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise DataError(f"{key} must be a positive integer")
    if settings["window_samples"] < 7 or settings["window_samples"] % 2 != 1:
        raise DataError("window_samples must be odd and at least 7")
    if type(settings["seed"]) is not int or settings["seed"] < 0:
        raise DataError("seed must be a nonnegative integer")
    for key in ("learning_rate", "active_loss_weight"):
        if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]) or settings[key] <= 0:
            raise DataError(f"{key} must be finite and positive")
    thresholds = settings["on_thresholds_w"]
    if not isinstance(thresholds, dict) or not thresholds:
        raise DataError("Per-appliance thresholds are required")
    if any(not isinstance(key, str) or type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for key, value in thresholds.items()):
        raise DataError("State thresholds must be finite positive Watts")
    return settings
