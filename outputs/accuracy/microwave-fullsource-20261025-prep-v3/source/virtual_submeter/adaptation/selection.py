"""Rank complete adaptation windows using only aggregate and model predictions."""

import hashlib
import math
import random

import numpy as np

from ..data.common import DataError, iso, read_json, utc
from .state import features, scores

STRATEGIES = ("random", "uncertainty", "proposed")


def load_settings(path):
    settings = read_json(path)
    expected = {"schema_version", "epochs", "learning_rate", "l2", "uncertainty_weight", "impact_weight", "answerability_weight", "event_threshold_w"}
    if set(settings) != expected or settings["schema_version"] != 1:
        raise DataError("Invalid M3 settings schema")
    if type(settings["epochs"]) is not int or not 1 <= settings["epochs"] <= 10000:
        raise DataError("M3 epochs must be in [1, 10000]")
    for name in expected - {"schema_version", "epochs"}:
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise DataError("M3 settings must be finite nonnegative numbers")
    if not 0 < settings["learning_rate"] <= 1 or settings["event_threshold_w"] <= 0:
        raise DataError("Invalid M3 learning rate or event threshold")
    if not math.isclose(sum(settings[k] for k in ("uncertainty_weight", "impact_weight", "answerability_weight")), 1.0):
        raise DataError("Query score weights must sum to one")
    return settings


def candidates(payload, partition, protocol, settings):
    if partition["role"] != "target_adaptation" or payload["partition_id"] != partition["partition_id"]:
        raise DataError("Selection requires an adaptation partition")
    interval = protocol["sample_interval_seconds"]
    duration = protocol["query_window_seconds"]
    width = duration // interval
    rows = payload["rows"]
    stamps = np.array([utc(r["timestamp"]) for r in rows])
    result = []
    for appliance in payload["appliances"]:
        threshold = payload["model_settings"]["on_thresholds_w"][appliance]
        values = scores(features(rows, appliance, threshold))
        # Disjoint grid slots per appliance prevent near-duplicate confirmations.
        for start in range(0, len(rows) - width + 1, width):
            end = start + width
            segment, times = rows[start:end], stamps[start:end]
            if (not np.isfinite(values[start:end]).all() or np.any(np.diff(times) != interval)
                    or times[0] < utc(partition["start"]) or times[-1] + interval > utc(partition["end"])):
                continue
            aggregate = np.array([r["aggregate_power_w"] for r in segment], dtype=float)
            energy = sum(r["appliances"][appliance]["raw_power_w"] for r in segment) * interval / 3600000
            uncertainty = 1 - 2 * abs(float(values[start:end].max()) - 0.5)
            events = int((np.abs(np.diff(aggregate)) >= settings["event_threshold_w"]).sum())
            answerability = 1 / (1 + events / 4)
            # Identifiers/tie-breaking depend on visible input, never on hidden
            # label checksums embedded in a protocol or model manifest.
            identity = f"{payload['input_sha256']}|{partition['household_id']}|{times[0]}|{appliance}"
            result.append({"query_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                           "household_id": partition["household_id"], "appliance_type": appliance,
                           "window_start": iso(int(times[0])), "window_end": iso(int(times[-1] + interval)),
                           "uncertainty": uncertainty, "estimated_energy_kwh": energy, "answerability_proxy": answerability})
    maximum = max((item["estimated_energy_kwh"] for item in result), default=0)
    for item in result:
        item["proposed_score"] = (settings["uncertainty_weight"] * item["uncertainty"]
                                  + settings["impact_weight"] * item["estimated_energy_kwh"] / max(maximum, 1e-12)
                                  + settings["answerability_weight"] * item["answerability_proxy"])
    return result


def choose(items, records, strategy, seed):
    if strategy not in STRATEGIES:
        raise DataError("Unknown query strategy")
    available = [item for item in items if not any(
        r["household_id"] == item["household_id"] and r["appliance_type"] == item["appliance_type"]
        and utc(r["window_start"]) < utc(item["window_end"]) and utc(item["window_start"]) < utc(r["window_end"])
        for r in records)]
    available.sort(key=lambda item: item["query_id"])
    random.Random(seed).shuffle(available)
    if strategy != "random":
        key = "uncertainty" if strategy == "uncertainty" else "proposed_score"
        available.sort(key=lambda item: item[key], reverse=True)
    return available[0] if available else None
