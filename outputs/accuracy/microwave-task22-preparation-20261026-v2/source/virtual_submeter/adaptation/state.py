"""A small max-pooling MIL state head; scores are not calibrated probabilities."""

import numpy as np

from ..data.common import DataError

INITIAL = [0.0, 1.0, 0.0]


def features(rows: list[dict], appliance: str, threshold: float):
    result = []
    for row in rows:
        raw = row["appliances"][appliance]["raw_power_w"]
        aggregate = row["aggregate_power_w"]
        result.append([1.0, np.log((raw + 1) / (threshold + 1)),
                       np.log1p(aggregate / (threshold + 1))] if raw is not None and aggregate is not None else [np.nan] * 3)
    return np.asarray(result, dtype=np.float64)


def scores(x, coefficients=INITIAL):
    return 1 / (1 + np.exp(-np.clip(x @ np.asarray(coefficients), -40, 40)))


def validate_head(head: dict, appliances: list[str]) -> None:
    if head.get("method") != "max_pool_mil_logistic_v1" or set(head.get("coefficients", {})) != set(appliances):
        raise DataError("Invalid state adapter")
    active = head.get("active_appliances")
    if not isinstance(active, list) or len(set(active)) != len(active) or any(a not in appliances for a in active):
        raise DataError("Invalid adapted appliance list")
    for values in head["coefficients"].values():
        array = np.asarray(values, dtype=float)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise DataError("Invalid state adapter coefficients")


def apply_head(rows: list[dict], metadata: dict) -> None:
    head = metadata.get("state_adapter")
    if head is None:
        return
    validate_head(head, metadata["appliances"])
    for appliance in metadata["appliances"]:
        if appliance not in head["active_appliances"]:
            continue
        threshold = metadata["settings"]["on_thresholds_w"][appliance]
        values = scores(features(rows, appliance, threshold), head["coefficients"][appliance])
        for row, value in zip(rows, values):
            item = row["appliances"][appliance]
            valid = bool(np.isfinite(value))
            item["state_score"] = float(value) if valid else None
            # The independent state head never gates or changes power estimates.
            item["state"] = item["raw_state"] = bool(value >= 0.5) if valid else None


def fit_head(rows, records, appliances, thresholds, settings):
    coefficients, history, counts = {}, {}, {}
    for appliance in appliances:
        x = features(rows, appliance, thresholds[appliance])
        bags = []
        for record in records:
            if record["appliance_type"] != appliance or record["response"] not in ("used", "not_used"):
                continue
            indices = [i for i, row in enumerate(rows) if record["window_start"] <= row["timestamp"] < record["window_end"]]
            bag = x[indices]
            if not len(bag) or not np.isfinite(bag).all():
                raise DataError("Confirmed window lacks complete model features")
            bags.append((bag, float(record["response"] == "used")))
        weight = np.asarray(INITIAL, dtype=float)
        prior = weight.copy()
        losses = []
        # Fixed steps and source-independent settings; no test-set early stopping.
        for _ in range(settings["epochs"] if bags else 0):
            gradient = settings["l2"] * (weight - prior)
            loss = 0.0
            for bag, label in bags:
                logits = bag @ weight
                index = int(np.argmax(logits))
                logit = float(logits[index])
                probability = float(scores(bag[index:index + 1], weight)[0])
                gradient += (probability - label) * bag[index] / len(bags)
                loss += float(np.logaddexp(0, logit) - label * logit) / len(bags)
            weight -= settings["learning_rate"] * gradient
            if not np.isfinite(weight).all():
                raise DataError("Non-finite MIL adapter")
            losses.append(loss)
        coefficients[appliance], history[appliance], counts[appliance] = weight.tolist(), losses, len(bags)
    return {"method": "max_pool_mil_logistic_v1", "coefficients": coefficients,
            "active_appliances": [a for a, count in counts.items() if count],
            "score_semantics": "uncalibrated state score; not confidence", "power_changed": False}, history, counts
