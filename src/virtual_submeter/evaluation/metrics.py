"""Power, state and energy metrics on exactly the same valid time support."""

import math

from ..data.common import DataError, utc


def _mean(values):
    return sum(values) / len(values) if values else None


def _scores(predicted, actual):
    tp = sum(p and y for p, y in zip(predicted, actual))
    fp = sum(p and not y for p, y in zip(predicted, actual))
    fn = sum(not p and y for p, y in zip(predicted, actual))
    return {"precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "actual_on_samples": sum(actual), "predicted_on_samples": sum(predicted)}


def appliance_metrics(rows: list[dict], labels: list[dict], appliance: str, threshold: float, interval: int, key="power_w") -> dict:
    truth = {}
    for row in labels:
        stamp = utc(row["timestamp"])
        identity = (row["household_id"], stamp)
        if identity in truth:
            raise DataError("Duplicate ground truth")
        value = float(row["power_w"]) if row["power_w"] != "" else None
        if value is not None and (not math.isfinite(value) or value < 0):
            raise DataError("Invalid ground truth power")
        truth[identity] = value
    pairs = []
    daily = {}
    seen = set()
    for row in rows:
        identity = (row["household_id"], utc(row["timestamp"]))
        if identity in seen:
            raise DataError("Duplicate prediction timestamp")
        seen.add(identity)
        predicted = row["appliances"][appliance][key]
        actual = truth.get(identity)
        if predicted is None or actual is None:
            continue
        if not math.isfinite(predicted) or predicted < 0:
            raise DataError("Invalid predicted power")
        pairs.append((predicted, actual))
        day = row["timestamp"][:10]
        totals = daily.setdefault(day, {"valid_samples": 0, "prediction_kwh": 0.0, "truth_kwh": 0.0})
        totals["valid_samples"] += 1
        totals["prediction_kwh"] += predicted * interval / 3600000
        totals["truth_kwh"] += actual * interval / 3600000
    predicted_on = [p >= threshold for p, y in pairs]
    actual_on = [y >= threshold for p, y in pairs]
    pred_energy = sum(p for p, y in pairs) * interval / 3600000
    true_energy = sum(y for p, y in pairs) * interval / 3600000
    for item in daily.values():
        item["coverage_of_24h"] = item["valid_samples"] * interval / 86400
        item["signed_error_kwh"] = item["prediction_kwh"] - item["truth_kwh"]
        item["absolute_error_kwh"] = abs(item["signed_error_kwh"])
    return {
        "valid_samples": len(pairs), "prediction_samples": len(rows), "evaluation_coverage": len(pairs) / len(rows) if rows else 0,
        "mae_w": _mean([abs(p - y) for p, y in pairs]),
        "active_mae_w": _mean([abs(p - y) for p, y in pairs if y >= threshold]),
        **_scores(predicted_on, actual_on), "on_threshold_w": threshold,
        "energy": {"prediction_kwh": pred_energy if pairs else None, "truth_kwh": true_energy if pairs else None,
                   "signed_error_kwh": pred_energy - true_energy if pairs else None,
                   "absolute_error_kwh": abs(pred_energy - true_energy) if pairs else None,
                   "support": "same valid prediction/label bins only; not full-day energy", "daily_utc": daily},
    }


def consistency_metrics(rows: list[dict], interval: int) -> dict:
    valid = [row for row in rows if row["raw_excess_w"] is not None]
    aggregate = sum(row["aggregate_power_w"] for row in valid)
    unexplained = sum(row["unexplained_power_w"] for row in valid)
    return {"valid_samples": len(valid), "prediction_coverage": len(valid) / len(rows) if rows else 0,
            "raw_exceedance_rate": sum(row["raw_excess_w"] > 0 for row in valid) / len(valid) if valid else None,
            "mean_raw_excess_w": _mean([row["raw_excess_w"] for row in valid]),
            "mean_removed_power_w": _mean([sum(-a["correction_w"] for a in row["appliances"].values()) for row in valid]),
            "unexplained_energy_kwh": unexplained * interval / 3600000 if valid else None,
            "attributed_energy_fraction": (aggregate - unexplained) / aggregate if aggregate > 0 else None,
            "notice": "Unexplained load includes modeling and measurement error, not identified unknown devices."}
