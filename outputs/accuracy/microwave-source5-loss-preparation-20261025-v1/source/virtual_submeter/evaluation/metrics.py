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
    predicted_on = []
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
        state_key = "raw_state" if key == "raw_power_w" else "state"
        state = row["appliances"][appliance].get(state_key, predicted >= threshold)
        if type(state) is not bool:
            raise DataError("A valid power prediction requires a Boolean state")
        predicted_on.append(state)
        day = row["timestamp"][:10]
        totals = daily.setdefault(day, {"valid_samples": 0, "prediction_kwh": 0.0, "truth_kwh": 0.0})
        totals["valid_samples"] += 1
        totals["prediction_kwh"] += predicted * interval / 3600000
        totals["truth_kwh"] += actual * interval / 3600000
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
            "corrected_exceedance_rate": sum(sum(a["power_w"] for a in row["appliances"].values()) > row["aggregate_power_w"] + 1e-5 for row in valid) / len(valid) if valid else None,
            "mean_absolute_balance_w": _mean([abs(row["aggregate_power_w"] - row["unexplained_power_w"] - sum(a["power_w"] for a in row["appliances"].values())) for row in valid]),
            "aggregate_energy_kwh": aggregate * interval / 3600000 if valid else None,
            "attributed_energy_kwh": (aggregate - unexplained) * interval / 3600000 if valid else None,
            "unexplained_energy_kwh": unexplained * interval / 3600000 if valid else None,
            "attributed_energy_fraction": (aggregate - unexplained) / aggregate if aggregate > 0 else None,
            "notice": "Unexplained load includes modeling and measurement error, not identified unknown devices."}


def interval_metrics(rows: list[dict], labels: list[dict], appliance: str) -> dict:
    truth = {(r["household_id"], utc(r["timestamp"])): float(r["power_w"]) for r in labels if r["power_w"] != ""}
    result = {}
    for version, lower, upper in (("raw", "raw_lower_w", "raw_upper_w"), ("corrected", "lower_w", "upper_w")):
        pairs = []
        for row in rows:
            value = row["appliances"][appliance]
            actual = truth.get((row["household_id"], utc(row["timestamp"])))
            if actual is not None and value.get(lower) is not None and value.get(upper) is not None:
                pairs.append((value[lower] <= actual <= value[upper], value[upper] - value[lower]))
        result[version] = {"valid_samples": len(pairs), "empirical_coverage": _mean([p[0] for p in pairs]),
                           "mean_width_w": _mean([p[1] for p in pairs]),
                           "evaluation_coverage": len(pairs) / len(rows) if rows else 0}
    return result


def attribution_metrics(rows: list[dict], data: dict, appliances: list[str], interval: int) -> dict:
    """Joint label support: penalize both false attribution and unassigned truth."""
    truth = {a: {(r["household_id"], utc(r["timestamp"])): float(r["power_w"]) for r in data[a] if r["power_w"] != ""} for a in appliances}
    common = []
    for row in rows:
        identity = (row["household_id"], utc(row["timestamp"]))
        if row["raw_excess_w"] is not None and all(identity in truth[a] for a in appliances):
            common.append((row, [truth[a][identity] for a in appliances]))
    result = {"joint_valid_samples": len(common), "joint_coverage": len(common) / len(rows) if rows else 0}
    total = sum(row["aggregate_power_w"] for row, _ in common)
    truth_energy = sum(sum(ys) for _, ys in common) * interval / 3600000
    for version, key in (("raw", "raw_power_w"), ("corrected", "power_w"), ("all_unexplained_control", None)):
        predicted, over, missed, errors = [], [], [], []
        for row, ys in common:
            ps = [row["appliances"][a][key] if key else 0.0 for a in appliances]
            predicted.extend(ps)
            over.extend(max(0.0, p - y) for p, y in zip(ps, ys))
            missed.extend(max(0.0, y - p) for p, y in zip(ps, ys))
            errors.extend(abs(p - y) for p, y in zip(ps, ys))
        result[version] = {"attributed_energy_fraction": sum(predicted) / total if total > 0 else None,
                           "mean_appliance_mae_w": _mean(errors), "known_truth_energy_kwh": truth_energy if common else None,
                           "overattributed_energy_kwh": sum(over) * interval / 3600000 if common else None,
                           "missed_known_energy_kwh": sum(missed) * interval / 3600000 if common else None}
    result["notice"] = "Report attribution and accuracy together. Assigning everything to unexplained has zero attribution and still misses known-device energy. Raw fraction may exceed one."
    return result
