"""Describe state regression on common support without changing the learner."""

from collections import Counter
import csv
import math

from ..data.common import DataError, utc
from ..evaluation.metrics import appliance_metrics


def state_summary(rows, labels, appliance, threshold, interval):
    metric = appliance_metrics(rows, labels, appliance, threshold, interval)
    count = metric["valid_samples"]
    on = metric["predicted_on_samples"]
    metric["predicted_on_fraction"] = on / count if count else None
    metric["actual_on_fraction"] = metric["actual_on_samples"] / count if count else None
    metric["state_pattern"] = ("unavailable" if not count else "all_off" if not on
                               else "all_on" if on == count else "mixed")
    return metric


def paired_summary(before, after, labels, appliance, threshold, interval):
    identity = lambda r: (r["household_id"], r["timestamp"])
    if [identity(r) for r in before] != [identity(r) for r in after]:
        raise DataError("Paired diagnosis requires identical ordered timelines")
    for left, right in zip(before, after):
        for key in ("raw_power_w", "power_w"):
            if left["appliances"][appliance][key] != right["appliances"][appliance][key]:
                raise DataError("A0 state-only adaptation changed power or valid support")
    base = state_summary(before, labels, appliance, threshold, interval)
    current = state_summary(after, labels, appliance, threshold, interval)
    return {"baseline": base, "adapted": current,
            "f1_delta": current["f1"] - base["f1"] if current["f1"] is not None and base["f1"] is not None else None,
            "power_unchanged": True}


def event_support(rows, threshold, interval):
    """Count only threshold events bracketed by observed off bins."""
    complete = censored = active = valid = 0
    previous_stamp, previous_on, in_event, bounded_start = None, None, False, False
    for row in rows:
        stamp = utc(row["timestamp"])
        value = float(row["power_w"]) if row["power_w"] != "" else None
        if value is not None and (not math.isfinite(value) or value < 0):
            raise DataError("Invalid event support label")
        contiguous = previous_stamp is not None and stamp - previous_stamp == interval
        if not contiguous or value is None:
            if in_event:
                censored += 1
            in_event, bounded_start, previous_on = False, False, None
        if value is not None:
            valid += 1
            on = value >= threshold
            active += int(on)
            if on and not in_event:
                in_event, bounded_start = True, previous_on is False and contiguous
            elif not on and in_event:
                complete += int(bounded_start)
                censored += int(not bounded_start)
                in_event = False
            previous_on = on
        previous_stamp = stamp
    censored += int(in_event)
    return {"valid_samples": valid, "observed_days": valid * interval / 86400,
            "missing_fraction": 1 - valid / len(rows) if rows else None,
            "active_samples": active, "complete_threshold_events": complete,
            "censored_threshold_events": censored}


def query_summary(records, appliances):
    result = {}
    for appliance in appliances:
        selected = [r for r in records if r["appliance_type"] == appliance]
        counts = Counter(r["response"] for r in selected)
        usable = counts["used"] + counts["not_used"]
        result[appliance] = {"queries": len(selected), "budget_fraction": len(selected) / len(records) if records else None,
                             **{name: counts[name] for name in ("used", "not_used", "unsure", "skipped")},
                             "usable": usable, "positive_fraction": counts["used"] / usable if usable else None,
                             "single_class": usable > 0 and (counts["used"] == 0 or counts["not_used"] == 0)}
    return result


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
