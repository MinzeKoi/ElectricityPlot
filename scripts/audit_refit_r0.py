"""R0 full-record support audit. Retains booleans only, never activity or energy.

Vectorized interval complements reproduce prepare.resample's complete-bin rule.
Requires the existing local numpy/pandas runtime; no training dependencies.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from virtual_submeter.data.common import DataError, iso, read_json, sha256, write_json


def invalidate(mask, first, last, origin):
    """Invalidate bins intersecting nonempty half-open intervals, using a union."""
    use = last > first
    left = np.clip((first[use] - origin) // 8, 0, len(mask))
    right = np.clip((last[use] - origin + 7) // 8, 0, len(mask))
    delta = np.zeros(len(mask) + 1, dtype=np.int32)
    np.add.at(delta, left, 1)
    np.add.at(delta, right, -1)
    mask[np.cumsum(delta[:-1]) > 0] = False


def strict_masks(stamps, validity, issues):
    if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
        raise DataError("Timestamps must be strictly increasing, with at least two rows")
    origin = int(stamps[0] // 8 * 8)
    size = (int(stamps[-1]) - origin + 7) // 8
    timing = np.ones(size, dtype=bool)
    invalidate(timing, np.array([origin, stamps[-1]]), np.array([stamps[0], origin + size * 8]), origin)
    invalidate(timing, stamps[:-1] + 8, stamps[1:], origin)
    ends = np.minimum(stamps[:-1] + 8, stamps[1:])
    before, after = {}, {}
    for channel, valid in validity.items():
        mask = timing.copy()
        bad = ~valid[:-1]
        invalidate(mask, stamps[:-1][bad], ends[bad], origin)
        before[channel] = mask
        mask = mask.copy()
        if channel != "aggregate":
            bad = issues[:-1]
            invalidate(mask, stamps[:-1][bad], ends[bad], origin)
        after[channel] = mask
    return origin, timing, before, after


def runs(mask, origin):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = ends - starts
    order = np.argsort(-lengths, kind="stable")[:5]
    return {"run_count": len(starts), "longest_seconds": int(lengths.max() * 8) if len(lengths) else 0,
            "top_five": [{"start": iso(origin + int(starts[i]) * 8), "end": iso(origin + int(ends[i]) * 8),
                          "seconds": int(lengths[i] * 8)} for i in order]}


def query_support(mask):
    width = 150
    count = len(mask) // width
    complete = mask[:count * width].reshape(count, width).all(axis=1)
    # Current CNN context is 31 bins, centered; require full input context too.
    prefix = np.r_[0, np.cumsum(~mask, dtype=np.int32)]
    starts = np.arange(count) * width
    left, right = starts - 15, starts + width + 15
    in_bounds = (left >= 0) & (right <= len(mask))
    contextual = np.zeros(count, dtype=bool)
    contextual[in_bounds] = prefix[right[in_bounds]] == prefix[left[in_bounds]]
    return {"complete_aggregate_slots": int(complete.sum()),
            "complete_slots_with_31_bin_context": int(contextual.sum()),
            "per_appliance_upper_bound": int(contextual.sum()),
            "household_three_appliance_upper_bound": int(contextual.sum()) * 3,
            "supports_budget_20_before_model_check": bool(contextual.sum() * 3 >= 20)}


def window_summary(masks, index, origin, plan):
    width = plan["window_days"] * 10800
    half = width // 2
    ranges = {"adaptation": (index, index + half), "future": (index + half + plan["split_gap_seconds"] // 8, index + width)}
    result = {"start": iso(origin + index * 8), "end": iso(origin + (index + width) * 8), "channels": {}}
    passed = True
    for channel, mask in masks.items():
        result["channels"][channel] = {}
        for name, (first, last) in ranges.items():
            valid = int(mask[first:last].sum())
            bins = last - first
            coverage = valid / bins
            ok = coverage >= plan["minimum_coverage"] and valid / 10800 >= plan["minimum_valid_days_per_partition"]
            passed &= ok
            result["channels"][channel][name] = {"coverage": coverage, "valid_days": valid / 10800, "passes": bool(ok)}
    result["all_present_channels_pass"] = bool(passed)
    a, b = ranges["adaptation"]
    result["query_input_support"] = query_support(masks["aggregate"][a:b])
    return result


def search_windows(masks, origin, plan):
    width = plan["window_days"] * 10800
    n = len(masks["aggregate"]) - width + 1
    if n <= 0:
        return {"windows_examined": 0, "first_pass": None, "best_support": None}
    half, gap = width // 2, plan["split_gap_seconds"] // 8
    minimum = np.ones(n, dtype=np.float64)
    passes = np.ones(n, dtype=bool)
    for mask in masks.values():
        prefix = np.r_[0, np.cumsum(mask, dtype=np.int32)]
        for offset, length in ((0, half), (half + gap, half - gap)):
            counts = prefix[offset + length:offset + length + n] - prefix[offset:offset + n]
            minimum = np.minimum(minimum, counts / length)
            passes &= (counts >= plan["minimum_coverage"] * length) & (counts >= plan["minimum_valid_days_per_partition"] * 10800)
    indices = np.flatnonzero(passes)
    first = int(indices[0]) if len(indices) else None
    best = int(np.argmax(minimum))
    return {"windows_examined": n, "passing_window_count": int(passes.sum()),
            "first_pass": window_summary(masks, first, origin, plan) if first is not None else None,
            "best_minimum_partition_channel_coverage": float(minimum[best]),
            "best_support": window_summary(masks, best, origin, plan),
            "best_support_purpose": "diagnostic only; not date selection"}


def audit_house(source, mapping, plan):
    fingerprint = sha256(source)
    receipt = read_json(source.with_suffix(".receipt.json"))
    if receipt["sha256"] != fingerprint or receipt["kind"] != "real_complete":
        raise DataError("Full-file receipt mismatch")
    names = {"aggregate": ["Aggregate"], **mapping}
    columns = list(dict.fromkeys(["Unix", "Issues", *(c for group in names.values() for c in group)]))
    times, issue_parts = [], []
    validity_parts = {key: [] for key in names}
    for frame in pd.read_csv(source, usecols=columns, chunksize=500000):
        t = frame["Unix"].to_numpy(dtype=np.int64)
        if not np.array_equal(t, frame["Unix"].to_numpy()):
            raise DataError("Noninteger timestamps")
        times.append(t)
        if not frame["Issues"].isin([0, 1]).all():
            raise DataError("Unknown Issues flag")
        issue_parts.append(frame["Issues"].to_numpy() == 1)
        for key, group in names.items():
            values = frame[group].to_numpy(dtype=float)
            validity_parts[key].append((np.isfinite(values) & (values >= 0)).all(axis=1))
        # No power values survive this loop; do not compute events/energy.
    stamps = np.concatenate(times)
    issues = np.concatenate(issue_parts)
    validity = {key: np.concatenate(parts) for key, parts in validity_parts.items()}
    origin, timing, before, masks = strict_masks(stamps, validity, issues)
    common = np.logical_and.reduce(list(masks.values()))
    histogram = Counter(map(int, np.diff(stamps)))
    rows = len(stamps)
    result = {"source_sha256": fingerprint, "receipt_sha256": sha256(source.with_suffix(".receipt.json")),
              "source_kind": receipt["kind"], "raw_rows": rows, "first_observation": iso(int(stamps[0])),
              "last_observation": iso(int(stamps[-1])), "grid_start": iso(origin), "grid_bins": len(timing),
              "interval_histogram_seconds": dict(sorted(histogram.items())),
              "issues_rows": int(issues.sum()), "issues_row_fraction": float(issues.mean()),
              "timestamp_support_complete_bin_coverage": float(timing.mean()),
              "channels": {}, "common_support": {"coverage": float(common.mean()), "valid_days": float(common.sum() / 10800), **runs(common, origin)},
              "original_earliest_28_days": window_summary(masks, 0, origin, plan) if len(timing) >= plan["window_days"] * 10800 else None,
              "all_aligned_28_day_windows": search_windows(masks, origin, plan)}
    for channel, mask in masks.items():
        result["channels"][channel] = {
            "raw_invalid_rows": int((~validity[channel]).sum()), "raw_nonmissing_supported_rows": int(validity[channel].sum()),
            "coverage": float(mask.mean()), "valid_days": float(mask.sum() / 10800),
            "missing_bin_decomposition": {"timestamp_gap_or_partial_boundary": int((~timing).sum()),
                                          "additional_invalid_channel_value": int((timing & ~before[channel]).sum()),
                                          "additional_issues_mask": int((before[channel] & ~mask).sum())},
            **runs(mask, origin)}
    if sha256(source) != fingerprint:
        raise DataError("Source changed during audit")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/r0_audit_20261022.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise DataError("Choose a fresh audit directory")
    plan = read_json(args.plan)
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "start.json", {"plan_sha256": sha256(args.plan), "audit_script_sha256": sha256(Path(__file__)),
                                            "status": "started_before_full_record_scan"})
    catalog = read_json(Path(plan["catalog"]))
    summary = {"schema_version": 1, "protocol_id": plan["protocol_id"], "plan_sha256": sha256(args.plan),
               "catalog_sha256": sha256(Path(plan["catalog"])), "scope": plan["scope"], "households": []}
    for house in sorted(catalog["households"], key=int):
        source = Path(plan["raw_directory"]) / f"CLEAN_House{house}.csv"
        result = audit_house(source, catalog["households"][house], plan)
        result.update(household_id=house, reserved_role=plan["roles"].get(house, plan["unlisted_role"]))
        missing = set(plan["required_channels"]) - set(result["channels"])
        result["missing_required_channels"] = sorted(missing)
        result["strict_time_support_pass"] = not missing and result["all_aligned_28_day_windows"]["first_pass"] is not None
        result["formal_admission"] = False
        write_json(args.output / f"house_{house}.json", result)
        summary["households"].append({k: result[k] for k in ("household_id", "reserved_role", "missing_required_channels", "strict_time_support_pass", "formal_admission")})
        print(f"House {house}: support_pass={result['strict_time_support_pass']}, best_min_coverage={result['all_aligned_28_day_windows'].get('best_minimum_partition_channel_coverage', 0):.4f}", flush=True)
    summary["strict_support_passing_households"] = [r["household_id"] for r in summary["households"] if r["strict_time_support_pass"]]
    summary["files"] = {p.name: sha256(p) for p in sorted(args.output.glob("*.json"))}
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
