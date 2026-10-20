"""Stream CSV into UTC, energy-weighted bins without filling missing observations."""

from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import tempfile

from .common import DataError, iso, read_json, sha256, utc, write_json


@dataclass
class Sample:
    timestamp: int
    values: dict[str, float | None]
    flags: dict[str, set[str]]


def power(value: str | None) -> tuple[float | None, set[str]]:
    if value is None or value.strip().lower() in ("", "nan", "null", "none"):
        return None, {"missing"}
    try:
        number = float(value)
    except ValueError as error:
        raise DataError(f"Power is not numeric: {value!r}") from error
    if not math.isfinite(number):
        return None, {"non_finite"}
    if number < 0:
        return None, {"negative_power_unsupported"}
    return number, set()


def read_samples(path: Path, house: str, channels: dict[str, list[str]], kind: str):
    previous = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"Unix", "Aggregate", "Issues"} if kind == "refit" else {"household_id", "timestamp", "aggregate_power_w"}
        required.update(column for group in channels.values() for column in group)
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise DataError(f"{path.name}: missing or duplicate columns; required {sorted(required)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise DataError(f"{path.name}:{line}: malformed CSV row")
            try:
                timestamp = int(row["Unix"]) if kind == "refit" else utc(row["timestamp"])
                iso(timestamp)  # Reject unsupported epoch values before persisting artifacts.
            except (ValueError, OSError, OverflowError) as error:
                raise DataError(f"{path.name}:{line}: invalid timestamp") from error
            if previous is not None and timestamp <= previous:
                raise DataError(f"{path.name}:{line}: duplicate or decreasing timestamp")
            previous = timestamp
            if kind != "refit" and row["household_id"] != house:
                raise DataError(f"{path.name}:{line}: mixed or unexpected household_id")
            value, flags = power(row["Aggregate" if kind == "refit" else "aggregate_power_w"])
            values = {"aggregate": value}
            quality = {"aggregate": flags}
            for appliance, columns in channels.items():
                pieces = [power(row[column]) for column in columns]
                values[appliance] = None if any(piece[0] is None for piece in pieces) else sum(piece[0] for piece in pieces)
                quality[appliance] = set().union(*(piece[1] for piece in pieces))
                if len(columns) > 1:
                    quality[appliance].add("multiple_meters_summed")
            if kind == "refit":
                if row["Issues"] not in ("0", "1"):
                    raise DataError(f"{path.name}:{line}: unknown Issues flag")
                for key in quality:
                    quality[key].add("upstream_cleaned_missingness_unknown")
                    if key != "aggregate" and row["Issues"] == "1":
                        quality[key].add("upstream_aggregate_discrepancy")
                        # Issues depends on IAM answers. Never filter or annotate model
                        # aggregate inputs with it, including target adaptation/test inputs.
                        values[key] = None
            else:
                try:
                    supplied = json.loads(row.get("quality_flags") or "[]")
                    if not isinstance(supplied, list) or any(not isinstance(flag, str) for flag in supplied):
                        raise ValueError("quality_flags must be a string array")
                    quality["aggregate"].update(supplied)
                except ValueError as error:
                    raise DataError(f"{path.name}:{line}: invalid quality_flags") from error
            yield Sample(timestamp, values, quality)


def resample(samples, interval: int, counters: dict):
    """Hold each observation at most one native interval; never bridge a long gap.

    Full bins contain the time-weighted mean. Partial bins return null and their
    actual coverage; partial-bin energy is deliberately not reported as full energy.
    """
    iterator = iter(samples)
    previous = next(iterator, None)
    if previous is None:
        raise DataError("Input contains no observations")
    start = previous.timestamp // interval * interval
    names = list(previous.values)
    covered = {name: 0 for name in names}
    weighted = {name: 0.0 for name in names}
    flags = {name: set() for name in names}
    count = 1
    counters["first_observation"] = iso(previous.timestamp)
    for current in iterator:
        count += 1
        delta = current.timestamp - previous.timestamp
        counters["raw_intervals_seconds"][str(delta)] += 1
        cursor = previous.timestamp
        hold_end = min(current.timestamp, cursor + interval)
        while cursor < current.timestamp:
            end = min(current.timestamp, start + interval)
            duration = max(0, min(end, hold_end) - cursor)
            for name in names:
                flags[name].update(previous.flags[name])
                if end > hold_end:
                    flags[name].add("observation_gap")
                if previous.values[name] is not None:
                    covered[name] += duration
                    weighted[name] += duration * previous.values[name]
            cursor = end
            if cursor == start + interval:
                yield _bin(start, interval, names, covered, weighted, flags)
                start += interval
                covered = {name: 0 for name in names}
                weighted = {name: 0.0 for name in names}
                flags = {name: set() for name in names}
        previous = current
    if any(covered.values()) or previous.timestamp > start:
        yield _bin(start, interval, names, covered, weighted, flags)
    if count < 2:
        raise DataError("At least two observations are required; final point is never extrapolated")
    counters["raw_rows"] = count
    counters["last_observation"] = iso(previous.timestamp)


def _bin(start, interval, names, covered, weighted, flags):
    result = {}
    for name in names:
        full = covered[name] == interval
        if not full:
            flags[name].add("incomplete_coverage")
        result[name] = {
            "power_w": weighted[name] / interval if full else None,
            "coverage": covered[name] / interval,
            "quality_flags": sorted(flags[name]),
        }
    return start, result


def prepare(inputs: dict[str, Path], output: Path, catalog: Path, interval: int = 8, kind: str = "refit") -> dict:
    if type(interval) is not int or interval != 8:
        raise DataError("M1 freezes the native 8-second grid; downsampling is a separate experiment")
    if kind not in ("refit", "aggregate"):
        raise DataError("Unknown input format")
    if not inputs or any(not isinstance(house, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", house) is None for house in inputs):
        raise DataError("Household identifiers must use 1-64 letters, digits, underscores or hyphens")
    if output.exists():
        raise DataError("Output already exists; choose a new versioned directory")
    mapping = read_json(catalog)
    for house in inputs:
        if kind == "refit" and house not in mapping["households"]:
            raise DataError(f"No verified channel mapping for house {house}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        manifest = {"schema_version": 1, "project": "ElectricityPlot", "dataset": mapping["dataset"] if kind == "refit" else "USER_AGGREGATE",
                    "license": mapping["license"] if kind == "refit" else None,
                    "citation": mapping["citation"] if kind == "refit" else None,
                    "sample_interval_seconds": interval, "catalog_sha256": sha256(catalog),
                    "upstream_limitations": mapping["upstream_limitations"] if kind == "refit" else [], "households": {}}
        for house, source in inputs.items():
            channels = mapping["households"][house] if kind == "refit" else {}
            keys = ["aggregate", *channels]
            receipt_path = source.with_suffix(".receipt.json")
            receipt = read_json(receipt_path) if receipt_path.exists() else None
            fingerprint = sha256(source)
            if receipt and receipt.get("sha256") != fingerprint:
                raise DataError(f"{source.name}: download receipt hash mismatch")
            stats = {"source_file": source.name, "source_sha256": fingerprint, "provenance": receipt,
                     "channel_mapping": channels, "raw_intervals_seconds": Counter(),
                     "bins": 0, "channels": {key: {"valid_bins": 0, "nonzero_bins": 0, "observed_energy_kwh": 0.0, "flags": Counter()} for key in keys}, "artifacts": {}}
            handles = []
            try:
                writers = {}
                for key in keys:
                    relative = f"aggregate/house_{house}.csv" if key == "aggregate" else f"labels/house_{house}/{key}.csv"
                    artifact = staging / relative
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    handle = artifact.open("w", encoding="utf-8", newline="")
                    handles.append(handle)
                    writer = csv.writer(handle)
                    writer.writerow(["household_id", "timestamp", *([] if key == "aggregate" else ["appliance_id", "appliance_type"]), "aggregate_power_w" if key == "aggregate" else "power_w", "coverage", "quality_flags"])
                    writers[key] = writer
                    stats["artifacts"][key] = {"path": relative}
                for stamp, values in resample(read_samples(source, house, channels, kind), interval, stats):
                    if stats["bins"] == 0:
                        stats["start"] = iso(stamp)
                    stats["end"] = iso(stamp + interval)
                    stats["bins"] += 1
                    for key, item in values.items():
                        value = item["power_w"]
                        identity = [] if key == "aggregate" else [f"{key}:{'+'.join(channels[key])}", key]
                        writers[key].writerow([house, iso(stamp), *identity, "" if value is None else format(value, ".12g"), item["coverage"], json.dumps(item["quality_flags"])])
                        summary = stats["channels"][key]
                        summary["flags"].update(item["quality_flags"])
                        if value is not None:
                            summary["valid_bins"] += 1
                            summary["nonzero_bins"] += int(value > 0)
                            summary["observed_energy_kwh"] += value * interval / 3600000
            finally:
                for handle in handles:
                    handle.close()
            if not stats["bins"] or not stats["channels"]["aggregate"]["valid_bins"]:
                raise DataError(f"House {house}: no fully covered valid aggregate bins")
            if sha256(source) != fingerprint:
                raise DataError(f"House {house}: source changed during preparation")
            for key in keys:
                stats["channels"][key]["coverage"] = stats["channels"][key]["valid_bins"] / stats["bins"]
                stats["artifacts"][key]["sha256"] = sha256(staging / stats["artifacts"][key]["path"])
            manifest["households"][house] = stats
        write_json(staging / "manifest.json", manifest)
        if kind == "refit":
            write_json(staging / "catalog.json", mapping)
        from .quality import render_quality
        render_quality(manifest, staging / "quality.html")
        staging.rename(output)
    return {"status": "data_prepared", "manifest": str(output / "manifest.json"), "households": list(inputs)}
