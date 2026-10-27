"""UK-DALE active-power import with native 1/6 second support and exact time arithmetic."""

import csv
import math
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np

from .common import DataError, iso, read_json, safe_child, sha256, utc, write_json

MICROSECOND = 1000000
GRID = 8 * MICROSECOND
DATASET_URL = "https://data.ceda.ac.uk/edc/d1/7d78f943-f9fe-413b-af52-1816f9d968b0"
DOWNLOAD_URLS = (
    "https://dap.ceda.ac.uk/edc/d1/7d78f943-f9fe-413b-af52-1816f9d968b0/data/version_0/ukdale.zip",
    "https://data.ukedc.rl.ac.uk/simplebrowse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.zip",
)


def timestamp_us(values):
    """Parse UNIX decimals to integer microseconds without float epoch rounding."""
    import pandas as pd
    values = pd.Series(values, dtype=str)
    if not values.str.fullmatch(r"[0-9]+(?:\.[0-9]{1,6})?").all():
        raise DataError("UK-DALE timestamps require nonnegative UNIX seconds, at most microsecond precision")
    split = values.str.partition(".")
    return split[0].astype("int64").to_numpy() * MICROSECOND + split[2].str.pad(6, side="right", fillchar="0").astype("int64").to_numpy()


def native_to_grid(stamps, powers, native_seconds, first, last):
    """Time-weight native observations, never extrapolating past the final sample.

    Each native interval is shorter than a grid bin and intersects at most two
    bins. Callers supply a predecessor and a closing observation.
    """
    stamps = np.asarray(stamps, dtype=np.int64)
    powers = np.asarray(powers, dtype=float)
    if len(stamps) < 2 or len(stamps) != len(powers) or np.any(np.diff(stamps) <= 0):
        raise DataError("UK-DALE requires strictly increasing nonduplicate timestamps")
    if native_seconds not in (1, 6) or first % 8 or last % 8 or last <= first:
        raise DataError("Invalid native support or eight-second range")
    start, stop = first * MICROSECOND, last * MICROSECOND
    n = (last - first) // 8
    weighted, covered = np.zeros(n), np.zeros(n, dtype=np.int64)
    left = np.maximum(stamps[:-1], start)
    right = np.minimum(np.minimum(stamps[1:], stamps[:-1] + native_seconds * MICROSECOND), stop)
    valid = (right > left) & np.isfinite(powers[:-1]) & (powers[:-1] >= 0)
    left, right, values = left[valid], right[valid], powers[:-1][valid]
    bins = (left - start) // GRID
    cut = np.minimum(right, start + (bins + 1) * GRID)
    for indices, duration in ((bins, cut - left), (bins + 1, right - cut)):
        keep = (duration > 0) & (indices >= 0) & (indices < n)
        np.add.at(covered, indices[keep], duration[keep])
        np.add.at(weighted, indices[keep], duration[keep] * values[keep])
    if np.any(covered > GRID):
        raise DataError("Native observation support overlaps")
    result = np.full(n, np.nan)
    result[covered == GRID] = weighted[covered == GRID] / GRID
    return result, covered / GRID


def read_dat(path, first, last, value_column=1):
    import pandas as pd
    pieces, previous = [], None
    for chunk in pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, value_column], dtype=str,
                             chunksize=200000, keep_default_na=False):
        stamps = timestamp_us(chunk[0])
        if len(stamps) > 1 and np.any(np.diff(stamps) <= 0):
            raise DataError("Duplicate/decreasing UK-DALE timestamps")
        before = stamps < first * MICROSECOND
        if before.any():
            previous = chunk.loc[before].tail(1)
        keep = (stamps >= first * MICROSECOND) & (stamps < last * MICROSECOND)
        if keep.any():
            if not pieces and previous is not None:
                pieces.append(previous)
            pieces.append(chunk.loc[keep])
        closing = stamps >= last * MICROSECOND
        if closing.any():
            pieces.append(chunk.loc[closing].head(1))
            break
    if not pieces:
        raise DataError("UK-DALE selected range is not represented")
    data = pd.concat(pieces)
    return timestamp_us(data[0]), pd.to_numeric(data[value_column], errors="coerce").to_numpy(dtype=float)


def find_channels(house_dir):
    channels = {"aggregate": house_dir / "mains.dat"}
    with (house_dir / "labels.dat").open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            number, label = parts
            normalized = label.lower().replace(" ", "_")
            if normalized in ("kettle", "microwave", "washing_machine"):
                if normalized in channels or not number.isdecimal():
                    raise DataError("Ambiguous independent UK-DALE device mapping")
                channels[normalized] = house_dir / f"channel_{number}.dat"
    return channels


def acquire(output):
    """Only accept a real ZIP from registered official hosts; retain failed receipts."""
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "ukdale.zip"
    errors = []
    if not archive.exists():
        for url in DOWNLOAD_URLS:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "ElectricityPlot/0.6 data research"})
                with urllib.request.urlopen(request, timeout=25) as response:
                    first = response.read(4)
                    if first != b"PK\x03\x04":
                        raise DataError(f"Official link returned non-ZIP content ({response.geturl()})")
                    temporary = output / "ukdale.zip.partial"
                    with temporary.open("wb") as stream:
                        stream.write(first)
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                    if not zipfile.is_zipfile(temporary):
                        raise DataError("Incomplete UK-DALE ZIP")
                    temporary.rename(archive)
                    write_json(output / "acquisition.json", {"status": "downloaded", "source_url": url,
                               "resolved_url": response.geturl(), "doi": "10.5286/UKERC.EDC.000004",
                               "license": "CC-BY-4.0", "sha256": sha256(archive), "publisher_checksum_verified": False})
                    break
            except (OSError, ValueError) as error:
                errors.append({"url": url, "error": str(error)})
        if not archive.exists():
            write_json(output / "acquisition_failed.json", {"status": "acquisition_failed", "attempts": errors})
            raise DataError("UK-DALE official download unavailable; see acquisition_failed.json")
    if not zipfile.is_zipfile(archive) or sha256(archive) != read_json(output / "acquisition.json")["sha256"]:
        raise DataError("UK-DALE archive or acquisition receipt changed")
    destination = output / "extracted"
    extraction_receipt = output / "extracted.json"
    if destination.exists() and (not extraction_receipt.exists() or read_json(extraction_receipt)["archive_sha256"] != sha256(archive)):
        raise DataError("Incomplete or mismatched UK-DALE extraction; preserve it and select a fresh location")
    if not destination.exists():
        destination.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path = Path(member.filename)
                if not any(p in ("house_1", "house_2", "house_5") for p in path.parts):
                    continue
                if path.name != "mains.dat" and path.name != "labels.dat" and not (path.name.startswith("channel_") and path.suffix == ".dat" and "button" not in path.name):
                    continue
                safe_child(destination, member.filename)
                bundle.extract(member, destination)
        write_json(output / "extracted.json", {"archive_sha256": sha256(archive), "status": "selected_members_crc_checked"})
    return destination


def normalize(root, output):
    """Dates depend only on channel availability; external home 5 gets no event audit."""
    if output.exists():
        raise DataError("UK-DALE normalized destination exists")
    output.mkdir(parents=True)
    manifest = {"schema_version": 1, "dataset": "UKDALE_2017_04", "license": "CC-BY-4.0", "sample_interval_seconds": 8,
                "upstream_limitations": ["Native 1-second active mains and 6-second IAM support; gaps preserved."], "households": {}}
    for house in ("1", "2", "5"):
        matches = [p for p in root.rglob(f"house_{house}") if p.is_dir() and (p / "mains.dat").is_file()]
        if len(matches) != 1:
            raise DataError(f"UK-DALE house {house}: unique active mains directory unavailable")
        channels = find_channels(matches[0])
        required = ("aggregate", "kettle", "microwave")
        if any(c not in channels or not channels[c].is_file() for c in required):
            raise DataError(f"UK-DALE house {house}: required independent active-power channels unavailable")
        first = 0
        for path in channels.values():
            with path.open(encoding="utf-8") as stream:
                stamp = timestamp_us([stream.readline().split()[0]])[0]
                first = max(first, int(math.ceil(int(stamp) / GRID) * 8))
        last = first + 28 * 86400
        info = {"start": iso(first), "end": iso(last), "artifacts": {}, "channel_mapping": {},
                "source_sha256": None, "provenance": {"kind": "real_complete", "dataset": "UKDALE_2017_04", "doi": "10.5286/UKERC.EDC.000004",
                "record_url": DATASET_URL, "license": "CC-BY-4.0", "selected_by": "earliest_common_channel_start_28_days", "original_channels": {}}}
        for channel, path in channels.items():
            stamps, powers = read_dat(path, first, last)
            values, coverage = native_to_grid(stamps, powers, 1 if channel == "aggregate" else 6, first, last)
            target = output / f"house_{house}/{channel}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            power_key = "aggregate_power_w" if channel == "aggregate" else "power_w"
            with target.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["household_id", "timestamp", power_key, "coverage", "quality_flags"])
                for index, value in enumerate(values):
                    writer.writerow([house, iso(first + index * 8), "" if not np.isfinite(value) else format(value, ".12g"),
                                     coverage[index], "[]" if np.isfinite(value) else '["incomplete_native_support"]'])
            info["artifacts"][channel] = {"path": target.relative_to(output).as_posix(), "sha256": sha256(target)}
            info["channel_mapping"][channel] = path.name
            info["provenance"]["original_channels"][channel] = {"file": path.name, "sha256": sha256(path), "power": "active_w", "native_seconds": 1 if channel == "aggregate" else 6}
        import hashlib
        import json
        info["source_sha256"] = hashlib.sha256(json.dumps(info["provenance"]["original_channels"], sort_keys=True).encode()).hexdigest()
        manifest["households"][house] = info
    write_json(output / "manifest.json", manifest)
    return output


def prepare_external(run, root=None):
    from ..accuracy import data_expansion as exp
    exp.check_registered(run)
    if not (run / "metrics_refit.json").exists():
        raise DataError("Complete the REFIT A/B/C comparison before external training")
    if (run / "protocols/D").exists():
        raise DataError("External protocol already exists")
    try:
        root = Path(root) if root else acquire(exp.ROOT / "data/raw/ukdale_2017_20261024")
        normalized = normalize(root, run / "ukdale_normalized")
        plan, settings = exp.load_plan(run / "plan.json"), read_json(run / "settings.json")
        destination = run / "protocols/D"
        shutil.copytree(run / "protocols/C", destination)
        protocol = read_json(destination / "protocol.json")
        protocol["protocol_id"] = plan["experiment_id"] + "-D"
        audit = read_json(normalized / "manifest.json")
        checks = []
        for house in ("1", "2", "5"):
            info = audit["households"][house]
            first, last = utc(info["start"]), utc(info["end"])
            spans = [(first, first + 14 * 86400), (first + 14 * 86400 + 3600, last)] if house != "5" else [(first, last)]
            for index, (start, end) in enumerate(spans):
                role = "source_train" if house != "5" else "frozen_test"
                part, check = exp.partition(normalized, house, role, start, end, f"{role}_ukdale_{house}_{index}", destination, settings, plan, "ukdale")
                checks.append(check)
                if house == "5" and any(not check["devices"][a]["included"] for a in ("kettle", "microwave")):
                    raise DataError("Reserved UK-DALE house 5 fails predeclared support gates")
                protocol["partitions"].append(part)
        # External admission uses coverage only; never read activity before the freeze.
        write_json(destination / "protocol.json", protocol)
        parts = [p for p in protocol["partitions"] if p["role"] == "source_train"]
        x, _, counts = exp.dataset(destination / "protocol.json", parts, list(exp.DEVICES), settings, 8)
        if not any(p["household_id"].startswith("ukdale_") and any(a in p["artifacts"] for a in exp.DEVICES) for p in parts):
            raise DataError("No UK-DALE source device passed admission; D cannot duplicate C")
        logged = np.log1p(x)
        protocol["expansion_training"].update(input_log_mean=float(logged.mean()), input_log_std=max(float(logged.std()), 1e-6), normalization_source_windows=counts)
        write_json(destination / "protocol.json", protocol)
        write_json(run / "ukdale_status.json", {"status": "prepared", "admission": checks, "external_house": "ukdale_5", "test_activity_inspected": False})
        return {"status": "external_prepared"}
    except (DataError, OSError, ValueError) as error:
        write_json(run / "ukdale_status.json", {"status": "blocked", "reason": str(error), "test_house_replaced": False})
        exp.report(run)
        return {"status": "external_blocked", "reason": str(error)}
