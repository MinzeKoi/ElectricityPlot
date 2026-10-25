"""Resume verified HTTP ranges from the official UK-DALE 2017 archive."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.data.common import DataError, read_json, sha256, write_json
from virtual_submeter.data.ukdale import DOWNLOAD_URLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args()
    deadline = time.monotonic() + args.max_seconds
    output = ROOT / "data/raw/ukdale_2017_20261024"
    output.mkdir(parents=True, exist_ok=True)
    url = DOWNLOAD_URLS[0]
    request = urllib.request.Request(url, headers={"Range": "bytes=0-3"})
    with urllib.request.urlopen(request, timeout=30) as response:
        header = response.headers.get("Content-Range", "")
        if response.status != 206 or response.read() != b"PK\x03\x04":
            raise DataError("Official server does not provide valid ZIP byte ranges")
        size = int(header.rsplit("/", 1)[1])
    chunk_size = 8 * 1024 * 1024
    chunks = output / "verified_ranges"
    chunks.mkdir(exist_ok=True)
    archive = output / "ukdale.zip"
    if archive.exists():
        raise DataError("Completed archive already exists")

    def fetch(index):
        first = index * chunk_size
        last = min(size, first + chunk_size) - 1
        path = chunks / f"{index:04d}.part"
        receipt = path.with_suffix(".json")
        progress = path.with_suffix(".progress.json")
        if receipt.exists() and path.exists():
            saved = read_json(receipt)
            if saved["size"] == last - first + 1 and sha256(path) == saved["sha256"]:
                return index
            raise DataError("Cached UK-DALE range changed")
        failure = None
        for attempt in range(3):
            if time.monotonic() >= deadline:
                raise DataError("Official download acquisition deadline exceeded; verified ranges retained")
            try:
                offset = 0
                if progress.exists() and path.exists():
                    saved = read_json(progress)
                    if sha256(path) != saved["sha256"] or path.stat().st_size != saved["size"]:
                        raise DataError("Incomplete cached range changed")
                    offset = saved["size"]
                request = urllib.request.Request(url, headers={"Range": f"bytes={first + offset}-{last}"})
                with urllib.request.urlopen(request, timeout=35) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {first + offset}-{last}/{size}":
                        raise DataError("Official byte-range response mismatch")
                    with path.open("ab" if offset else "wb") as stream:
                        total = offset
                        while True:
                            if time.monotonic() >= deadline:
                                raise DataError("Official download acquisition deadline exceeded")
                            data = response.read1(65536)
                            if not data:
                                break
                            stream.write(data)
                            total += len(data)
                    if total != last - first + 1:
                        raise DataError("Truncated UK-DALE range")
                write_json(receipt, {"source_url": url, "first": first, "last": last, "size": total, "sha256": sha256(path)})
                return index
            except (OSError, ValueError) as error:
                failure = error
                if path.exists():
                    write_json(progress, {"size": path.stat().st_size, "sha256": sha256(path), "source_url": url,
                                          "first": first, "last": last, "incomplete": True})
        raise DataError(f"Range {index} failed after three attempts: {failure}")

    count = (size + chunk_size - 1) // chunk_size
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = [executor.submit(fetch, index) for index in range(count)]
            for task in as_completed(tasks):
                task.result()
                completed += 1
                if completed % 10 == 0 or completed == count:
                    print(f"Verified ranges {completed}/{count}", flush=True)
    except (OSError, ValueError) as error:
        write_json(output / "range_acquisition_failed.json", {"status": "acquisition_incomplete", "reason": str(error),
                   "verified_ranges": len(list(chunks.glob("[0-9][0-9][0-9][0-9].json"))), "total_ranges": count,
                   "source_url": url, "archive_bytes": size, "deadline_seconds": args.max_seconds})
        raise
    partial = output / "ukdale.ranges-complete.zip"
    with partial.open("wb") as stream:
        for index in range(count):
            with (chunks / f"{index:04d}.part").open("rb") as incoming:
                import shutil
                shutil.copyfileobj(incoming, stream)
    if partial.stat().st_size != size or not zipfile.is_zipfile(partial):
        raise DataError("Reassembled archive is incomplete")
    partial.rename(archive)
    write_json(output / "acquisition.json", {"status": "downloaded", "source_url": url, "resolved_url": url,
               "doi": "10.5286/UKERC.EDC.000004", "license": "CC-BY-4.0", "sha256": sha256(archive),
               "size": size, "publisher_checksum_verified": False, "method": "verified_http_ranges"})
    print(json.dumps({"status": "downloaded", "bytes": size}), flush=True)


if __name__ == "__main__":
    main()
