"""Download bounded real REFIT prefixes or checksum-verified complete files."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from .common import DataError, sha256, write_json

API = "https://zenodo.org/api/records/5063428"


def fetch_refit(destination: Path, households: list[str], sample_bytes: int = 2 * 1024 * 1024) -> dict:
    if not households or len(set(households)) != len(households):
        raise DataError("Household list must be non-empty and unique")
    if type(sample_bytes) is not int or sample_bytes < 0 or 0 < sample_bytes < 1024:
        raise DataError("sample_bytes must be zero (full download) or at least 1024")
    destination.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(API, headers={"User-Agent": "ElectricityPlot/0.2"}), timeout=45) as response:
        metadata = json.load(response)
    if metadata["metadata"].get("license", {}).get("id") != "cc-by-4.0":
        raise DataError("REFIT license changed or is unknown; inspect record before downloading")
    files = {item["key"]: item for item in metadata["files"]}
    selected = []
    for house in households:
        key = f"CLEAN_House{house}.csv"
        if key not in files:
            raise DataError(f"House {house} is not available in this record")
        path = destination / key
        if path.exists() or path.with_suffix(".receipt.json").exists():
            raise DataError(f"Refusing to replace existing download: {path.name}")
        selected.append((house, files[key], path))
    write_json(destination / "zenodo_record.json", metadata)
    receipts = []
    for house, item, path in selected:
        size = item["size"]
        limit = min(sample_bytes, size) if sample_bytes else size
        partial = limit < size
        url = f"https://zenodo.org/records/5063428/files/{item['key']}?download=1"
        headers = {"User-Agent": "ElectricityPlot/0.2", "Accept-Encoding": "identity"}
        if partial:
            headers["Range"] = f"bytes=0-{limit - 1}"
        temporary = path.with_suffix(".download")
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if partial and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes 0-{limit - 1}/")):
                    raise DataError("Server did not honor the bounded range request")
                with temporary.open("xb") as stream:
                    remaining = limit
                    while remaining:
                        block = response.read(min(65536, remaining))
                        if not block:
                            raise DataError("Download truncated before requested byte count")
                        stream.write(block)
                        remaining -= len(block)
            if partial:
                # Discard the final potentially incomplete CSV row; no synthetic rows.
                with temporary.open("r+b") as stream:
                    stream.seek(max(0, limit - 65536))
                    tail = stream.read()
                    last_newline = tail.rfind(b"\n")
                    if last_newline < 0:
                        raise DataError("No complete CSV row at the end of the prefix")
                    stream.truncate(max(0, limit - 65536) + last_newline + 1)
            else:
                algorithm, expected = item["checksum"].split(":", 1)
                with temporary.open("rb") as stream:
                    if hashlib.file_digest(stream, algorithm).hexdigest() != expected:
                        raise DataError("Publisher checksum mismatch")
            with temporary.open("r", encoding="utf-8-sig") as stream:
                if not stream.readline().startswith("Time,Unix,Aggregate,"):
                    raise DataError("Downloaded content is not a REFIT CSV")
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
        receipt = {
            "schema_version": 1, "household_id": house, "source_url": url,
            "license": "CC-BY-4.0", "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "kind": "real_prefix" if partial else "real_complete", "requested_bytes": limit,
            "stored_bytes": path.stat().st_size, "publisher_bytes": size,
            "publisher_checksum": item["checksum"], "publisher_checksum_verified": not partial,
            "sha256": sha256(path), "trailing_incomplete_row_removed": partial,
        }
        write_json(path.with_suffix(".receipt.json"), receipt)
        receipts.append(receipt)
    return {"status": "downloaded", "files": receipts}
