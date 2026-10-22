"""Extract a complete official REFIT archive with member CRC and source receipts.

Requires py7zr (isolated tools are also accepted through --tools-dir).
Does not compute events, predictions, or model metrics.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zlib


SOURCE = "https://pure.strath.ac.uk/ws/portalfiles/portal/62090184/CLEAN_REFIT_081116.7z"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tools-dir", type=Path)
    args = parser.parse_args()
    if args.tools_dir:
        sys.path.insert(0, str(args.tools_dir.resolve()))
    import py7zr

    if args.output.exists():
        raise ValueError("Choose a new extraction directory; existing data is never replaced")
    archive_hash = digest(args.archive)
    expected = {f"CLEAN_House{h}.csv" for h in range(1, 22) if h != 14}
    with py7zr.SevenZipFile(args.archive) as archive:
        members = archive.list()
        names = [item.filename for item in members]
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("Archive must contain exactly the 20 expected root-level CSV files")
        if any(item.is_directory or item.is_symlink for item in members):
            raise ValueError("Archive must contain regular files only")
        args.output.mkdir(parents=True)
        print("Extracting 20 complete household files", flush=True)
        archive.extractall(args.output)
    receipts = []
    now = datetime.now(timezone.utc).isoformat()
    for item in members:
        path = args.output / item.filename
        if path.stat().st_size != item.uncompressed:
            raise ValueError(f"Wrong extracted size: {path.name}")
        crc = 0
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            header = stream.readline()
            if not header.startswith(b"Time,Unix,Aggregate,") or b"Issues" not in header:
                raise ValueError(f"Not the expected cleaned CSV: {path.name}")
            crc = zlib.crc32(header, crc)
            sha.update(header)
            while block := stream.read(4 * 1024 * 1024):
                crc = zlib.crc32(block, crc)
                sha.update(block)
        if item.crc32 is None or crc & 0xFFFFFFFF != item.crc32:
            raise ValueError(f"Archive member CRC mismatch: {path.name}")
        receipt = {
            "schema_version": 1,
            "household_id": re.search(r"House(\d+)", path.name)[1],
            "source_url": SOURCE,
            "license": "CC-BY-4.0",
            "retrieved_at": now,
            "kind": "real_complete",
            "stored_bytes": path.stat().st_size,
            "sha256": sha.hexdigest(),
            "archive_sha256": archive_hash,
            "archive_member": item.filename,
            "archive_member_crc32": f"{item.crc32:08x}",
            "archive_member_crc32_verified": True,
            "publisher_checksum_verified": False,
            "integrity_note": "Archive member CRC and local SHA-256 verified; no independently published archive checksum obtained.",
            "trailing_incomplete_row_removed": False,
        }
        path.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        receipts.append(receipt)
        print(f"Verified house {receipt['household_id']}: {receipt['stored_bytes']} bytes", flush=True)
    if digest(args.archive) != archive_hash:
        raise ValueError("Archive changed during extraction")
    inventory = {
        "schema_version": 1, "status": "complete_archive_extracted_and_verified",
        "archive_sha256": archive_hash, "archive_bytes": args.archive.stat().st_size,
        "source_url": SOURCE, "license": "CC-BY-4.0", "verified_at": now,
        "household_count": len(receipts), "total_csv_bytes": sum(r["stored_bytes"] for r in receipts),
        "files": receipts,
    }
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in inventory.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
