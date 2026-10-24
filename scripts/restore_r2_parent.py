"""Restore the sealed R1 parent from fetched Git objects and local LFS bytes.

Run git fetch and a scoped git lfs fetch first. Never modify an existing run,
the checkout/index, historical metadata, or parse target label CSV contents.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.common import DataError, publish_directory, sha256

PARENT = "outputs/accuracy/r1-fixed-20261022-v2"
REF = "f266b762b752b32a31810d916d198ccc06f7ec51"
POINTER = re.compile(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n")


def sealed_text(data, expected):
    """Recover checkout line endings only when the original seal proves them."""
    if hashlib.sha256(data).hexdigest() == expected:
        return data
    restored = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if hashlib.sha256(restored).hexdigest() != expected:
        raise DataError("Git text bytes cannot reproduce the original experiment seal")
    return restored


def restore(ref, output, check_only=False):
    if output.exists() and not check_only:
        raise DataError("Destination already exists; preserve the original run")
    archive = subprocess.run(["git", "-c", "filter.lfs.process=", "-c", "filter.lfs.smudge=",
                              "-c", "filter.lfs.required=false", "archive", "--format=tar", ref, PARENT], cwd=ROOT,
                             check=True, capture_output=True).stdout
    lfs_root = subprocess.run(["git", "rev-parse", "--git-path", "lfs/objects"], cwd=ROOT,
                              check=True, capture_output=True, text=True).stdout.strip()
    lfs_root = (ROOT / lfs_root).resolve()
    entries, missing, checked, total_lfs_bytes = [], [], {}, 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tree:
        for member in tree.getmembers():
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or ".." in name.parts
                    or "\\" in member.name or not member.name.startswith(PARENT + "/")):
                raise DataError("Unexpected Git archive member")
            relative = Path(*name.relative_to(PARENT).parts)
            data = tree.extractfile(member).read()
            pointer = POINTER.fullmatch(data)
            source = None
            if pointer:
                oid, size = pointer[1].decode("ascii"), int(pointer[2])
                source = lfs_root / oid[:2] / oid[2:4] / oid
                total_lfs_bytes += size
                if oid not in checked:
                    checked[oid] = source.is_file() and source.stat().st_size == size and sha256(source) == oid
                if not checked[oid]:
                    missing.append({"path": member.name, "sha256": oid, "bytes": size})
            elif data.startswith(b"version https://git-lfs.github.com/spec/"):
                raise DataError("Unsupported LFS pointer")
            entries.append((relative, source, data))
    # Git normalizes text on commit, whereas these Windows runs sealed CRLF
    # JSON files. Recover only the exact bytes certified by the historical seal;
    # never rewrite a manifest/hash to accommodate a changed file.
    text = {relative.as_posix(): data for relative, source, data in entries if source is None}
    state = json.loads(text["run.json"])
    barrier, evidence = json.loads(text["freeze.json"]), json.loads(text["evidence.json"])
    expected = {"freeze.json": state["freeze_sha256"], "evidence.json": state["evidence_sha256"],
                **{"frozen/" + k: v for k, v in barrier["artifacts_sha256"].items()},
                **evidence["artifacts_sha256"]}
    restored_text, verified_entries = 0, []
    for relative, source, data in entries:
        seal = expected.get(relative.as_posix())
        if seal:
            if source:
                if POINTER.fullmatch(data)[1].decode("ascii") != seal:
                    raise DataError("LFS identity differs from the original experiment seal")
            else:
                restored = sealed_text(data, seal)
                restored_text += restored != data
                data = restored
        verified_entries.append((relative, source, data))
    result = {"status": "r1_lfs_missing" if missing else "r1_restore_ready", "ref": ref,
              "files": len(entries), "lfs_objects": len(checked), "lfs_bytes_in_tree": total_lfs_bytes,
              "text_files_restored_to_original_sealed_line_endings": restored_text,
              "missing": missing, "target_labels_parsed": False, "output": str(output)}
    if missing or check_only:
        return result
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-r1-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        for relative, source, data in verified_entries:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source:
                shutil.copyfile(source, destination)
            else:
                destination.write_bytes(data)
        publish_directory(staging, output)
    result["status"] = "r1_parent_restored"
    result["notice"] = "Git/LFS bytes restored; run R2 preflight to verify the original experiment seals."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=REF)
    parser.add_argument("--output", type=Path, default=ROOT / PARENT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        result = restore(args.ref, args.output.resolve(), args.check_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result["missing"] else 0
    except (DataError, OSError, subprocess.CalledProcessError, tarfile.TarError, KeyError, ValueError) as error:
        print(json.dumps({"status": "r1_restore_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
