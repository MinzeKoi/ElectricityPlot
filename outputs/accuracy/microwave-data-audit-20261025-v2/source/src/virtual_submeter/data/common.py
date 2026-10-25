"""Shared formats. All persisted timestamps are UTC and intervals are half-open."""

from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from pathlib import Path


class DataError(ValueError):
    """Input cannot satisfy the declared data protocol."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(directory: Path) -> dict[str, str]:
    """Hash every file under a root using sorted, relative POSIX paths."""
    return {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def utc(value: str) -> int:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("timezone required")
        result = stamp.timestamp()
        if not result.is_integer():
            raise ValueError("whole seconds required")
        return int(result)
    except (ValueError, OverflowError, OSError) as error:
        raise DataError(f"Invalid timezone-aware timestamp: {value!r}") from error


def iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("JSON object required")
        return result
    except (ValueError, OSError) as error:
        raise DataError(f"Cannot read {path.name}: {error}") from error


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_child(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise DataError("Artifact path escapes its manifest directory")
    return target


def publish_directory(staging: Path, output: Path) -> None:
    """Publish a sibling staging tree; retry transient Windows sharing errors.

    Never fall back to a partial copy or overwrite an existing run. A permanent
    permission failure is still raised and handled as a failed operation.
    """
    if not staging.resolve().is_relative_to(output.parent.resolve()):
        raise DataError("Staging tree must be inside the declared destination parent")
    for attempt in range(6):
        if output.exists():
            raise DataError("Destination already exists; previous artifacts are immutable")
        try:
            staging.rename(output)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))
