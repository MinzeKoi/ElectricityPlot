"""Load complete, consistent results or fail without returning stale curves."""

from pathlib import Path

from ..data.common import DataError, read_json, safe_child, sha256, write_json
from ..trust.output import validate_output


def write_result_manifest(directory: Path, payload: dict) -> None:
    files = ["predictions.json", "predictions.csv", "report.html"]
    if payload.get("calibration"):
        files.append("calibration.json")
    write_json(directory / "run.json", {**{k: v for k, v in payload.items() if k != "rows"},
               "artifacts_sha256": {name: sha256(directory / name) for name in files}})


def load_result(directory: Path) -> dict:
    payload = read_json(directory / "predictions.json")
    validate_output(payload)
    if not (directory / "report.html").is_file() or not (directory / "predictions.csv").is_file():
        raise DataError("Result export is incomplete; run analysis again")
    if payload.get("calibration") and sha256(directory / "calibration.json") != payload["calibration"]["sha256"]:
        raise DataError("Calibration snapshot has changed; run analysis again")
    record = read_json(directory / "run.json")
    hashes = record.get("artifacts_sha256")
    # Legacy M2/M3 exports have no hash manifest; new M4 exports bind every
    # downloadable file to the displayed prediction, including report and CSV.
    if hashes is not None:
        required = {"predictions.json", "predictions.csv", "report.html"}
        if not isinstance(hashes, dict) or not required <= hashes.keys():
            raise DataError("Result export manifest is incomplete; run analysis again")
        for name, expected in hashes.items():
            if sha256(safe_child(directory, name)) != expected:
                raise DataError(f"Result export changed: {name}; run analysis again")
        if any(record.get(k) != v for k, v in payload.items() if k != "rows"):
            raise DataError("Result metadata has changed; run analysis again")
    return payload
