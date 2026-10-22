"""Verify M4 evidence without reopening any source or target submeter labels."""

from pathlib import Path

from ..data.common import DataError, read_json, safe_child, sha256, source_fingerprint
from .output import validate_output


def verify(directory: Path) -> dict:
    if (directory / "run.json").is_file():
        record = read_json(directory / "run.json")
        if record.get("status") == "m4_complete":
            directory = safe_child(directory, record["stress"]).parent
    result = read_json(directory / "stress.json")
    if result.get("status") != "stress_evaluated" or not result.get("predictions_frozen_before_test_labels"):
        raise DataError("M4 stress is incomplete or was not frozen before evaluation")
    hashes = result.get("artifacts_sha256", {})
    required = {"baseline/predictions.json", "injected/predictions.json", "baseline/metrics.json", "injected/metrics.json",
                "frozen_predictions.json", "injection.json", "injected_aggregate.csv", "settings.json", "protocol.json", "report.html"}
    if not required <= hashes.keys():
        raise DataError("M4 evidence manifest is incomplete")
    for name, expected in hashes.items():
        if sha256(safe_child(directory, name)) != expected:
            raise DataError(f"M4 artifact hash mismatch: {name}")
    frozen = read_json(directory / "frozen_predictions.json")
    if frozen["settings_sha256"] != hashes["settings.json"] or frozen["injected_input_sha256"] != hashes["injected_aggregate.csv"] or frozen["injection_sha256"] != hashes["injection.json"]:
        raise DataError("M4 frozen inputs mismatch")
    if result["settings_sha256"] != hashes["settings.json"] or result["protocol_sha256"] != hashes["protocol.json"]:
        raise DataError("M4 settings/protocol identity mismatch")
    for name in ("baseline", "injected"):
        payload = read_json(directory / name / "predictions.json")
        validate_output(payload)
        evaluation = read_json(directory / name / "metrics.json")
        expected = hashes[f"{name}/predictions.json"]
        if frozen[f"{name}_sha256"] != expected or evaluation["prediction_sha256"] != expected or result[name] != evaluation:
            raise DataError("M4 prediction/evaluation identity mismatch")
        if payload.get("calibration") and payload["calibration"]["sha256"] != hashes.get(f"{name}/calibration.json"):
            raise DataError("M4 calibration snapshot mismatch")
    return {"status": "m4_artifacts_verified", "files_checked": len(hashes), "partition_id": result["partition_id"],
            "source_code_matches_current": result["code_sha256"] == source_fingerprint(),
            "test_labels_reopened": False, "notice": result["notice"]}
