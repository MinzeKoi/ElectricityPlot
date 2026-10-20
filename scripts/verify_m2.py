"""Verify saved M2 evidence without training or choosing models using test answers."""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_submeter.data.common import DataError, read_json, sha256
from virtual_submeter.models.artifacts import load_model


def verify(run: Path, manifest: Path):
    protocol = read_json(manifest)
    protocol_hash = sha256(manifest)
    targets = {p["household_id"] for p in protocol["partitions"] if p["role"] in ("target_adaptation", "frozen_test")}
    summary = {}
    for name in ("single_kettle", "seq2point", "constant_mean", "zero"):
        model_dir = run / name
        model, _ = load_model(model_dir)
        payload_path = run / f"predictions_{name}/predictions.json"
        payload = read_json(payload_path)
        evaluation = read_json(run / f"evaluation_{name}/metrics.json")
        if model["protocol_sha256"] != protocol_hash or payload["protocol_sha256"] != protocol_hash or evaluation["protocol_sha256"] != protocol_hash:
            raise DataError("Evidence refers to a different protocol")
        if model["model_id"] != payload["model_id"] or model["model_id"] != evaluation["model_id"]:
            raise DataError("Model identity differs between artifacts")
        if payload["model_manifest_sha256"] != sha256(model_dir / "model.json") or evaluation["prediction_sha256"] != sha256(payload_path):
            raise DataError("Model/prediction artifacts changed after evaluation")
        if targets & (set(model["training_households"]) | set(model["validation_households"])) or model["target_labels_read"]:
            raise DataError("Target labels entered model fitting")
        if not (run / f"evaluation_{name}/report.html").is_file() or not (run / f"predictions_{name}/predictions.csv").is_file():
            raise DataError("Report or CSV export is missing")
        valid = 0
        for row in payload["rows"]:
            if any(value["confidence"] is not None for value in row["appliances"].values()):
                raise DataError("M2 must not invent calibrated confidence")
            if row["unexplained_power_w"] is None:
                if any(value["power_w"] is not None or value["state"] is not None for value in row["appliances"].values()):
                    raise DataError("Unavailable prediction was represented as an appliance state")
                continue
            valid += 1
            total = row["aggregate_power_w"]
            assigned = sum(value["power_w"] for value in row["appliances"].values())
            tolerance = max(0.0001, total * 0.000001)
            if any(not math.isfinite(value["power_w"]) or value["power_w"] < 0 for value in row["appliances"].values()):
                raise DataError("Invalid device prediction")
            if assigned > total + tolerance or abs(assigned + row["unexplained_power_w"] - total) > tolerance:
                raise DataError("Aggregate consistency failed")
        summary[name] = {"model_id": model["model_id"], "rows": len(payload["rows"]), "valid_prediction_rows": valid,
                         "appliances": {a: {key: value["corrected"][key] for key in ("mae_w", "active_mae_w", "f1", "evaluation_coverage")} for a, value in evaluation["metrics"].items()}}
    return {"status": "m2_artifacts_verified", "purpose": protocol["purpose"], "baselines": summary,
            "notice": "Real prefix pilot only. This verification does not train, tune or read new test labels."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    import json
    try:
        result = verify(args.run, args.manifest)
    except (DataError, OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "verification_failed", "error": str(error)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
