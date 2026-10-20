"""Freeze source calibration, then run paired background sensitivity evaluation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_submeter.data.common import DataError, publish_directory, write_json


def main():
    parser = argparse.ArgumentParser(description="ElectricityPlot M4 trustworthy output")
    parser.add_argument("--model", type=Path, required=True, help="Unadapted frozen power backbone")
    parser.add_argument("--prediction-model", type=Path, help="Optional P1 state-adapted model with identical power backbone")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--settings", type=Path, default=Path("configs/m4_trust.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from virtual_submeter.trust.calibration import calibrate
        from virtual_submeter.evaluation.stress import stress_test
        import tempfile
        if args.output.exists():
            raise DataError("Output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".m4-", dir=args.output.parent) as temporary:
            staging = Path(temporary) / "run"
            staging.mkdir()
            calibrate(args.model, args.manifest, args.settings, staging / "calibration")
            stress_test(args.prediction_model or args.model, args.manifest, args.partition, args.settings, staging / "stress", staging / "calibration")
            from virtual_submeter.trust.verification import verify
            write_json(staging / "verification.json", verify(staging / "stress"))
            write_json(staging / "run.json", {"status": "m4_complete", "calibration": "calibration/calibration.json", "stress": "stress/stress.json", "report": "stress/report.html"})
            publish_directory(staging, args.output)
        print(json.dumps({"status": "m4_complete", "report": str(args.output / "stress/report.html")}))
        return 0
    except (DataError, OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "m4_failed", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
