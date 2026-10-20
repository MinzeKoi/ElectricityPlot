"""Reproduce the four M2 pilot baselines; all fitting precedes test evaluation."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_submeter import __version__
from virtual_submeter.data.common import DataError, sha256, source_fingerprint, write_json
from virtual_submeter.evaluation.runner import evaluate
from virtual_submeter.models.prediction import predict
from virtual_submeter.models.training import train


def main():
    parser = argparse.ArgumentParser(description="ElectricityPlot M2 pilot reproduction")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/refit_m1_protocol/protocol.json"))
    parser.add_argument("--settings", type=Path, default=Path("configs/m2_baseline.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", default="frozen_test_house_3")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new run directory")
    args.output.mkdir(parents=True)
    record = {"project": "ElectricityPlot", "version": __version__, "status": "running",
              "created_at": datetime.now(timezone.utc).isoformat(), "code_sha256": source_fingerprint(),
              "query_budget_used": 0, "adaptation": "none", "training": {}, "predictions": {}, "evaluations": {}}
    try:
        record.update({"protocol_sha256": sha256(args.manifest), "settings_sha256": sha256(args.settings)})
        write_json(args.output / "run.json", record)
        (args.output / "settings.json").write_bytes(args.settings.read_bytes())
        for name, kind, appliances in (("single_kettle", "small_seq2point", ["kettle"]), ("seq2point", "small_seq2point", None), ("constant_mean", "constant_mean", None), ("zero", "zero", None)):
            record["training"][name] = train(args.manifest, args.settings, args.output / name, kind, appliances)
        # No target test labels have been read before every checkpoint is fixed.
        for name in record["training"]:
            record["predictions"][name] = predict(args.output / name, args.output / f"predictions_{name}", args.manifest, args.partition)
            record["evaluations"][name] = evaluate(args.output / f"predictions_{name}", args.manifest, args.output / f"evaluation_{name}")
        record["status"] = "pilot_baselines_evaluated"
    except (DataError, OSError, ValueError, RuntimeError, KeyError) as error:
        record.update({"status": "failed", "error": str(error)})
        write_json(args.output / "run.json", record)
        print(str(error), file=sys.stderr)
        return 2
    write_json(args.output / "run.json", record)
    print(f"M2 pilot complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
