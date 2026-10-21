"""Run and verify development-only A0 diagnosis; no network access required."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.runner import run, verify
from virtual_submeter.data.common import DataError, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="A0 development accuracy diagnosis")
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--model", type=Path, default=ROOT / "outputs/m2-20261020/seq2point")
    execute.add_argument("--manifest", type=Path, default=ROOT / "data/processed/refit_m1_protocol/protocol.json")
    execute.add_argument("--prepared", type=Path, default=ROOT / "data/processed/refit_m1")
    execute.add_argument("--settings", type=Path, default=ROOT / "configs/m3_adaptation.json")
    execute.add_argument("--output", type=Path, required=True)
    for name in ("verify", "check-regression"):
        commands.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run(args.model, args.manifest, args.settings, args.output, ROOT, args.prepared)
        else:
            result = verify(args.run)
        if args.command == "check-regression":
            diagnosis = read_json(args.run / "diagnosis.json")
            regressions = diagnosis["regressions"]
            print(json.dumps({"status": "regression_detected" if regressions else "no_f1_regression", "regressions": regressions}, indent=2))
            return 1 if regressions else 0
    except (DataError, OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(json.dumps({"status": "a0_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
