"""Freeze, evaluate, verify or reproduce an M5 experiment without overwriting runs."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.common import DataError
from virtual_submeter.experiments.runner import freeze_experiment, evaluate_experiment, verify_experiment
from virtual_submeter.experiments.summary import compare_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--experiment", type=Path, default=ROOT / "configs/m5_experiment.json")
        sub.add_argument("--training", type=Path, default=ROOT / "configs/m2_baseline.json")
        sub.add_argument("--adaptation", type=Path, default=ROOT / "configs/m3_adaptation.json")
        sub.add_argument("--trust", type=Path, default=ROOT / "configs/m4_trust.json")
        sub.add_argument("--output", type=Path, required=True)
    for name in ("evaluate", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--run", type=Path, required=True)
        if name == "evaluate":
            sub.add_argument("--manifest", type=Path, help="Same frozen protocol on another machine")
    sub = commands.add_parser("compare")
    sub.add_argument("--left", type=Path, required=True)
    sub.add_argument("--right", type=Path, required=True)
    sub.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    try:
        if args.command in ("freeze", "run"):
            result = freeze_experiment(args.manifest, args.experiment, args.training, args.adaptation, args.trust, args.output)
            if args.command == "run":
                result = evaluate_experiment(args.output)
        elif args.command == "evaluate":
            result = evaluate_experiment(args.run, args.manifest)
        elif args.command == "verify":
            result = verify_experiment(args.run)
        else:
            result = compare_runs(args.left, args.right, args.tolerance)
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result["status"] == "reproduction_differs" else 0
    except (DataError, OSError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "m5_failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
