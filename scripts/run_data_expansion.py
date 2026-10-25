"""Versioned data expansion: prepare, train/freeze, evaluate, verify and external import."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy import data_expansion as expansion
from virtual_submeter.data.common import DataError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-additional", "prepare", "train", "evaluate", "verify", "external", "train-external", "evaluate-external"])
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/accuracy/data-expansion-20261024-v1")
    parser.add_argument("--additional", type=Path, default=ROOT / "data/processed/refit_expansion_additional_20261024_v1")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/data_expansion_20261024.json")
    parser.add_argument("--ukdale", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare-additional":
            result = expansion.prepare_additional(args.additional, args.plan)
        elif args.command == "prepare":
            result = expansion.prepare(args.run, args.additional, args.plan)
        elif args.command in ("train", "train-external"):
            if args.command == "train-external":
                from virtual_submeter.data.common import read_json
                if read_json(args.run / "ukdale_status.json").get("status") != "prepared":
                    raise DataError("External source acquisition/admission has not completed")
            result = expansion.train_groups(args.run, ("D",) if args.command == "train-external" else ("A", "B", "C"))
        elif args.command in ("evaluate", "evaluate-external"):
            result = expansion.evaluate(args.run, external=args.command == "evaluate-external")
        elif args.command == "external":
            from virtual_submeter.data.ukdale import prepare_external
            result = prepare_external(args.run, args.ukdale)
        else:
            result = expansion.verify(args.run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("status") == "external_blocked" else 0
    except (DataError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "expansion_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
