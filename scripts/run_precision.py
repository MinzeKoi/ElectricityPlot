"""Prepare, train/freeze, evaluate and verify the finite microwave precision experiment."""
import argparse
import json
from pathlib import Path
import sys

# A copied entrypoint runs against its sealed package without adding cache files.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ("source" if (ROOT / "source/virtual_submeter").is_dir() else "src")))
from virtual_submeter.accuracy import precision
from virtual_submeter.data.common import DataError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "verify", "report"])
    parser.add_argument("--run", type=Path, default=ROOT if (ROOT / "registration.json").exists() else ROOT / "outputs/accuracy/precision-20261024-v1")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/precision_20261024.json")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete training with identical registered inputs/source")
    args = parser.parse_args()
    try:
        if args.resume and args.command != "train":
            raise DataError("Resume is only supported for incomplete training")
        if args.command == "prepare":
            result = precision.prepare(args.run, args.plan)
        elif args.command == "train":
            result = precision.train_and_freeze(args.run, args.resume)
        else:
            if args.command == "report":
                precision.verify(args.run)
            result = getattr(precision, args.command)(args.run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(json.dumps({"status": "precision_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
