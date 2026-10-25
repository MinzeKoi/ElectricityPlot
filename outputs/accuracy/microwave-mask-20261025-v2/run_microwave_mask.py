"""Immutable full-source microwave training and development acceptance workflow."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE/"registration.json").exists() else HERE.parent
sys.path.insert(0, str(ROOT/("source" if (HERE/"registration.json").exists() else "src")))
from virtual_submeter.accuracy import microwave_mask as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "report", "train", "freeze", "evaluate"])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=ROOT/"configs/microwave_mask_20261025.json")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = experiment.prepare(args.run, args.plan)
        else:
            result = getattr(experiment, args.command)(args.run, args.authorization)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
