"""Prepare/verify the registered full-source microwave workflow."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE/"registration.json").exists() else HERE.parent
sys.path.insert(0, str(ROOT/("source" if (HERE/"registration.json").exists() else "src")))
from virtual_submeter.accuracy import microwave_fullsource as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "report", "train", "freeze", "evaluate"])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=ROOT/"configs/microwave_fullsource_20261025.json")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = experiment.prepare(args.run, args.plan)
        elif args.command in ("train", "freeze", "evaluate"):
            raise ValueError("Execution adapters pending upstream recipe lock and coordinator execution message; no training or target label access")
        else:
            result = getattr(experiment, args.command)(args.run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
