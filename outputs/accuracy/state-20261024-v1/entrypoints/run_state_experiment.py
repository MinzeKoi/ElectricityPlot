"""Prepare, train/freeze, evaluate or verify the finite microwave state experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ("source" if (ROOT / "source/virtual_submeter").is_dir() else "src")))
from virtual_submeter.accuracy import state_experiment
from virtual_submeter.data.common import DataError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "verify"])
    parser.add_argument("--run", type=Path, default=ROOT if (ROOT / "registration.json").exists() else ROOT / "outputs/accuracy/state-20261024-v1")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/state_20261024.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.resume and args.command != "train":
            raise DataError("Resume is only for incomplete training")
        result = (state_experiment.prepare(args.run, args.plan) if args.command == "prepare" else
                  state_experiment.train_and_freeze(args.run, args.resume) if args.command == "train" else
                  getattr(state_experiment, args.command)(args.run))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(json.dumps({"status": "state_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
