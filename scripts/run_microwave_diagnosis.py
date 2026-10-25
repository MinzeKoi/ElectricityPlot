"""Registered source-only microwave capability and transfer diagnosis."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "registration.json").exists() else HERE.parent
sys.path.insert(0, str(ROOT / ("source" if (ROOT / "source/virtual_submeter").exists() else "src")))
from virtual_submeter.accuracy import microwave_diagnosis as diagnosis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "audit", "small", "train", "evaluate", "verify"])
    parser.add_argument("--run", type=Path, default=ROOT if (ROOT / "registration.json").exists() else ROOT / "outputs/accuracy/microwave-diagnosis-20261025-v2")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/microwave_diagnosis_20261025.json")
    args = parser.parse_args()
    try:
        result = diagnosis.prepare(args.run, args.plan) if args.command == "prepare" else getattr(diagnosis, args.command)(args.run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
