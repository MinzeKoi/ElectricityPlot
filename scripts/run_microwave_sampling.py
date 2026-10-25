"""Audited, source-only six-fit microwave sampling experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "registration.json").exists() else HERE.parent
sys.path.insert(0, str(ROOT / ("source" if (ROOT / "registration.json").exists() else "src")))
from virtual_submeter.accuracy import microwave_sampling as experiment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "train", "freeze", "evaluate", "verify", "report"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--plan", type=Path, default=ROOT / "configs/microwave_sampling_20261025.json")
    p.add_argument("--data-manifest", type=Path)
    p.add_argument("--authorization", type=Path)
    a = p.parse_args()
    try:
        if a.command == "prepare":
            if a.data_manifest is None:
                p.error("prepare requires --data-manifest from completed audit")
            result = experiment.prepare(a.run, a.plan, a.data_manifest)
        elif a.command == "train":
            if a.authorization is None:
                p.error("train requires --authorization issued after audit and execution message")
            result = experiment.train(a.run, a.authorization)
        else:
            result = getattr(experiment, a.command)(a.run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as e:
        print(json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
