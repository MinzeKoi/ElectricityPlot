"""Prepare R1 dates, compare three source candidates, or verify frozen evidence."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.r1 import prepare, run_models, verify
from virtual_submeter.data.common import DataError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        p = commands.add_parser(name)
        p.add_argument("--sources", type=Path, default=ROOT / "data/processed/refit_r1_sources_20261022")
        p.add_argument("--development", type=Path, default=ROOT / "data/processed/refit_r1_development_20261022")
        p.add_argument("--plan", type=Path, default=ROOT / "configs/r1_baseline_20261022.json")
        p.add_argument("--settings", type=Path, default=ROOT / "configs/m2_baseline.json")
        p.add_argument("--output", type=Path, required=True)
    for name in ("train", "verify"):
        commands.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command in ("run", "prepare"):
            result = prepare(args.sources, args.development, args.plan, args.settings, args.output)
            if args.command == "run":
                result = run_models(args.output)
        else:
            result = run_models(args.run) if args.command == "train" else verify(args.run)
    except (DataError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "r1_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("acceptance", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
