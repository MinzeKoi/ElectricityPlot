"""Prepare, compare, freeze and verify the A1 source baseline."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.baseline import prepare, train_comparison, verify
from virtual_submeter.data.common import DataError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument("--sources", type=Path, default=ROOT / "data/processed/refit_a1_sources")
        p.add_argument("--development", type=Path, default=ROOT / "data/processed/refit_a1_development")
        p.add_argument("--plan", type=Path, default=ROOT / "configs/a1_baseline.json")
        p.add_argument("--settings", type=Path, default=ROOT / "configs/m2_baseline.json")
        p.add_argument("--historical-model", type=Path, default=ROOT / "outputs/m2-20261020/seq2point")
        p.add_argument("--output", type=Path, required=True)
    for name in ("train", "verify"):
        sub.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ("prepare", "run"):
            result = prepare(args.sources, args.development, args.plan, args.settings, args.historical_model, args.output)
            if args.command == "run":
                result = train_comparison(args.output)
        elif args.command == "train":
            result = train_comparison(args.run)
        else:
            result = verify(args.run)
    except (DataError, OSError, ImportError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "a1_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("acceptance", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
