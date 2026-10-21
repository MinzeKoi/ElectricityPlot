"""Freeze, evaluate and verify A2 on the fixed A1 development protocol."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.fixed_queries import freeze, evaluate, verify
from virtual_submeter.data.common import DataError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "run"):
        command = commands.add_parser(name)
        command.add_argument("--a1-run", type=Path, default=ROOT / "outputs/accuracy/a1-20261021-v3")
        command.add_argument("--plan", type=Path, default=ROOT / "configs/a2_learners.json")
        command.add_argument("--legacy-settings", type=Path, default=ROOT / "configs/m3_adaptation.json")
        command.add_argument("--output", type=Path, required=True)
    for name in ("evaluate", "verify"):
        commands.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ("run", "freeze"):
            result = freeze(args.a1_run, args.plan, args.legacy_settings, args.output)
            if args.command == "run":
                result = evaluate(args.output)
        else:
            result = evaluate(args.run) if args.command == "evaluate" else verify(args.run)
    except (DataError, OSError, ImportError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "a2_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
