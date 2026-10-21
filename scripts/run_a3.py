"""Freeze, evaluate, resume and verify A3 dynamic development queries."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.dynamic_queries import freeze, evaluate, verify
from virtual_submeter.data.common import DataError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "freeze", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--a2-run", type=Path, default=ROOT / "outputs/accuracy/a2-20261021")
        command.add_argument("--plan", type=Path, default=ROOT / "configs/a3_queries.json")
        command.add_argument("--output", type=Path, required=True)
    for name in ("evaluate", "verify"):
        commands.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ("run", "freeze", "resume"):
            result = freeze(args.a2_run, args.plan, args.output, resume=args.command == "resume")
            if args.command == "run":
                result = evaluate(args.output)
        else:
            result = evaluate(args.run) if args.command == "evaluate" else verify(args.run)
    except (DataError, OSError, ImportError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "a3_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
