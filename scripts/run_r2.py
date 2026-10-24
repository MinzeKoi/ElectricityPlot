"""Preflight, freeze, resume, evaluate and verify the limited R2 development run."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy import dynamic_queries as engine
from virtual_submeter.accuracy.dynamic_session import load_plan
from virtual_submeter.accuracy.r2 import preflight
from virtual_submeter.data.common import DataError, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "freeze", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--r1-fixed", type=Path, default=ROOT / "outputs/accuracy/r1-fixed-20261022-v2")
        command.add_argument("--plan", type=Path, default=ROOT / "configs/r2_queries_20261022.json")
        if name == "preflight":
            command.add_argument("--report", type=Path, help="Save the input check as JSON")
        else:
            command.add_argument("--output", type=Path, required=True)
    for name in ("evaluate", "verify"):
        commands.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ("preflight", "run", "freeze", "resume"):
            result = preflight(args.r1_fixed, args.plan)
            if args.command == "preflight" and args.report:
                if args.report.exists():
                    raise DataError("Preflight report exists; choose a new path")
                args.report.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.report, result)
            if args.command != "preflight" and result["status"] == "r2_ready":
                result = engine.freeze(args.r1_fixed, args.plan, args.output, resume=args.command == "resume")
                if args.command == "run":
                    result = engine.evaluate(args.output)
        else:
            if load_plan(args.run / "frozen/query_plan.json")["schema_version"] not in (2, 3):
                raise DataError("The supplied run is not an R2 experiment")
            result = engine.evaluate(args.run) if args.command == "evaluate" else engine.verify(args.run)
        if result["status"].startswith("a3_"):
            result["status"] = result["status"].replace("a3_", "r2_", 1)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result["status"] == "r2_inputs_missing" else 0
    except (DataError, OSError, ImportError, KeyError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "r2_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
