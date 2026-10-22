"""Prepare, freeze, evaluate and verify the A5 multi-household matrix."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.common import DataError, read_json


def preflight(plan_path):
    from virtual_submeter.accuracy.final_evaluation import validate_plan
    plan = validate_plan(read_json(plan_path))
    available, unavailable = {}, []
    for directory in sorted((ROOT / "data/processed").iterdir()):
        if not directory.is_dir():
            continue
        manifest = directory / "manifest.json"
        try:
            manifest.stat()
            audit = read_json(manifest)
        except FileNotFoundError:
            continue  # Protocol-only directories do not contain prepared audits.
        except (DataError, OSError) as error:
            unavailable.append({"directory": directory.name, "reason": str(error)})
            continue
        for house, info in audit.get("households", {}).items():
            available.setdefault(house, []).append({"manifest": str(manifest.relative_to(ROOT)),
                "kind": (info.get("provenance") or {}).get("kind"), "channels": sorted(info.get("artifacts", {}))})
    eligible = {h: records for h, records in available.items() if h not in plan["excluded_households"]
                and any(r["kind"] == "real_complete" and {"aggregate", "kettle", "microwave", "washing_machine"} <= set(r["channels"]) for r in records)}
    return {"status": "a5_preflight", "formal_evaluation_completed": False, "frozen_target_roster": plan["target_households"],
            "available_households": available, "unavailable_directories": unavailable, "inventory_complete": not unavailable,
            "eligible_complete_new_households": sorted(eligible),
            "minimum_households": plan["minimum_households"],
            "data_ready": True if len(eligible) >= plan["minimum_households"] else None if unavailable else False,
            "missing_requirements": (["Fewer than three eligible complete new households verified in accessible inventory; acquire files or resolve unavailable inventory"] if len(eligible) < plan["minimum_households"] else [])
                + (["Final target roster is still empty"] if not plan["target_households"] else [])
                + ["Named, dated unseen-future-label declaration is required for final protocol preparation",
                   "Independent member rerun is not established by local engineering tests"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--plan", type=Path, default=ROOT / "configs/a5_final.json")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--a2-run", type=Path, default=ROOT / "outputs/accuracy/a2-20261021")
    prepare.add_argument("--prepared-targets", type=Path, required=True)
    prepare.add_argument("--plan", type=Path, default=ROOT / "configs/a5_final.json")
    prepare.add_argument("--declaration", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("run", "freeze"):
        command = commands.add_parser(name)
        command.add_argument("--a2-run", type=Path, default=ROOT / "outputs/accuracy/a2-20261021")
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--plan", type=Path, default=ROOT / "configs/a5_final.json")
        command.add_argument("--query-plan", type=Path, default=ROOT / "configs/a5_queries.json")
        command.add_argument("--output", type=Path, required=True)
    for name in ("evaluate", "verify"):
        commands.add_parser(name).add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.plan)
        elif args.command == "prepare":
            from virtual_submeter.data.final_protocol import prepare as prepare_protocol
            result = prepare_protocol(args.a2_run / "frozen/protocol/protocol.json", args.prepared_targets,
                                      read_json(args.plan), args.output, args.declaration)
        else:
            from virtual_submeter.accuracy.final_evaluation import freeze, evaluate, verify
            if args.command in ("run", "freeze"):
                result = freeze(args.a2_run, args.manifest, args.plan, args.query_plan, args.output)
                if args.command == "run":
                    result = evaluate(args.output)
            else:
                result = evaluate(args.run) if args.command == "evaluate" else verify(args.run)
    except (DataError, OSError, ImportError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "a5_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
