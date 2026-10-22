"""Assemble R1 validation using artifact hashes and saved metrics only."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.accuracy.r1 import verify as verify_baseline
from virtual_submeter.accuracy.fixed_queries import verify as verify_fixed
from virtual_submeter.data.common import DataError, read_json, sha256, source_fingerprint, utc, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/accuracy/r1-20261022")
    parser.add_argument("--fixed", type=Path, default=ROOT / "outputs/accuracy/r1-fixed-20261022-v2")
    parser.add_argument("--boundary", type=Path, default=ROOT / "outputs/accuracy/r1-boundary-check-20261022.json")
    parser.add_argument("--handoff", type=Path, default=ROOT / "outputs/accuracy/r1-handoff-20261022-v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise DataError("Validation evidence destination must be fresh")
    baseline, fixed = verify_baseline(args.baseline), verify_fixed(args.fixed)
    protocol = read_json(args.baseline / "protocol/protocol.json")
    houses = {p["household_id"] for p in protocol["partitions"]}
    if houses != {"2", "5", "17", "19"} or houses & {"8", "9", "20"}:
        raise DataError("R1 household isolation failed")
    partitions = {p["partition_id"]: p for p in protocol["partitions"]}
    gap = utc(partitions["frozen_test_house_17"]["start"]) - utc(partitions["target_adaptation_house_17"]["end"])
    if gap != 3600:
        raise DataError("R1 temporal isolation changed")
    date_bindings = {}
    for name in ("sources", "development"):
        selection = read_json(ROOT / f"configs/r0_dates_{name}_20261022.json")
        path = ROOT / f"data/processed/refit_r1_{name}_20261022/manifest.json"
        audit = read_json(path)
        audit_key = "source_audit_sha256" if name == "sources" else "development_audit_sha256"
        if sha256(path) != protocol[audit_key] or sha256(ROOT / f"configs/r0_dates_{name}_20261022.json") != protocol["r0_date_hashes"][name]:
            raise DataError("R1 prepared manifest or R0 date version changed")
        if audit["catalog_sha256"] != selection["catalog_sha256"]:
            raise DataError("Prepared R1 channel catalog differs from R0")
        for house, spec in selection["households"].items():
            info = audit["households"][house]
            if any(info[k] != spec[k] for k in ("start", "end", "source_sha256")):
                raise DataError("Prepared R1 raw identity/date differs from R0")
        date_bindings[name] = {"manifest_sha256": sha256(path), "r0_selection_sha256": sha256(ROOT / f"configs/r0_dates_{name}_20261022.json")}
    before = args.baseline / "source/virtual_submeter"
    current = ROOT / "src/virtual_submeter"
    changed = [p.relative_to(current).as_posix() for p in current.rglob("*.py")
               if not (before / p.relative_to(current)).exists() or sha256(p) != sha256(before / p.relative_to(current))]
    if sorted(changed) not in ([], ["accuracy/fixed_queries.py", "adaptation/session.py"]):
        raise DataError("Unexpected source changes after baseline freeze")
    boundary = read_json(args.boundary)
    if boundary["status"] != "passed" or boundary["cases"] != 72 or not boundary["real_model_reload_score_parity"]:
        raise DataError("R1 real bounds/reload verification incomplete")
    tests = ROOT / "outputs/accuracy/r1-20261022-tests-final.log"
    log = tests.read_text(encoding="utf-8")
    if "Ran 138 tests" not in log or not log.rstrip().endswith("OK"):
        raise DataError("Expected complete R1 regression result missing")
    report = (args.fixed / "report.html").read_text(encoding="utf-8")
    if "House 17" not in report or "House 3" in report:
        raise DataError("Fixed comparison report household incorrect")
    write_json(args.output, {"status": "passed", "task": "R1", "formal_evaluation": False,
                            "baseline": baseline, "fixed_comparison": fixed, "boundary_checks": boundary,
                            "regression": {"tests": 138, "seconds": 107.098, "log_sha256": sha256(tests)},
                            "households_used": sorted(houses), "final_households_excluded": ["8", "9", "20"], "split_gap_seconds": gap,
                            "r0_date_bindings": date_bindings, "changed_modules_after_baseline_freeze": changed,
                            "source_training_modules_unchanged_after_freeze": True, "current_code_sha256": source_fingerprint(),
                            "restart_prefix_check": read_json(ROOT / "outputs/accuracy/r1-restarted-prefix-check-20261022.json"),
                            "handoff_results_sha256": sha256(args.handoff / "results.json"),
                            "baseline_evidence_sha256": sha256(args.baseline / "evidence.json"),
                            "fixed_freeze_sha256": sha256(args.fixed / "freeze.json"),
                            "script_sha256": {name: sha256(ROOT / "scripts" / name) for name in ("run_r1.py", "summarize_r1.py", "verify_r1_boundaries.py", "verify_r1.py")},
                            "independent_member_reproduction": False, "second_machine_reproduction": False,
                            "future_labels_reopened": False})
    print({"status": "passed", "output": str(args.output)})


if __name__ == "__main__":
    main()
