"""Seal report-only semantics before evaluation; deliver final report from frozen metrics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from virtual_submeter.data.common import DataError, read_json, sha256, write_json
from virtual_submeter.accuracy.microwave_fullsource_reporting import improvement_flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["register", "report"])
    p.add_argument("--run", required=True, type=Path)
    args = p.parse_args()
    run = args.run
    if args.command == "register":
        if (run/"evaluation").exists() or (run/"reporting_registration.json").exists():
            raise DataError("Reporting semantics must be sealed once before target evaluation")
        write_json(run/"reporting_registration.json", {
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_thread_id": "01a16d7b-478e-7780-a999-88abe71e8c94",
            "coordinator_message": "合格source预选者包括natural，相对W50通过原开发全部门槛即完整方案改善；另设sampling_or_background_improvement_supported=passed且selected!=natural，不改训练、选择、数值门槛或读取顺序。",
            "timing": "request_received_after_training_started_before_any_development_evaluation",
            "training_registration_sha256": sha256(run/"registration.json"),
            "training_started_sha256": sha256(run/"training_started.json"),
            "reporting_module_sha256": sha256(ROOT/"src/virtual_submeter/accuracy/microwave_fullsource_reporting.py"),
            "reporting_tests_sha256": sha256(ROOT/"tests/test_microwave_fullsource_reporting.py"),
            "script_sha256": sha256(Path(__file__)),
            "declared_improvement_definition": "qualified_source_selected_including_natural_passes_all_original_W50_development_gates",
            "sampling_or_background_improvement_supported_definition": "development_acceptance_passed_and_selected_not_natural",
            "development_scores_opened": False})
        print(json.dumps({"status": "report_semantics_sealed_before_evaluation"}))
        return
    r = read_json(run/"reporting_registration.json")
    if r["training_registration_sha256"] != sha256(run/"registration.json") or r["reporting_module_sha256"] != sha256(ROOT/"src/virtual_submeter/accuracy/microwave_fullsource_reporting.py"):
        raise DataError("Report registration or semantics changed")
    summary = read_json(run/"evaluation/summary.json")
    final = {**summary, **improvement_flags(summary), "reporting_registration_sha256": sha256(run/"reporting_registration.json"),
             "engine_summary_sha256": sha256(run/"evaluation/summary.json"), "freeze_sha256": sha256(run/"freeze.json")}
    path = run/"final_summary.json"
    if path.exists() and read_json(path) != final:
        raise DataError("Completed final summary is immutable")
    if not path.exists():
        write_json(path, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
