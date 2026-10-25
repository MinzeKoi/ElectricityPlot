"""Publish an immutable local full-source handoff after all verification receipts pass."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from virtual_submeter.data.common import DataError, inventory, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    handoff = ROOT/"outputs/accuracy/microwave-fullsource-20261025-handoff-v1"
    docs_report = ROOT/"docs/MICROWAVE_FULLSOURCE_REPORT_20261025.md"
    docs_results = ROOT/"docs/MICROWAVE_FULLSOURCE_RESULTS_20261025.json"
    docs_validation = ROOT/"docs/MICROWAVE_FULLSOURCE_EXECUTION_VALIDATION_20261025.json"
    if any(p.exists() for p in (handoff, docs_report, docs_results, docs_validation, run/"delivery_manifest.json")):
        raise DataError("Completed handoff paths must not be overwritten")
    receipts = {name: read_json(run/name) for name in (
        "independent_verification.json", "verification_after_evaluation.json",
        "original_W50_support_verification.json", "reporting_readback.json")}
    expected = ["independently_verified", "fullsource_evaluation_verified", "original_W50_reproduced_exactly", "reporting_readback_verified"]
    if [r["status"] for r in receipts.values()] != expected:
        raise DataError("All required verification stages must pass")
    if read_json(run/"training_complete.json")["fits"] != 9 or len(read_json(run/"freeze.json")["predictions"]) != 9:
        raise DataError("Final nine-model/prediction matrix differs")
    log = ROOT/"tmp/microwave-fullsource-final-tests-20261025.log"
    if "Ran 22 tests" not in log.read_text(encoding="utf-8-sig") or "OK" not in log.read_text(encoding="utf-8-sig"):
        raise DataError("Final twenty-two-test evidence missing")
    validation = {"status": "fullsource_completed_verified_development_target_not_met",
        "run": str(run), "actual_new_real_fits": 9, "additional_real_fits_after_resume": 0,
        "predictions_frozen_before_target_evaluation": 9, "tests_passed": 22,
        "source_selection": "event", "development_acceptance_passed": False,
        "verification": {name: r["status"] for name, r in receipts.items()},
        "original_W50_support_samples_per_seed": 137523,
        "truth_csv_parses": read_json(run/"evaluation_complete.json")["truth_csv_parses"],
        "target_truth_csv_reopened_by_verification": False, "final_houses_accessed": False,
        "default_model_changed": False, "other_device_raw_power_and_state_unchanged": True,
        "original_preparation_validation_preserved": "docs/MICROWAVE_FULLSOURCE_VALIDATION_20261025.json",
        "execution_validation_path": str(docs_validation),
        "artifacts": {name: sha256(run/name) for name in receipts}, "tests_log_sha256": sha256(log)}
    result = read_json(run/"final_results.json")
    if result["development_acceptance_passed"] is not False or result["selected_before_development"] != "event":
        raise DataError("Handoff decision does not match frozen outcome")
    handoff.mkdir(parents=True)
    report = (run/"final_report.md").read_text(encoding="utf-8")
    report += "\n## 完成交付\n\n22项测试、评价后完整复算、原W50掩码/指标逐项核对和报告语义读回全部通过。恢复后没有新增真实拟合，未覆盖已有模型、预测、评价或训练快照。\n\n正式执行验证见MICROWAVE_FULLSOURCE_EXECUTION_VALIDATION_20261025.json；旧MICROWAVE_FULLSOURCE_VALIDATION_20261025.json保留准备阶段记录。\n"
    for p in (docs_report, handoff/"report.md"):
        p.write_text(report, encoding="utf-8")
    for p in (docs_results, handoff/"results.json"):
        write_json(p, result)
    for p in (docs_validation, handoff/"validation.json"):
        write_json(p, validation)
    (handoff/"scripts").mkdir()
    scripts = ["verify_microwave_fullsource.py", "verify_microwave_fullsource_support.py", "verify_microwave_fullsource_reporting.py",
               "summarize_microwave_fullsource.py", "report_microwave_fullsource.py", "finish_microwave_fullsource_delivery.py"]
    for name in scripts:
        shutil.copyfile(ROOT/"scripts"/name, handoff/"scripts"/name)
    shutil.copyfile(log, handoff/"final_tests.log")
    manifest = {"status": validation["status"], "run": str(run),
                "artifacts": inventory(run), "handoff": str(handoff), "handoff_artifacts": inventory(handoff),
                "documents": {str(p.relative_to(ROOT)): sha256(p) for p in (docs_report, docs_results, docs_validation)},
                "exclude_manifest_self": True, "actual_new_real_fits": 9, "tests_passed": 22}
    write_json(run/"delivery_manifest.json", manifest)
    write_json(handoff/"delivery_manifest.json", manifest)
    print(json.dumps({"status": validation["status"], "report": str(docs_report), "handoff": str(handoff),
                      "manifest_sha256": sha256(run/"delivery_manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
