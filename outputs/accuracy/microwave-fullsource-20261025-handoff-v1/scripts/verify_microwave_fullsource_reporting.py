"""Read back the pre-evaluation report semantics and archived implementation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from virtual_submeter.data.common import DataError, read_json, sha256, write_json
from virtual_submeter.accuracy.microwave_fullsource_reporting import improvement_flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    dest = run/"reporting_readback.json"
    if dest.exists():
        raise DataError("Completed readback receipt cannot be overwritten")
    r = read_json(run/"reporting_registration.json")
    for key, path in (("training_registration_sha256", run/"registration.json"),
                      ("training_started_sha256", run/"training_started.json"),
                      ("reporting_module_sha256", run/"reporting_addendum_files/microwave_fullsource_reporting.py"),
                      ("reporting_tests_sha256", run/"reporting_addendum_files/test_microwave_fullsource_reporting.py"),
                      ("script_sha256", run/"reporting_addendum_files/summarize_microwave_fullsource.py")):
        if r[key] != sha256(path):
            raise DataError("Supplementary report registration mismatch: "+key)
    before = datetime.fromisoformat(r["registered_at_utc"]).timestamp() < (run/"evaluation/started.json").stat().st_mtime
    if not before or r["development_scores_opened"] is not False:
        raise DataError("Report semantics were not fixed before evaluation")
    engine, final = read_json(run/"evaluation/summary.json"), read_json(run/"final_summary.json")
    if final != {**engine, **improvement_flags(engine), "reporting_registration_sha256": sha256(run/"reporting_registration.json"),
                 "engine_summary_sha256": sha256(run/"evaluation/summary.json"), "freeze_sha256": sha256(run/"freeze.json")}:
        raise DataError("Final report changes more than the registered reporting semantics")
    value = {"status": "reporting_readback_verified", "registered_before_evaluation": True,
             "archived_module_tests_and_script_match": True, "training_snapshot_unchanged": True,
             "numeric_gates_and_source_selection_unchanged": True,
             "reporting_registration_sha256": sha256(run/"reporting_registration.json"),
             "final_summary_sha256": sha256(run/"final_summary.json"), "script_sha256": sha256(Path(__file__))}
    write_json(dest, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
