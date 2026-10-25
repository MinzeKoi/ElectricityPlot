"""Run an immutable state snapshot with an audited PR decompression cache repair."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "evaluate"])
    parser.add_argument("--run", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/accuracy/state-20261024-v2")
    args = parser.parse_args()
    run = args.run.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(run / "source"))
    from virtual_submeter.accuracy import state_experiment
    from virtual_submeter.data.common import read_json, write_json, sha256, DataError
    # Cached array values are identical. Selection rule, ordering and labels do not change.
    original = state_experiment.constrained_point
    def cached_curve(curve, minimum_recall=.85):
        return original({key: curve[key] for key in curve}, minimum_recall)
    state_experiment.constrained_point = cached_curve
    script = Path(__file__).resolve()
    repair = run / "execution_repair"
    if args.command == "evaluate":
        repair.mkdir(exist_ok=True)
        copied = repair / "run_state_snapshot.py"
        if copied.exists() and sha256(copied) != sha256(script):
            raise DataError("Snapshot repair code changed")
        if not copied.exists():
            shutil.copyfile(script, copied)
        receipt = {"freeze_sha256": sha256(run / "freeze.json"), "wrapper_sha256": sha256(script),
                   "reason": "Materialize lazy NPZ arrays once before exact PR ranking; no score, threshold, model or selection changes",
                   "snapshot_code_sha256": read_json(run / "registration.json")["code_sha256"],
                   "development_labels_parsed_before_repair": False}
        path = repair / "receipt.json"
        if path.exists() and read_json(path) != receipt:
            raise DataError("Snapshot repair receipt changed")
        if not path.exists():
            write_json(path, receipt)
    result = getattr(state_experiment, args.command)(run)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
