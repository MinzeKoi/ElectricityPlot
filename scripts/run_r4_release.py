"""Build R4/R5 evidence from a previously frozen House 17 trajectory."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy import live, release
from virtual_submeter.data.common import DataError, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/accuracy/r2-stability-20261023-v1")
    args = parser.parse_args()
    output, run = args.output.resolve(), args.run.resolve()
    if output.exists():
        raise DataError("Use a fresh R4 evidence directory")
    frozen = run / "frozen"
    trajectory = frozen / "trajectories/seed_42/house_17/query_2026/proposed"
    checkpoint = trajectory / "models/budget_10"
    barrier = read_json(run / "freeze.json")
    # Verify the exact retained inputs/answers without parsing any submeter CSV.
    paths = [frozen / "query_plan.json", frozen / "inputs/plan.json", frozen / "inputs/legacy_settings.json",
             frozen / "inputs/protocol/protocol.json", checkpoint / "queries.json", checkpoint / "model.json"]
    paths += [p for p in (frozen / "inputs/bases/seed_42").rglob("*") if p.is_file()]
    for path in paths:
        relative = path.relative_to(frozen).as_posix()
        if sha256(path) != barrier["artifacts_sha256"][relative]:
            raise DataError("Historical input differs: " + relative)
    records = read_json(checkpoint / "queries.json")["records"]
    original_open = Path.open
    def guard(path, *a, **kw):
        if any(p in path.parts for p in ("oracle_labels", "evaluation_labels")):
            raise AssertionError("R4/R5 attempted target submeter access")
        return original_open(path, *a, **kw)
    with patch.object(Path, "open", guard):
        directory = output / "live"
        live.create(directory, frozen / "inputs/bases/seed_42", frozen / "inputs/protocol/protocol.json",
                    "target_adaptation_house_17", frozen / "inputs/plan.json", frozen / "query_plan.json",
                    frozen / "inputs/legacy_settings.json", "proposed", 2026, 10)
        for index, record in enumerate(records):
            pending = live.status(directory)["pending"]
            if pending is None or any(pending[k] != record[k] for k in pending):
                raise DataError("Live question/model differs from frozen experiment")
            if live.answer(directory, pending["query_id"], record["response"], record["label_source"]) != record:
                raise DataError("Live answer differs from retained experiment")
            print(f"Replayed confirmation {index + 1}/10", flush=True)
        current = live.status(directory)
        actual, expected = read_json(Path(current["model"]) / "model.json"), read_json(checkpoint / "model.json")
        if actual["accuracy_state_head"] != expected["accuracy_state_head"] or actual["model_id"] != expected["model_id"]:
            raise DataError("Live adaptation differs from frozen R2 model")
        write_json(output / "experiment_parity.json", {
            "status": "same_questions_records_adapter_and_model_id", "source": str(run),
            "historical_freeze_sha256": sha256(run / "freeze.json"),
            "source_seed": 42, "query_seed": 2026, "strategy": "proposed", "budget": 10,
            "learner": "L1_stable", "model_id": actual["model_id"], "formal_method_approved": False,
            "target_pointwise_labels_read": False, "r3": "skipped_by_user"})
        result = release.build(directory, output / "bundle", ROOT, historical_run=run)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
