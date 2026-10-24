"""Run the one-candidate R2 repair with predeclared inputs and label guards."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy import dynamic_queries as engine
from virtual_submeter.accuracy.r2 import preflight
from virtual_submeter.data.common import DataError, read_json, sha256, source_fingerprint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "resume", "evaluate", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/accuracy/r2-stability-20261023-v1")
    parser.add_argument("--evidence", type=Path, default=ROOT / "outputs/r2-stability-start-20261023")
    args = parser.parse_args()
    parent = ROOT / "outputs/accuracy/r1-fixed-20261022-v2"
    plan = ROOT / "configs/r2_stability_20261023.json"
    declaration = ROOT / "docs/R2_STABILITY_PLAN_20261023.md"
    registration = args.evidence / "registration.json"
    if args.evidence.resolve().is_relative_to(args.output.resolve()):
        raise DataError("Execution evidence must be outside the sealed experiment")
    identity = {"plan_sha256": sha256(plan), "declaration_sha256": sha256(declaration),
                "code_sha256": source_fingerprint(), "runner_sha256": sha256(Path(__file__)),
                "parent_evidence_sha256": sha256(parent / "evidence.json"),
                "original_r2_freeze_sha256": sha256(ROOT / "outputs/accuracy/r2-20261023/freeze.json"),
                "original_r2_evidence_sha256": sha256(ROOT / "outputs/accuracy/r2-20261023/evidence.json"),
                "output": str(args.output.resolve())}
    original_open = Path.open

    def guarded_open(path, *values, **options):
        mode = values[0] if values else options.get("mode", "r")
        forbidden = ("evaluation_labels",) if args.phase in ("freeze", "resume") else ("oracle_labels", "evaluation_labels")
        if any(name in path.parts for name in forbidden) and "r" in mode and "b" not in mode:
            raise DataError("This R2 phase may not parse target pointwise labels")
        return original_open(path, *values, **options)

    args.evidence.mkdir(parents=True, exist_ok=True)
    receipt = args.evidence / f"{args.phase}-receipt.json"
    if receipt.exists():
        raise DataError("Phase receipt already exists")
    if args.phase == "freeze":
        if registration.exists() or args.output.exists():
            raise DataError("Choose fresh output and evidence paths")
        with patch.object(Path, "open", guarded_open):
            checked = preflight(parent, plan)
        if checked["status"] != "r2_ready":
            raise DataError("R2 repair inputs are not ready")
        write_json(registration, {"registered_at": datetime.now(timezone.utc).isoformat(), **identity,
                                 "preflight": checked, "future_labels_parsed": False,
                                 "candidate_count": 1, "formal_evaluation": False})
    else:
        saved = read_json(registration)
        if any(saved.get(k) != v for k, v in identity.items()):
            raise DataError("R2 repair source, declaration, inputs or original evidence changed")
    if args.phase in ("freeze", "resume"):
        with patch.object(Path, "open", guarded_open):
            result = engine.freeze(parent, plan, args.output, resume=args.phase == "resume")
    elif args.phase == "evaluate":
        # Only engine.evaluate may open future labels, after verify_frozen has
        # checked the complete matrix. Never use this phase for model choice.
        result = engine.evaluate(args.output)
    else:
        with patch.object(Path, "open", guarded_open):
            result = engine.verify(args.output)
    result.update(registration_sha256=sha256(registration), phase=args.phase,
                  completed_at=datetime.now(timezone.utc).isoformat())
    write_json(receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
