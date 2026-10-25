"""Compare development W50 support and confusion counts with its original sealed report."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from virtual_submeter.data.common import DataError, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    args = p.parse_args()
    run = args.run.resolve()
    output = run/"original_W50_support_verification.json"
    if output.exists():
        raise DataError("Completed support verification cannot be overwritten")
    original = Path(read_json(run/"baseline_identity.json")["reference_run"])
    evidence = read_json(original/"evidence.json")
    for name in ("metrics.json",):
        if evidence["artifacts"][name] != sha256(original/name):
            raise DataError("Original W50 metric evidence changed")
    old = read_json(original/"metrics.json")["rows"]
    new = read_json(run/"evaluation/metrics.json")
    checks = []
    for seed in read_json(run/"plan.json")["seeds"]:
        a = next(r for r in old if r["candidate"] == "W50" and r["seed"] == seed and r["partition"] == "frozen_test_refit_17_0" and r["appliance"] == "microwave" and r["view"] == "raw")
        b = next(r for r in new["rows"] if r["arm"] == "W50" and r["seed"] == seed)
        for key, old_key in (("tp", "true_positive"), ("fp", "false_positive"), ("fn", "false_negative"), ("n", "valid_samples"), ("precision", "precision"), ("recall", "recall"), ("f1", "f1")):
            if b[key] != a[old_key]:
                raise DataError(f"Original W50 reproduction differs: {seed}/{key}")
        if new["support"][str(seed)]["mask_sha256"] != a["support_sha256"]:
            raise DataError("Effective evaluation timestamp mask differs from original W50")
        checks.append({"seed": seed, "valid_samples": b["n"], "exact_support_mask_and_metrics_match": True})
    result = {"status": "original_W50_reproduced_exactly", "checks": checks,
              "original_metrics_sha256": sha256(original/"metrics.json"),
              "original_evidence_sha256": sha256(original/"evidence.json"),
              "evaluation_sha256": sha256(run/"evaluation_complete.json"), "target_labels_reopened": False,
              "script_sha256": sha256(Path(__file__))}
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
