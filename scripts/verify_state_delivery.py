"""Independent delivery checks using sealed artifacts, without reopening target labels."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.accuracy.state_experiment import verify, SEEDS, GROUPS
from virtual_submeter.data.common import DataError, read_json, write_json, sha256, safe_child


def require(condition, message):
    if not condition:
        raise DataError(message)


def cli_result(script, *arguments):
    result = subprocess.run([sys.executable, str(script), *map(str, arguments)], cwd=ROOT,
                            capture_output=True, encoding="utf-8")
    require(result.returncode == 0, "Snapshot check failed: " + result.stderr)
    return json.loads(result.stdout)


def main():
    run = ROOT / "outputs/accuracy/state-20261024-v2"
    parent = ROOT / "outputs/accuracy/precision-20261024-v1"
    current = verify(run)
    snapshot = cli_result(ROOT / "scripts/run_state_snapshot.py", "verify", "--run", run)
    require(snapshot["execution_source_matches_current"], "Saved execution source differs from registration")
    historical = cli_result(parent / "entrypoints/run_precision.py", "verify")
    rows = read_json(run / "metrics.json")["rows"]
    old_rows = read_json(ROOT / "outputs/accuracy/data-expansion-20261024-v1/metrics_refit.json")["metrics"]
    comparisons = 0
    for old in old_rows:
        if old["group"] != "A" or old["view"] != "raw":
            continue
        new = next(r for r in rows if r["candidate"] == "W50" and r["support"] == "native"
                   and all(r[k] == old[k] for k in ("seed", "partition", "appliance")))
        for key in old:
            if key in ("group", "seed", "partition", "appliance", "view"):
                continue
            require(new[key] == old[key], "Historical A metric changed: " + key)
        comparisons += 1
    require(comparisons == 18, "Incomplete historical metric readback")
    precision_rows = read_json(parent / "metrics.json")["rows"]
    precision_power_differences = {k: 0. for k in ("mae_w", "active_mae_w", "energy_absolute_error_kwh")}
    precision_state_comparisons = 0
    for old in precision_rows:
        if old["candidate"] != "W50" or old["view"] != "raw":
            continue
        new = next(r for r in rows if r["candidate"] == "W50" and r["support"] == "native"
                   and all(r[k] == old[k] for k in ("seed", "partition", "appliance")))
        for key in old:
            if key in ("candidate", "seed", "partition", "appliance", "view"):
                continue
            if key in precision_power_differences:
                delta = abs(new[key] - old[key])
                precision_power_differences[key] = max(precision_power_differences[key], delta)
                require(delta <= (1e-6 if key == "energy_absolute_error_kwh" else 1e-4),
                        "Previous W50 replay has material power metric differences")
            else:
                require(new[key] == old[key], "Previous W50 state metric changed")
        precision_state_comparisons += 1
    recovery = []
    for group, seed in [("R_PR", s) for s in SEEDS] + [("C31", 42)]:
        previous = torch.load(ROOT / f"outputs/accuracy/state-20261024-v1/models/{group}/seed_{seed}/weights.pt", weights_only=True)
        actual = torch.load(run / f"models/{group}/seed_{seed}/weights.pt", weights_only=True)
        require(previous.keys() == actual.keys() and all(torch.equal(v, actual[k]) for k, v in previous.items()),
                "Engineering rerun changed learned parameters")
        recovery.append({"candidate": group, "seed": seed, "parameters_equal": True, "maximum_difference": 0})
    unchanged = 0
    for group in GROUPS:
        for seed in SEEDS:
            for partition in ("source_validation_refit_5_0", "frozen_test_refit_17_0"):
                with np.load(run / f"predictions/{group}/seed_{seed}/{partition}.npz", allow_pickle=False) as a, np.load(
                        run / f"reference/predictions/seed_{seed}/{partition}.npz", allow_pickle=False) as b:
                    columns = [0, 2] if group == "R_PR" else [0, 1, 2]
                    require(np.array_equal(a["power"][:, columns], b["raw"][:, columns], equal_nan=True),
                            "Independent original power output changed")
                    unchanged += 1
    readback_path = ROOT / "tmp/state-full-readback-20261024.json"
    readback = read_json(readback_path)
    require(len(readback["checks"]) == 18 and readback["all_passed"], "Incomplete public inference readback")
    require(readback["freeze_sha256"] == sha256(run / "freeze.json"), "Readback belongs to another freeze")
    require(all(c["rows"] == 151200 and c["states_equal"] and c["missing_equal"] for c in readback["checks"]),
            "Incomplete source-validation public inference")
    repair = read_json(run / "execution_repair/receipt.json")
    require(repair["freeze_sha256"] == sha256(run / "freeze.json")
            and repair["wrapper_sha256"] == sha256(run / "execution_repair/run_state_snapshot.py")
            and repair["wrapper_sha256"] == sha256(ROOT / "scripts/run_state_snapshot.py"), "Repair wrapper evidence changed")
    handoff = ROOT / "outputs/accuracy/state-20261024-handoff-v2"
    for name, digest in read_json(handoff / "evidence.json")["artifacts"].items():
        require(sha256(safe_child(handoff, name)) == digest, "Derived PR report changed")
    svgs = list(handoff.glob("*.svg"))
    require(len(svgs) == 18, "Incomplete PR chart matrix")
    for svg in svgs:
        ET.parse(svg)
    log = ROOT / "tmp/state-regression-delivery-20261024.log"
    log_text = log.read_text(encoding="utf-8", errors="replace")
    require("Ran 183 tests" in log_text and log_text.rstrip().endswith("OK"), "Full regression did not pass")
    result = {
        "status": "state_delivery_verified", "current_source_verification": current,
        "snapshot_verification_with_pr_cache": snapshot, "historical_precision_verification": historical,
        "full_regression": {"passed": 183, "failed": 0, "log_sha256": sha256(log)},
        "focused_tests": {"passed": 9, "log_sha256": sha256(ROOT / "tmp/state-focused-final-20261024.log")},
        "checkpoints_and_exact_pr_curves_verified": 216, "models": 18, "frozen_predictions": 36,
        "device_seed_partition_support_metric_rows": len(rows),
        "historical_A_native_metrics_exactly_reproduced": comparisons,
        "previous_precision_W50_state_metric_rows_exactly_reproduced": precision_state_comparisons,
        "previous_precision_W50_maximum_power_metric_differences": precision_power_differences,
        "engineering_recovery_parameter_checks": recovery, "cumulative_fits_including_failed_run": 22,
        "independent_power_prediction_checks": unchanged,
        "full_source_validation_public_readback": readback,
        "full_readback_evidence_sha256": sha256(readback_path), "execution_repair": repair,
        "execution_repair_receipt_sha256": sha256(run / "execution_repair/receipt.json"),
        "source_pr_charts_verified": len(svgs), "source_pr_report_evidence_sha256": sha256(handoff / "evidence.json"),
        "freeze_sha256": sha256(run / "freeze.json"), "evaluation_evidence_sha256": sha256(run / "evidence.json"),
        "handoff_sha256": sha256(ROOT / "docs/STATE_HANDOFF_20261024.md"),
        "results_sha256": sha256(ROOT / "docs/STATE_RESULTS_20261024.json"),
        "target_labels_reopened_by_delivery_verification": False, "final_households_accessed": False,
        "default_model_changed": False, "independent_member_or_second_machine_reproduction": False,
        "new_development_household_check": "not_completed_no_additional_designated_household",
    }
    destination = ROOT / "docs/STATE_VALIDATION_20261024.json"
    if destination.exists():
        require(read_json(destination) == result, "Existing delivery record differs")
    else:
        write_json(destination, result)
    print(json.dumps({"status": result["status"], "tests": 183, "models": 18, "predictions": 36,
                      "historical_A_metric_rows": comparisons, "full_readback_models": 18}, ensure_ascii=False))


if __name__ == "__main__":
    main()
