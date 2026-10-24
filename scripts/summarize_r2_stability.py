"""Collect sealed R2 repair results and paired historical controls, labels closed."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.dynamic_queries import verify
from virtual_submeter.accuracy.fixed_queries import mean_complete
from virtual_submeter.data.common import DataError, read_json, safe_child, sha256, source_fingerprint, write_json


def collect():
    run = ROOT / "outputs/accuracy/r2-stability-20261023-v1"
    old = ROOT / "outputs/accuracy/r2-20261023"
    evidence = ROOT / "outputs/r2-stability-start-20261023"
    reg = read_json(evidence / "registration.json")
    for key, path in (("plan_sha256", ROOT / "configs/r2_stability_20261023.json"),
                      ("declaration_sha256", ROOT / "docs/R2_STABILITY_PLAN_20261023.md"),
                      ("runner_sha256", ROOT / "scripts/run_r2_stability.py"),
                      ("original_r2_freeze_sha256", old / "freeze.json"),
                      ("original_r2_evidence_sha256", old / "evidence.json")):
        if reg[key] != sha256(path):
            raise DataError("Predeclared inputs or original R2 seals changed")
    checked, historical = verify(run), verify(old)
    if reg["code_sha256"] != source_fingerprint() or not checked["current_source_matches_execution"]:
        raise DataError("Repair execution source differs from the predeclared source")
    boundary, readback = read_json(evidence / "boundaries.json"), read_json(evidence / "readback.json")
    if (boundary["status"] != "passed" or boundary["checkpoints"] != 135
            or boundary["future_labels_opened_at_check"] is not False or readback["status"] != "passed"
            or any(r["code_sha256"] != reg["code_sha256"] for r in (boundary, readback))):
        raise DataError("Real boundary and ordinary-inference evidence is incomplete")
    summary, original = read_json(run / "summary.json"), read_json(old / "summary.json")
    metrics, prior = read_json(run / "metrics.json")["rows"], read_json(old / "metrics.json")["rows"]
    keys = ("source_seed", "query_seed", "household_id", "variant", "budget", "appliance")
    indexed = {tuple(r[k] for k in keys): r for r in prior}
    for row in metrics:
        before = indexed[tuple(row[k] for k in keys)]
        if any(row[k] != before[k] for k in ("baseline_f1", "valid_samples", "actual_on_samples")):
            raise DataError("Historical comparison changed its paired baseline or support")
        for mode in ("raw", "corrected"):
            if any(row[mode][k] != before[mode][k] for k in ("mae_w", "active_mae_w", "energy")):
                raise DataError("Repair changed power or energy relative to original R2")
    groups = []
    for group in summary["groups"]:
        before = next(g for g in original["groups"] if (g["variant"], g["budget"]) == (group["variant"], group["budget"]))
        groups.append({**group, "original_r2_f1": before["f1"],
                       "f1_vs_original_r2": group["f1"] - before["f1"] if group["f1"] is not None and before["f1"] is not None else None,
                       "original_r2_degenerate_cases": before["adaptation_degenerate_cases"]})
    main_devices = []
    for strategy in ("random", "uncertainty_static", "proposed"):
        for appliance in ("kettle", "microwave", "washing_machine"):
            rows = [r for r in metrics if (r["variant"], r["budget"], r["appliance"]) == (strategy, 10, appliance)]
            main_devices.append({"strategy": strategy, "appliance": appliance, "cases": len(rows),
                                 **{k: mean_complete([r[k] for r in rows]) for k in ("precision", "recall", "f1", "baseline_f1", "f1_delta", "f1_vs_random")},
                                 "missing_precision": sum(r["precision"] is None for r in rows),
                                 "new_degenerate_cases": sum(r["adaptation_caused_degeneracy"] for r in rows)})
    state, answers = read_json(run / "run.json"), []
    for strategy in ("random", "uncertainty_static", "proposed"):
        records = []
        for trajectory in sorted({c["trajectory"] for c in state["cases"] if c["variant"] == strategy}):
            records.extend(read_json(safe_child(run / "frozen", trajectory) / "session.json")["records"][:10])
        identity = lambda r: tuple(r[k] for k in ("household_id", "appliance_type", "window_start", "window_end"))
        answers.append({"strategy": strategy, "charged_answers_in_repeated_trajectories": len(records),
                        "responses": dict(Counter(r["response"] for r in records)),
                        "unique_appliance_windows": len({identity(r) for r in records}),
                        "unique_used_windows": len({identity(r) for r in records if r["response"] == "used"}),
                        "by_appliance": {a: dict(Counter(r["response"] for r in records if r["appliance_type"] == a))
                                         for a in ("kettle", "microwave", "washing_machine")}})
    degeneracy = [{k: r[k] for k in (*keys, "valid_samples", "actual_on_samples", "predicted_on_fraction", "f1")}
                  for r in metrics if r["adaptation_caused_degeneracy"]]
    return {"schema_version": 1, "completed_at": datetime.now(timezone.utc).isoformat(),
            "experiment_id": summary["experiment_id"], "run": str(run), "formal_evaluation": False,
            "development_household_already_inspected": "17", "reserved_final_households": ["8", "9", "20"],
            "verification": checked, "historical_verification": historical,
            "registration": reg, "registration_sha256": sha256(evidence / "registration.json"),
            "summary": summary, "groups_vs_original_r2": groups,
            "main_device_means": main_devices, "main_budget_answers": answers,
            "all_new_degenerate_cases": degeneracy, "boundaries": boundary, "normal_readback": readback,
            "policy_controls": read_json(evidence / "policy-controls.json"),
            "all_324_rows_same_baselines_support_power_and_energy_as_original": True,
            "target_pointwise_labels_parsed_by_collection": False,
            "run_artifacts_sha256": {name: sha256(run / name) for name in
                                     ("freeze.json", "evidence.json", "summary.json", "metrics.json", "comparison.csv", "query_distribution.json", "report.html")},
            "execution_evidence_sha256": {p.name: sha256(p) for p in sorted(evidence.glob("*.json"))}}


def main():
    output = ROOT / "docs/R2_STABILITY_RESULTS.json"
    if output.exists():
        raise DataError("Results exist; preserve the previous collection")
    original_open = Path.open

    def deny_labels(path, *values, **options):
        mode = values[0] if values else options.get("mode", "r")
        if any(area in path.parts for area in ("oracle_labels", "evaluation_labels")) and "r" in mode and "b" not in mode:
            raise DataError("Results collection may not parse target labels")
        return original_open(path, *values, **options)

    with patch.object(Path, "open", deny_labels):
        result = collect()
    write_json(output, result)
    print({"status": "collected", "output": str(output),
           "development_exit_passed": result["summary"]["development_exit_passed"],
           "new_degenerate_cases": len(result["all_new_degenerate_cases"])})


if __name__ == "__main__":
    main()
