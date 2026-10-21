"""Controlled development probes; preserve the original diagnosed run unchanged."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from virtual_submeter.accuracy.diagnosis import state_summary, write_csv
from virtual_submeter.accuracy.runner import inventory, verify
from virtual_submeter.adaptation.state import INITIAL, apply_head, features, fit_head, scores
from virtual_submeter.data.common import DataError, read_json, sha256, source_fingerprint, write_json
from virtual_submeter.data.protocol import read_partition


HYPOTHESES = [
    {"id": "H1", "prediction": "Negative-only queried windows can drive global suppression. One negative window may reproduce all-off; one update or a stronger fixed prior may reduce collapse."},
    {"id": "H2", "prediction": "An appliance with no usable queries retains its original state and coefficients. Concentrated query budgets leave other devices unadapted."},
    {"id": "H3", "prediction": "Window max-pool loss can fall while future pointwise F1 falls. Sparse positive-window activity would show the weak/pointwise supervision mismatch; it alone cannot establish causality."},
    {"id": "H4", "prediction": "Activating INITIAL coefficients alone switches active appliances from corrected-power thresholds to raw-power thresholds. Its F1 difference isolates this convention change from learned updates."},
]


def variants(rows, records, appliances, thresholds, settings):
    head, history, _ = fit_head(rows, records, appliances, thresholds, settings)
    initial = copy.deepcopy(head)
    initial["coefficients"] = {a: INITIAL[:] for a in appliances}
    yield "replay", head, history
    yield "initial_head_only", initial, {}
    yield "one_epoch", *fit_head(rows, records, appliances, thresholds, {**settings, "epochs": 1})[:2]
    yield "stronger_prior_l2_1", *fit_head(rows, records, appliances, thresholds, {**settings, "l2": 1.0})[:2]
    for name, response in (("negative_only", "not_used"), ("positive_only", "used")):
        selected = [r for r in records if r["response"] == response]
        yield name, *fit_head(rows, selected, appliances, thresholds, settings)[:2]
    for appliance in appliances:
        negative = next((r for r in records if r["appliance_type"] == appliance and r["response"] == "not_used"), None)
        if negative:
            yield f"one_negative_{appliance}", *fit_head(rows, [negative], appliances, thresholds, settings)[:2]


def run_probes(run_dir, output):
    verified = verify(run_dir)
    if not verified["source_code_matches_current"]:
        raise DataError("Restore A0 source snapshot before running controlled probes")
    if output.exists():
        raise DataError("Probe output exists; choose a new directory")
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output / "probe_source.py")
    plan = {"schema_version": 1, "status": "plan_frozen_before_probes", "hypotheses": HYPOTHESES,
            "parent_diagnosis_sha256": sha256(run_dir / "diagnosis.json"),
            "parent_evidence_sha256": sha256(run_dir / "evidence.json"), "code_sha256": source_fingerprint(),
            "variant_scope": "Development mechanism probes only, not learner selection or confirmatory performance.",
            "fitting_labels": "Same saved budgeted weak answers only; no pointwise labels in fit_head."}
    write_json(output / "plan.json", plan)
    write_json(output / "run.json", {"status": "running", "plan_sha256": sha256(output / "plan.json")})
    diagnosis = read_json(run_dir / "diagnosis.json")
    manifest = run_dir / "development/protocol.json"
    protocol = read_json(manifest)
    rows_out, detail, labels_cache, base_cache = [], [], {}, {}
    try:
        for case in diagnosis["cases"]:
            if not case["budget"]:
                continue
            directory = run_dir / "comparison" / case["path"]
            metadata = read_json(directory / "model/model.json")
            records = read_json(directory / "model/queries.json")["records"]
            training_rows = read_json(directory / "session/prediction/predictions.json")["rows"]
            baseline = next(c for c in diagnosis["cases"] if c["budget"] == 0 and all(c[k] == case[k] for k in ("household_id", "seed", "strategy")))
            if baseline["path"] not in base_cache:
                base_cache[baseline["path"]] = read_json(run_dir / "comparison" / baseline["path"] / "prediction/predictions.json")
            base = base_cache[baseline["path"]]
            thresholds = metadata["settings"]["on_thresholds_w"]
            heads = list(variants(training_rows, records, metadata["appliances"], thresholds, metadata["adaptation_settings"]))
            # Commit every counterfactual head before evaluating that case.
            name = case["path"].replace("/", "_")
            write_json(output / "heads" / f"{name}.json", {"case": case["path"], "variants": {n: h for n, h, _ in heads}})
            if case["test_partition"] not in labels_cache:
                labels_cache[case["test_partition"]] = read_partition(manifest, case["test_partition"], "development_diagnosis", True)
            labels = labels_cache[case["test_partition"]]
            if heads[0][1]["coefficients"] != metadata.get("state_adapter", heads[0][1])["coefficients"]:
                raise DataError("Probe replay does not reproduce saved adapter parameters")
            for variant, head, _ in heads:
                predicted = copy.deepcopy(base["rows"])
                apply_head(predicted, {**metadata, "state_adapter": head})
                for appliance in metadata["appliances"]:
                    metric = state_summary(predicted, labels[appliance], appliance, thresholds[appliance], protocol["sample_interval_seconds"])
                    original = case["devices"][appliance]["adapted"]
                    if variant == "replay" and metric != original:
                        raise DataError("Probe replay changed the saved diagnostic metrics")
                    rows_out.append({"case": case["path"], "strategy": case["strategy"], "seed": case["seed"], "budget": case["budget"],
                                     "appliance": appliance, "variant": variant, "f1": metric["f1"],
                                     "precision": metric["precision"], "recall": metric["recall"],
                                     "predicted_on_fraction": metric["predicted_on_fraction"], "pattern": metric["state_pattern"],
                                     "predicted_state_sha256": hashlib.sha256(json.dumps([r["appliances"][appliance]["state"] for r in predicted]).encode()).hexdigest()})
            for appliance in metadata["appliances"]:
                x = features(training_rows, appliance, thresholds[appliance])
                bags = []
                for record in records:
                    if record["appliance_type"] != appliance or record["response"] not in ("used", "not_used"):
                        continue
                    indices = [i for i, r in enumerate(training_rows) if record["window_start"] <= r["timestamp"] < record["window_end"]]
                    bag = x[indices]
                    fitted = heads[0][1]["coefficients"][appliance]
                    label = float(record["response"] == "used")
                    losses = {}
                    for key, weight in (("initial", INITIAL), ("fitted", fitted)):
                        logit = float(np.max(bag @ weight))
                        losses[key] = float(np.logaddexp(0, logit) - label * logit)
                    bags.append({"query_id": record["query_id"], "response": record["response"], "bag_loss": losses,
                                 "initial_max_score": float(scores(bag).max()), "fitted_max_score": float(scores(bag, fitted).max())})
                detail.append({"case": case["path"], "appliance": appliance, "queries": case["queries"][appliance],
                               "coefficients": case["devices"][appliance]["coefficients"], "bags": bags,
                               "mean_initial_bag_loss": sum(b["bag_loss"]["initial"] for b in bags) / len(bags) if bags else None,
                               "mean_fitted_bag_loss": sum(b["bag_loss"]["fitted"] for b in bags) / len(bags) if bags else None,
                               "future_f1_delta": case["devices"][appliance]["f1_delta"]})
        write_csv(output / "probe_metrics.csv", rows_out, list(rows_out[0]))
        write_json(output / "diagnosis.json", {"status": "controlled_development_probes_completed", "hypotheses": HYPOTHESES,
                                               "cases": detail, "metrics": rows_out, "formal_evaluation": False,
                                               "notice": "Diagnostic interventions do not establish the best learner or generalisation to new households."})
        write_json(output / "evidence.json", {"artifacts_sha256": {k: v for k, v in inventory(output).items() if k != "run.json"}})
        write_json(output / "run.json", {"status": "completed", "plan_sha256": sha256(output / "plan.json"),
                                         "evidence_sha256": sha256(output / "evidence.json")})
        return {"status": "a0_probes_completed", "metric_rows": len(rows_out)}
    except Exception as error:
        write_json(output / "run.json", {"status": "failed", "error": str(error)})
        raise


def verify_probes(output):
    state = read_json(output / "run.json")
    if state.get("status") != "completed" or sha256(output / "evidence.json") != state["evidence_sha256"]:
        raise DataError("Probe evidence is incomplete or changed")
    if sha256(output / "plan.json") != state["plan_sha256"]:
        raise DataError("Probe plan changed")
    from virtual_submeter.data.common import safe_child
    for relative, digest in read_json(output / "evidence.json")["artifacts_sha256"].items():
        if sha256(safe_child(output, relative)) != digest:
            raise DataError("Probe artifact changed")
    return {"status": "a0_probe_artifacts_verified", "labels_reopened": False}


def single_negative(run_dir, case_path, appliance):
    verify(run_dir)
    directory = run_dir / "comparison" / case_path
    metadata = read_json(directory / "model/model.json")
    records = read_json(directory / "model/queries.json")["records"]
    record = next((r for r in records if r["appliance_type"] == appliance and r["response"] == "not_used"), None)
    if record is None:
        raise DataError("Case has no negative answer for the specified appliance")
    rows = read_json(directory / "session/prediction/predictions.json")["rows"]
    # A single observed query bag is sufficient; all other training rows removed.
    bag = [r for r in rows if record["window_start"] <= r["timestamp"] < record["window_end"]]
    baseline_dir = directory.parent / "budget_0/prediction/predictions.json"
    base = read_json(baseline_dir)
    labels = read_partition(run_dir / "development/protocol.json", base["partition_id"], "development_diagnosis", True)[appliance]
    thresholds = metadata["settings"]["on_thresholds_w"]
    result = {"case": case_path, "appliance": appliance, "weak_answers": 1, "training_samples": len(bag), "fitting_pointwise_labels": False}
    for name, settings in (("original", metadata["adaptation_settings"]), ("one_epoch_control", {**metadata["adaptation_settings"], "epochs": 1})):
        head, _, _ = fit_head(bag, [record], metadata["appliances"], thresholds, settings)
        predicted = copy.deepcopy(base["rows"])
        apply_head(predicted, {**metadata, "state_adapter": head})
        result[name] = state_summary(predicted, labels, appliance, thresholds[appliance], base["sample_interval_seconds"])
    result["status"] = "collapse_reproduced" if result["original"]["state_pattern"] == "all_off" and result["original"]["actual_on_samples"] > 0 else "no_all_off_collapse"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Verify a probe output directory without reopening labels")
    mode.add_argument("--check-single-negative", action="store_true", help="Return 1 if one saved negative answer reproduces all-off")
    parser.add_argument("--case", default="house_3/proposed/seed_42/budget_5")
    parser.add_argument("--appliance", default="washing_machine")
    args = parser.parse_args()
    try:
        if args.verify:
            result = verify_probes(args.run)
        elif args.check_single_negative:
            result = single_negative(args.run, args.case, args.appliance)
        else:
            if args.output is None:
                parser.error("--output is required when running probes")
            result = run_probes(args.run, args.output)
    except (DataError, OSError, KeyError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "a0_probes_failed", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "collapse_reproduced" else 0


if __name__ == "__main__":
    raise SystemExit(main())
