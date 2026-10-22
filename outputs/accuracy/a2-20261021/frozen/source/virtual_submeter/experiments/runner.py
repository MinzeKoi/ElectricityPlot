"""Freeze the entire comparison before exposing any target test answers."""

from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

from ..data.common import DataError, read_json, safe_child, sha256, source_fingerprint, utc, write_json

GROUPS = {"random": "B2", "uncertainty": "B3", "proposed": "P1"}


def inventory(root):
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def check_hashes(root, hashes):
    if not hashes:
        raise DataError("Empty evidence inventory")
    for name, expected in hashes.items():
        if sha256(safe_child(root, name)) != expected:
            raise DataError(f"Experiment artifact changed: {name}")


def validate_plan(plan, protocol):
    fields = {"schema_version", "mode", "backbone", "budgets", "seeds", "background_stress",
              "test_history", "test_history_note"}
    if set(plan) != fields or plan["schema_version"] != 1:
        raise DataError("Invalid M5 experiment schema")
    if plan["mode"] not in ("pilot", "final") or plan["backbone"] not in ("constant_mean", "small_seq2point"):
        raise DataError("Invalid experiment mode or backbone")
    if type(plan["background_stress"]) is not bool:
        raise DataError("background_stress must be Boolean")
    if plan["test_history"] not in ("previously_evaluated", "never_evaluated"):
        raise DataError("Declare whether test results have previously been inspected")
    if not isinstance(plan["test_history_note"], str) or not plan["test_history_note"].strip():
        raise DataError("Test history requires an explicit explanation")
    for key, allowed in (("budgets", protocol["query_budgets"]), ("seeds", protocol["random_seeds"])):
        values = plan[key]
        if (not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values)
                or len(set(values)) != len(values) or any(v not in allowed for v in values)):
            raise DataError(f"{key} must be unique frozen protocol choices")
    if 0 not in plan["budgets"] or not any(v > 0 for v in plan["budgets"]):
        raise DataError("Include zero and positive confirmation budgets")
    partitions = protocol["partitions"]
    if len({p["partition_id"] for p in partitions}) != len(partitions):
        raise DataError("Duplicate partition IDs")
    roles = {role: [p for p in partitions if p["role"] == role]
             for role in ("source_train", "source_validation", "target_adaptation", "frozen_test")}
    houses = {r: {p["household_id"] for p in ps} for r, ps in roles.items()}
    if any(not ps for ps in roles.values()):
        raise DataError("All four protocol roles are required")
    if (houses["source_train"] & houses["source_validation"]
            or (houses["source_train"] | houses["source_validation"]) & (houses["target_adaptation"] | houses["frozen_test"])
            or houses["target_adaptation"] != houses["frozen_test"]):
        raise DataError("Source and target household isolation is invalid")
    for house in houses["target_adaptation"]:
        adaptation = [p for p in roles["target_adaptation"] if p["household_id"] == house]
        tests = [p for p in roles["frozen_test"] if p["household_id"] == house]
        if len(adaptation) != 1 or len(tests) != 1:
            raise DataError("Each target needs exactly one adaptation and test partition")
        if utc(tests[0]["start"]) - utc(adaptation[0]["end"]) < max(
                protocol["split_gap_seconds"], protocol["max_model_context_seconds"], protocol["query_window_seconds"]):
            raise DataError("Target isolation gap is too short")
    if plan["mode"] == "final":
        if (protocol["purpose"] != "final_evaluation"
                or protocol.get("data_kind") == "synthetic_contract_fixture"
                or plan["test_history"] != "never_evaluated"
                or len(houses["target_adaptation"]) < 3 or len(plan["seeds"]) < 3
                or plan["budgets"] != protocol["query_budgets"] or not plan["background_stress"]
                or plan["backbone"] != "small_seq2point"):
            raise DataError("Final M5 requires a final protocol, unseen tests, >=3 target homes/seeds, all budgets and CNN/stress")
    return roles


def freeze_experiment(manifest, experiment, training, adaptation, trust, output):
    """Persist settings, fit every seed/strategy/budget, then freeze predictions."""
    from ..adaptation.runner import run_comparison
    from ..adaptation.verification import verify as verify_adaptation
    from ..models.training import train
    from ..models.prediction import predict
    from ..models.settings import load_settings as training_settings
    from ..adaptation.selection import load_settings as adaptation_settings
    from ..trust.calibration import calibrate, load_settings as trust_settings
    from ..environment import environment_report

    if output.exists():
        raise DataError("Experiment destination exists; use a new run ID")
    protocol, plan = read_json(manifest), read_json(experiment)
    roles = validate_plan(plan, protocol)
    settings = training_settings(training)
    adaptation_settings(adaptation)
    trust_settings(trust)
    started = time.perf_counter()
    output.mkdir(parents=True)
    record = {"schema_version": 1, "status": "freezing", "created_at": datetime.now(timezone.utc).isoformat(),
              "code_sha256": source_fingerprint(), "manifest_path": str(manifest.resolve()),
              "test_labels_opened": False}
    write_json(output / "run.json", record)
    try:
        frozen = output / "frozen"
        frozen.mkdir()
        for source, name in ((manifest, "protocol"), (experiment, "experiment"), (training, "training"),
                             (adaptation, "adaptation"), (trust, "trust")):
            shutil.copyfile(source, frozen / f"{name}.json")
        package = Path(__file__).resolve().parents[1]
        for source in package.rglob("*.py"):
            target = frozen / "source/virtual_submeter" / source.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        project = package.parents[1]
        for source in [project / "pyproject.toml", *project.glob("requirements*.lock")]:
            shutil.copyfile(source, frozen / source.name)
        write_json(frozen / "environment.json", environment_report())
        cases, comparisons = [], []
        for seed in plan["seeds"]:
            print(f"M5 freeze: source training and all adaptation cases, seed={seed}", file=sys.stderr, flush=True)
            settings["seed"] = seed
            seed_root = frozen / f"seed_{seed}"
            write_json(seed_root / "training.json", settings)
            train(manifest, seed_root / "training.json", seed_root / "B0", "zero")
            train(manifest, seed_root / "training.json", seed_root / "B1", plan["backbone"])
            calibrate(seed_root / "B1", manifest, frozen / "trust.json", seed_root / "calibration")
            comparison = seed_root / "adapted"
            result = run_comparison(seed_root / "B1", manifest, frozen / "adaptation.json",
                                    comparison, plan["budgets"], [seed], evaluate_test=False)
            verify_adaptation(comparison)
            comparisons.append(comparison.relative_to(output).as_posix())
            for target in roles["frozen_test"]:
                for group in ("B0", "B1"):
                    cases.append({"id": f"{group}_house_{target['household_id']}_seed_{seed}",
                                  "group": group, "household_id": target["household_id"], "seed": seed,
                                  "budget": 0, "test_partition": target["partition_id"],
                                  "model": (seed_root / group).relative_to(output).as_posix()})
            for case in result["cases"]:
                cases.append({"id": f"{GROUPS[case['strategy']]}_house_{case['household_id']}_seed_{seed}_budget_{case['budget']}",
                              "group": GROUPS[case["strategy"]], "household_id": case["household_id"],
                              "seed": seed, "budget": case["budget"], "test_partition": case["test_partition"],
                              "model": (comparison / case["path"] / "model").relative_to(output).as_posix()})
        # All fitting has ended. No target test labels have been read.
        for case in cases:
            print(f"M5 freeze: test aggregate prediction {case['id']}", file=sys.stderr, flush=True)
            prediction = frozen / "predictions" / case["id"]
            calibration = frozen / f"seed_{case['seed']}/calibration" if case["group"] == "P1" else None
            predict(output / case["model"], prediction, manifest, case["test_partition"], calibration_dir=calibration)
            case["prediction"] = prediction.relative_to(output).as_posix()
        write_json(frozen / "design.json", {
            "cases": cases,
            "group_output": {"B0": "raw", "B1": "raw", "B2": "raw", "B3": "raw", "P1": "raw", "P2": "corrected"},
            "P2": "Paired view of the exact P1 prediction and queries, with source intervals and scale-down correction. No extra labels.",
            "primary_metrics": ["mae_w", "active_mae_w", "f1", "energy.absolute_error_kwh"],
            "aggregation": "Mean over seeds within each household, then equal-weight mean over households. Nulls excluded with counts.",
            "comparisons": comparisons,
            "test_history_is_operator_declaration": True,
        })
        if source_fingerprint() != record["code_sha256"] or sha256(manifest) != sha256(frozen / "protocol.json"):
            raise DataError("Source or protocol changed while freezing")
        write_json(output / "freeze.json", {"schema_version": 1, "status": "all_models_and_predictions_frozen",
                   "created_at": datetime.now(timezone.utc).isoformat(), "code_sha256": record["code_sha256"],
                   "artifacts_sha256": inventory(frozen), "test_labels_opened": False})
        record.update(status="frozen", freeze_sha256=sha256(output / "freeze.json"),
                      freeze_seconds=time.perf_counter() - started)
        write_json(output / "run.json", record)
        return {"status": "frozen", "cases": len(cases), "mode": plan["mode"]}
    except Exception as error:
        record.update(status="failed_freeze", error=str(error))
        write_json(output / "run.json", record)
        raise


def check_frozen(output):
    record = read_json(output / "run.json")
    freeze = read_json(output / "freeze.json")
    if (record.get("freeze_sha256") != sha256(output / "freeze.json")
            or freeze.get("status") != "all_models_and_predictions_frozen"
            or freeze.get("test_labels_opened") is not False):
        raise DataError("Invalid experiment freeze")
    check_hashes(output / "frozen", freeze["artifacts_sha256"])
    if inventory(output / "frozen") != freeze["artifacts_sha256"]:
        raise DataError("Frozen experiment file inventory changed")
    design = read_json(output / "frozen/design.json")
    return record, freeze, design


def evaluate_experiment(output, manifest=None):
    # Exclusive creation prevents two processes from evaluating the same run.
    from ..adaptation.session import locked
    with locked(output):
        return _evaluate_experiment(output, manifest)


def _evaluate_experiment(output, manifest=None):
    from ..evaluation.runner import evaluate
    from ..evaluation.stress import stress_test
    from .summary import build_summary, export_summary

    record, freeze, design = check_frozen(output)
    if record["status"] != "frozen":
        raise DataError("Only an unevaluated frozen run can be evaluated; failed/evaluated runs cannot be reused")
    if freeze["code_sha256"] != source_fingerprint():
        raise DataError("Code changed after freeze; restore the saved source before evaluation")
    manifest = manifest or Path(record["manifest_path"])
    if sha256(manifest) != sha256(output / "frozen/protocol.json"):
        raise DataError("Evaluation protocol changed")
    plan = read_json(output / "frozen/experiment.json")
    started = time.perf_counter()
    record.update(status="evaluating", test_labels_opened=True)
    write_json(output / "run.json", record)
    try:
        for case in design["cases"]:
            print(f"M5 evaluate: {case['id']}", file=sys.stderr, flush=True)
            evaluate(output / case["prediction"], manifest, output / "evaluation" / case["id"])
        if plan["background_stress"]:
            # No fitting or parameter selection occurs after the freeze barrier.
            # Each stress pair also freezes its predictions before reading labels.
            for case in design["cases"]:
                if case["group"] != "P1":
                    continue
                print(f"M5 background stress: {case['id']}", file=sys.stderr, flush=True)
                stress_test(output / case["model"], manifest, case["test_partition"],
                            output / "frozen/trust.json", output / "stress" / case["id"],
                            output / f"frozen/seed_{case['seed']}/calibration")
        summary = build_summary(output)
        export_summary(output, summary)
        evidence = {}
        for name in ("evaluation", "stress"):
            for relative, digest in inventory(output / name).items():
                evidence[f"{name}/{relative}"] = digest
        for name in ("summary.json", "summary.csv", "report.html", "freeze.json"):
            evidence[name] = sha256(output / name)
        write_json(output / "evidence.json", {"artifacts_sha256": evidence})
        record.update(status="evaluated", evidence_sha256=sha256(output / "evidence.json"),
                      evaluation_seconds=time.perf_counter() - started)
        write_json(output / "run.json", record)
        return {"status": "evaluated", "report": str(output / "report.html"),
                "formal_acceptance": summary["formal_acceptance"]}
    except Exception as error:
        record.update(status="failed_evaluation", error=str(error))
        write_json(output / "run.json", record)
        raise


def verify_experiment(output):
    from .summary import build_summary
    record, freeze, design = check_frozen(output)
    if record["status"] not in ("frozen", "evaluated"):
        raise DataError("Experiment is incomplete or failed")
    if record["status"] == "evaluated":
        if sha256(output / "evidence.json") != record["evidence_sha256"]:
            raise DataError("Evaluation evidence manifest changed")
        check_hashes(output, read_json(output / "evidence.json")["artifacts_sha256"])
        for case in design["cases"]:
            prediction = read_json(output / case["prediction"] / "predictions.json")
            metric = read_json(output / "evaluation" / case["id"] / "metrics.json")
            if (metric["prediction_sha256"] != sha256(output / case["prediction"] / "predictions.json")
                    or prediction["model_manifest_sha256"] != sha256(output / case["model"] / "model.json")
                    or prediction["partition_id"] != case["test_partition"]):
                raise DataError("Model/prediction/evaluation identity mismatch")
        if build_summary(output) != read_json(output / "summary.json"):
            raise DataError("Report numbers differ from saved per-case metrics")
        from ..trust.verification import verify
        for directory in sorted((output / "stress").glob("*")):
            verify(directory)
    return {"status": "m5_artifacts_verified", "phase": record["status"], "cases": len(design["cases"]),
            "source_code_matches_current": freeze["code_sha256"] == source_fingerprint(),
            "test_labels_reopened": False, "independent_member_reproduction": "not_attested"}
