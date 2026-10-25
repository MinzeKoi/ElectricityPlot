"""Freeze all equal-budget strategy/seed models before any test evaluation."""

from datetime import datetime, timezone
from pathlib import Path

from ..data.common import DataError, read_json, sha256, source_fingerprint, write_json
from ..data.oracle import answer_pending
from ..environment import environment_report
from ..models.prediction import predict
from .selection import STRATEGIES, load_settings
from .session import confirm, create_session, select_next
from .training import adapt


def run_comparison(model_dir: Path, manifest_path: Path, settings_path: Path, output: Path,
                   budgets=None, seeds=None, strategies=STRATEGIES, evaluate_test=True):
    if output.exists():
        raise DataError("M3 run exists; choose a new output directory")
    protocol = read_json(manifest_path)
    settings = load_settings(settings_path)
    budgets = list(protocol["query_budgets"] if budgets is None else budgets)
    seeds = list(protocol["random_seeds"] if seeds is None else seeds)
    strategies = list(strategies)
    for values, allowed in ((budgets, protocol["query_budgets"]), (seeds, protocol["random_seeds"]), (strategies, STRATEGIES)):
        if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
            raise DataError("Comparison budgets, seeds and strategies must be unique frozen choices")
    targets = [p for p in protocol["partitions"] if p["role"] == "target_adaptation"]
    if not targets:
        raise DataError("No target adaptation partitions")
    output.mkdir(parents=True)
    record = {"schema_version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha256(manifest_path), "base_model_manifest_sha256": sha256(model_dir / "model.json"),
              "code_sha256": source_fingerprint(), "settings": settings, "budgets": budgets, "seeds": seeds,
              "strategies": strategies, "purpose": protocol["purpose"], "label_source": "simulated_from_submeter",
              "power_output_frozen": True, "cases": [], "test_evaluation_started": False}
    write_json(output / "run.json", record)
    (output / "protocol.json").write_bytes(manifest_path.read_bytes())
    write_json(output / "settings.json", settings)
    write_json(output / "environment.json", environment_report())
    try:
        for target in targets:
            test = [p for p in protocol["partitions"] if p["role"] == "frozen_test" and p["household_id"] == target["household_id"]]
            if len(test) != 1:
                raise DataError("Each target requires exactly one frozen test partition")
            for seed in seeds:
                for strategy in strategies:
                    for budget in budgets:
                        name = f"house_{target['household_id']}/{strategy}/seed_{seed}/budget_{budget}"
                        case_dir = output / name
                        create_session(model_dir, manifest_path, target["partition_id"], case_dir / "session", settings_path, budget, strategy, seed)
                        for _ in range(budget):
                            if select_next(case_dir / "session") is None:
                                raise DataError("Not enough complete disjoint windows for equal-budget comparison")
                            confirm(case_dir / "session", **answer_pending(case_dir / "session"))
                        result = adapt(case_dir / "session", case_dir / "model")
                        record["cases"].append({"path": name, "household_id": target["household_id"], "strategy": strategy,
                                                "seed": seed, "budget": budget, "model_id": result["model_id"], "test_partition": test[0]["partition_id"]})
                        write_json(output / "run.json", record)
        record["status"] = "all_models_frozen"
        write_json(output / "run.json", record)
        if evaluate_test:
            # Only now import and call the module permitted to read test labels.
            from ..evaluation.runner import evaluate
            record["test_evaluation_started"] = True
            write_json(output / "run.json", record)
            for case in record["cases"]:
                directory = output / case["path"]
                predict(directory / "model", directory / "prediction", manifest_path, case["test_partition"])
                evaluate(directory / "prediction", manifest_path, directory / "evaluation")
                metrics = read_json(directory / "evaluation/metrics.json")
                case["metrics"] = {a: v["corrected"] for a, v in metrics["metrics"].items()}
                case["query_summary"] = read_json(directory / "model/model.json")["query_summary"]
            record["status"] = "m3_pilot_evaluated"
        write_json(output / "run.json", record)
        write_json(output / "summary.json", record)
        return record
    except (DataError, OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        record.update(status="failed", error=str(error))
        write_json(output / "run.json", record)
        raise
