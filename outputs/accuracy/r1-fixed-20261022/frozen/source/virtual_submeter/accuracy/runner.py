"""Replay historical M3 on a separate, explicitly development-only protocol."""

from datetime import datetime, timezone
from importlib import metadata as packages
from pathlib import Path
import shutil
import sys

from ..data.common import DataError, read_json, safe_child, sha256, source_fingerprint, write_json
from ..data.protocol import read_partition
from ..environment import environment_report
from .diagnosis import event_support, paired_summary, query_summary, write_csv


def inventory(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def development_copy(model_dir, manifest, output):
    """Copy immutable inputs; rebind metadata only, retaining exact weights."""
    from ..models.artifacts import load_model
    protocol = read_json(manifest)
    metadata, _ = load_model(model_dir)
    if protocol["purpose"] != "engineering_pilot_only" or metadata.get("adaptation") != "none":
        raise DataError("A0 requires a previously inspected pilot protocol and unadapted model")
    if metadata["protocol_sha256"] != sha256(manifest):
        raise DataError("Base model does not belong to the input protocol")
    dev = output / "development"
    for partition in protocol["partitions"]:
        for artifact in partition["artifacts"].values():
            source = safe_child(manifest.parent, artifact["path"])
            if sha256(source) != artifact["sha256"]:
                raise DataError("Input protocol artifact hash mismatch")
            target = safe_child(dev, artifact["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    protocol.update(accuracy_usage="development_only", parent_protocol_sha256=sha256(manifest),
                    development_notice="Previously inspected pilot households; never final blind evaluation.")
    protocol["protocol_id"] += "-accuracy-a0-dev-v1"
    write_json(dev / "protocol.json", protocol)
    (dev / "parent_protocol.json").write_bytes(manifest.read_bytes())
    base = output / "base_model"
    shutil.copytree(model_dir, base)
    metadata.update(parent_model_manifest_sha256=sha256(model_dir / "model.json"),
                    protocol_sha256=sha256(dev / "protocol.json"), protocol_id=protocol["protocol_id"],
                    accuracy_usage="development_only", weights_changed_for_development=False)
    write_json(base / "model.json", metadata)
    return base, dev / "protocol.json"


def availability(root, prepared, protocol, thresholds):
    result = {"environment": environment_report(), "packages": {}, "historical_runs": [],
              "prepared_households": {}, "partition_support": [],
              "final_evaluation": {"required_new_households": 3, "verified_eligible_households": [],
                                   "status": "not_available_in_current_inventory"}}
    for name in ("numpy", "torch", "streamlit"):
        try:
            result["packages"][name] = packages.version(name)
        except packages.PackageNotFoundError:
            result["packages"][name] = None
    for name in ("M3", "M4", "M5"):
        paths = sorted(p.relative_to(root).as_posix() for p in (root / "outputs").glob(f"{name.lower()}*") if p.is_dir())
        result["historical_runs"].append({"milestone": name, "local_directories": paths,
                                          "status": "located_not_verified" if paths else "not_found"})
    if prepared is not None:
        audit = read_json(prepared / "manifest.json")
        result["prepared_manifest_sha256"] = sha256(prepared / "manifest.json")
        for house, info in audit["households"].items():
            # Unused household labels are hash-checked only, never analysed.
            checked = {}
            for channel, artifact in info["artifacts"].items():
                checked[channel] = sha256(safe_child(prepared, artifact["path"])) == artifact["sha256"]
            receipt = info.get("provenance") or {}
            raw = root / "data/raw/refit_m1" / info["source_file"]
            result["prepared_households"][house] = {
                "start": info["start"], "end": info["end"], "channel_mapping": info["channel_mapping"],
                "kind": receipt.get("kind"), "publisher_checksum_verified": receipt.get("publisher_checksum_verified", False),
                "artifacts_verified": checked,
                "raw_source_sha256_matches": sha256(raw) == info["source_sha256"] if raw.exists() else None,
                "artifact_counts": {a: {k: v for k, v in artifact.items() if k not in ("path", "sha256")}
                                    for a, artifact in info["artifacts"].items()},
                "final_eligible": False,
                "limitation": "Local prefix only; 2-4 weeks and final household eligibility not established."}
            if not all(checked.values()):
                raise DataError("Prepared data inventory hash mismatch")
    manifest = read_json(protocol)
    for partition in manifest["partitions"]:
        data = read_partition(protocol, partition["partition_id"], "development_diagnosis", include_labels=True)
        for appliance, threshold in thresholds.items():
            result["partition_support"].append({"partition": partition["partition_id"], "role": partition["role"],
                                                "household_id": partition["household_id"], "appliance": appliance,
                                                **event_support(data[appliance], threshold, manifest["sample_interval_seconds"])})
    return result


def build_diagnosis(output):
    from ..adaptation.state import INITIAL
    comparison = output / "comparison"
    record = read_json(comparison / "run.json")
    manifest = output / "development/protocol.json"
    protocol = read_json(manifest)
    base_metadata = read_json(output / "base_model/model.json")
    baselines, data_cache = {}, {}
    metrics_rows, distribution_rows, query_rows, cases = [], [], [], []
    for case in record["cases"]:
        if case["budget"] == 0:
            baselines[(case["household_id"], case["strategy"], case["seed"])] = read_json(comparison / case["path"] / "prediction/predictions.json")
    for case in record["cases"]:
        directory = comparison / case["path"]
        prediction = read_json(directory / "prediction/predictions.json")
        model = read_json(directory / "model/model.json")
        records = read_json(directory / "model/queries.json")["records"]
        history = read_json(directory / "model/adaptation_history.json")
        baseline = baselines[(case["household_id"], case["strategy"], case["seed"])]
        partition = case["test_partition"]
        if partition not in data_cache:
            data_cache[partition] = read_partition(manifest, partition, "development_diagnosis", include_labels=True)
        distribution = query_summary(records, prediction["appliances"])
        result = {k: case[k] for k in ("path", "household_id", "strategy", "seed", "budget", "test_partition")}
        result.update(devices={}, queries=distribution, training_history=history,
                      base_model_manifest_sha256=model["base_model_manifest_sha256"])
        for appliance in prediction["appliances"]:
            threshold = prediction["model_settings"]["on_thresholds_w"][appliance]
            paired = paired_summary(baseline["rows"], prediction["rows"], data_cache[partition][appliance],
                                    appliance, threshold, protocol["sample_interval_seconds"])
            coefficients = model.get("state_adapter", {}).get("coefficients", {}).get(appliance, INITIAL)
            delta = [x - y for x, y in zip(coefficients, INITIAL)]
            result["devices"][appliance] = {**paired, "coefficients": coefficients, "coefficient_delta": delta,
                                          "coefficient_delta_l2": sum(x*x for x in delta)**0.5}
            common = {"household_id": case["household_id"], "strategy": case["strategy"], "seed": case["seed"],
                      "budget": case["budget"], "appliance": appliance}
            metric = paired["adapted"]
            metrics_rows.append({**common, **{k: metric[k] for k in ("precision", "recall", "f1", "predicted_on_fraction", "actual_on_fraction", "valid_samples", "state_pattern")},
                                 "f1_delta": paired["f1_delta"], "coefficient_delta_l2": result["devices"][appliance]["coefficient_delta_l2"]})
            distribution_rows.append({**common, **distribution[appliance]})
        # Post-freeze label analysis describes queried activity, never used in fitting.
        adaptation = next(p for p in protocol["partitions"] if p["role"] == "target_adaptation" and p["household_id"] == case["household_id"])
        aid = adaptation["partition_id"]
        if aid not in data_cache:
            data_cache[aid] = read_partition(manifest, aid, "development_diagnosis", include_labels=True)
        for query in records:
            a = query["appliance_type"]
            labels = [r for r in data_cache[aid][a] if query["window_start"] <= r["timestamp"] < query["window_end"]]
            values = [float(r["power_w"]) for r in labels if r["power_w"] != ""]
            threshold = base_metadata["settings"]["on_thresholds_w"][a]
            query_rows.append({"strategy": case["strategy"], "seed": case["seed"], "budget": case["budget"], **query,
                               "valid_label_samples": len(values),
                               "active_fraction_observed": sum(v >= threshold for v in values) / len(values) if values else None,
                               "analysis_only_after_freeze": True})
        cases.append(result)
    write_csv(output / "state_metrics.csv", metrics_rows, list(metrics_rows[0]))
    write_csv(output / "query_distribution.csv", distribution_rows, list(distribution_rows[0]))
    if query_rows:
        write_csv(output / "queries.csv", query_rows, list(query_rows[0]))
    else:
        write_csv(output / "queries.csv", [], ["strategy", "seed", "budget", "query_id"])
    regressions = [{"case": c["path"], "appliance": a, "f1_delta": d["f1_delta"],
                    "pattern": d["adapted"]["state_pattern"]}
                   for c in cases for a, d in c["devices"].items() if d["f1_delta"] is not None and d["f1_delta"] < 0]
    return {"schema_version": 1, "status": "development_diagnosed", "accuracy_usage": "development_only",
            "seed_semantics": "query selection seeds; all cases share the same already-trained source weights",
            "label_source": "simulated_from_submeter", "power_output_frozen": True,
            "formal_evaluation": False, "regression_reproduced": bool(regressions), "regressions": regressions,
            "cases": cases, "root_cause_status": "descriptive_evidence_only_requires_controlled_probes",
            "scope": "Replay of available M2 model with historical M3; missing M5 originals are not reconstructed evidence."}


def run(model, manifest, settings, output, root, prepared=None):
    from ..adaptation.runner import run_comparison
    from ..models.prediction import predict
    from ..evaluation.runner import evaluate
    from .report import render_report
    if output.exists():
        raise DataError("A0 output exists; choose a fresh run directory")
    output.mkdir(parents=True)
    state = {"schema_version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
             "source_protocol_sha256": sha256(manifest), "source_model_sha256": sha256(model / "model.json"),
             "code_sha256": source_fingerprint(), "development_labels_opened": False}
    write_json(output / "run.json", state)
    try:
        shutil.copytree(Path(__file__).resolve().parents[1], output / "source/virtual_submeter",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        base, dev_manifest = development_copy(model, manifest, output)
        run_comparison(base, dev_manifest, settings, output / "comparison",
                       strategies=("random", "proposed"), evaluate_test=False)
        comparison = read_json(output / "comparison/run.json")
        for case in comparison["cases"]:
            print(f"A0 freeze prediction: {case['path']}", file=sys.stderr, flush=True)
            directory = output / "comparison" / case["path"]
            predict(directory / "model", directory / "prediction", dev_manifest, case["test_partition"])
        frozen = {"status": "all_models_queries_predictions_frozen", "development_labels_opened": False,
                  "comparison_sha256": inventory(output / "comparison"),
                  "development_sha256": inventory(output / "development"), "base_model_sha256": inventory(base),
                  "code_sha256": state["code_sha256"]}
        write_json(output / "freeze.json", frozen)
        state.update(status="frozen", freeze_sha256=sha256(output / "freeze.json"))
        write_json(output / "run.json", state)
        verify_frozen(output)
        state.update(status="diagnosing", development_labels_opened=True)
        write_json(output / "run.json", state)
        for case in comparison["cases"]:
            directory = output / "comparison" / case["path"]
            evaluate(directory / "prediction", dev_manifest, output / "evaluation" / case["path"])
        diagnosis = build_diagnosis(output)
        write_json(output / "diagnosis.json", diagnosis)
        info = availability(root, prepared, dev_manifest, read_json(base / "model.json")["settings"]["on_thresholds_w"])
        write_json(output / "availability.json", info)
        render_report(output, diagnosis)
        evidence = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
                    if p.is_file() and p.name not in ("run.json", "evidence.json")}
        write_json(output / "evidence.json", {"artifacts_sha256": evidence})
        state.update(status="diagnosed", evidence_sha256=sha256(output / "evidence.json"),
                     regression_reproduced=diagnosis["regression_reproduced"])
        write_json(output / "run.json", state)
        return verify(output)
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_json(output / "run.json", state)
        raise


def verify_frozen(output):
    state = read_json(output / "run.json")
    freeze = read_json(output / "freeze.json")
    if sha256(output / "freeze.json") != state["freeze_sha256"] or freeze["development_labels_opened"] is not False:
        raise DataError("Invalid A0 freeze barrier")
    for name, key in (("comparison", "comparison_sha256"), ("development", "development_sha256"), ("base_model", "base_model_sha256")):
        if inventory(output / name) != freeze[key]:
            raise DataError("A0 frozen artifact inventory changed")
    return state, freeze


def verify(output):
    state, freeze = verify_frozen(output)
    if state["status"] != "diagnosed" or sha256(output / "evidence.json") != state["evidence_sha256"]:
        raise DataError("A0 diagnosis is incomplete or evidence changed")
    evidence = read_json(output / "evidence.json")["artifacts_sha256"]
    for relative, digest in evidence.items():
        if sha256(safe_child(output, relative)) != digest:
            raise DataError("A0 diagnosis evidence hash mismatch")
    diagnosis = read_json(output / "diagnosis.json")
    return {"status": "a0_artifacts_verified", "cases": len(diagnosis["cases"]),
            "regression_reproduced": diagnosis["regression_reproduced"],
            "source_code_matches_current": freeze["code_sha256"] == source_fingerprint(),
            "development_labels_reopened": False, "formal_evaluation": False}
