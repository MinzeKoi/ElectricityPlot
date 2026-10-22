"""A2 fixed-query comparison with a full model/prediction freeze barrier."""

import copy
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from ..data.common import DataError, read_json, safe_child, sha256, source_fingerprint, write_json
from ..data.protocol import read_partition, validate_confirmation_history
from ..environment import environment_report
from .diagnosis import query_summary, write_csv
from .learners import (VARIANTS, composed_rows, design, fit_adapter, normalization,
                       score_adapter, source_fit, temporal_features)
from .runner import inventory


def load_plan(path):
    plan = read_json(path)
    expected = {"schema_version", "source_seeds", "query_seeds", "budgets", "primary_appliances", "extension_appliances",
                "variants", "source_epochs", "source_learning_rate", "source_max_samples", "source_positive_weight_cap",
                "target_epochs", "target_learning_rate", "bias_bound", "single_class_bias_bound", "parameter_radius",
                "replay_samples", "replay_weight", "prior_l2", "smooth_temperature", "main_budget",
                "minimum_primary_f1_gain", "maximum_device_f1_drop"}
    if set(plan) != expected or plan["schema_version"] != 1:
        raise DataError("Invalid A2 configuration schema")
    for key in ("source_seeds", "query_seeds", "budgets"):
        values = plan[key]
        if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values) or len(values) != len(set(values)):
            raise DataError("A2 seeds and budgets must be unique nonnegative integers")
    if plan["budgets"] != sorted(plan["budgets"]) or plan["budgets"][0] != 0 or plan["main_budget"] not in plan["budgets"]:
        raise DataError("A2 needs nested budgets with a zero baseline and a main budget")
    if not isinstance(plan["variants"], list) or not plan["variants"] or len(set(plan["variants"])) != len(plan["variants"]) or not set(plan["variants"]) <= set(VARIANTS):
        raise DataError("Invalid A2 learner variants")
    for key in ("source_epochs", "target_epochs", "source_max_samples", "replay_samples"):
        if type(plan[key]) is not int or not 2 <= plan[key] <= 100000:
            raise DataError("Invalid A2 iteration/sample count")
    for key in expected - {"schema_version", "main_budget", "source_seeds", "query_seeds", "budgets", "primary_appliances", "extension_appliances", "variants", "source_epochs", "target_epochs", "source_max_samples", "replay_samples"}:
        value = plan[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise DataError("A2 numerical settings must be finite positive numbers")
    if plan["single_class_bias_bound"] > plan["bias_bound"]:
        raise DataError("Single-class bias bound exceeds the mixed-class bound")
    devices = plan["primary_appliances"] + plan["extension_appliances"]
    if not plan["primary_appliances"] or any(not isinstance(a, str) for a in devices) or len(devices) != len(set(devices)):
        raise DataError("A2 primary and extension appliances must be disjoint")
    return plan


def validate_inputs(a1, plan):
    from ..models.artifacts import load_model
    freeze = read_json(a1 / "baseline_freeze.json")
    manifest = a1 / "protocol/protocol.json"
    protocol = read_json(manifest)
    if (freeze.get("status") != "source_baseline_weights_frozen" or not freeze.get("primary_exit_passed")
            or freeze["protocol_sha256"] != sha256(manifest)
            or protocol.get("purpose") != "engineering_pilot_only" or protocol.get("accuracy_usage") != "development_only"):
        raise DataError("A2 requires eligible frozen A1 development inputs")
    if not set(plan["budgets"]) <= set(protocol["query_budgets"]) or not set(plan["query_seeds"]) <= set(protocol["random_seeds"]):
        raise DataError("A2 budget/query seeds are outside the input protocol")
    if set(plan["primary_appliances"] + plan["extension_appliances"]) != set(protocol["target_appliances"]):
        raise DataError("A2 device scope differs from A1")
    role_houses = {role: {p["household_id"] for p in protocol["partitions"] if p["role"] == role}
                   for role in ("source_train", "source_validation", "target_adaptation", "frozen_test")}
    if (not all(role_houses.values()) or role_houses["target_adaptation"] != role_houses["frozen_test"]
            or role_houses["source_train"] & role_houses["source_validation"]
            or (role_houses["source_train"] | role_houses["source_validation"]) & role_houses["target_adaptation"]):
        raise DataError("A2 source, validation and development households must be isolated")
    for seed in plan["source_seeds"]:
        entry = freeze["models"].get(str(seed))
        if entry is None:
            raise DataError("A2 source seed has no frozen A1 weights")
        model_dir = safe_child(a1, entry["path"])
        metadata, _ = load_model(model_dir)
        if (sha256(model_dir / "model.json") != entry["model_manifest_sha256"]
                or metadata["weights"]["sha256"] != entry["weights_sha256"]
                or metadata["protocol_sha256"] != sha256(manifest)
                or metadata.get("state_threshold_view") != "raw_power_w"
                or metadata.get("adaptation") != "none" or metadata["settings"]["seed"] != seed):
            raise DataError("A1 source identity or state view differs from its freeze")
    return freeze, protocol


def source_design(metadata, model, x, normalizer=None):
    import torch
    settings = metadata["settings"]
    torch.set_num_threads(settings["cpu_threads"])
    normalized = (np.log1p(x) - metadata["input_log_mean"]) / metadata["input_log_std"]
    hidden, powers = [], []
    with torch.inference_mode():
        for batch in torch.from_numpy(normalized).split(settings["batch_size"]):
            features = model.head[:-1](model.features(batch[:, None, :]))
            hidden.append(features.numpy())
            powers.append(torch.nn.functional.softplus(model.head[-1](features)).numpy())
    hidden = np.concatenate(hidden).astype(np.float64)
    powers = np.concatenate(powers) * np.asarray(metadata["power_scales_w"])
    normalizer = normalization(hidden) if normalizer is None else normalizer
    temporal = np.clip((hidden - normalizer["mean"]) / normalizer["std"], -8, 8)
    result = {}
    for column, appliance in enumerate(metadata["appliances"]):
        threshold = settings["on_thresholds_w"][appliance]
        result[appliance] = np.column_stack((np.ones(len(x)), np.log((powers[:, column] + 1) / (threshold + 1)), temporal))
    return result, normalizer


def train_source_head(model_dir, manifest, plan, directory):
    from ..models.artifacts import load_model
    from ..models.training import dataset
    metadata, model = load_model(model_dir)
    protocol = read_json(manifest)
    arrays = {}
    norm = None
    for role in ("source_train", "source_validation"):
        partitions = [p for p in protocol["partitions"] if p["role"] == role]
        x, y, counts = dataset(manifest, partitions, metadata["appliances"], metadata["settings"], metadata["sample_interval_seconds"])
        matrices, norm = source_design(metadata, model, x, norm)
        states = {a: np.where(np.isfinite(y[:, c]), (y[:, c] >= metadata["settings"]["on_thresholds_w"][a]).astype(float), np.nan)
                  for c, a in enumerate(metadata["appliances"])}
        arrays[role] = (matrices, states, counts)
    head = {"schema_version": 1, "normalizer": norm, "coefficients": {}, "base_model_sha256": sha256(model_dir / "model.json"),
            "weights_sha256": metadata["weights"]["sha256"], "protocol_sha256": sha256(manifest), "target_labels_read": False}
    replay, histories, cache = {}, {}, {}
    for a in metadata["appliances"]:
        train_x, train_y, _ = arrays["source_train"]
        valid_x, valid_y, _ = arrays["source_validation"]
        coefficients, replay[a], histories[a] = source_fit(train_x[a], train_y[a], valid_x[a], valid_y[a], plan, metadata["settings"]["seed"])
        head["coefficients"][a] = coefficients.tolist()
        cache[a + "_x"], cache[a + "_y"] = replay[a]
    write_json(directory / "source_head.json", head)
    write_json(directory / "source_history.json", histories)
    np.savez_compressed(directory / "source_replay.npz", **cache)
    return head, replay


def export_model(base, metadata, adapter, records, plan, history, destination, source_head):
    destination.mkdir(parents=True)
    shutil.copyfile(base / "weights.pt", destination / "weights.pt")
    write_json(destination / "queries.json", {"records": records, "budget_used": len(records)})
    payload = copy.deepcopy(metadata)
    payload.update(accuracy_state_head=adapter, adaptation="a2_fixed_query_state_only",
                   base_model_id=metadata["model_id"], base_model_manifest_sha256=sha256(base / "model.json"),
                   code_sha256=source_fingerprint(), power_output_frozen=True, target_pointwise_labels_read=False,
                   target_weak_labels_read=any(r["response"] in ("used", "not_used") for r in records),
                   query_summary={"budget_used": len(records), "responses": query_summary(records, metadata["appliances"]), "strategy": "random"},
                   queries_sha256=sha256(destination / "queries.json"), accuracy_settings=plan,
                   source_state_head=source_head, state_definition="A2 independent state score >= 0.5; frozen raw and corrected power")
    if records:
        payload["adaptation_household_id"] = records[0]["household_id"]
    identity = json.dumps({"base": payload["base_model_manifest_sha256"], "head": adapter, "queries": payload["queries_sha256"], "plan": plan}, sort_keys=True)
    payload["model_id"] = "a2-state-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    write_json(destination / "model.json", payload)
    write_json(destination / "adaptation_history.json", history)
    return payload


def freeze(a1, plan_path, legacy_path, output):
    from ..adaptation.session import create_session, select_next, confirm, read_session
    from ..adaptation.selection import load_settings
    from ..data.oracle import answer_pending
    from ..models.artifacts import load_model
    from ..models.prediction import predict
    if output.exists():
        raise DataError("A2 output exists; choose a fresh directory")
    plan, legacy = load_plan(plan_path), load_settings(legacy_path)
    baseline, protocol = validate_inputs(a1, plan)
    output.mkdir(parents=True)
    state = {"schema_version": 1, "status": "freezing", "created_at": datetime.now(timezone.utc).isoformat(),
             "code_sha256": source_fingerprint(), "a1_freeze_sha256": sha256(a1 / "baseline_freeze.json"),
             "development_future_labels_opened": False, "formal_evaluation": False, "cases": [], "environment": environment_report()}
    write_json(output / "run.json", state)
    try:
        frozen = output / "frozen"
        shutil.copytree(a1 / "protocol", frozen / "protocol")
        shutil.copytree(Path(__file__).resolve().parents[1], frozen / "source/virtual_submeter", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (frozen / "plan.json").write_bytes(plan_path.read_bytes())
        (frozen / "legacy_settings.json").write_bytes(legacy_path.read_bytes())
        (frozen / "a1_freeze.json").write_bytes((a1 / "baseline_freeze.json").read_bytes())
        manifest = frozen / "protocol/protocol.json"
        for seed in plan["source_seeds"]:
            shutil.copytree(safe_child(a1, baseline["models"][str(seed)]["path"]), frozen / "bases" / f"seed_{seed}")
        # One longest random trajectory per household/query seed. Every learner
        # and every source seed receives exactly the same answered prefix.
        for partition in protocol["partitions"]:
            if partition["role"] != "target_adaptation":
                continue
            for query_seed in plan["query_seeds"]:
                directory = frozen / "trajectories" / f"house_{partition['household_id']}" / f"query_{query_seed}"
                print(f"A2 fixed random trajectory house={partition['household_id']} query_seed={query_seed}", file=sys.stderr, flush=True)
                create_session(frozen / "bases" / f"seed_{plan['source_seeds'][0]}", manifest, partition["partition_id"], directory,
                               frozen / "legacy_settings.json", max(plan["budgets"]), "random", query_seed)
                for _ in range(max(plan["budgets"])):
                    query = select_next(directory)
                    if query is None:
                        raise DataError("A2 fixed query pool cannot fill the shared household budget")
                    answer = answer_pending(directory)
                    confirm(directory, **answer)
                read_session(directory)
        for seed in plan["source_seeds"]:
            print(f"A2 source temporal head seed={seed}", file=sys.stderr, flush=True)
            base = frozen / "bases" / f"seed_{seed}"
            metadata, model = load_model(base)
            source_dir = frozen / "source_heads" / f"seed_{seed}"
            source_dir.mkdir(parents=True)
            head, replay = train_source_head(base, manifest, plan, source_dir)
            for partition in protocol["partitions"]:
                if partition["role"] != "target_adaptation":
                    continue
                house, aid = partition["household_id"], partition["partition_id"]
                future = [p for p in protocol["partitions"] if p["role"] == "frozen_test" and p["household_id"] == house]
                if len(future) != 1:
                    raise DataError("A2 requires exactly one development future partition per household")
                fid = future[0]["partition_id"]
                cache = frozen / "cache" / f"seed_{seed}" / f"house_{house}"
                arrays, bases = {}, {}
                for pid, consumer in ((aid, "adaptation"), (fid, "inference")):
                    aggregate = read_partition(manifest, pid, consumer)["aggregate"]
                    predict(base, cache / pid, manifest, pid)
                    arrays[pid] = temporal_features(metadata, model, aggregate)
                    bases[pid] = read_json(cache / pid / "predictions.json")
                    np.savez_compressed(cache / f"{pid}_features.npz", hidden=arrays[pid])
                for query_seed in plan["query_seeds"]:
                    trajectory = read_json(frozen / "trajectories" / f"house_{house}" / f"query_{query_seed}" / "session.json")["records"]
                    validate_confirmation_history(trajectory, protocol)
                    for variant in plan["variants"]:
                        for budget in plan["budgets"]:
                            started = time.perf_counter()
                            records = trajectory[:budget]
                            adapter, history = fit_adapter(bases[aid]["rows"], records, metadata, plan, variant, head,
                                                           arrays[aid], replay, legacy)
                            relative = f"cases/seed_{seed}/house_{house}/query_{query_seed}/{variant}/budget_{budget}"
                            directory = frozen / relative
                            model_meta = export_model(base, metadata, adapter, records, plan, history, directory / "model", sha256(source_dir / "source_head.json"))
                            values = score_adapter(bases[fid]["rows"], metadata, adapter, arrays[fid])
                            np.savez_compressed(directory / "states.npz", scores=values)
                            case = {"path": relative, "source_seed": seed, "query_seed": query_seed, "household_id": house,
                                    "variant": variant, "budget": budget, "future_partition": fid,
                                    "base_prediction": (cache / fid / "predictions.json").relative_to(frozen).as_posix(),
                                    "query_prefix_sha256": sha256(directory / "model/queries.json"), "model_id": model_meta["model_id"],
                                    "adaptation_seconds": time.perf_counter() - started}
                            write_json(directory / "case.json", case)
                            state["cases"].append(case)
                    print(f"A2 frozen seed={seed} house={house} query_seed={query_seed}", file=sys.stderr, flush=True)
                write_json(output / "run.json", state)
        write_json(output / "freeze.json", {"status": "all_queries_models_future_predictions_frozen", "development_future_labels_opened": False,
                                             "artifacts_sha256": inventory(frozen), "cases": state["cases"]})
        state.update(status="frozen", freeze_sha256=sha256(output / "freeze.json"))
        write_json(output / "run.json", state)
        return {"status": "a2_frozen", "cases": len(state["cases"]), "development_future_labels_opened": False}
    except Exception as error:
        state.update(status="failed_freeze", error=str(error))
        write_json(output / "run.json", state)
        raise


def verify_frozen(output):
    state = read_json(output / "run.json")
    barrier = read_json(output / "freeze.json")
    if (sha256(output / "freeze.json") != state["freeze_sha256"] or barrier.get("development_future_labels_opened") is not False
            or inventory(output / "frozen") != barrier["artifacts_sha256"] or state["cases"] != barrier["cases"]):
        raise DataError("A2 frozen inputs/models/predictions changed")
    frozen = output / "frozen"
    plan, protocol = load_plan(frozen / "plan.json"), read_json(frozen / "protocol/protocol.json")
    houses = {p["household_id"] for p in protocol["partitions"] if p["role"] == "target_adaptation"}
    expected = {(s, q, h, v, b) for s in plan["source_seeds"] for q in plan["query_seeds"] for h in houses for v in plan["variants"] for b in plan["budgets"]}
    actual = {(c["source_seed"], c["query_seed"], c["household_id"], c["variant"], c["budget"]) for c in state["cases"]}
    if actual != expected or len(actual) != len(state["cases"]):
        raise DataError("A2 comparison matrix is incomplete")
    trajectories = {}
    for case in state["cases"]:
        key = (case["household_id"], case["query_seed"])
        if key not in trajectories:
            session = read_json(frozen / "trajectories" / f"house_{key[0]}" / f"query_{key[1]}" / "session.json")
            validate_confirmation_history(session["records"], protocol)
            trajectories[key] = session["records"]
        directory = safe_child(frozen, case["path"])
        queries = read_json(directory / "model/queries.json")
        if queries["records"] != trajectories[key][:case["budget"]] or queries["budget_used"] != case["budget"]:
            raise DataError("A2 learners do not share the exact nested confirmation prefix")
        if sha256(directory / "model/queries.json") != case["query_prefix_sha256"]:
            raise DataError("A2 prefix identity mismatch")
    return state, plan


def mean_complete(values):
    return sum(values) / len(values) if values and all(v is not None for v in values) else None


def summarize(metrics, plan):
    """Equal query/source seeds, then devices, then households; nulls propagate."""
    result = []
    for variant in plan["variants"]:
        for budget in plan["budgets"]:
            selected = [m for m in metrics if m["variant"] == variant and m["budget"] == budget and m["appliance"] in plan["primary_appliances"]]
            per_house = []
            for house in sorted({m["household_id"] for m in selected}):
                devices = []
                for appliance in plan["primary_appliances"]:
                    rows = [m for m in selected if m["household_id"] == house and m["appliance"] == appliance]
                    devices.append({"appliance": appliance, "f1": mean_complete([m["f1"] for m in rows]),
                                    "f1_delta": mean_complete([m["f1_delta"] for m in rows]),
                                    "precision": mean_complete([m["precision"] for m in rows]),
                                    "recall": mean_complete([m["recall"] for m in rows])})
                per_house.append({"household_id": house, "devices": devices,
                                  "f1": mean_complete([d["f1"] for d in devices]), "f1_delta": mean_complete([d["f1_delta"] for d in devices])})
            result.append({"variant": variant, "budget": budget, "f1": mean_complete([h["f1"] for h in per_house]),
                           "f1_delta": mean_complete([h["f1_delta"] for h in per_house]), "households": per_house,
                           "missing_f1": sum(m["f1"] is None for m in selected),
                           "adaptation_degenerate_cases": sum(m["adaptation_caused_degeneracy"] for m in selected),
                           "device_drops_over_tolerance": sum(d["f1_delta"] is None or d["f1_delta"] < -plan["maximum_device_f1_drop"] for h in per_house for d in h["devices"])})
    candidates = [r for r in result if r["variant"] in ("L1", "L2_max", "L2_smooth") and r["budget"] == plan["main_budget"]]
    positive = [r for r in candidates if r["f1_delta"] is not None and r["f1_delta"] > 0 and not r["adaptation_degenerate_cases"] and not r["device_drops_over_tolerance"]]
    target = [r for r in positive if r["f1_delta"] >= plan["minimum_primary_f1_gain"]]
    selected = max(positive, key=lambda r: r["f1_delta"])["variant"] if positive else None
    return {"schema_version": 1, "status": "development_evaluated", "formal_evaluation": False, "label_source": "simulated_from_submeter",
            "power_output_frozen": True, "groups": result, "main_budget": plan["main_budget"],
            "development_exit_passed": bool(positive), "internal_gain_target_met": bool(target), "selected_development_learner": selected,
            "selection_rule": "positive own-baseline gain at main budget, no caused all-off/on, no primary device mean loss over tolerance; largest gain among L1/L2_max/L2_smooth",
            "notice": "Previously inspected development household, ideal simulated answers; seeds are not independent households. No final H1/H2/H3 or power improvement claim."}


def evaluate(output):
    from ..evaluation.metrics import appliance_metrics
    state, plan = verify_frozen(output)
    if state["status"] != "frozen":
        raise DataError("Only a frozen, unevaluated A2 run can open development future labels")
    state.update(status="evaluating", development_future_labels_opened=True)
    write_json(output / "run.json", state)
    frozen = output / "frozen"
    try:
        data_cache, prediction_cache, zero = {}, {}, {}
        metrics = []
        for case in state["cases"]:
            pid = case["future_partition"]
            if pid not in data_cache:
                data_cache[pid] = read_partition(frozen / "protocol/protocol.json", pid, "evaluation", include_labels=True)
            base_path = case["base_prediction"]
            if base_path not in prediction_cache:
                prediction_cache[base_path] = read_json(safe_child(frozen, base_path))
            base = prediction_cache[base_path]
            with np.load(safe_child(frozen, case["path"]) / "states.npz", allow_pickle=False) as saved:
                rows = composed_rows(base["rows"], base["appliances"], saved["scores"])
            case_metrics = []
            for appliance in base["appliances"]:
                key = (case["source_seed"], case["query_seed"], case["household_id"], case["variant"], appliance)
                threshold = base["model_settings"]["on_thresholds_w"][appliance]
                raw = appliance_metrics(rows, data_cache[pid][appliance], appliance, threshold, base["sample_interval_seconds"], "raw_power_w")
                corrected = appliance_metrics(rows, data_cache[pid][appliance], appliance, threshold, base["sample_interval_seconds"])
                if case["budget"] == 0:
                    zero[key] = raw
                before = zero[key]
                if before["valid_samples"] != raw["valid_samples"]:
                    raise DataError("A2 paired evaluation changed valid support")
                degenerate = raw["valid_samples"] > 0 and raw["predicted_on_samples"] in (0, raw["valid_samples"])
                before_degenerate = before["valid_samples"] > 0 and before["predicted_on_samples"] in (0, before["valid_samples"])
                item = {**{k: case[k] for k in ("source_seed", "query_seed", "household_id", "variant", "budget")}, "appliance": appliance,
                        "precision": raw["precision"], "recall": raw["recall"], "f1": raw["f1"], "baseline_f1": before["f1"],
                        "f1_delta": raw["f1"] - before["f1"] if raw["f1"] is not None and before["f1"] is not None else None,
                        "valid_samples": raw["valid_samples"], "predicted_on_fraction": raw["predicted_on_samples"] / raw["valid_samples"] if raw["valid_samples"] else None,
                        "actual_on_samples": raw["actual_on_samples"], "degenerate": degenerate,
                        "adaptation_caused_degeneracy": degenerate and not before_degenerate, "adaptation_seconds": case["adaptation_seconds"],
                        "raw": raw, "corrected": corrected}
                for power_key in ("mae_w", "active_mae_w", "energy"):
                    if raw[power_key] != before[power_key]:
                        raise DataError("A2 state adaptation changed power or energy")
                case_metrics.append(item)
                metrics.append(item)
            write_json(output / "evaluation" / case["path"] / "metrics.json", {"case": case, "metrics": case_metrics})
        summary = summarize(metrics, plan)
        write_json(output / "metrics.json", {"rows": metrics})
        write_json(output / "summary.json", summary)
        fields = ["source_seed", "query_seed", "household_id", "variant", "budget", "appliance", "precision", "recall", "f1", "baseline_f1", "f1_delta", "valid_samples", "predicted_on_fraction", "actual_on_samples", "degenerate", "adaptation_caused_degeneracy", "adaptation_seconds"]
        write_csv(output / "comparison.csv", [{k: m[k] for k in fields} for m in metrics], fields)
        distributions = []
        for house in sorted({c["household_id"] for c in state["cases"]}):
            for seed in plan["query_seeds"]:
                records = read_json(frozen / "trajectories" / f"house_{house}" / f"query_{seed}" / "session.json")["records"]
                for budget in plan["budgets"]:
                    for appliance, counts in query_summary(records[:budget], plan["primary_appliances"] + plan["extension_appliances"]).items():
                        distributions.append({"household_id": house, "query_seed": seed, "budget": budget, "appliance": appliance, **counts})
        write_json(output / "query_distribution.json", {"rows": distributions})
        render_report(output, summary, distributions)
        evidence = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and not p.is_relative_to(frozen) and p.name not in ("run.json", "evidence.json")}
        write_json(output / "evidence.json", {"artifacts_sha256": evidence})
        state.update(status="evaluated", evidence_sha256=sha256(output / "evidence.json"))
        write_json(output / "run.json", state)
        return verify(output)
    except Exception as error:
        state.update(status="failed_evaluation", error=str(error))
        write_json(output / "run.json", state)
        raise


def render_report(output, summary, distributions):
    def number(value):
        return f"{value:.4f}" if value is not None else "—"
    rows = "".join(f"<tr><td>{escape(g['variant'])}</td><td>{g['budget']}</td><td>{number(g['f1'])}</td><td>{number(g['f1_delta'])}</td><td>{g['adaptation_degenerate_cases']}</td></tr>" for g in summary["groups"])
    queries = "".join(f"<tr><td>{d['query_seed']}</td><td>{d['budget']}</td><td>{escape(d['appliance'])}</td><td>{d['used']}</td><td>{d['not_used']}</td><td>{d['unsure']}</td></tr>" for d in distributions)
    html = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>A2 固定确认学习器比较</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#152a38}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.notice{{background:#fff1d1;padding:18px}}a{{color:#005b8c}}</style>
<h1>A2 · 固定确认学习器比较</h1><p class="notice">仅 House 3 已查看开发数据，理想模拟确认；三训练种子 × 三查询种子不代表九户。功率、电量保持不变，未验证 A3 选片收益及最终多户效果。</p>
<p>主预算：{summary['main_budget']} 次／户；开发出口：{summary['development_exit_passed']}；内部 +0.05 目标：{summary['internal_gain_target_met']}；开发候选：{escape(str(summary['selected_development_learner']))}。</p>
<p>下表按主设备等权，比较各结构自身零确认基线。L2 零确认的来源状态训练收益单列，不能算成确认贡献。全关／全开统计只计由确认新引起的案例，已有退化仍保留在完整数据中。</p>
<table><tr><th>学习器／消融</th><th>预算</th><th>主设备 F1</th><th>相对自身零确认 ΔF1</th><th>确认导致全关／全开</th></tr>{rows}</table>
<h2>固定查询前缀的回答分布</h2><p>三设备共享总预算；无有效回答严格返回来源决策。受保护学习器的单类回答仅允许预先固定的小幅偏置更新。</p>
<table><tr><th>查询种子</th><th>预算</th><th>设备</th><th>用过</th><th>未用</th><th>不确定</th></tr>{queries}</table>
<p><a href="comparison.csv">完整逐种子指标 CSV</a> · <a href="metrics.json">原始／修正功率与电量指标</a> · <a href="summary.json">汇总及出口规则</a></p></html>"""
    (output / "report.html").write_text(html, encoding="utf-8")


def verify(output):
    state, plan = verify_frozen(output)
    if state["status"] != "evaluated" or sha256(output / "evidence.json") != state.get("evidence_sha256"):
        raise DataError("A2 evaluation is incomplete or its evidence changed")
    for relative, digest in read_json(output / "evidence.json")["artifacts_sha256"].items():
        if sha256(safe_child(output, relative)) != digest:
            raise DataError("A2 evaluation artifact hash mismatch")
    metrics = read_json(output / "metrics.json")["rows"]
    summary = read_json(output / "summary.json")
    if summary != summarize(metrics, plan):
        raise DataError("A2 summary does not reproduce from per-case metrics")
    return {"status": "a2_verified", "cases": len(state["cases"]), "metric_rows": len(metrics),
            "development_exit_passed": summary["development_exit_passed"], "internal_gain_target_met": summary["internal_gain_target_met"],
            "selected_development_learner": summary["selected_development_learner"],
            "current_source_matches_execution": source_fingerprint() == state["code_sha256"], "formal_evaluation": False,
            "report": str(output / "report.html")}
