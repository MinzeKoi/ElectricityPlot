"""A4 paired power experiment, immutable freeze and source residual calibration."""

from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from ..data.common import DataError, read_json, safe_child, sha256, source_fingerprint, utc, write_json
from ..data.protocol import read_partition, validate_confirmation_history
from ..environment import environment_report
from ..trust.calibration import METHOD, NOTICE, power_fingerprint, apply_intervals, load_calibration
from ..trust.output import validate_output
from . import dynamic_queries as dynamic
from .diagnosis import write_csv
from .fixed_queries import mean_complete, source_design
from .learners import composed_rows, sigmoid
from .power import load_plan, fit_branch, replace_power
from .runner import inventory

MODES = ("state_only", "coupled")
NOTICE_A4 = ("Inspected House 3 development data and ideal simulated confirmations; A3 state gains remain weak. "
             "Seeds are not independent households. Exploratory H3 only, no final cross-household or real-user claim.")


def source_arrays(base, manifest, normalizer):
    from ..models.artifacts import load_model
    from ..models.training import dataset
    metadata, model = load_model(base)
    protocol = read_json(manifest)
    result = {}
    for role in ("source_train", "source_validation"):
        partitions = [p for p in protocol["partitions"] if p["role"] == role]
        x, y, counts = dataset(manifest, partitions, metadata["appliances"], metadata["settings"], metadata["sample_interval_seconds"])
        matrices, _ = source_design(metadata, model, x, normalizer)
        result[role] = {"matrices": matrices, "labels": y, "totals": x[:, metadata["settings"]["window_samples"] // 2], "counts": counts}
    return result


def source_calibration(model_dir, metadata, branch, arrays, manifest, plan, destination):
    """Exact frozen branch on source validation centers, without target labels."""
    raw = []
    for appliance in metadata["appliances"]:
        x = arrays["matrices"][appliance]
        scores = sigmoid(x @ np.asarray(metadata["accuracy_state_head"]["coefficients"][appliance]))
        amplitude = np.clip(np.delete(x, 1, axis=1) @ np.asarray(branch["coefficients"][appliance]), *branch["bounds_w"][appliance])
        raw.append(scores * amplitude)
    raw = np.column_stack(raw)
    totals = arrays["totals"]
    factors = np.minimum(1, np.divide(totals, raw.sum(axis=1), out=np.ones(len(totals)), where=raw.sum(axis=1) > 0))
    corrected = raw * factors[:, None]
    quantiles = {}
    for column, appliance in enumerate(metadata["appliances"]):
        actual = arrays["labels"][:, column]
        valid = np.isfinite(actual)
        if int(valid.sum()) < plan["minimum_calibration_samples"]:
            raise DataError("A4 source residual calibration lacks support")
        quantiles[appliance] = {}
        for version, values in (("raw", raw), ("corrected", corrected)):
            errors = np.sort(np.abs(values[valid, column] - actual[valid]))
            radius = float(errors[int(np.ceil(len(errors) * plan["interval_level"])) - 1])
            quantiles[appliance][version] = {"radius_w": radius, "samples": len(errors),
                                           "source_empirical_coverage": float(np.mean(errors <= radius))}
    settings = {"schema_version": 1, "interval_level": plan["interval_level"], "minimum_samples": plan["minimum_calibration_samples"],
                "stress": {"power_w": 500, "period_samples": 300, "duration_samples": 30, "offset_samples": 0}}
    write_json(destination / "settings.json", settings)
    shutil.copyfile(manifest, destination / "protocol.json")
    write_json(destination / "calibration.json", {
        "schema_version": 1, "status": "calibrated", "method": METHOD, "model_id": metadata["model_id"],
        "created_at": datetime.now(timezone.utc).isoformat(), "power_fingerprint": power_fingerprint(metadata),
        "model_manifest_sha256": sha256(model_dir / "model.json"), "protocol_sha256": sha256(manifest),
        "settings_sha256": sha256(destination / "settings.json"), "code_sha256": source_fingerprint(),
        "interval_level": plan["interval_level"], "sample_interval_seconds": metadata["sample_interval_seconds"],
        "appliances": metadata["appliances"], "target_labels_read": False, "coverage_guarantee": False,
        "validation_reused_for_model_selection": True, "notice": NOTICE,
        "validation_partitions": arrays["counts"], "source_aggregate_range_w": [float(totals.min()), float(totals.max())],
        "quantiles": quantiles, "support": "source validation complete centers with at least one observed appliance label"})
    return load_calibration(destination, metadata)


def export_model(parent_model, branch, destination):
    shutil.copytree(parent_model, destination)
    metadata = read_json(destination / "model.json")
    metadata.update(accuracy_power_head=branch, adaptation="a4_state_power_coupled", power_output_frozen=False,
                    source_amplitude_frozen=True, target_power_labels_read=False,
                    state_definition="A4 independent state score >= 0.5; raw power = score * source conditional on-power",
                    parent_a3_model_sha256=sha256(parent_model / "model.json"), code_sha256=source_fingerprint())
    identity = json.dumps({"parent": metadata["parent_a3_model_sha256"], "branch": branch}, sort_keys=True)
    metadata["model_id"] = "a4-power-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    write_json(destination / "model.json", metadata)
    return metadata


def freeze(a3, plan_path, output):
    from ..models.artifacts import load_model
    from .power import conditional_power
    if output.exists():
        raise DataError("A4 output exists; choose a fresh directory")
    plan = load_plan(plan_path)
    parent_result = dynamic.verify(a3)
    parent_state = read_json(a3 / "run.json")
    parent_plan = read_json(a3 / "frozen/query_plan.json")
    if not set(plan["strategies"]) <= set(parent_plan["strategies"]) or parent_plan["learner"] != "L2_max":
        raise DataError("A4 policies/learner absent from frozen A3")
    parent = a3 / "frozen"
    learner = read_json(parent / "inputs/plan.json")
    protocol = read_json(parent / "inputs/protocol/protocol.json")
    if protocol.get("accuracy_usage") != "development_only":
        raise DataError("A4 requires an explicit development protocol")
    output.mkdir(parents=True)
    frozen = output / "frozen"
    state = {"schema_version": 1, "status": "freezing", "code_sha256": source_fingerprint(), "cases": [],
             "created_at": datetime.now(timezone.utc).isoformat(), "environment": environment_report(),
             "development_future_labels_opened": False, "formal_evaluation": False, "notice": NOTICE_A4}
    write_json(output / "run.json", state)
    try:
        shutil.copytree(parent / "inputs/protocol", frozen / "protocol")
        shutil.copytree(Path(__file__).resolve().parents[1], frozen / "source/virtual_submeter", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(plan_path, frozen / "plan.json")
        write_json(frozen / "learner_plan.json", learner)
        write_json(frozen / "parent_receipt.json", {"verification": parent_result,
                   "a3_freeze_sha256": sha256(a3 / "freeze.json"), "a3_evidence_sha256": sha256(a3 / "evidence.json"),
                   "historical_development_labels_already_viewed": True})
        manifest = frozen / "protocol/protocol.json"
        for seed in learner["source_seeds"]:
            base = parent / "inputs/bases" / f"seed_{seed}"
            normalizer = read_json(parent / "inputs/source_heads" / f"seed_{seed}" / "source_head.json")["normalizer"]
            arrays = source_arrays(base, manifest, normalizer)
            selected = [c for c in parent_state["cases"] if c["source_seed"] == seed and c["variant"] in plan["strategies"]]
            example, _ = load_model(safe_child(parent, selected[0]["model_path"]))
            branch = fit_branch(arrays["source_train"]["matrices"], arrays["source_train"]["labels"], example, normalizer, plan)
            write_json(frozen / "branches" / f"seed_{seed}.json", branch)
            validation = arrays["source_validation"]
            np.savez_compressed(frozen / "branches" / f"validation_{seed}.npz", labels=validation["labels"], totals=validation["totals"],
                                **{a: x for a, x in validation["matrices"].items()})
            del arrays
            caches = {}
            for case in selected:
                pid = case["future_partition"]
                if pid not in caches:
                    cache_path = frozen / "cache" / f"seed_{seed}" / pid
                    cache_path.mkdir(parents=True)
                    shutil.copyfile(safe_child(parent, case["base_prediction"]), cache_path / "predictions.json")
                    hidden_path = parent / "inputs/cache" / f"seed_{seed}" / f"house_{case['household_id']}" / f"{pid}_features.npz"
                    with np.load(hidden_path, allow_pickle=False) as saved:
                        amplitudes = conditional_power(branch, example["appliances"], saved["hidden"])
                    payload = read_json(cache_path / "predictions.json")
                    caches[pid] = payload, amplitudes
                payload, amplitudes = caches[pid]
                with np.load(safe_child(parent, case["path"]) / "states.npz", allow_pickle=False) as saved:
                    scores = saved["scores"].copy()
                raw_base = np.asarray([[r["appliances"][a]["raw_power_w"] if r["appliances"][a]["raw_power_w"] is not None else np.nan
                                        for a in payload["appliances"]] for r in payload["rows"]])
                for mode in MODES:
                    relative = f"cases/seed_{seed}/house_{case['household_id']}/query_{case['query_seed']}/{case['variant']}/budget_{case['budget']}/{mode}"
                    destination = frozen / relative
                    destination.mkdir(parents=True)
                    if mode == "coupled":
                        metadata = export_model(safe_child(parent, case["model_path"]), branch, destination / "model")
                        source_calibration(destination / "model", metadata, branch, validation, manifest, plan, destination / "calibration")
                    else:
                        shutil.copytree(safe_child(parent, case["model_path"]), destination / "model")
                        metadata, _ = load_model(destination / "model")
                    raw = raw_base if mode == "state_only" else scores * amplitudes
                    np.savez_compressed(destination / "prediction.npz", scores=scores, raw=raw)
                    record = {k: case[k] for k in ("source_seed", "query_seed", "household_id", "variant", "budget", "future_partition")}
                    record.update(path=relative, mode=mode, model_id=metadata["model_id"],
                                  base_prediction=f"cache/seed_{seed}/{pid}/predictions.json")
                    write_json(destination / "case.json", record)
                    state["cases"].append(record)
                print(f"A4 frozen seed={seed} policy={case['variant']} query={case['query_seed']} budget={case['budget']}", file=sys.stderr, flush=True)
                write_json(output / "run.json", state)
        write_json(output / "freeze.json", {"status": "all_power_models_calibrations_future_predictions_frozen",
                   "development_future_labels_opened": False, "artifacts_sha256": inventory(frozen), "cases": state["cases"]})
        state.update(status="frozen", freeze_sha256=sha256(output / "freeze.json"))
        write_json(output / "run.json", state)
        verify_frozen(output)
        return {"status": "a4_frozen", "cases": len(state["cases"]), "development_future_labels_opened": False}
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="failed_freeze", error=str(error))
        write_json(output / "run.json", state)
        raise


def verify_frozen(output):
    from ..models.artifacts import load_model
    frozen = output / "frozen"
    state, barrier = read_json(output / "run.json"), read_json(output / "freeze.json")
    if (sha256(output / "freeze.json") != state["freeze_sha256"] or barrier["development_future_labels_opened"] is not False
            or barrier["cases"] != state["cases"] or inventory(frozen) != barrier["artifacts_sha256"]):
        raise DataError("A4 frozen evidence changed")
    plan, learner = load_plan(frozen / "plan.json"), read_json(frozen / "learner_plan.json")
    protocol = read_json(frozen / "protocol/protocol.json")
    houses = {p["household_id"] for p in protocol["partitions"] if p["role"] == "target_adaptation"}
    expected = {(s, q, h, v, b, m) for s in learner["source_seeds"] for q in learner["query_seeds"] for h in houses
                for v in plan["strategies"] for b in learner["budgets"] for m in MODES}
    actual = {(c["source_seed"], c["query_seed"], c["household_id"], c["variant"], c["budget"], c["mode"]) for c in state["cases"]}
    if actual != expected or len(actual) != len(state["cases"]):
        raise DataError("A4 paired matrix incomplete")
    prefixes = {}
    for case in state["cases"]:
        directory = safe_child(frozen, case["path"])
        metadata, _ = load_model(directory / "model")
        records = read_json(directory / "model/queries.json")["records"]
        validate_confirmation_history(records, protocol)
        if len(records) != case["budget"] or metadata["model_id"] != case["model_id"]:
            raise DataError("A4 confirmation prefix/model mismatch")
        identity = (case["source_seed"], case["query_seed"], case["household_id"], case["variant"])
        longest = prefixes.get(identity, [])
        if longest[:min(len(longest), len(records))] != records[:min(len(longest), len(records))]:
            raise DataError("A4 modes/budgets do not share nested prefixes")
        prefixes[identity] = records if len(records) > len(longest) else longest
        if case["mode"] == "coupled":
            sibling = directory.parent / "state_only/model"
            control = read_json(sibling / "model.json")
            if (metadata["accuracy_state_head"] != control["accuracy_state_head"]
                    or metadata["weights"] != control["weights"] or metadata["queries_sha256"] != control["queries_sha256"]):
                raise DataError("A4 paired outputs use different states, weights or confirmations")
            if metadata.get("accuracy_power_head") != read_json(frozen / "branches" / f"seed_{case['source_seed']}.json"):
                raise DataError("A4 amplitude changed across budgets")
            load_calibration(directory / "calibration", metadata)
    return state, plan, learner


def materialize(frozen, case, payload):
    directory = safe_child(frozen, case["path"])
    with np.load(directory / "prediction.npz", allow_pickle=False) as saved:
        rows = composed_rows(payload["rows"], payload["appliances"], saved["scores"])
        if case["mode"] == "coupled":
            replace_power(rows, payload["appliances"], saved["raw"])
        else:
            original = np.asarray([[r["appliances"][a]["raw_power_w"] if r["appliances"][a]["raw_power_w"] is not None else np.nan
                                    for a in payload["appliances"]] for r in rows])
            if not np.array_equal(original, saved["raw"], equal_nan=True):
                raise DataError("A4 state-only control changed power")
    calibration = read_json(directory / "calibration/calibration.json") if case["mode"] == "coupled" else None
    apply_intervals(rows, calibration)
    validate_output({**payload, "rows": rows, "calibration": calibration})
    return rows


def summarize(metrics, plan, learner):
    groups = []
    primary = learner["primary_appliances"]
    def hierarchical(rows, key):
        houses = sorted({r["household_id"] for r in rows})
        return mean_complete([mean_complete([mean_complete([r[key] for r in rows if r["household_id"] == h and r["appliance"] == a])
                                             for a in primary]) for h in houses])
    for strategy in plan["strategies"]:
        for mode in MODES:
            for budget in learner["budgets"]:
                rows = [m for m in metrics if m["variant"] == strategy and m["mode"] == mode and m["budget"] == budget and m["appliance"] in primary]
                group = {"variant": strategy, "mode": mode, "budget": budget, "households": sorted({r["household_id"] for r in rows})}
                for key in ("active_mae_w", "energy_absolute_error_kwh", "active_mae_reduction", "energy_error_reduction", "f1", "off_state_attributed_kwh"):
                    group[key] = hierarchical(rows, key)
                group["missing_primary_metrics"] = sum(m["active_mae_w"] is None or m["energy_absolute_error_kwh"] is None for m in rows)
                group["adaptation_caused_degeneracy"] = sum(m["adaptation_caused_degeneracy"] for m in rows)
                groups.append(group)
    main = next(g for g in groups if g["variant"] == plan["main_strategy"] and g["mode"] == "coupled" and g["budget"] == learner["main_budget"])
    mae, energy = main["active_mae_reduction"], main["energy_error_reduction"]
    complete = mae is not None and energy is not None and not main["missing_primary_metrics"]
    passed = complete and ((mae >= plan["minimum_relative_improvement"] and energy >= -plan["maximum_relative_degradation"])
                           or (energy >= plan["minimum_relative_improvement"] and mae >= -plan["maximum_relative_degradation"]))
    household_pairs = []
    for house in main["households"]:
        rows = [m for m in metrics if m["household_id"] == house and m["variant"] == plan["main_strategy"] and m["mode"] == "coupled"
                and m["budget"] == learner["main_budget"] and m["appliance"] in primary]
        household_pairs.append({"household_id": house, "active_mae_reduction": hierarchical(rows, "active_mae_reduction"),
                                "energy_error_reduction": hierarchical(rows, "energy_error_reduction")})
    return {"schema_version": 1, "groups": groups, "main_strategy": plan["main_strategy"], "main_budget": learner["main_budget"],
            "h3_development_target_met": bool(passed), "primary_comparison_complete": complete,
            "household_pairs": household_pairs, "formal_evaluation": False, "notice": NOTICE_A4,
            "normalization": "paired reduction (zero - adapted) / max(zero, frozen near-zero floor), seeds then devices then households equally weighted",
            "background_semantics": "Off-state attributed energy is false device attribution while that device is off; no specific unknown appliance identity."}


def evaluate(output):
    from ..evaluation.metrics import appliance_metrics, consistency_metrics, attribution_metrics, interval_metrics
    state, plan, learner = verify_frozen(output)
    if state["status"] != "frozen":
        raise DataError("Only a frozen unevaluated A4 matrix may open development future labels")
    if state["code_sha256"] != source_fingerprint():
        raise DataError("A4 source changed after freeze; create a fresh run before opening labels")
    state.update(status="evaluating", development_future_labels_opened=True)
    write_json(output / "run.json", state)
    frozen = output / "frozen"
    try:
        labels, bases, zero, metrics, extra = {}, {}, {}, [], []
        for case in state["cases"]:
            pid = case["future_partition"]
            if pid not in labels:
                labels[pid] = read_partition(frozen / "protocol/protocol.json", pid, "evaluation", include_labels=True)
            base_path = case["base_prediction"]
            if base_path not in bases:
                bases[base_path] = read_json(safe_child(frozen, base_path))
            payload = bases[base_path]
            rows = materialize(frozen, case, payload)
            extra.append({"path": case["path"], "consistency": consistency_metrics(rows, payload["sample_interval_seconds"]),
                          "attribution": attribution_metrics(rows, labels[pid], payload["appliances"], payload["sample_interval_seconds"])})
            for appliance in payload["appliances"]:
                threshold = payload["model_settings"]["on_thresholds_w"][appliance]
                results = {version: appliance_metrics(rows, labels[pid][appliance], appliance, threshold, payload["sample_interval_seconds"], key)
                           for version, key in (("raw", "raw_power_w"), ("corrected", "power_w"))}
                corrected = results["corrected"]
                identity = tuple(case[k] for k in ("source_seed", "query_seed", "household_id", "variant", "mode")) + (appliance,)
                if case["budget"] == 0:
                    zero[identity] = results
                before = zero[identity]["corrected"]
                if corrected["valid_samples"] != before["valid_samples"]:
                    raise DataError("A4 paired support changed")
                def reduction(value, old, floor):
                    return (old - value) / max(old, floor) if old is not None and value is not None else None
                truth = {(r["household_id"], utc(r["timestamp"])): float(r["power_w"]) for r in labels[pid][appliance] if r["power_w"] != ""}
                inactive = [r["appliances"][appliance]["power_w"] for r in rows if r["appliances"][appliance]["power_w"] is not None
                            and (r["household_id"], utc(r["timestamp"])) in truth and truth[(r["household_id"], utc(r["timestamp"]))] < threshold]
                degenerate = corrected["valid_samples"] > 0 and corrected["predicted_on_samples"] in (0, corrected["valid_samples"])
                old_degenerate = before["valid_samples"] > 0 and before["predicted_on_samples"] in (0, before["valid_samples"])
                metrics.append({**{k: case[k] for k in ("source_seed", "query_seed", "household_id", "variant", "budget", "mode")},
                                "appliance": appliance, "valid_samples": corrected["valid_samples"], "active_mae_w": corrected["active_mae_w"],
                                "energy_absolute_error_kwh": corrected["energy"]["absolute_error_kwh"], "f1": corrected["f1"],
                                "active_mae_reduction": reduction(corrected["active_mae_w"], before["active_mae_w"], plan["near_zero_mae_w"]),
                                "energy_error_reduction": reduction(corrected["energy"]["absolute_error_kwh"], before["energy"]["absolute_error_kwh"], plan["near_zero_energy_kwh"]),
                                "off_state_attributed_kwh": sum(inactive) * payload["sample_interval_seconds"] / 3600000 if inactive else None,
                                "adaptation_caused_degeneracy": degenerate and not old_degenerate,
                                "raw": results["raw"], "corrected": corrected,
                                "intervals": interval_metrics(rows, labels[pid][appliance], appliance)})
        summary = summarize(metrics, plan, learner)
        write_json(output / "metrics.json", {"rows": metrics})
        write_json(output / "consistency_attribution.json", {"rows": extra})
        write_json(output / "summary.json", summary)
        fields = [k for k in metrics[0] if k not in ("raw", "corrected", "intervals")]
        write_csv(output / "comparison.csv", [{k: m[k] for k in fields} for m in metrics], fields)
        render_report(output, summary)
        write_json(output / "evidence.json", {"artifacts_sha256": {name: sha256(output / name) for name in
                   ("metrics.json", "summary.json", "comparison.csv", "consistency_attribution.json", "report.html")}})
        state.update(status="evaluated", evidence_sha256=sha256(output / "evidence.json"))
        write_json(output / "run.json", state)
        return verify(output)
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="failed_evaluation", error=str(error))
        write_json(output / "run.json", state)
        raise


def render_report(output, summary):
    def number(value):
        return "—" if value is None else f"{value:.4f}"
    table = "".join(f"<tr><td>{escape(g['variant'])}</td><td>{g['mode']}</td><td>{g['budget']}</td>"
                    f"<td>{number(g['active_mae_w'])}</td><td>{number(g['energy_absolute_error_kwh'])}</td>"
                    f"<td>{number(g['active_mae_reduction'])}</td><td>{number(g['energy_error_reduction'])}</td></tr>" for g in summary["groups"])
    (output / "report.html").write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>A4 功率联动验证</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#172c38}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.notice{{background:#fff1d1;padding:18px}}</style>
<h1>A4 · 确认能否改善功率和电量</h1><p class="notice">仅已查看 House 3 开发数据和理想模拟确认。A3 状态收益尚弱，本轮按用户指示启动探索。种子不代表独立家庭；不构成最终跨户 H3 或真实用户效果。</p>
<p>主策略 {escape(summary['main_strategy'])}，主预算 {summary['main_budget']}；H3 开发目标达到：{summary['h3_development_target_met']}。</p>
<p>固定关系为状态分数 × 来源训练的条件开机功率；目标回答不提供真实瓦数。state_only 为同确认纯状态对照；coupled 始终与自身零确认比较。正降幅表示误差减少；近零误差使用预先冻结容差。显示修正功率主指标，原始功率、逐设备负结果、误激活/漏检、来源经验区间及归因保留在完整指标中。</p>
<table><tr><th>策略</th><th>输出</th><th>预算</th><th>运行段 MAE W</th><th>有效区间电量绝对误差 kWh</th><th>自身零确认 MAE 降幅</th><th>自身零确认电量误差降幅</th></tr>{table}</table>
<p><a href="comparison.csv">逐设备／逐种子配对</a> · <a href="metrics.json">原始及修正指标</a> · <a href="consistency_attribution.json">一致性及背景误归因</a> · <a href="summary.json">逐户汇总与出口</a></p></html>''', encoding="utf-8")


def verify(output):
    state, plan, learner = verify_frozen(output)
    if state["status"] != "evaluated" or sha256(output / "evidence.json") != state.get("evidence_sha256"):
        raise DataError("A4 evaluation incomplete or evidence changed")
    for path, expected in read_json(output / "evidence.json")["artifacts_sha256"].items():
        if sha256(safe_child(output, path)) != expected:
            raise DataError("A4 evaluation artifact changed")
    metrics = read_json(output / "metrics.json")["rows"]
    summary = read_json(output / "summary.json")
    if summary != summarize(metrics, plan, learner) or len(metrics) != len(state["cases"]) * len(learner["primary_appliances"] + learner["extension_appliances"]):
        raise DataError("A4 summary/matrix does not reproduce")
    return {"status": "a4_verified", "cases": len(state["cases"]), "metric_rows": len(metrics),
            "h3_development_target_met": summary["h3_development_target_met"], "formal_evaluation": False,
            "current_source_matches_execution": source_fingerprint() == state["code_sha256"], "report": str(output / "report.html")}
