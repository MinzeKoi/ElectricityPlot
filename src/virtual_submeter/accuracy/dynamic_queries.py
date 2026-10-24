"""A3 development matrix with dynamic queries and a pre-evaluation barrier."""

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

from ..data.common import DataError, read_json, safe_child, sha256, source_fingerprint, write_json, publish_directory
from ..environment import environment_report
from . import dynamic_session as sessions
from . import fixed_queries as fixed
from .diagnosis import query_summary, write_csv
from .learners import composed_rows, fit_adapter, score_adapter
from .runner import inventory


def validate_parent(a2):
    """Accept retained A2 evidence for A3, without repairing historical runs.

    Old caches/sessions can be absent or inaccessible in a restricted view. Every retained byte,
    model, query prefix, head, feature array and evaluation artifact must still
    match its original seal. Missing core evidence is always fatal.
    """
    from ..data.protocol import validate_confirmation_history
    state, barrier = read_json(a2 / "run.json"), read_json(a2 / "freeze.json")
    current = inventory(a2 / "frozen")
    expected = barrier["artifacts_sha256"]
    missing = sorted(set(expected) - set(current))
    if not missing:
        fixed.verify(a2)
        return {"status": "a2_full_evidence_verified", "unavailable_derived_artifacts": []}
    if (state["status"] != "evaluated" or sha256(a2 / "freeze.json") != state["freeze_sha256"]
            or state["cases"] != barrier["cases"] or barrier["development_future_labels_opened"] is not False
            or any(k not in expected or expected[k] != v for k, v in current.items())
            or sha256(a2 / "evidence.json") != state["evidence_sha256"]):
        raise DataError("A2 retained evidence changed")
    for path in missing:
        parts = Path(path).parts
        allowed = ((parts[0] == "cache" and parts[-1] in ("predictions.json", "predictions.csv", "report.html", "run.json"))
                   or (parts[0] == "trajectories" and parts[-1] in ("session.json", "protocol.json", "predictions.json", "predictions.csv", "report.html", "run.json")))
        if not allowed:
            raise DataError("A2 core model/feature/query/evaluation evidence is missing")
    for path, value in read_json(a2 / "evidence.json")["artifacts_sha256"].items():
        if sha256(safe_child(a2, path)) != value:
            raise DataError("A2 evaluation evidence changed")
    learner = fixed.load_plan(a2 / "frozen/plan.json")
    if read_json(a2 / "summary.json") != fixed.summarize(read_json(a2 / "metrics.json")["rows"], learner):
        raise DataError("A2 selected learner does not reproduce from retained metrics")
    protocol = read_json(a2 / "frozen/protocol/protocol.json")
    prefixes = {}
    for case in state["cases"]:
        records = read_json(safe_child(a2 / "frozen", case["path"]) / "model/queries.json")["records"]
        validate_confirmation_history(records, protocol)
        if len(records) != case["budget"] or any(r["selection_strategy"] != "random" for r in records):
            raise DataError("A2 retained query budget/strategy differs")
        key = case["household_id"], case["query_seed"]
        longest = prefixes.get(key, [])
        if longest[:min(len(longest), len(records))] != records[:min(len(longest), len(records))]:
            raise DataError("A2 retained models do not share one nested random trajectory")
        if len(records) > len(longest):
            prefixes[key] = records
    return {"status": "a2_retained_core_and_evaluation_verified", "historical_full_verification_available": False,
            "unavailable_derived_artifacts": missing,
            "notice": "Derived artifacts are unavailable in this filesystem view; this does not establish physical deletion. A3 regenerates aggregate predictions in its own directory. Full historical A2 verification is unavailable in this view."}


def _export(base, metadata, adapter, records, settings, history, destination, head_hash, strategy):
    if destination.exists():
        saved = read_json(destination / "model.json")
        if (saved["accuracy_state_head"] != adapter or read_json(destination / "queries.json")["records"] != records
                or saved["accuracy_settings"] != settings or saved["queries_sha256"] != sha256(destination / "queries.json")
                or sha256(destination / "weights.pt") != metadata["weights"]["sha256"]):
            raise DataError("A3 resumed checkpoint differs from deterministic source-initialized fit")
        return saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".a3-model-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "model"
        result = fixed.export_model(base, metadata, adapter, records, settings, history, staging, head_hash, strategy, "a3")
        publish_directory(staging, destination)
    return result


def freeze(a2, plan_path, output, resume=False):
    from ..data.oracle import answer_pending
    from ..models.artifacts import load_model
    from ..models.prediction import predict
    plan = sessions.load_plan(plan_path)
    if plan["schema_version"] in (2, 3):
        from .r2 import validate_parent as validate_r2_parent
        selection = validate_r2_parent(a2, plan)
    parent_receipt = validate_parent(a2)
    selected = read_json(a2 / "summary.json")
    if plan["schema_version"] in (2, 3):
        parent_receipt["r2_selection"] = selection
    elif not selected["development_exit_passed"] or plan["learner"] != selected["selected_development_learner"]:
        raise DataError("A3 requires the frozen learner selected by a passing A2 development run")
    learner = fixed.load_plan(a2 / "frozen/plan.json")
    if any(b % plan["update_every"] for b in learner["budgets"]):
        raise DataError("A3 budget checkpoints must align with the update frequency")
    frozen = output / "frozen"
    if output.exists():
        if not resume:
            raise DataError("A3 output exists; use resume or a fresh output")
        state = read_json(output / "run.json")
        if (state["status"] not in ("freezing", "failed_freeze") or state["code_sha256"] != source_fingerprint()
                or state["a2_evidence_sha256"] != sha256(a2 / "evidence.json")
                or sha256(frozen / "query_plan.json") != sha256(plan_path)
                or inventory(frozen / "inputs") != state["input_inventory"]):
            raise DataError("A3 resume inputs/source changed or run is already frozen/evaluated")
        state.update(status="freezing", cases=[])
    else:
        output.mkdir(parents=True)
        inputs = frozen / "inputs"
        # Source heads/replay and temporal features retain A2 hashes. Aggregate
        # predictions are generated afresh, never restored into historical A2.
        for name in ("protocol", "bases", "source_heads"):
            shutil.copytree(a2 / "frozen" / name, inputs / name)
        for path in sorted((a2 / "frozen/cache").rglob("*_features.npz")):
            destination = inputs / "cache" / path.relative_to(a2 / "frozen/cache")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        for name in ("plan.json", "legacy_settings.json"):
            shutil.copyfile(a2 / "frozen" / name, inputs / name)
        shutil.copyfile(plan_path, frozen / "query_plan.json")
        write_json(inputs / "parent_receipt.json", parent_receipt)
        copied_protocol = read_json(inputs / "protocol/protocol.json")
        for seed in learner["source_seeds"]:
            for partition in copied_protocol["partitions"]:
                if partition["role"] in ("target_adaptation", "frozen_test"):
                    predict(inputs / "bases" / f"seed_{seed}",
                            inputs / "cache" / f"seed_{seed}" / f"house_{partition['household_id']}" / partition["partition_id"],
                            inputs / "protocol/protocol.json", partition["partition_id"])
        shutil.copytree(Path(__file__).resolve().parents[1], frozen / "source/virtual_submeter",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        state = {"schema_version": 1, "status": "freezing", "created_at": datetime.now(timezone.utc).isoformat(),
                 "code_sha256": source_fingerprint(), "a2_evidence_sha256": sha256(a2 / "evidence.json"),
                 "development_future_labels_opened": False, "formal_evaluation": False,
                 "input_inventory": inventory(inputs), "environment": environment_report(), "cases": []}
    write_json(output / "run.json", state)
    try:
        inputs = frozen / "inputs"
        manifest = inputs / "protocol/protocol.json"
        protocol = read_json(manifest)
        legacy = read_json(inputs / "legacy_settings.json")
        for seed in learner["source_seeds"]:
            base = inputs / "bases" / f"seed_{seed}"
            metadata, _ = load_model(base)
            source_dir = inputs / "source_heads" / f"seed_{seed}"
            head = read_json(source_dir / "source_head.json")
            with np.load(source_dir / "source_replay.npz", allow_pickle=False) as saved:
                replay = {a: (saved[a + "_x"].copy(), saved[a + "_y"].copy()) for a in metadata["appliances"]}
            for partition in protocol["partitions"]:
                if partition["role"] != "target_adaptation":
                    continue
                house, aid = partition["household_id"], partition["partition_id"]
                future = [p for p in protocol["partitions"] if p["role"] == "frozen_test" and p["household_id"] == house]
                if len(future) != 1:
                    raise DataError("A3 needs one development future partition per household")
                fid = future[0]["partition_id"]
                cache = inputs / "cache" / f"seed_{seed}" / f"house_{house}"
                payloads, hidden = {}, {}
                for pid in (aid, fid):
                    payloads[pid] = read_json(cache / pid / "predictions.json")
                    with np.load(cache / f"{pid}_features.npz", allow_pickle=False) as saved:
                        hidden[pid] = saved["hidden"].copy()
                pool = sessions.build_pool(payloads[aid], partition, protocol, legacy)
                for query_seed in learner["query_seeds"]:
                    for strategy in plan["strategies"]:
                        relative = f"trajectories/seed_{seed}/house_{house}/query_{query_seed}/{strategy}"
                        directory = frozen / relative
                        if not directory.exists():
                            sessions.create_session(directory, manifest, base, aid, pool, plan, strategy, query_seed, max(learner["budgets"]))
                        session = sessions.read_session(directory)
                        if session["plan"] != plan or read_json(directory / "pool.json")["items"] != pool:
                            raise DataError("A3 resumed policy/candidate pool changed")
                        export_settings = {"learner_settings": learner, "query_plan": plan, "strategy": strategy}
                        for count in range(0, max(learner["budgets"]) + 1, plan["update_every"]):
                            session = sessions.read_session(directory)
                            if len(session["records"]) < count:
                                raise DataError("A3 trajectory has missing answered prefix")
                            records = session["records"][:count]
                            started = time.perf_counter()
                            adapter, history = fit_adapter(payloads[aid]["rows"], records, metadata, learner,
                                                           plan["learner"], head, hidden[aid], replay, legacy)
                            model_dir = directory / "models" / f"budget_{count}"
                            model_meta = _export(base, metadata, adapter, records, export_settings, history, model_dir,
                                                 sha256(source_dir / "source_head.json"), strategy)
                            fit_seconds = time.perf_counter() - started
                            if len(session["rounds"]) <= count // plan["update_every"]:
                                if plan["schema_version"] in (2, 3):
                                    from .r2 import selection_scores
                                    selection_values = selection_scores(payloads[aid]["rows"], metadata, adapter)
                                else:
                                    selection_values = score_adapter(payloads[aid]["rows"], metadata, adapter, hidden[aid])
                                sessions.commit_round(directory, model_dir, selection_values)
                            if count in learner["budgets"]:
                                case_path = f"cases/seed_{seed}/house_{house}/query_{query_seed}/{strategy}/budget_{count}"
                                destination = frozen / case_path
                                destination.mkdir(parents=True, exist_ok=True)
                                values = score_adapter(payloads[fid]["rows"], metadata, adapter, hidden[fid])
                                np.savez_compressed(destination / "states.npz", scores=values)
                                case = {"path": case_path, "source_seed": seed, "query_seed": query_seed, "household_id": house,
                                        "variant": strategy, "budget": count, "future_partition": fid,
                                        "trajectory": relative, "model_path": model_dir.relative_to(frozen).as_posix(),
                                        "base_prediction": (cache / fid / "predictions.json").relative_to(frozen).as_posix(),
                                        "query_prefix_sha256": sha256(model_dir / "queries.json"), "model_id": model_meta["model_id"],
                                        "adaptation_seconds": fit_seconds}
                                write_json(destination / "case.json", case)
                                state["cases"].append(case)
                            end = min(count + plan["update_every"], max(learner["budgets"]))
                            while len(sessions.read_session(directory)["records"]) < end:
                                if sessions.select_next(directory) is None:
                                    raise DataError("A3 complete candidate pool exhausted")
                                sessions.confirm(directory, **answer_pending(directory))
                        print(f"A3 frozen seed={seed} house={house} query_seed={query_seed} strategy={strategy}", file=sys.stderr, flush=True)
                        write_json(output / "run.json", state)
        write_json(output / "freeze.json", {"status": "all_queries_models_future_predictions_frozen",
                   "development_future_labels_opened": False, "artifacts_sha256": inventory(frozen), "cases": state["cases"]})
        state.update(status="frozen", freeze_sha256=sha256(output / "freeze.json"))
        write_json(output / "run.json", state)
        verify_frozen(output)
        return {"status": "a3_frozen", "cases": len(state["cases"]), "development_future_labels_opened": False}
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="failed_freeze", error=str(error))
        write_json(output / "run.json", state)
        raise


def verify_frozen(output):
    state, barrier = read_json(output / "run.json"), read_json(output / "freeze.json")
    frozen = output / "frozen"
    if (sha256(output / "freeze.json") != state["freeze_sha256"] or barrier["development_future_labels_opened"] is not False
            or state["cases"] != barrier["cases"] or inventory(frozen) != barrier["artifacts_sha256"]):
        raise DataError("A3 frozen evidence changed")
    plan = sessions.load_plan(frozen / "query_plan.json")
    learner = fixed.load_plan(frozen / "inputs/plan.json")
    protocol = read_json(frozen / "inputs/protocol/protocol.json")
    if plan["schema_version"] in (2, 3):
        from .r2 import validate_frozen_inputs
        validate_frozen_inputs(frozen / "inputs", plan)
    houses = {p["household_id"] for p in protocol["partitions"] if p["role"] == "target_adaptation"}
    expected = {(s, q, h, v, b) for s in learner["source_seeds"] for q in learner["query_seeds"] for h in houses
                for v in plan["strategies"] for b in learner["budgets"]}
    actual = {(c["source_seed"], c["query_seed"], c["household_id"], c["variant"], c["budget"]) for c in state["cases"]}
    if actual != expected or len(actual) != len(state["cases"]):
        raise DataError("A3 comparison matrix is incomplete")
    seen = {}
    for case in state["cases"]:
        trajectory = case["trajectory"]
        if trajectory not in seen:
            seen[trajectory] = sessions.read_session(safe_child(frozen, trajectory))
        session = seen[trajectory]
        model = safe_child(frozen, case["model_path"])
        metadata = read_json(model / "model.json")
        if (session["plan"] != plan or len(session["records"]) != max(learner["budgets"]) or session["pending"] is not None
                or sha256(model / "queries.json") != case["query_prefix_sha256"]
                or read_json(model / "queries.json")["records"] != session["records"][:case["budget"]]
                or metadata["model_id"] != case["model_id"] or metadata["accuracy_state_head"]["variant"] != plan["learner"]):
            raise DataError("A3 model does not match its budgeted trajectory prefix")
    return state, plan, learner


def summarize(metrics, plan, learner):
    summary = fixed.summarize(metrics, {**learner, "variants": plan["strategies"]})
    groups = summary["groups"]
    for group in groups:
        random = next(g for g in groups if g["variant"] == "random" and g["budget"] == group["budget"])
        group["f1_vs_random"] = group["f1"] - random["f1"] if group["f1"] is not None and random["f1"] is not None else None
        for house in group["households"]:
            other = next(h for h in random["households"] if h["household_id"] == house["household_id"])
            house["f1_vs_random"] = house["f1"] - other["f1"] if house["f1"] is not None and other["f1"] is not None else None
    if plan["schema_version"] in (2, 3):
        from .r2 import summarize as summarize_r2
        return summarize_r2(metrics, plan, learner, summary)
    main = next(g for g in groups if g["variant"] == "proposed" and g["budget"] == learner["main_budget"])
    paired = [m for m in metrics if m["variant"] == "proposed" and m["budget"] == learner["main_budget"] and m["appliance"] in learner["primary_appliances"]]
    by_seed = []
    for seed in learner["query_seeds"]:
        values = [m["f1_vs_random"] for m in paired if m["query_seed"] == seed]
        by_seed.append({"query_seed": seed, "f1_vs_random": fixed.mean_complete(values)})
    repeatable = sum(s["f1_vs_random"] is not None and s["f1_vs_random"] > 0 for s in by_seed) >= (2 * len(by_seed) + 2) // 3
    stable = not main["adaptation_degenerate_cases"] and not main["device_drops_over_tolerance"]
    gain = main["f1_vs_random"]
    summary.update(schema_version=1, selected_development_learner=plan["learner"],
                   development_exit_passed=gain is not None and gain > 0 and repeatable and stable,
                   internal_gain_target_met=gain is not None and gain >= plan["minimum_random_f1_gain"] and repeatable and stable,
                   query_seed_pairs=by_seed, selection_rule="Fixed proposed policy; positive main-budget paired gain over random in at least two thirds of query seeds, with no caused degeneracy/device drop",
                   notice="Previously inspected development households and ideal simulated answers. Query/training seeds are not households; no final H1/H2/H3 or power improvement claim.")
    return summary


def evaluate(output):
    from ..data.protocol import read_partition
    from ..evaluation.metrics import appliance_metrics
    state, plan, learner = verify_frozen(output)
    if state["status"] != "frozen":
        raise DataError("Only a frozen unevaluated A3 matrix may open development future labels")
    state.update(status="evaluating", development_future_labels_opened=True)
    write_json(output / "run.json", state)
    frozen = output / "frozen"
    try:
        data, bases, zero, metrics = {}, {}, {}, []
        for case in state["cases"]:
            pid = case["future_partition"]
            if pid not in data:
                data[pid] = read_partition(frozen / "inputs/protocol/protocol.json", pid, "evaluation", include_labels=True)
            path = case["base_prediction"]
            if path not in bases:
                bases[path] = read_json(safe_child(frozen, path))
            base = bases[path]
            with np.load(safe_child(frozen, case["path"]) / "states.npz", allow_pickle=False) as saved:
                rows = composed_rows(base["rows"], base["appliances"], saved["scores"])
            for appliance in base["appliances"]:
                threshold = base["model_settings"]["on_thresholds_w"][appliance]
                raw = appliance_metrics(rows, data[pid][appliance], appliance, threshold, base["sample_interval_seconds"], "raw_power_w")
                corrected = appliance_metrics(rows, data[pid][appliance], appliance, threshold, base["sample_interval_seconds"])
                key = (case["source_seed"], case["query_seed"], case["household_id"], case["variant"], appliance)
                if case["budget"] == 0:
                    zero[key] = raw
                before = zero[key]
                if raw["valid_samples"] != before["valid_samples"] or any(raw[k] != before[k] for k in ("mae_w", "active_mae_w", "energy")):
                    raise DataError("A3 changed power/energy or paired support")
                degenerate = raw["valid_samples"] > 0 and raw["predicted_on_samples"] in (0, raw["valid_samples"])
                before_degenerate = before["valid_samples"] > 0 and before["predicted_on_samples"] in (0, before["valid_samples"])
                metrics.append({**{k: case[k] for k in ("source_seed", "query_seed", "household_id", "variant", "budget")},
                                "appliance": appliance, "precision": raw["precision"], "recall": raw["recall"], "f1": raw["f1"],
                                "baseline_f1": before["f1"], "f1_delta": raw["f1"] - before["f1"] if raw["f1"] is not None and before["f1"] is not None else None,
                                "valid_samples": raw["valid_samples"], "actual_on_samples": raw["actual_on_samples"],
                                "predicted_on_fraction": raw["predicted_on_samples"] / raw["valid_samples"] if raw["valid_samples"] else None,
                                "adaptation_caused_degeneracy": degenerate and not before_degenerate, "degenerate": degenerate,
                                "adaptation_seconds": case["adaptation_seconds"], "raw": raw, "corrected": corrected})
        indexed = {(m["source_seed"], m["query_seed"], m["household_id"], m["budget"], m["appliance"]): m for m in metrics if m["variant"] == "random"}
        for row in metrics:
            other = indexed[(row["source_seed"], row["query_seed"], row["household_id"], row["budget"], row["appliance"])]
            if row["valid_samples"] != other["valid_samples"] or row["baseline_f1"] != other["baseline_f1"]:
                raise DataError("A3 policies differ in zero baseline or paired support")
            row["f1_vs_random"] = row["f1"] - other["f1"] if row["f1"] is not None and other["f1"] is not None else None
        summary = summarize(metrics, plan, learner)
        distributions = []
        for trajectory in sorted({c["trajectory"] for c in state["cases"]}):
            session = sessions.read_session(safe_child(frozen, trajectory))
            for budget in learner["budgets"]:
                for appliance, counts in query_summary(session["records"][:budget], learner["primary_appliances"] + learner["extension_appliances"]).items():
                    distributions.append({"trajectory": trajectory, "budget": budget, "appliance": appliance, **counts})
        write_json(output / "metrics.json", {"rows": metrics})
        write_json(output / "summary.json", summary)
        write_json(output / "query_distribution.json", {"rows": distributions})
        fields = [k for k in metrics[0] if k not in ("raw", "corrected")]
        write_csv(output / "comparison.csv", [{k: m[k] for k in fields} for m in metrics], fields)
        render_report(output, summary)
        evidence = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
                    if p.is_file() and not p.is_relative_to(frozen) and p.name not in ("run.json", "evidence.json")}
        write_json(output / "evidence.json", {"artifacts_sha256": evidence})
        state.update(status="evaluated", evidence_sha256=sha256(output / "evidence.json"))
        write_json(output / "run.json", state)
        return verify(output)
    except Exception as error:
        state.update(status="failed_evaluation", error=str(error))
        write_json(output / "run.json", state)
        raise


def render_report(output, summary):
    if summary.get("experiment_id") in ("r2-l1-limited-v1", "r2-l1-stable-v1"):
        from .r2 import render_report as render_r2_report
        return render_r2_report(output, summary)
    def number(value):
        return "—" if value is None else f"{value:.4f}"
    rows = "".join(f"<tr><td>{escape(g['variant'])}</td><td>{g['budget']}</td><td>{number(g['f1'])}</td><td>{number(g['f1_delta'])}</td><td>{number(g['f1_vs_random'])}</td><td>{g['adaptation_degenerate_cases']}</td></tr>" for g in summary["groups"])
    html = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>A3 动态选片比较</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#152a38}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.notice{{background:#fff1d1;padding:18px}}</style>
<h1>A3 · 动态选片与反馈循环</h1><p class="notice">已查看开发家庭、理想模拟确认；种子不是独立家庭。固定 L2_max，每 5 次从来源重新适配，状态分数未经校准；功率与电量保持不变。本报告不构成最终多户 H1/H2/H3 证据。</p>
<p>主预算 {summary['main_budget']}；选片开发出口：{summary['development_exit_passed']}；相对随机 +0.02 内部目标：{summary['internal_gain_target_met']}。</p>
<table><tr><th>策略／消融</th><th>预算</th><th>主设备 F1</th><th>自身零确认 ΔF1</th><th>相对随机 ΔF1</th><th>确认导致全关／全开</th></tr>{rows}</table>
<p><a href="comparison.csv">逐设备逐种子指标</a> · <a href="metrics.json">功率和状态原始指标</a> · <a href="query_distribution.json">各前缀回答分布</a> · <a href="summary.json">逐户汇总与出口</a></p></html>'''
    (output / "report.html").write_text(html, encoding="utf-8")


def verify(output):
    state, plan, learner = verify_frozen(output)
    if state["status"] != "evaluated" or sha256(output / "evidence.json") != state.get("evidence_sha256"):
        raise DataError("A3 evaluation incomplete or evidence changed")
    for relative, expected in read_json(output / "evidence.json")["artifacts_sha256"].items():
        if sha256(safe_child(output, relative)) != expected:
            raise DataError("A3 evaluation artifact changed")
    metrics = read_json(output / "metrics.json")["rows"]
    summary = read_json(output / "summary.json")
    if summary != summarize(metrics, plan, learner) or len(metrics) != len(state["cases"]) * len(learner["primary_appliances"] + learner["extension_appliances"]):
        raise DataError("A3 summary/metric matrix does not reproduce")
    return {"status": "a3_verified", "cases": len(state["cases"]), "metric_rows": len(metrics),
            "development_exit_passed": summary["development_exit_passed"], "internal_gain_target_met": summary["internal_gain_target_met"],
            "current_source_matches_execution": source_fingerprint() == state["code_sha256"],
            "formal_evaluation": False, "report": str(output / "report.html")}
