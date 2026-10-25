"""R2 limited L1 comparison and a separately versioned uniform-bound repair.

No query score terms, windows or final-household access are added.
The A3 engine supplies replay, fresh source fits and the all-matrix barrier.
"""

from html import escape
from pathlib import Path

import numpy as np

from ..data.common import DataError, read_json, sha256
from . import fixed_queries as fixed

ROOT = Path(__file__).resolve().parents[3]


def selection_scores(rows, metadata, adapter):
    """Continuous L1 logits also for untouched devices; no hidden-label input.

    Ordinary prediction retains exact source decisions for untouched devices.
    Ranking needs continuous scores rather than that output's boolean values.
    """
    from .learners import design, sigmoid, validate_adapter
    validate_adapter(adapter, metadata)
    if adapter["variant"] not in ("L1", "L1_stable"):
        raise DataError("R2 selection scores require the frozen L1 learner")
    return np.column_stack([sigmoid(design(rows, a, metadata["settings"]["on_thresholds_w"][a])
                                    @ np.asarray(adapter["coefficients"][a])) for a in metadata["appliances"]])


def validate_scope(protocol, learner):
    roles = {role: {p["household_id"] for p in protocol["partitions"] if p["role"] == role}
             for role in ("source_train", "source_validation", "target_adaptation", "frozen_test")}
    expected = {"source_train": {"2", "19"}, "source_validation": {"5"},
                "target_adaptation": {"17"}, "frozen_test": {"17"}}
    dates = {name: sha256(ROOT / f"configs/r0_dates_{name}_20261022.json")
             for name in ("sources", "development")}
    if (roles != expected or {p["household_id"] for p in protocol["partitions"]} != {"2", "5", "17", "19"}
            or protocol.get("protocol_id") != "refit-r1-r0-dates-development-v1"
            or protocol.get("data_kind") != "real_REFIT_complete_date_selected"
            or protocol.get("accuracy_usage") != "development_only"
            or protocol.get("purpose") != "engineering_pilot_only"
            or protocol.get("r0_date_hashes") != dates
            or protocol.get("sample_interval_seconds") != 8
            or protocol.get("query_window_seconds") != 1200
            or protocol.get("split_gap_seconds") != 3600
            or learner != fixed.load_plan(ROOT / "configs/r1_learners_20261022.json")):
        raise DataError("R2 requires unchanged R1 dates, roles, learner settings, seeds and budgets; final 8/9/20 excluded")


def selection_receipt(summary, plan):
    """Separate the historical own-gain winner from the new absolute-F1 choice."""
    groups = [g for g in summary["groups"] if g["budget"] == summary["main_budget"]]
    eligible = [g for g in groups if g["variant"] in ("L1", "L2_max") and g["f1"] is not None
                and not g["missing_f1"] and not g["adaptation_degenerate_cases"]
                and not g["device_drops_over_tolerance"]]
    best = max(eligible, key=lambda g: (g["f1"], g["variant"] == "L1")) if eligible else None
    repair = plan["schema_version"] == 3
    if (not summary["development_exit_passed"] or best is None or best["variant"] != ("L1" if repair else plan["learner"])
            or best["f1_delta"] is None or best["f1_delta"] <= 0):
        raise DataError("R2's predeclared L1 choice is unsupported by guarded R1 absolute performance")
    return {"experiment_id": plan["experiment_id"], "learner": plan["learner"],
            "selection_rule": plan["learner_selection_rule"],
            "historical_own_gain_selection": summary["selected_development_learner"],
            "main_budget": summary["main_budget"],
            "absolute_f1_comparison": [{k: g[k] for k in ("variant", "f1", "f1_delta")} for g in groups],
            "notice": ("R2 repair based on R1 L1 source with uniform existing single-class bound. R1 does not validate this candidate; no final method inheritance."
                       if repair else "New R2 identity. R1 own-gain winner is retained; no repair/retraining or final A5 method inheritance.")}


def validate_parent(parent, plan):
    learner = fixed.load_plan(parent / "frozen/plan.json")
    validate_scope(read_json(parent / "frozen/protocol/protocol.json"), learner)
    receipt = selection_receipt(read_json(parent / "summary.json"), plan)
    receipt.update(parent_evidence_sha256=sha256(parent / "evidence.json"),
                   parent_summary_sha256=sha256(parent / "summary.json"))
    return receipt


def validate_frozen_inputs(inputs, plan):
    validate_scope(read_json(inputs / "protocol/protocol.json"), fixed.load_plan(inputs / "plan.json"))
    receipt = read_json(inputs / "parent_receipt.json").get("r2_selection", {})
    if (receipt.get("experiment_id") != plan["experiment_id"] or receipt.get("learner") != plan["learner"]
            or receipt.get("selection_rule") != plan["learner_selection_rule"]):
        raise DataError("R2 is missing its explicit R1 learner selection receipt")


def preflight(parent, plan_path):
    """Check local artifacts without creating a run or parsing future labels."""
    from .dynamic_queries import validate_parent as verify_parent
    from .dynamic_session import load_plan
    plan = load_plan(plan_path)
    if plan["schema_version"] not in (2, 3):
        raise DataError("Use an R2 version-2 or version-3 plan")
    expected_learner = fixed.load_plan(ROOT / "configs/r1_learners_20261022.json")
    counts = {"cases": len(expected_learner["source_seeds"]) * len(expected_learner["query_seeds"])
              * len(plan["strategies"]) * len(expected_learner["budgets"])}
    counts["metric_rows"] = counts["cases"] * len(expected_learner["primary_appliances"] + expected_learner["extension_appliances"])
    required = [parent / name for name in ("run.json", "freeze.json", "evidence.json", "summary.json",
                                          "frozen/plan.json", "frozen/protocol/protocol.json")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return {"status": "r2_inputs_missing", "formal_evaluation": False, "missing": missing,
                **counts, "future_pointwise_labels_parsed": False,
                "required_run": str(parent), "reserved_final_households": ["8", "9", "20"],
                "notice": "Restore the complete R1 fixed run, including weights, source heads/replay, temporal features, protocol data and query evidence. GitHub synchronization may be incomplete; Git LFS pointers alone are insufficient. Preflight has not executed R2."}
    # Reject a changed scope before the general verifier can visit its data.
    receipt = validate_parent(parent, plan)
    parent_receipt = verify_parent(parent)
    return {"status": "r2_ready", "formal_evaluation": False, "learner_selection": receipt,
            "parent_verification": parent_receipt, **counts,
            "future_pointwise_labels_parsed": False}


def summarize(metrics, plan, learner, summary):
    """Guarded absolute selection; H1 (own gain) and H2 (vs random) stay separate."""
    outcomes = []
    for strategy in plan["strategies"]:
        main = next(g for g in summary["groups"] if g["variant"] == strategy and g["budget"] == learner["main_budget"])
        rows = [m for m in metrics if m["variant"] == strategy and m["budget"] == learner["main_budget"]
                and m["appliance"] in learner["primary_appliances"]]
        pairs = []
        for query in learner["query_seeds"]:
            selected = [r for r in rows if r["query_seed"] == query]
            pairs.append({"query_seed": query,
                          "f1_delta": fixed.mean_complete([r["f1_delta"] for r in selected]),
                          "f1_vs_random": fixed.mean_complete([r["f1_vs_random"] for r in selected])})
        needed = (2 * len(pairs) + 2) // 3
        stable = not main["adaptation_degenerate_cases"] and not main["device_drops_over_tolerance"] and not main["missing_f1"]
        positive_own = sum(p["f1_delta"] is not None and p["f1_delta"] > 0 for p in pairs)
        positive_random = sum(p["f1_vs_random"] is not None and p["f1_vs_random"] > 0 for p in pairs)
        own, random = main["f1_delta"], main["f1_vs_random"]
        outcomes.append({"strategy": strategy, "f1": main["f1"], "f1_delta": own, "f1_vs_random": random,
                         "guard_passed": stable, "query_seed_pairs": pairs,
                         "confirmation_development_exit_passed": stable and own is not None and own > 0 and positive_own >= needed,
                         "h1_development_target_met": stable and own is not None and own >= learner["minimum_primary_f1_gain"] and positive_own >= needed,
                         "h2_development_target_met": stable and random is not None and random >= plan["minimum_random_f1_gain"] and positive_random >= needed})
    eligible = [o for o in outcomes if o["guard_passed"] and o["f1"] is not None]
    # max retains input order on ties: random, static uncertainty, proposed.
    chosen = max(eligible, key=lambda o: o["f1"]) if eligible else None
    proposed = next(o for o in outcomes if o["strategy"] == "proposed")
    summary.update(schema_version=plan["schema_version"], experiment_id=plan["experiment_id"], selected_development_learner=plan["learner"],
                   selected_development_strategy=chosen["strategy"] if chosen else None,
                   strategy_selection_rule=plan["strategy_selection_rule"], strategy_outcomes=outcomes,
                   development_exit_passed=bool(chosen and chosen["confirmation_development_exit_passed"]),
                   internal_gain_target_met=proposed["h2_development_target_met"],
                   proposed_h2_development_target_met=proposed["h2_development_target_met"],
                   final_h1_h2_h3_evaluated=False,
                   selection_rule="Guarded absolute main-budget F1; ties prefer random then static uncertainty then proposed. H1/H2 checked separately.",
                   notice="House 17 inspected development only; L1 state adaptation, unchanged power. No real-user or final cross-household claim. A new final configuration/protocol is still required.")
    if plan["schema_version"] == 3:
        summary["notice"] = "House 17 already inspected development; one predeclared L1_stable candidate with uniform existing +/-0.25 bias bound. Historical R2 negatives retained. No calibrated, real-user, final cross-household or power improvement claim."
    return summary


def render_report(output, summary):
    def number(v):
        return "—" if v is None else f"{v:.4f}"
    rows = "".join(f"<tr><td>{escape(g['variant'])}</td><td>{g['budget']}</td><td>{number(g['f1'])}</td><td>{number(g['f1_delta'])}</td><td>{number(g['f1_vs_random'])}</td></tr>" for g in summary["groups"])
    html = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>R2 有限选片比较</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}.notice{{background:#fff1d1;padding:18px}}</style>
<h1>R2 · {escape(summary['selected_development_learner'])} 有限选片比较</h1><p class="notice">{escape(summary['notice'])}</p>
<p>绝对性能排序候选：{escape(str(summary['selected_development_strategy']))}；确认开发出口：{summary['development_exit_passed']}；主预算 {summary['main_budget']} 次／户；原拟议 H2 开发目标：{summary['proposed_h2_development_target_met']}。</p>
<p>随机、静态不确定性、原完整拟议共享基础权重、候选池与预算。每五次从来源重新适配；静态策略一直用零确认分数选片。跳过和不确定回答均计费。状态分数未校准。</p>
<table><tr><th>策略</th><th>预算</th><th>主设备 F1</th><th>自身零确认 ΔF1</th><th>相对随机 ΔF1</th></tr>{rows}</table>
<p>选择依据为受保护的主预算绝对 F1。自身确认 +0.05 与选片相对随机 +0.02 分别记录，种子不等于家庭。</p>
<p><a href="comparison.csv">逐设备逐种子指标</a> · <a href="query_distribution.json">完整回答分布</a> · <a href="summary.json">逐户汇总、选择与 H1/H2</a></p></html>'''
    (output / "report.html").write_text(html, encoding="utf-8")
