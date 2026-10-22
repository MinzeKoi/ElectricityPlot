"""Build small R1 handoff evidence without reopening any measurement labels."""

import argparse
import csv
from html import escape
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.r1 import verify
from virtual_submeter.accuracy.fixed_queries import verify as verify_fixed
from virtual_submeter.data.common import DataError, read_json, sha256, write_json


def table(rows):
    if not rows:
        return "<p>无记录</p>"
    fields = list(rows[0])
    return "<table><tr>" + "".join(f"<th>{escape(k)}</th>" for k in fields) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{escape(str(r[k]))}</td>" for k in fields) + "</tr>" for r in rows) + "</table>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, default=ROOT / "configs/r1_acceptance_20261022.json")
    args = parser.parse_args()
    if args.output.exists():
        raise DataError("Summary destination must be fresh")
    baseline_check, fixed_check = verify(args.baseline), verify_fixed(args.fixed)
    state = read_json(args.baseline / "run.json")
    summary = read_json(args.fixed / "summary.json")
    support = read_json(args.baseline / "protocol/admission.json")
    baseline_rows = []
    training_costs = []
    for case in state["cases"]:
        cid, seed = case["candidate"], case["seed"]
        if seed == 42:
            model_dir = args.baseline / f"candidates/{cid}/model"
            component_dir = model_dir.parent / "model_components"
        else:
            model_dir = args.baseline / f"selected/seed_{seed}"
            component_dir = args.baseline / f"selected/seed_{seed}_components"
        model_meta = read_json(model_dir / "model.json")
        component_metadata = [read_json(p) for p in component_dir.glob("*/model.json")] if cid == "single_mae" else [model_meta]
        training_costs.append({"candidate": cid, "seed": seed, "independent_trainings": len(component_metadata),
                               "total_fit_seconds": sum(m["timing_seconds"]["fit"] for m in component_metadata),
                               "total_training_pipeline_seconds": sum(m["timing_seconds"]["total"] for m in component_metadata),
                               "packed_parameter_count": model_meta["parameter_count"],
                               "component_parameter_count": sum(m["parameter_count"] for m in component_metadata),
                               "checkpoint_epochs": [m["selected_epoch"] for m in component_metadata]})
        for e in case["evaluations"]:
            m = e["raw"]
            baseline_rows.append({"candidate": case["candidate"], "seed": case["seed"], "house": e["household_id"],
                                  "device": e["appliance"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                                  "active_mae_w": m["active_mae_w"], "energy_error_kwh": m["energy"]["absolute_error_kwh"],
                                  "valid_samples": m["valid_samples"], "predicted_on_samples": m["predicted_on_samples"],
                                  "actual_on_samples": m["actual_on_samples"], "complete_events": e["events"]["complete_events"],
                                  "complete_detected": e["events"]["complete_detected"], "eligible": case["selection"]["eligible"]})
    groups = summary["groups"]
    query = read_json(args.fixed / "query_distribution.json")
    metrics = read_json(args.fixed / "metrics.json")
    main_budget = summary["main_budget"]
    paired = []
    for variant in ("L1", "L2_max"):
        for source in (42, 1019, 2026):
            for seed in (42, 1019, 2026):
                rows = [m for m in metrics["rows"] if m["variant"] == variant and m["source_seed"] == source
                        and m["query_seed"] == seed and m["budget"] == main_budget and m["appliance"] in ("kettle", "microwave")]
                paired.append({"variant": variant, "source_seed": source, "query_seed": seed,
                               "f1_delta": sum(m["f1_delta"] for m in rows) / len(rows) if rows and all(m["f1_delta"] is not None for m in rows) else None,
                               "new_degenerate_devices": sum(m["adaptation_caused_degeneracy"] for m in rows)})
    learner_rows = [{"learner": g["variant"], "budget": g["budget"], "f1": g["f1"], "own_baseline_delta": g["f1_delta"],
                     "new_degenerate_cases": g["adaptation_degenerate_cases"], "device_drops": g["device_drops_over_tolerance"]} for g in groups]
    degeneracy = []
    for g in groups:
        rows = [m for m in metrics["rows"] if m["variant"] == g["variant"] and m["budget"] == g["budget"]]
        degeneracy.append({"variant": g["variant"], "budget": g["budget"], "device_cases": len(rows),
                           "primary_degenerate_device_cases": sum(m["degenerate"] for m in rows if m["appliance"] in ("kettle", "microwave")),
                           "extension_degenerate_device_cases": sum(m["degenerate"] for m in rows if m["appliance"] == "washing_machine")})
    acceptance = read_json(args.acceptance)
    repeatability = []
    for variant in ("L1", "L2_max"):
        group = next(g for g in groups if g["variant"] == variant and g["budget"] == main_budget)
        pairs = [p for p in paired if p["variant"] == variant]
        positives = sum(p["f1_delta"] is not None and p["f1_delta"] > 0 for p in pairs)
        passed = (len(pairs) == 9 and positives >= acceptance["minimum_positive_seed_pairs_per_household"]
                  and group["f1_delta"] is not None and group["f1_delta"] > 0
                  and not group["adaptation_degenerate_cases"] and not group["device_drops_over_tolerance"])
        repeatability.append({"variant": variant, "positive_pairs": positives, "total_pairs": len(pairs), "r1_development_exit_passed": passed})
    structure = {"source_state_head_f1_delta_at_zero_confirmation": next(g["f1"] for g in groups if g["variant"] == "L2_max" and g["budget"] == 0)
                 - next(g["f1"] for g in groups if g["variant"] == "L1" and g["budget"] == 0),
                 "notice": "Source state head/features only; excluded from confirmation gain."}
    r1_internal_met = any(r["r1_development_exit_passed"] and next(g["f1_delta"] for g in groups if g["variant"] == r["variant"] and g["budget"] == main_budget)
                          >= acceptance["minimum_primary_f1_gain_internal"] for r in repeatability)
    r1_eligible = [g for g in groups if g["budget"] == main_budget and any(r["variant"] == g["variant"] and r["r1_development_exit_passed"] for r in repeatability)]
    r1_selected = max(r1_eligible, key=lambda g: g["f1_delta"])["variant"] if r1_eligible else None
    source_zero = next(g["f1"] for g in groups if g["variant"] == "L1" and g["budget"] == 0)
    l2_zero = next(g["f1"] for g in groups if g["variant"] == "L2_max" and g["budget"] == 0)
    l2_main = next(g["f1"] for g in groups if g["variant"] == "L2_max" and g["budget"] == main_budget)
    best_absolute = max((g for g in groups if g["budget"] == main_budget), key=lambda g: g["f1"])
    absolute_audit = {"unadapted_source_f1": source_zero, "l2_zero_f1": l2_zero, "l2_main_f1": l2_main,
                      "l2_main_minus_unadapted_source": l2_main - source_zero,
                      "highest_absolute_f1_learner_at_main_budget": best_absolute["variant"], "highest_absolute_f1": best_absolute["f1"],
                      "selection_by_own_gain_rule_is_unchanged": r1_selected,
                      "post_evaluation_interpretation": "Own-baseline confirmation gain does not offset source-state-head regression. Preserve the predeclared gate result, but do not promote L2 as the better end-to-end method; retain the stronger absolute reference and diagnose source-head transfer before a new main-method freeze."}
    result = {"schema_version": 1, "task": "R1", "formal_evaluation": False, "development_households": ["17"],
              "selected_candidate": state["selected_candidate"], "source_seeds": [42, 1019, 2026],
              "baseline": baseline_check, "fixed_comparison": fixed_check, "baseline_rows": baseline_rows,
              "training_costs": training_costs,
              "degeneracy_by_budget": degeneracy,
              "training_timing_notice": "Packed model metadata inherits its first component timing; aggregate costs here sum every component model timing. Zero cross-block parameters increase packed storage/compute and are not independently trained.",
              "support": support, "confirmation_summary": summary, "structure_gain": structure, "paired_main_budget": paired,
              "repeatability": repeatability, "r1_development_exit_passed": any(r["r1_development_exit_passed"] for r in repeatability),
              "r1_internal_gain_target_met": r1_internal_met,
              "r1_selected_development_learner": r1_selected,
              "absolute_performance_audit": absolute_audit,
              "acceptance_rule": acceptance,
              "query_distribution": query, "cases": len(read_json(args.fixed / "run.json")["cases"]), "device_metric_rows": len(metrics["rows"]),
              "unsure_semantics": "Offline oracle returns unsure only for incomplete label support; it imposes no minimum complete-event duration. These are not reports of user unwillingness or measured recall errors.",
              "limits": ["One new inspected development household; historical House 3 has PV limitations.",
                         "No final H1/H2/H3 acceptance; final 8/9/20 excluded.",
                         "Power is frozen across target confirmations.", "Independent member/second machine reproduction pending."],
              "inputs_sha256": {"baseline_evidence": sha256(args.baseline / "evidence.json"), "fixed_freeze": sha256(args.fixed / "freeze.json"),
                                "acceptance": sha256(args.acceptance),
                                "fixed_summary": sha256(args.fixed / "summary.json"), "fixed_metrics": sha256(args.fixed / "metrics.json")}}
    write_json(args.output / "results.json", result)
    for name, rows in (("baseline.csv", baseline_rows), ("learners.csv", learner_rows), ("paired.csv", paired)):
        with (args.output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>R1 开发结果</title>'
            '<style>body{font:15px system-ui;margin:32px;color:#18243b}table{border-collapse:collapse;margin:20px 0}td,th{padding:7px;border:1px solid #ccd2df}th{background:#edf1ff}p{max-width:1000px}</style>'
            '<h1>R1 基础识别与固定确认</h1><p>来源 2/19，验证 5，开发 17。最终 8/9/20 未进入本轮。模拟回答、单开发家庭，不是最终跨户效果验收。</p>'
            f'<p>选定模型：{escape(state["selected_candidate"])}；主预算 {main_budget} 次/户；R1 可重复性开发出口 {any(r["r1_development_exit_passed"] for r in repeatability)}；R1 +0.05 内部目标 {r1_internal_met}。</p>'
            f'<p style="background:#fff0df;padding:16px"><strong>关键负结果：</strong>L2 来源状态头将零确认 F1 从 {source_zero:.4f} 降到 {l2_zero:.4f}，确认后 {l2_main:.4f} 仍低于未适配来源模型。按主预算绝对 F1，{escape(best_absolute["variant"])} 最高，为 {best_absolute["f1"]:.4f}。L2 相对较差自身基线的恢复不能当作整体优于原模型；预存的自身增益判据保持原样。</p>'
            '<h2>固定确认：相对各自零确认</h2>' + table(learner_rows) + '<h2>逐来源/查询种子主设备配对差值</h2>' + table(paired)
            + '<h2>R1 可重复性：固定九个种子组合至少六个为正</h2>' + table(repeatability)
            + '<h2>全部全关/全开检查：含零确认与扩展设备</h2>' + table(degeneracy)
            + '<h2>基础识别：统一支持的完整候选与种子</h2>' + table(baseline_rows)
            + '<h2>实际训练成本与参数量</h2>' + table(training_costs)
            + '<h2>回答分布</h2>' + table(query["rows"])
            + '<p>unsure 来自模拟代理缺失标签支持，不等于用户不愿回答。事件按固定阈值连续段计数；分表缺口造成截断，不能称物理使用周期。功率/电量不随目标确认改变。所有负结果保留。</p>'
            + '<p><a href="results.json">完整证据摘要</a> · <a href="baseline.csv">基础比较</a> · <a href="learners.csv">学习器比较</a> · <a href="paired.csv">配对差值</a></p></html>')
    (args.output / "report.html").write_text(page, encoding="utf-8")
    print({"output": str(args.output), "development_exit_passed": summary["development_exit_passed"], "internal_gain_target_met": summary["internal_gain_target_met"]})


if __name__ == "__main__":
    main()
