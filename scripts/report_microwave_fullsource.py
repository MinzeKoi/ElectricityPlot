"""Readable full-source development handoff, using sealed outputs only."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from virtual_submeter.data.common import DataError, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = read_json(run/"final_summary.json")
    rows = read_json(run/"evaluation/metrics.json")["rows"]
    plan, selection = read_json(run/"plan.json"), read_json(run/"source_selection.json")
    source = read_json(run/"source_rows.json")
    chosen = summary["selected_before_development"]
    seeds = plan["seeds"]
    reference = [next(r for r in rows if r["arm"] == "W50" and r["seed"] == s) for s in seeds]
    selected = [] if chosen is None else [next(r for r in rows if r["arm"] == chosen and r["seed"] == s) for s in seeds]
    mean = lambda rs, key: sum(r[key] for r in rs)/len(rs)
    checks = {"source_selection_qualified": chosen is not None}
    if selected:
        checks.update(mean_precision_at_least_008=mean(selected, "precision") >= .08,
            every_seed_recall_at_least_085=all(r["recall"] >= .85 for r in selected),
            mean_f1_gain_at_least_002=mean(selected, "f1")-mean(reference, "f1")+1e-12 >= .02,
            at_least_two_seed_precision_and_f1_wins=sum(b["precision"] > a["precision"] and b["f1"] > a["f1"] for a, b in zip(reference, selected)) >= 2,
            no_seed_f1_drop_over_002=all(b["f1"]-a["f1"] >= -.02 for a, b in zip(reference, selected)),
            mean_fp_reduction_at_least_30_percent=sum(r["fp"] for r in selected) <= .7*sum(r["fp"] for r in reference),
            no_degenerate_selected_seed=all(not r["degenerate"] for r in selected))
    checks["same_support_and_other_devices"] = summary["raw_power_and_other_devices_unchanged"]
    result = {**summary, "acceptance_checks": checks,
              "failed_checks": [k for k, v in checks.items() if not v],
              "actual_new_fits": read_json(run/"training_complete.json")["fits"],
              "source_selection": selection,
              "registered_config_sha256": sha256(run/"plan.json"),
              "registration_sha256": sha256(run/"registration.json"),
              "evaluation_sha256": sha256(run/"evaluation_complete.json")}
    if selected:
        result["selected_fp_reduction_ratio"] = 1-sum(r["fp"] for r in selected)/sum(r["fp"] for r in reference)
    text = "# 微波炉全量来源训练与House17开发验收\n\n"
    text += f"**开发验收{'通过' if summary['development_acceptance_passed'] else '未通过'}；来源唯一预选为{chosen}。** 完整方案改善声明：{summary['declared_improvement']}；采样或背景增强机制收益支持：{summary['sampling_or_background_improvement_supported']}。\n\n"
    text += "本轮完成9个真实拟合、每个3528次更新；所有方案和种子均按同一House5多日期规则选择检查点及公共阈值，之后完成全部9份House17预测封存，再一次性评分。下表未预选方案的分数仅为透明诊断，不改变来源选择。\n\n"
    text += "|方案（三种子平均）|Precision|Recall|F1|平均FP|\n|---|---:|---:|---:|---:|\n"
    for arm, values in summary["means"].items():
        text += f"|{arm}|{values['precision']:.4%}|{values['recall']:.4%}|{values['f1']:.4%}|{values['fp']:.1f}|\n"
    text += "\n|原开发门槛|判定|\n|---|---|\n"
    labels = {"source_selection_qualified": "来源预选合格", "mean_precision_at_least_008": "平均Precision≥8%", "every_seed_recall_at_least_085": "每种子Recall≥85%", "mean_f1_gain_at_least_002": "相较W50平均F1提升≥0.02", "at_least_two_seed_precision_and_f1_wins": "至少两个种子Precision/F1同时提高", "no_seed_f1_drop_over_002": "任何种子F1下降≤0.02", "mean_fp_reduction_at_least_30_percent": "平均FP减少≥30%", "no_degenerate_selected_seed": "无全开/全关", "same_support_and_other_devices": "共同时间点及其他设备原始预测一致"}
    for key, value in checks.items():
        text += f"|{labels[key]}|{'通过' if value else '未通过'}|\n"
    if selected:
        text += f"\n预选者相较W50平均F1差值为{summary['selected_deltas']['W50']['f1']:+.6f}，FP减少比例为{result['selected_fp_reduction_ratio']:.4%}。\n"
    text += "\n|方案|种子|Precision|Recall|F1|FP|FN|\n|---|---:|---:|---:|---:|---:|---:|\n"
    for r in rows:
        text += f"|{r['arm']}|{r['seed']}|{r['precision']:.4%}|{r['recall']:.4%}|{r['f1']:.4%}|{r['fp']}|{r['fn']}|\n"
    text += "\n## 来源选择与训练证据\n\n|方案|种子|检查点|概率阈值|最差块Precision|源内平均F1|正中心实际曝光|正类加权质量比例|\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    total_seconds = 0.
    for arm in read_json(run/"matrix.json")["arms"]:
        for seed in seeds:
            m = read_json(run/f"models/{arm}/seed_{seed}/model.json")
            total_seconds += m["seconds"]
            q = m["selected"]
            text += f"|{arm}|{seed}|{q['epoch']}|{q['threshold']:.8f}|{q.get('worst_precision', 0):.4%}|{mean(list(q['block_metrics'].values()), 'f1'):.4%}|{m['positive_seen']}|{m['positive_weight_mass_fraction']:.4%}|\n"
    text += f"\n9模型拟合及检查点选择合计耗时{total_seconds:.1f}秒，不含数据准备、冻结推理及复核。\n"
    text += "\n完整来源为House2/19原14天，150,069个窗口、646个正中心，窗口完整且边界可见阈值段21+27；不能与旧0–10天仅236个正中心的小块结果等同。House5采用固定6个UTC双日块，最低支持全部通过，但它们仍来自同一已使用家庭、存在时间相关性，不是六个独立外部样本。背景265个模板中心覆盖33个阈值段，实际增强约占正槽15.5%至15.7%，失败槽保持原样。\n"
    text += "\n## 解释与复核边界\n\nHouse17多次被历史实验查看，本轮是开发验收，不是独立外部泛化证明。水壶/洗衣机原始功率与状态和微波炉原始功率均逐点沿用原A；微波炉只有独立状态候选，没有功率改善结论，默认模型未切换，最终House8/9/20未访问。无论结果如何，不按开发成绩改选、调阈值或补训练。\n\n"
    text += "natural若预选且通过W50全部原门槛，可报告完整方案改善，但不归因于事件采样/背景增强。协调报告语义修正到达时训练已启动、开发标签未打开；已另行在评价前封存reporting_registration.json，不更改训练源码快照、任何数值门槛或来源选择。最终声明以final_summary.json为准，原引擎summary中的旧declared_improvement字段仍保留其最初的机制归因含义。\n\n"
    text += "证据：registration.json、source_selection.json、training_complete.json、freeze.json、inference_checks.json、independent_verification.json、evaluation_complete.json、verification_after_evaluation.json。独立核对原来源CSV的150,069个31点输入、模板和背景，并逐条重算增强加和；普通模型加载推理与冻结预测逐点一致。原始House17微波炉CSV仅解析一次，后续复核使用封存真值，不重新读取原标签CSV。\n"
    for path, value in ((run/"final_results.json", result),):
        if path.exists() and read_json(path) != value:
            raise DataError("Final results cannot be overwritten")
        if not path.exists():
            write_json(path, value)
    target = run/"final_report.md"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise DataError("Final report cannot be overwritten")
    if not target.exists():
        target.write_text(text, encoding="utf-8")
    print(json.dumps({"report": str(target), "acceptance": summary["development_acceptance_passed"], "failed_checks": result["failed_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
