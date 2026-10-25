"""Write the Chinese task3 conclusion from already frozen, verified results."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    sys.path.insert(0, str(run / "source"))
    import numpy as np
    from virtual_submeter.data.common import DataError, inventory, read_json, sha256, write_json
    from virtual_submeter.accuracy import microwave_background as b
    destinations = [root / f"docs/MICROWAVE_BACKGROUND_RESULTS_20261025.{suffix}" for suffix in ("md", "json")]
    curve_path = root / "docs/MICROWAVE_BACKGROUND_VALIDATION_CURVES_20261025.svg"
    if any(path.exists() for path in [*destinations, curve_path]):
        raise DataError("Results already exist; refuse overwrite")
    independent = read_json(run / "independent_verification.json")
    b.check_artifacts(run, read_json(run / "independent_delivery_manifest.json")["artifacts"])
    results, selection = read_json(run / "evaluation/results.json"), read_json(run / "source_selection.json")
    aggregates = []
    for house in b.HOUSES:
        for arm in b.ARMS:
            rows = [r for r in results["rows"] if r["partition"] == house+"/outer" and r["arm"] == arm]
            aggregates.append({"house": house, "arm": arm, "seeds": len(rows),
                               **{k: float(np.mean([r[k] for r in rows])) for k in ("precision", "recall", "f1", "ap")},
                               "fp_total": sum(r["fp"] for r in rows), "all_seed_recall_85": all(r["recall"] >= .85 for r in rows)})
    comparison_details = []
    for house in b.HOUSES:
        candidate = next(r for r in aggregates if r["house"] == house and r["arm"] == "event_background")
        for control in ("event", "natural"):
            baseline = next(r for r in aggregates if r["house"] == house and r["arm"] == control)
            pairs = [(next(r for r in results["rows"] if r["partition"] == house+"/outer" and r["arm"] == control and r["seed"] == seed),
                      next(r for r in results["rows"] if r["partition"] == house+"/outer" and r["arm"] == "event_background" and r["seed"] == seed)) for seed in b.SEEDS]
            comparison_details.append({"house": house, "control": control,
                "mean_f1_delta": candidate["f1"]-baseline["f1"], "fp_ratio": candidate["fp_total"]/baseline["fp_total"],
                "gates": {"all_seed_recall_85": candidate["all_seed_recall_85"], "mean_precision_08": candidate["precision"] >= .08,
                          "mean_f1_delta_02": candidate["f1"]-baseline["f1"] >= .02,
                          "fp_ratio_70": candidate["fp_total"] <= .7*baseline["fp_total"],
                          "two_seeds_p_and_f1_improve": sum(y["precision"] > x["precision"] and y["f1"] > x["f1"] for x,y in pairs) >= 2,
                          "no_seed_f1_drop_over_02": all(y["f1"]-x["f1"] >= -.02 for x,y in pairs)}})
    summary = {"status": "complete_failed_registered_improvement_gate", "new_fits": 3, "reused_control_fits": 6,
               "source_selection": selection, "outer_aggregates": aggregates, "comparison_details": comparison_details,
               "improvement_demonstrated": results["improvement_demonstrated"],
               "pool_support": read_json(run / "pool_support.json"), "independent_verification": independent,
               "files": {name: sha256(run / name) for name in ("registration.json", "training_complete.json", "freeze.json", "evaluation_complete.json", "delivery_manifest.json", "independent_verification.json", "independent_delivery_manifest.json")},
               "house17_or_final_house_accessed": False, "default_model_changed": False}
    svg = (run / "validation_curves.svg").read_text(encoding="utf-8")
    ticks = []
    for row in range(3):
        for column in range(3):
            left, top, width, height = 50+column*330, 70+row*255, 270, 190
            for value in (0, .5, 1):
                ticks.append(f'<text x="{left-25}" y="{top+height*(1-value)+4}" font-size="10">{value:g}</text>')
            for epoch in (1, 6, 12):
                ticks.append(f'<text x="{left+(epoch-1)*width/11-3}" y="{top+height+15}" font-size="10">{epoch}</text>')
            ticks.append(f'<text x="{left+100}" y="{top+height+28}" font-size="10">checkpoint</text>')
    curve_path.write_text(svg.replace('</svg>', '\n'.join(ticks)+'\n</svg>'), encoding="utf-8")
    text = "# 微波炉背景组合受控实验：结果\n\n2026-10-25，任务3已完成。严格新增3次拟合，复用任务2的6个对照；全部18份预测先冻结后统一评分。未追加搜索、未扩大数据、未修改默认模型。\n\n"
    text += "**结论：源内预选分数提高，但来源时间开发诊断未通过预登记改善门槛，不能认定跨时段预测效果改善。**背景依赖仍是待检验假设。House17及最终House8/9/20未访问。\n\n"
    text += "|源内方案|平均P|平均F1|资格|\n|---|---:|---:|---|\n"
    for row in selection["arms"]:
        text += f"|{row['arm']}|{row['mean_precision']:.4%}|{row['mean_f1']:.4%}|{row['eligible']}|\n"
    text += "\n源内按既定规则选中event_background；这是源内预选，不代表部署验收。event因平均F1低于natural而未获资格，背景候选没有据此修改8槽配方。\n\n"
    text += "|来源时间开发诊断|方案|三种子均P|均R|均F1|均AP|FP总数|\n|---|---|---:|---:|---:|---:|---:|\n"
    for row in aggregates:
        text += f"|{row['house']}|{row['arm']}|{row['precision']:.4%}|{row['recall']:.4%}|{row['f1']:.4%}|{row['ap']:.4f}|{row['fp_total']}|\n"
    text += "\nHouse2对event平均F1下降约1.81个百分点，召回明显下降；House19平均P/F1有所提高，但平均P不足8%、各种子召回均不足85%，FP比例也未达到70%。两户对两对照共4项整体改善判断全部失败。来源外层已使用过，仅是开发诊断，不能称独立泛化验证。\n\n"
    text += "训练75,222窗、236正中心；完整31点目标模板130中心（105/25），来自10个阈值段（7/3），其中原数据完整段8个（5/3）。关闭背景69,151窗（56,083/13,068）。其余106正中心没有完整标签上下文，严格回退原正窗，不补缺失。House2/19训练完整段总数14/4；House2第二内验证块完整段0，必须保留支持限制。\n\n"
    text += "|种子|尝试增强|实际增强|回退|占全部正槽|独立阈值段|\n|---|---:|---:|---:|---:|---:|\n"
    for seed in b.SEEDS:
        row = read_json(run / f"models/event_background/seed_{seed}/augmentation_summary.json")
        text += f"|{seed}|{row['attempts']}|{row['synthesized']}|{row['fallbacks']}|{row['actual_fraction_of_positive_slots']:.4%}|{row['unique_template_threshold_segments']}|\n"
    text += "\n所有实际增强均跨户。重复次数不能算新增事件；日期/事件均衡曝光使可增强尝试率不同于简单中心比例。\n\n"
    text += f"独立复核从训练原CSV重建{independent['templates_raw_checked']}个模板与{independent['backgrounds_raw_checked']}个背景，逐条复算全部84,672条尝试及16,152次实际加和；三个种子的原采样流和逐窗曝光与任务2event完全一致。普通模型接口复核18份概率容差及逐点状态一致，原始行推理一致；源内检查点/阈值可重选重放、配对初始化相同、归一化沿用真实训练统计。11项有意义测试及合成生命周期通过。\n\n"
    text += f"完整运行：`{run}`。主报告`report.md`；逐种子混淆/FPR/AP/PR/事件命中及配对误报片段在`evaluation/`；曲线`validation_curves.svg/.csv`，附坐标标记的版本见`docs/MICROWAVE_BACKGROUND_VALIDATION_CURVES_20261025.svg`；身份/源哈希/物理输出轨迹在`models/event_background/*/augmentation.jsonl`。登记、训练、冻结、评估、交付及独立复核各有哈希清单。稳定接口说明见`docs/MICROWAVE_BACKGROUND_INTERFACE_20261025.md`；后续全来源重训须另行登记，不能把本开发诊断转换成独立测试。\n"
    destinations[0].write_text(text, encoding="utf-8")
    write_json(destinations[1], summary)
    write_json(root / "docs/MICROWAVE_BACKGROUND_DELIVERY_20261025.json", {"run": str(run),
               "artifacts": {str(path.relative_to(root)).replace('\\', '/'): sha256(path) for path in [*destinations, curve_path,
                   root / "docs/MICROWAVE_BACKGROUND_PLAN_20261025.md", root / "docs/MICROWAVE_BACKGROUND_INTERFACE_20261025.md",
                   root / "configs/microwave_background_20261025.json", root / "src/virtual_submeter/accuracy/microwave_background.py",
                   root / "scripts/run_microwave_background.py", root / "scripts/verify_microwave_background.py",
                   Path(__file__), root / "tests/test_microwave_background.py"]}})
    print(json.dumps({"status": summary["status"], "results": str(destinations[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
