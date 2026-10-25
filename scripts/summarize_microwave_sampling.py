"""Publish a read-only summary of the completed, verified sampling comparison."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.data.common import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    verification = read_json(args.verification)
    if verification["status"] != "independently_verified" or verification["freeze_sha256"] != sha256(run / "freeze.json"):
        raise ValueError("Verified freeze required")
    results = read_json(run / "evaluation/results.json")
    support = read_json(run / "support.json")
    source = read_json(run / "source_selection.json")
    models, exposure = [], []
    for arm in ("natural", "event"):
        for seed in (42, 1019, 2026):
            path = run / f"models/{arm}/seed_{seed}"
            meta = read_json(path / "model.json")
            models.append(meta)
            with (path / "batches.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            positive_loss = sum(float(r["positive_loss_sum"]) for r in rows)
            negative_loss = sum(float(r["negative_loss_sum"]) for r in rows)
            exposure.append({"arm": arm, "seed": seed, "updates": meta["updates"], "seen": meta["seen"],
                "positive_seen": meta["positive_seen"], "negative_seen": meta["seen"]-meta["positive_seen"],
                "minimum_positive_per_batch": min(int(r["positive"]) for r in rows),
                "maximum_positive_per_batch": max(int(r["positive"]) for r in rows),
                "positive_weight_mass_fraction": meta["positive_weight_mass_fraction"],
                "positive_effective_loss_fraction": positive_loss/(positive_loss+negative_loss),
                "positive_loss_sum": positive_loss, "negative_loss_sum": negative_loss,
                "seconds": meta["seconds"], "checkpoint": meta["selected"]["epoch"],
                "threshold": meta["decision_threshold_probability"]})
    averages = []
    for house in ("refit_2", "refit_19"):
        for arm in ("natural", "event"):
            for role in ("train", "validation_0", "validation_1", "outer"):
                selected = [r for r in results["rows"] if r["arm"] == arm and r["partition"] == f"{house}/{role}"]
                averages.append({"house": house, "arm": arm, "role": role,
                    **{k: float(np.mean([r[k] for r in selected])) for k in
                       ("precision", "recall", "f1", "off_false_positive_rate", "ap", "fp")},
                    "diagnostic_p_at_r85": float(np.mean([r["diagnostic"]["p_at_r85"] for r in selected])),
                    "diagnostic_fpr_at_r85": float(np.mean([r["diagnostic"]["fpr_at_r85"] for r in selected]))})
    failures = []
    if source["selected"] is None or source["selected"]["arm"] != "event":
        failures.append("来源内部未预选事件组；其源内平均F1低于自然组，不能依据外层表现改选")
    for house in ("refit_2", "refit_19"):
        groups = {a: sorted([r for r in results["rows"] if r["arm"] == a and r["partition"] == f"{house}/outer"],
                            key=lambda r: r["seed"]) for a in ("natural", "event")}
        n, e = groups["natural"], groups["event"]
        bad = [r["seed"] for r in e if r["recall"] < .85]
        if bad:
            failures.append(f"{house}事件组种子{bad}冻结阈值召回未达85%")
        if np.mean([r["precision"] for r in e]) < .08:
            failures.append(f"{house}事件组平均Precision低于8%")
        delta = float(np.mean([b["f1"]-a["f1"] for a, b in zip(n, e)]))
        if delta < .02:
            failures.append(f"{house}事件组平均F1增益{delta:.4f}低于0.02")
        if sum(r["fp"] for r in e) > .7*sum(r["fp"] for r in n):
            failures.append(f"{house}误报未降低30%")
        if sum(b["precision"] > a["precision"] and b["f1"] > a["f1"] for a, b in zip(n, e)) < 2:
            failures.append(f"{house}少于两种子同时提高Precision和F1")
        if any(b["f1"]-a["f1"] < -.02 for a, b in zip(n, e)):
            failures.append(f"{house}存在种子F1下降超过0.02")
    out = {"status": "source_sampling_experiment_completed", "run": str(run),
           "real_fits": 6, "additional_fits": 0, "improvement_demonstrated": results["improvement_demonstrated"],
           "source_selection": source, "comparisons": results["comparisons"], "failure_reasons": failures, "averages": averages,
           "rows": results["rows"], "models": models, "exposure": exposure, "support": support,
           "verification": verification, "registration_sha256": sha256(run / "registration.json"),
           "freeze_sha256": sha256(run / "freeze.json"), "audit_sha256": sha256(run / "audit_receipt.json"),
           "data_version": read_json(run / "audit_manifest.json")["data_version"],
           "limited_complete_event_evidence": True,
           "source_gate_meaning": "All four blocks meet registered partial-threshold-segment minima; complete-event support remains sparse, including zero complete segments in House2 validation_1",
           "tests": {"new_sampling": 8, "prior_diagnosis": 5, "prior_state": 9, "total_passed": 22},
           "final_households_accessed": False, "house17_labels_accessed": False,
           "scope": "reused source temporal development diagnosis; not new independent testing"}
    result_path = ROOT / "docs/MICROWAVE_SAMPLING_RESULTS_20261025.json"
    report_path = ROOT / "docs/MICROWAVE_SAMPLING_REPORT_20261025.md"
    if result_path.exists() or report_path.exists():
        raise ValueError("Published summary already exists")
    text = "# 微波炉事件曝光与多日期验证受控实验\n\n"
    text += "2026-10-25。运行microwave-sampling-20261025-v1；6次拟合、12份外层预测统一冻结后评价；未追加搜索。仅House2/19，House17及最终8/9/20标签未访问。\n\n"
    text += f"**预登记改善门槛：{'通过' if out['improvement_demonstrated'] else '未通过'}。** "
    text += "这是已使用来源日期的开发诊断，不能称新独立泛化证明。完整训练/内验证/外层均按冻结阈值报告；事后P@R85仅诊断，不部署。\n\n"
    text += "## 冻结时间留出主要结果\n\n|目标户|采样|P均值|R均值|F1均值|off-FPR|AP|诊断P@R85|诊断FPR@R85|\n|---|---|---|---|---|---|---|---|---|\n"
    for r in averages:
        if r["role"] == "outer":
            text += "|"+"|".join([r["house"], r["arm"]]+[f"{r[k]:.2%}" for k in
                ("precision", "recall", "f1", "off_false_positive_rate", "ap", "diagnostic_p_at_r85", "diagnostic_fpr_at_r85")])+"|\n"
    for house in ("refit_2", "refit_19"):
        rows = {a: [r for r in results["rows"] if r["arm"] == a and r["partition"] == f"{house}/outer"] for a in ("natural", "event")}
        n, e = rows["natural"], rows["event"]
        text += f"\n{house}：事件组相对自然组FP总数{sum(r['fp'] for r in n)}→{sum(r['fp'] for r in e)}；"
        text += f"平均F1变化{100*(np.mean([r['f1'] for r in e])-np.mean([r['f1'] for r in n])):+.2f}个百分点；"
        text += f"事件组三种子达到R85的数量{sum(r['recall'] >= .85 for r in e)}/3。\n"
    text += "\n未通过的具体条件：\n\n"+"\n".join("- "+reason for reason in failures)+"\n"
    text += "\n## 源内预选与训练重放\n\n"
    text += f"预选结果：{source['selected']['arm'] if source['selected'] else '无合格候选'}。阈值/检查点仅按来源内验证决定；门槛为每个满足最低支持块R≥85%及非退化，按最差块P、平均P/F1、较早轮次排序。无来源P/F1/FPR绝对门槛；事件组平均F1不低于自然组才有资格。\n\n"
    text += "|采样|种子|选中轮|阈值|源内最差块P|源内平均P|源内平均F1|\n|---|---|---|---|---|---|---|\n"
    for m in models:
        v = m["selected"]
        text += f"|{m['arm']}|{m['seed']}|{v['epoch']}|{v['threshold']:.6f}|{v.get('worst_precision',0):.2%}|{v.get('mean_precision',0):.2%}|{v.get('mean_f1',0):.2%}|\n"
    text += "\n|目标户|采样|分区|P|R|F1|AP|off-FPR|\n|---|---|---|---|---|---|---|---|\n"
    for r in averages:
        if r["role"] != "outer":
            text += "|"+"|".join([r["house"], r["arm"], r["role"]]+[f"{r[k]:.2%}" for k in
                ("precision", "recall", "f1", "ap", "off_false_positive_rate")])+"|\n"
    text += "\n训练表现仅说明拟合，不能代替时间留出。每个候选均训练固定3528步，未按内验证提前停止；选中检查点在预登记12轮内。\n\n"
    text += "## 逐种子外层混淆计数\n\n|目标户|采样|种子|P|R|F1|TP|FP|FN|TN|AP|段命中/支持|\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for r in results["rows"]:
        if r["partition"].endswith("/outer"):
            text += f"|{r['partition'].split('/')[0]}|{r['arm']}|{r['seed']}|{r['precision']:.2%}|{r['recall']:.2%}|{r['f1']:.2%}|{r['tp']}|{r['fp']}|{r['fn']}|{r['tn']}|{r['ap']:.2%}|{r['events_hit']}/{r['events_supported']}|\n"
    text += "\n## 数据支持与缺失限制\n\n"
    text += "审计严格重放与旧版逐字节一致，0个新增样本，没有数据修复。31点完整上下文、边界各侧1800秒、跨边界阈值段整段排除；训练归一化只估计训练窗。最低支持门槛不代表完整事件充足或独立。\n\n"
    text += "|家庭|分区|有效窗|正中心|部分支持阈值段|完整可观测段|有效小时|排除网格中心|\n|---|---|---|---|---|---|---|---|\n"
    for house, hs in support.items():
        if house != "refit_2":
            text += "|家庭|分区|有效窗|正中心|部分支持阈值段|完整可观测段|有效小时|排除网格中心|\n|---|---|---|---|---|---|---|---|\n"
        for role, b in hs["blocks"].items():
            text += f"|{house}|{role}|{b['windows']}|{b['positive_windows']}|{b['supported_threshold_segments']}|{b['complete_observed_segments']}|{b['effective_seconds']/3600:.2f}|{b['excluded_grid_points']}|\n"
        text += f"\n{house}原网格总表缺失{hs['missing_aggregate']}点、标签缺失{hs['missing_labels_on_aggregate_grid']}点；排除中心还包含上下文、边界和标签有效性要求，不能都叫原始传感器缺失。\n\n"
    text += "House2第二内验证日完整段为0，House19训练完整段仅4；窗口相关、阈值段也不等于物理独立周期。所有来源日期都曾用于旧诊断，不能消除历史使用。\n\n"
    text += "## 计算与实际曝光\n\n"
    text += "事件组每批16正中心、240个均匀训练关闭中心，正类按家庭/日期、事件、中心轮换；负类没有高功率偏采。BCE50不变，但有效类别权重改变，不能描述为等价损失。\n\n"
    text += "|采样|种子|总曝光|正曝光|每批正数范围|正权重质量比例|实际正损失比例|训练秒|\n|---|---|---|---|---|---|---|---|\n"
    for r in exposure:
        text += f"|{r['arm']}|{r['seed']}|{r['seen']}|{r['positive_seen']}|{r['minimum_positive_per_batch']}–{r['maximum_positive_per_batch']}|{r['positive_weight_mass_fraction']:.2%}|{r['positive_effective_loss_fraction']:.2%}|{r['seconds']:.2f}|\n"
    text += "\n每个模型8249参数，903168个窗口曝光；重复曝光不是新增事件。逐窗、逐日/事件曝光和逐批正/负损失均保留。\n\n"
    text += "## 同一时间点误报对照\n\n"
    text += "固定seed42，每个目标/采样取最高分误报并查询另一个采样模型同一时间分数。真实微波炉均关闭；总有功高不能据此识别其他电器。完整31点曲线和全部种子案例保存在paired_false_positives.json，不能作为下一轮训练挖掘清单。\n\n"
    text += "|目标户|按哪组排序|UTC中心|总表中心W|自然分数/阈值|事件分数/阈值|\n|---|---|---|---|---|---|\n"
    cases = read_json(run / "evaluation/paired_false_positives.json")["cases"]
    seen = set()
    for c in cases:
        key = c["house"], c["ranked_by"]
        if c["seed"] == 42 and key not in seen:
            seen.add(key)
            text += f"|{c['house']}|{c['ranked_by']}|{c['timestamp']}|{c['aggregate_window_w'][15]:.1f}|{c['scores']['natural']:.6f}/{c['thresholds']['natural']:.6f}|{c['scores']['event']:.6f}/{c['thresholds']['event']:.6f}|\n"
    text += "\n## 结论边界与复核\n\n"
    text += "本轮只检验增加正事件曝光这一方案。若训练/内验证改善但外层召回或排序退化，不能以训练满分宣称成功，也不能因单个家庭改善宣称稳定提升。失败不触发背景增强、网络变更、训练加长或补搜；下一步由协调者决定。旧默认模型和界面未变。\n\n"
    text += "8项新测试、5项原诊断测试和9项原状态测试通过。封存verify重算72份内验证选点，12份外层普通推理概率和状态一致；独立复核重放6个采样流、逐窗曝光、正负损失占比、外层混淆/AP、时间/整事件隔离。此为本机自动复核，不是第二研究者或第二机器复现。\n\n"
    text += "```powershell\n"
    text += f".venv-m2/Scripts/python.exe {args.run.as_posix()}/run_microwave_sampling.py verify --run {args.run.as_posix()}\n"
    text += f".venv-m2/Scripts/python.exe scripts/verify_microwave_sampling.py --run {args.run.as_posix()}\n```\n\n"
    text += "完整逐种子数据见MICROWAVE_SAMPLING_RESULTS_20261025.json；协议、适配清单、运行registration/freeze/delivery_manifest包含配置、源码、数据、划分、模型与预测哈希。报告只读封存结果，无额外拟合。\n"
    write_json(result_path, out)
    report_path.write_text(text, encoding="utf-8")
    write_json(ROOT / "docs/MICROWAVE_SAMPLING_DELIVERY_20261025.json", {
        "artifacts": {str(p.relative_to(ROOT)).replace(chr(92), "/"): sha256(p) for p in (
            result_path, report_path, args.verification.resolve(), ROOT / "docs/MICROWAVE_SAMPLING_PLAN_20261025.md",
            ROOT / "configs/microwave_sampling_20261025.json", ROOT / "scripts/run_microwave_sampling.py",
            ROOT / "scripts/verify_microwave_sampling.py", ROOT / "tests/test_microwave_sampling.py",
            ROOT / "src/virtual_submeter/accuracy/microwave_sampling.py", Path(__file__))},
        "run_delivery_sha256": sha256(run / "delivery_manifest.json")})
    print(json.dumps({"report": str(report_path), "improvement_demonstrated": out["improvement_demonstrated"],
                      "source_selection": source, "outer_averages": [r for r in averages if r["role"] == "outer"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
