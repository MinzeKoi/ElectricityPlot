"""Publish a Chinese handoff from already sealed state results; never opens labels."""
import argparse
import json
from pathlib import Path
import sys
from html import escape
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.accuracy.state_experiment import verify, GROUPS, SEEDS, DEVELOPMENT
from virtual_submeter.data.common import read_json, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/accuracy/state-20261024-v2")
    args = parser.parse_args()
    run = args.run
    checked = verify(run)
    summary = read_json(run / "summary.json")
    rows = read_json(run / "metrics.json")["rows"]
    means = {}
    for support in ("native", "common61"):
        for candidate in ["W50", "W50_THR", "R_FIXED"] + GROUPS:
            for appliance in ("kettle", "microwave", "washing_machine"):
                rs = [r for r in rows if r["partition"] == DEVELOPMENT and r["support"] == support
                      and r["candidate"] == candidate and r["appliance"] == appliance]
                values = {k: sum(r[k] for r in rs) / 3 if all(r[k] is not None for r in rs) else None
                          for k in ("precision", "recall", "f1", "false_positive", "false_negative", "mae_w", "active_mae_w", "energy_absolute_error_kwh")}
                means[f"{support}/{candidate}/{appliance}"] = values
    selection = read_json(run / "selection.json")
    checkpoints = {}
    for g in GROUPS:
        for s in SEEDS:
            p = run / f"models/{g}/seed_{s}"
            meta = read_json(p / "model.json")
            checkpoints[f"{g}/{s}"] = {k: meta[k] for k in ("selected_epoch", "optimizer_steps", "parameter_count", "source_train_windows", "source_validation_windows")}
            checkpoints[f"{g}/{s}"].update(
                decision_threshold=meta["source_state_selection"]["point"]["threshold"] if g == "R_PR" else meta["decision_threshold_probability"],
                model_sha256=sha256(p / "model.json"), history_sha256=sha256(p / "history.json"))
    result = {"status": "state_development_completed", "run": str(run), "verification": checked,
              "freeze_sha256": sha256(run / "freeze.json"), "evidence_sha256": sha256(run / "evidence.json"),
              "execution_code_sha256": read_json(run / "registration.json")["code_sha256"],
              "source_selection": selection, "summary": summary, "development_means": means,
              "checkpoints": checkpoints}
    write_json(ROOT / "docs/STATE_RESULTS_20261024.json", result)
    gate = summary["common_support_gates"][summary["selected"]]
    fmt = lambda x: f"{x:.6f}" if x is not None else "未定义"
    text = "# 微波炉状态识别交接\n\n日期：2026-10-24。已完成18个网络、36份预测、216个检查点及各自精确PR数组。全部目标预测冻结后统一评价。House17为已反复查看的开发户；训练种子不是独立家庭，不构成新户泛化证明。最终8/9/20未访问，界面默认模型未替换。\n\n"
    text += f"来源验证预选 **{summary['selected']}**，来源资格门槛{'通过' if selection['source_gate_passed'] else '未通过，按预案仅回退C31作诊断'}；其开发共同支持完整出口**{'通过' if gate['precision_exit_passed'] else '未通过'}**。其他候选即使单项成绩较高，也不替换开发评价前的选择。\n\n"
    text += "## 开发共同支持结果\n\n|候选|Precision|Recall均值|F1|FP均值|FN均值|出口|\n|---|---:|---:|---:|---:|---:|---|\n"
    for g in ["W50", "W50_THR", "R_FIXED"] + GROUPS:
        m = means[f"common61/{g}/microwave"]
        passed = summary["common_support_gates"].get(g, summary["selection_diagnostics"].get(g, {})).get("precision_exit_passed")
        text += f"|{g}|" + "|".join(fmt(m[k]) for k in ("precision", "recall", "f1", "false_positive", "false_negative")) + f"|{'基准' if g == 'W50' else '通过' if passed else '未通过'}|\n"
    text += "\n每种子Recall和全部逐项出口见结果JSON与comparison.csv；均值不能替代每种子≥85%的保护。主表对全部候选使用61点共同有效时间，另有native支持完整报表；C61原生支持小于31点，因此其原生指标不直接当公平出口。support.csv列出来源验证/开发的微波炉有效中心、正例中心及阈值连续段数；连续段不等于物理周期。\n\n来源训练31点支持150,069窗口、646开机中心、103阈值连续段；61点为134,673窗口、559开机中心、87段。长上下文丢失部分正例，不能把覆盖变化当识别收益。\n\n"
    text += "## 对照解释与功率\n\nR_PR沿用原L2功率训练，新增召回约束下选检查点及预测阈值；W50_THR仅对历史A功率按来源PR选阈值，R_FIXED对新选中轮仍用200W决策，两者用于分离选点与阈值作用。三个MAE最优检查点均要求逐参数重现历史A，证据位于models/R_PR/seed_*/mae_reference_check.json。\n\nC31从随机初始化端到端训练状态，C_EVENT/C_HARD/C_FEATURES/C61分别只改变采样、困难样本覆盖、总表局部特征、上下文。C_EVENT按事件和家庭分层，但保留原正例采样总质量；C_HARD也保持类别总质量，只改变类内难例覆盖。困难样本只由对应C31来源训练预测与训练标签确定。没有联合功率训练、冻结L2特征线性头或用户确认。\n\n分类每模型3528步，回归7044步，不能声称分类已充分收敛或训练量相等。C61因Flatten结构参数增加；完整参数量见结果JSON。所有分类概率只通过独立状态接口返回，功率继续由A模型独立输出；不能把功率误差保持不变说成状态改进。R_PR/R_FIXED另报选中回归网络的功率误差。\n\n"
    text += "|设备，独立A功率，共同支持|原始状态F1均值|MAE W|开机MAE W|电量绝对误差 kWh|\n|---|---:|---:|---:|---:|\n"
    for a in ("kettle", "microwave", "washing_machine"):
        m = means[f"common61/W50/{a}"]
        text += f"|{a}|" + "|".join(fmt(m[k]) for k in ("f1", "mae_w", "active_mae_w", "energy_absolute_error_kwh")) + "|\n"
    text += "\n水壶/洗衣机原始功率逐点与历史参考一致；共同支持上的状态不变。当前表不启用组合功率一致性修正，避免把独立状态概率混入瓦数。错误段与曲线对所有候选按同一规则在评价后提取，仅供诊断，未回流训练。\n\n"
    text += "失败初跑state-20261024-v1因来源读回过度严格的逐位概率比较中止：差约1.9e-9，状态/缺失完全一致。修复为严格数值容差后另建v2，全部配方不变，v1保留且未评价开发。最终18模型之外还有4次失败运行中的拟合，累计22次；已在开发评价前登记工程预算修订，未增加候选。\n\n"
    text += "## 复跑与边界\n\n预登记见docs/STATE_REGISTRATION_20261024.md和configs/state_20261024.json。新输出目录必须不存在，依次运行：\n\n```powershell\n.venv-m2/Scripts/python.exe scripts/run_state_experiment.py prepare --run outputs/accuracy/state-reproduction\n.venv-m2/Scripts/python.exe scripts/run_state_experiment.py train --run outputs/accuracy/state-reproduction\n.venv-m2/Scripts/python.exe scripts/run_state_experiment.py evaluate --run outputs/accuracy/state-reproduction\n.venv-m2/Scripts/python.exe scripts/run_state_experiment.py verify --run outputs/accuracy/state-reproduction\n```\n\n仅相同源码/输入的未完成运行支持train --resume；不完整单模型目录拒绝覆盖，需另建运行。完整冻结/评价不可重写。\n\n本次冻结后发现核验器会在每个阈值访问时重复解压PR列。当前源码已缓存这些数组，并通过字典/压缩文件选择结果等价测试。历史冻结源码原样保留，因此工作区指纹与执行快照不同，这是明确记录的核验性能修复，不是模型或选择变更。此次评价使用封存源码加仅缓存PR读取的scripts/run_state_snapshot.py包装器，精确重建216轮选择；包装器副本与收据位于运行execution_repair/并单独哈希。复核现有结果可运行：\n\n```powershell\n.venv-m2/Scripts/python.exe scripts/run_state_snapshot.py verify --run outputs/accuracy/state-20261024-v2\n```\n\n当前普通verify也能复核本次结果且不重新解析开发标签；不要把工作区指纹差异误报成历史预测改变。原始封存入口直接verify保留旧版重复解压行为，建议用上述包装器。\n\n独立状态加载与推理为models.state_classifier.load_state_model和predict_state_rows，输出probability_on/state；不满足完整上下文或缺失总表的中心返回None。旧功率加载器和推理行为保留。训练入口默认normalized_mae，只有显式recall_precision启用新规则，checkpoint_evidence可单独保存每轮证据。R_PR在实验中显式消费来源阈值，普通功率入口仍遵守旧功率状态口径；完整读回对其比较raw_power_w及显式来源阈值，而不声称更改了旧接口默认状态。\n\n现有合法配置仅指定House17为开发家庭，本轮未征用其他家庭，新户稳定性检查未完成。只能报告单户开发观察，不能外推一般跨户收益；同一开发户多次使用已有选择偏差。负结果不触发追加权重/阈值/上下文搜索。\n\n产物包含源码/协议/配置/参考输入登记哈希、18模型与216轮权重/PR、36预测、来源选择、冻结与评价证据；summary.json列出通过与未通过项，report.html为易读入口。测试和额外独立读回范围见docs/STATE_VALIDATION_20261024.json；这些是本机自动核验，不是第二成员或第二机器复现。\n"
    (ROOT / "docs/STATE_HANDOFF_20261024.md").write_text(text, encoding="utf-8")
    handoff = ROOT / "outputs/accuracy/state-20261024-handoff-v2"
    if handoff.exists():
        raise RuntimeError("Derived handoff already exists; do not overwrite")
    handoff.mkdir()
    html = '<!doctype html><meta charset="utf-8"><title>来源PR与检查点证据</title><style>body{font:16px system-ui;margin:36px;max-width:1200px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}img{max-width:100%}</style><h1>来源验证PR与检查点</h1><p>只用House5来源验证；虚线为85%召回保护，浅灰为所有轮，深色为所选轮，圆点为冻结阈值。图线均匀抽样最多1500点用于显示，精确全PR保存在对应npz文件。</p><p><a href="../' + run.name + '/report.html">开发评价与逐设备报表</a></p>'
    for g in GROUPS:
        for s in SEEDS:
            model = run / f"models/{g}/seed_{s}"
            meta = read_json(model / "model.json")
            history = read_json(model / "history.json")["epochs"]
            svg = '<svg xmlns="http://www.w3.org/2000/svg" width="680" height="440" viewBox="0 0 680 440"><rect width="680" height="440" fill="white"/><path d="M60 30V380H640" fill="none" stroke="#333"/><path d="M553 30V380" stroke="#aa3333" stroke-dasharray="5 5"/><text x="300" y="424">Recall (0—1)</text><text x="4" y="22">Precision</text><text x="40" y="396">0</text><text x="637" y="396">1</text><text x="39" y="35">1</text>'
            table_rows = []
            for r in sorted(history, key=lambda r: r["epoch"] == meta["selected_epoch"]):
                epoch = r["epoch"]
                path = model / (f"checkpoints/epoch_{epoch}_microwave_pr.npz" if g == "R_PR" else f"epoch_{epoch}_pr.npz")
                with np.load(path, allow_pickle=False) as curve:
                    indices = np.unique(np.linspace(0, len(curve["threshold"])-1, min(1500, len(curve["threshold"]))).astype(int))
                    coordinates = " ".join(f'{60+580*x:.2f},{380-350*y:.2f}' for x, y in zip(curve["recall"][indices], curve["precision"][indices]))
                selected_epoch = epoch == meta["selected_epoch"]
                svg += f'<polyline fill="none" stroke="{"#155b87" if selected_epoch else "#c8cdd1"}" stroke-width="{2 if selected_epoch else 1}" points="{coordinates}"/>'
                point = r["state_evidence"][0]["constrained"] if g == "R_PR" else r["constrained"]
                if selected_epoch:
                    svg += f'<circle cx="{60+580*point["recall"]}" cy="{380-350*point["precision"]}" r="5" fill="#bd3f2d"/>'
            svg += f'<text x="70" y="22">{g} / seed {s} / selected epoch {meta["selected_epoch"]}</text></svg>'
            filename = f"{g}_{s}_source_pr.svg"
            (handoff / filename).write_text(svg, encoding="utf-8")
            for r in history:
                point = r["state_evidence"][0]["constrained"] if g == "R_PR" else r["constrained"]
                fixed = {k: r["state_evidence"][0]["fixed_"+k] for k in ("precision", "recall", "f1")} if g == "R_PR" else r["fixed_decision"]
                values = [r["epoch"], fixed["precision"], fixed["recall"], fixed["f1"], point["threshold"], point["precision"], point["recall"], point["f1"], r.get("validation_normalized_mae", "独立功率输出")]
                table_rows.append("<tr>" + "".join("<td>"+escape(str(v))+"</td>" for v in values) + "</tr>")
            html += f'<h2>{g} / seed{s}</h2><img src="{filename}" alt="来源PR曲线"><table><tr><th>轮</th><th>固定P</th><th>固定R</th><th>固定F1</th><th>阈值</th><th>约束P</th><th>约束R</th><th>约束F1</th><th>功率MAE</th></tr>' + "".join(table_rows) + "</table>"
    (handoff / "source_pr.html").write_text(html, encoding="utf-8")
    write_json(handoff / "evidence.json", {"parent_freeze_sha256": sha256(run / "freeze.json"),
               "parent_evidence_sha256": sha256(run / "evidence.json"),
               "artifacts": {p.name: sha256(p) for p in handoff.iterdir() if p.is_file()}})
    print(json.dumps({"status": "handoff_written", "selected": summary["selected"], "passed": gate["precision_exit_passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
