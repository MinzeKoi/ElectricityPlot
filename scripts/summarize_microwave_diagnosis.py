"""Build the Chinese diagnosis report and standard vector/raster charts."""
import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return "\n".join(["|"+"|".join(headers)+"|", "|"+"|".join(["---"]*len(headers))+"|"] + ["|"+"|".join(map(str, r))+"|" for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/accuracy/microwave-diagnosis-20261025-v2")
    args = parser.parse_args()
    run = args.run
    if (run / "report.md").exists():
        raise ValueError("Report already exists; refuse overwrite")
    result = read(run / "evaluation/results.json")
    rows, stats = result["rows"], read(run / "support.json")
    small = read(run / "small_complete.json")["results"]
    plan = read(run / "plan.json")
    validation = read(ROOT / "docs/MICROWAVE_DIAGNOSIS_VALIDATION_20261025.json")
    figdir = run / "figures"
    figdir.mkdir()
    # ReportLab's plotting component produces vector plots; PDFium rasterizes
    # the same drawings for the Markdown viewer. No training runtime changes.
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics import renderPDF, renderSVG
    from reportlab.lib import colors
    import pypdfium2 as pdfium

    palette = [colors.HexColor(c) for c in ["#147d92", "#c56b27", "#7545a0", "#50a47c", "#e89553", "#b38fce"]]
    def chart(name, title, series, legends, xmin, xmax, ymin, ymax, xlabel, ylabel):
        drawing = Drawing(720, 390)
        drawing.add(String(65, 368, title, fontName="Helvetica-Bold", fontSize=15))
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = 65, 68, 610, 265
        plot.data = series
        plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = xmin, xmax
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = ymin, ymax
        plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 9
        for i in range(len(series)):
            plot.lines[i].strokeColor = palette[i % len(palette)]
            plot.lines[i].strokeWidth = 1.5
        drawing.add(plot)
        drawing.add(String(335, 34, xlabel, fontSize=11))
        drawing.add(String(8, 342, ylabel, fontSize=10))
        for i, label in enumerate(legends):
            drawing.add(String(70+(i%3)*208, 15-(i//3)*12, label, fontSize=8, fillColor=palette[i % len(palette)]))
        renderSVG.drawToFile(drawing, str(figdir / f"{name}.svg"))
        renderPDF.drawToFile(drawing, str(figdir / f"{name}.pdf"))
        doc = pdfium.PdfDocument(str(figdir / f"{name}.pdf"))
        page = doc[0]
        bitmap = page.render(scale=1.7)
        bitmap.to_pil().save(figdir / f"{name}.png")
        bitmap.close()
        page.close()
        doc.close()

    curves = []
    for r in small:
        hist = read(run / f"small/seed_{r['seed']}/history.json")["rows"]
        curves.append([(h["step"], float(np.log10(max(h["bce"], 1e-5)))) for h in hist])
    chart("small_loss", "Fixed small set: weighted BCE during optimization", curves,
          [f"seed {r['seed']}" for r in small], 0, 2000, -2, 2, "Optimizer updates", "log10 BCE")
    for target in plan["households"]:
        series, legends = [], []
        for house in plan["households"]:
            for seed in plan["seeds"]:
                with np.load(run / f"evaluation/pr_{house}_{seed}_{target}.npz") as data:
                    # Compress constant recall portions by taking their first point.
                    ix = np.r_[0, np.flatnonzero(np.diff(data["recall"]) > 0)+1]
                    series.append([(float(data["recall"][i]), float(data["precision"][i])) for i in ix])
                legends.append(f"{'same' if house == target else 'cross'} / seed {seed}")
        chart(f"pr_{target}", f"Outer holdout {target}: frozen model PR (diagnostic only)", series, legends,
              0, 1, 0, 1, "Recall", "Precision")

    daily, cases = [], []
    from datetime import datetime
    for house in plan["households"]:
        with (run / f"inputs/{house}/microwave.csv").open(encoding="utf-8", newline="") as f:
            raw = list(csv.DictReader(f))
        start = datetime.fromisoformat(raw[0]["timestamp"].replace("Z", "+00:00")).timestamp()
        stamps = np.array([datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp() for r in raw])
        yraw = np.array([float(r["power_w"]) if r["power_w"] else np.nan for r in raw])
        with (run / f"inputs/{house}/aggregate.csv").open(encoding="utf-8", newline="") as f:
            ar = list(csv.DictReader(f))
        a = np.array([float(r["aggregate_power_w"]) if r["aggregate_power_w"] else np.nan for r in ar])
        # All rows in these source files are on the same regular 8s grid.
        assert len(a) == len(yraw) and np.all(np.diff(stamps) == 8)
        valid = np.zeros(len(a), bool)
        valid[15:-15] = np.lib.stride_tricks.sliding_window_view(np.isfinite(a),31).all(axis=1)
        for day in range(14):
            m = (stamps >= start+day*86400) & (stamps < start+(day+1)*86400)
            daily.append({"household": house, "day_from_zero": day, "raw_on_centers": int((m & (yraw >= 200)).sum()),
                          "complete_aggregate_windows": int((m & valid).sum()),
                          "usable_on_centers": int((m & valid & (yraw >= 200)).sum())})
        with np.load(run / f"arrays/{house}/outer.npz") as z:
            outer = {k: z[k] for k in z}
        with np.load(run / f"outer_labels/{house}.npz") as z:
            y = z["y"]
        other = next(h for h in plan["households"] if h != house)
        scores = {}
        thresholds = {}
        for train_house in [house, other]:
            with np.load(run / f"predictions/train_{train_house}_seed_42_target_{house}.npz") as z:
                scores[train_house] = z["scores"]
            thresholds[train_house] = read(run / f"models/{train_house}/seed_42/model.json")["selected"]["threshold"]
        for train_house in [house, other]:
            fp = np.flatnonzero((y < 200) & (scores[train_house] >= thresholds[train_house]))
            j = int(fp[np.argmax(scores[train_house][fp])])
            stamp = datetime.fromtimestamp(int(outer["t"][j]), __import__("datetime").timezone.utc).isoformat()
            cases.append({"target_house": house, "ranked_by_train_house": train_house, "seed": 42, "timestamp": stamp,
                          "truth_w": float(y[j]), "center_aggregate_w": float(outer["x"][j,15]),
                          "same_score": float(scores[house][j]), "same_threshold": thresholds[house],
                          "cross_score": float(scores[other][j]), "cross_threshold": thresholds[other],
                          "aggregate_window_w": outer["x"][j].tolist()})
    (run / "daily_support.json").write_text(json.dumps({"rows": daily}, indent=2), encoding="utf-8")
    (run / "paired_error_cases.json").write_text(json.dumps({"rule": "seed42 highest-score FP per train/target pair; compare same timestamp under other train model", "rows": cases}, indent=2), encoding="utf-8")
    chart("false_positive_windows", "False-positive aggregate contexts (truth below 200 W)",
          [[(i*8-120, v) for i,v in enumerate(c["aggregate_window_w"])] for c in cases],
          [f"target {c['target_house']} / train {c['ranked_by_train_house']}" for c in cases],
          -120, 120, 0, max(max(c["aggregate_window_w"]) for c in cases)*1.1, "Seconds from center", "Total power W")
    support_table = []
    for house in plan["households"]:
        for role, v in stats[house]["blocks"].items():
            support_table.append([house, role, v["windows"], v["positive_windows"], v["positive_segments_supported"], f"{v['prevalence']:.3%}", f"{v['effective_seconds']/3600:.2f}"])
    summary_rows = []
    for target in plan["households"]:
        for role in ["same_house_outer", "cross_house_outer"]:
            group = [r for r in rows if r["target_house"] == target and r["role"] == role]
            summary_rows.append([target, "同户" if role.startswith("same") else "跨户"] + [f"{np.mean([r[k] for r in group]):.2%}" for k in ["precision","recall","f1","ap","off_false_positive_rate"]] + [f"{np.mean([r['diagnostic']['p_at_r85'] for r in group]):.2%}", f"{np.mean([r['diagnostic']['fpr_at_r85'] for r in group]):.2%}"])
    per_seed = [[r["train_house"],r["target_house"],r["seed"]]+[f"{r[k]:.2%}" for k in ["precision","recall","f1","ap"]]+[r[k] for k in ["tp","fp","fn","tn"]]+[f"{r['off_false_positive_rate']:.2%}", f"{r['threshold']:.6f}",f"{r['positive_segments_hit']}/{r['positive_segments_supported']}"] for r in rows if r["role"].endswith("outer")]
    budget = []
    for house in plan["households"]:
        for seed in plan["seeds"]:
            m = read(run / f"models/{house}/seed_{seed}/model.json")
            budget.append([house,seed,m["steps"],m["seen"],m["positive_seen"],m["selected"]["epoch"],f"{m['seconds']:.2f}"])
    internal = []
    for house in plan["households"]:
        for role in ["train","validation","same_house_outer"]:
            g = [r for r in rows if r["train_house"] == house and r["role"] == role]
            internal.append([house,role]+[f"{np.mean([r[k] for r in g]):.2%}" for k in ["ap","recall"]]+[f"{np.mean([r['diagnostic']['p_at_r85'] for r in g]):.2%}",f"{np.mean([r['diagnostic']['fpr_at_r85'] for r in g]):.2%}"])
    small_table = [[r["seed"],r["updates"],r["final"]["tp"],r["final"]["fp"],r["final"]["fn"],r["final"]["tn"],f"{r['final']['precision']:.2%}",f"{r['final']['recall']:.2%}",f"{r['final']['bce']:.5f}","通过" if r["success"] else "未通过"] for r in small]
    error_table = [[c["target_house"],c["ranked_by_train_house"],c["timestamp"],f"{c['truth_w']:.1f}",f"{c['center_aggregate_w']:.1f}",f"{c['same_score']:.6f} / {c['same_threshold']:.6f}",f"{c['cross_score']:.6f} / {c['cross_threshold']:.6f}"] for c in cases]
    parts = ["# 微波炉小样本能力检查与同户/跨户诊断报告\n\n日期：2026-10-25。运行：`microwave-diagnosis-20261025-v2`。9次拟合全部完成，12份外层预测先冻结后统一评价。仅House2/19；House17未重新验收，最终House8/9/20未访问。旧默认模型未替换。",
    "## 结论\n\n小集合能够学会，3/3种子达到预定标准；不能把当前低精确率主要解释为网络完全不能优化、无参数更新或窗口取中心的实现错位（原始传感器时间偏差仍未排除）。同户时间留出仍明显退化：两户平均Recall仅38.65%/64.29%，即使事后重设阈值到Recall≥85%，Precision也只有3.44%/7.91%。因此户间变化不是唯一瓶颈。\n\n同户优势有方向差异。目标House19：同户AP12.60%对跨户5.72%，诊断R85处FPR2.08%对8.28%，满足登记优势标准。目标House2：同户AP12.20%对跨户8.61%，差3.59个百分点未达5个百分点，诊断R85处FPR反而10.70%对5.07%。两个方向并不一致，不能宣布“同户明显优于跨户”是普遍结论。\n\n证据支持：数据有效支持稀缺、同户时间/负载组合变化、阈值迁移与分数排序退化共同作用。不能给出各原因的因果贡献百分比，也不能据一次小规模诊断断言8秒单通道任务不可解。",
    "## 数据可用性与训练前修订\n\nv1在准备阶段发现House19前7天没有任何完整C31开机窗，原定7/3/4天划分不可行，0次拟合。其失败目录和日志保留。按可用性在任何训练前改为v2：两户共同使用相对起点第8–11天训练、11–12天内部验证、12–14天外层留出，第11/12天边界各侧剔除1800秒。完整31点上下文位于所属块内，跨界阈值段整段排除，无插值、无缺失填零。\n\nHouse19前8天原标签存在开机，但每日完整总表窗仅0/40/8/19/7/0/23/16个，开机窗全为0；原聚合记录含coverage=.875、incomplete_coverage、observation_gap。说明原始标签正数不等于模型可见正窗数。归一化文件中总表缺失House2为14216/151200、House19为12960/151200，标签缺失均15120/151200。无法恢复上游已前填/填零及IAM截断带来的信息。逐日统计见`daily_support.json`。",
    table(["家庭","分区","有效窗","开机窗","阈值连续段","开机占比","有效小时"], support_table),
    "上述阈值连续段不等于独立物理周期；相邻窗口高度相关。训练名义3天，实际有效时长47.37/55.85小时；内部验证只有7/18段，外层仅8/7段。不同户时间并非相同日历天。支持筛选限制结论外推。",
    "## 小集合能否学会\n\n每户最早4个完整可用训练开机段，共8段66个正中心；按同户总表中心功率最近匹配66个易混淆关闭窗，按低功率且时间最近匹配66个易关闭窗，共198窗。清单含户、UTC中心、窗口标识、原始行、事件、真实功率及配对原因。负窗彼此可邻近相关，此集合不作为独立泛化样本。\n\n沿用现有C31输入形式、历史来源log1p统计、两层CNN、Adam .001、BCE pos_weight50，固定全集合优化；阈值.5。开始三个种子TP均0、FN66，加权BCE约50–52；每100步检查。P/R≥.98、AP≥.99、BCE≤.1连续两次则停止，上限2000步。所有实际梯度与参数更新非零、输入有限，原API标签与独立时间映射逐点一致。",
    table(["种子","更新数","TP","FP","FN","TN","P","R","加权BCE","判据"], small_table),
    "![小集合损失](figures/small_loss.png)\n\nAP最终三者均为1。训练初始、每次检查、最终逐样本得分与正负分位数均保存。1019最终1个关闭窗仍过.5，但正样本排序完整领先负样本；符合预登记98%标准。小集合成功表明能在有限样本上学习，不能排除总体容量/特征表达不足、全量优化或稀疏正例采样问题。",
    "## 同户/跨户控制结果\n\n6模型均从头训练、自然类别比例shuffle、训练块独立估计log1p统计；没有困难挖掘或新网络变体。内部验证以Recall≥.85的精确PR点选阈值，再按P/F1/较早轮次选检查点。外层标签物理分离，所有模型和阈值及12份预测封存后才统一评分。对同一目标户，同户/跨户使用完全相同的有效时间点与正类比例。以下均为3种子算术均值，不能视作3个独立家庭。末两列是各留出分区事后阈值的诊断上界，禁止部署或作为通过验收依据。",
    table(["目标","训练来源","P","R","F1","AP","冻结阈值FPR","诊断P@R85","诊断FPR@R85"], summary_rows),
    "冻结阈值下两户同户Recall均未达到85%。House2跨户F1均值略高于同户，且其R85处误报率更低；不能只报告同户AP略高而忽略这一反例。AP比较仅在相同目标支持上可直接解读，不以两户Precision落差直接推断迁移。",
    "![目标House2 PR](figures/pr_refit_2.png)\n\n![目标House19 PR](figures/pr_refit_19.png)",
    "## 分清阈值漂移与排序退化\n\n下表为每个训练户所选模型在训练重放、内部验证、同户外层上的均值。训练重放只检查拟合程度。即使独立放宽留出阈值达到85%召回，外层误报率仍明显高于训练及内部验证；AP也降低。因此仅重新校准阈值不能恢复原排序质量。",
    table(["训练户","分区","AP","冻结阈值R","诊断P@R85","诊断FPR@R85"], internal),
    "House2内部验证到同户外层AP下降41.58个百分点，House19下降31.32个百分点。分区类别比例不同会影响AP/P，但诊断固定召回处FPR同样升高（1.54%→10.70%、1.07%→2.08%），支持存在超出先验比例变化的性能退化。仍不能把它唯一归因于硬件容量或负载变化：验证仅一天，事件支持少，且检查点选择有内验证乐观偏差。",
    "## 逐种子冻结阈值结果\n\n事件命中指该阈值段至少一个有效开机中心被识别；不能代替逐点Recall或完整事件召回。连续分数完整PR存于evaluation/pr_*.npz，固定.5的全部计数与指标存于results.json的fixed_half。",
    table(["训练户","目标户","种子","P","R","F1","AP","TP","FP","FN","TN","FPR","阈值","段命中"], per_seed),
    "## 训练预算与选择\n\n每模型3528次更新，批量上限256、末批可短；约89.5万窗口曝光。House2正例曝光6972次，House19约5017–5031次，窗口重复并非新事件。名义时长相同、更新预算相同，有效窗口/事件支持不同。House2约42次全训练集曝光，House19约35.6次；原C31来源全训练集约150069窗、646正窗、103段，3528步约6次曝光，因此不能将本轮与历史训练预算说成完全同等学习任务。",
    table(["训练户","种子","更新数","实际窗口曝光","正窗曝光","选中检查点","拟合秒数"], budget),
    "内部验证阈值约.56–.93；高阈值随训练冻结后迁移召回不足。较早最佳轮与末轮均在登记范围内保存，未按外层结果追加训练、改阈值或搜索模型。",
    "## 误报案例对照\n\n固定seed42，每个训练/目标组合选分数最高的误报，并在同一目标时间点查询另一来源模型分数。表内为“得分 / 各自冻结阈值”，真实微波炉功率均小于200W。窗口曲线是总有功功率；仅凭本图不能识别究竟是哪一种其他设备。",
    table(["目标户","按哪个训练户排序","UTC中心","真实W","总表中心W","同户得分/阈值","跨户得分/阈值"], error_table),
    "![误报总表上下文](figures/false_positive_windows.png)\n\n完整逐窗31点功率、两个来源分数见paired_error_cases.json。所有模型另保留每类至多3个、相隔至少248秒的TP/FP/FN案例（evaluation/error_cases.json）。最高分样例是机制线索，不能代表全部误报频率，更不能直接当成下一轮训练集。",
    "## 既有证据核对\n\n重新以旧C61权重对来源2/19的61点共同支持134673窗、559正中心推理，3种子的AP、P@R85、FPR@R85均与旧只读诊断JSON一致；重算训练P@R85均值17.37%、FPR1.71%。旧检查点历史的选中轮、末轮BCE和来源验证Precision也逐项一致。见docs/MICROWAVE_PRIOR_EVIDENCE_READBACK_20261025.json。原House5/17数字仅引用旧摘要，无新标签读取；旧开发P@R85约4.26%、FPR约5.5%仍是独立事后阈值诊断，不能部署。末轮训练损失下降而来源验证P下降的记录支持“不直接延长训练”。state-v2的PR读取缓存修复与模型训练不同，本轮没有因源码指纹不同而否定旧实验。",
    "## 能支持、不能支持与下一步\n\n已支持：①原C31实现能优化固定小集合；②有效完整上下文使可用开机支持显著缩水，名义14天不能代替有效事件统计；③同户时间变化已足以造成大幅性能退化；④House19有额外跨户损失，House2没有一致同户优势；⑤冻结阈值漂移和排序质量退化都存在。\n\n不能排除：更丰富负载组合下的特征/容量瓶颈、稀疏正例梯度与采样不足、单日内部验证不稳定、上游预处理造成的信息损失。不能把低精确率唯一归因于损失权重或跨户，也不能证明信号不可辨识。训练/内部验证选择和外层差异不是随机因果试验，无法量化每个原因占比。\n\n下一轮最小建议：先只做来源数据的独立质量审计，定位House19前8天coverage=.875及碎片化缺失的生成原因，逐时核对原始采样、重采样规则、标签和总表时间戳，不填补缺失来凑训练样本；以“有效完整事件支持”决定能否建立至少多个日期块的内验证。若数据核对正确，再单独预登记一个“真正增加每批正事件曝光”的采样对照：同一C31、归一化、唯一正负清单、更新数和损失保持一致，仅改变批次正事件曝光，按事件/日期分块，冻结阈值后再看新的来源时间留出；必须明确正例有效权重随采样改变。不要同时增加网络或延长训练，不复用本轮外层反复选配方。本轮未执行该建议。",
    "## 复核与交付\n\n入口scripts/run_microwave_diagnosis.py支持prepare/audit/small/train/evaluate/verify。新运行目录必须不存在，已完成阶段或单模型目录拒绝覆盖。封存入口可在工作区日后变化时使用；当前verify已完成12份外层预测的权重读回，状态逐点一致。补充独立核验另重建72份内部验证PR与选择、12份外层混淆/AP、3小集合权重与成功判据、训练归一化、上下文和事件隔离，全部通过。这是本机自动复核，不是第二成员或第二机器复现。\n\n```powershell\n.venv-m2/Scripts/python.exe scripts/run_microwave_diagnosis_snapshot.py verify --run outputs/accuracy/microwave-diagnosis-20261025-v2\n.venv-m2/Scripts/python.exe scripts/verify_microwave_diagnosis.py\n.venv-m2/Scripts/python.exe -m unittest discover -s tests -p test_microwave_diagnosis.py -v\n```\n\n重跑依次prepare、audit、small、train、evaluate、verify，各命令指定同一个全新--run。源码/输入/模型/预测哈希分别位于registration.json、models/*/complete.json、freeze.json，评价封存位于evaluation_complete.json，交付总清单delivery_manifest.json。window_manifest.csv、small_manifest.csv、events.json、support.json提供划分与样本身份；source快照可检查没有外层选择路径。\n\n训练早期误调用一次evaluate，由缺少freeze.json的门禁阻止，未创建评价目录、未解析外层标签；随后在训练进程退出、freeze.json存在后正式评价成功。v1准备失败和该门禁日志保留。无额外拟合、无留出后搜索。必要测试5项和已有状态实验回归9项全部通过，共14项。交付复核时发现注册封存入口的目录定位错误，直接启动会在导入项目代码前失败。原封存入口未改写；新增scripts/run_microwave_diagnosis_snapshot.py只定位并加载登记源码以verify，修复副本和哈希收据存于execution_repair/，12份预测读回已通过。当前工作区入口亦修正了未来封存目录定位，不改变本轮训练源码。",
    ]
    report = "\n\n".join(parts) + "\n"
    (run / "report.md").write_text(report, encoding="utf-8")
    (ROOT / "docs/MICROWAVE_DIAGNOSIS_REPORT_20261025.md").write_text(report.replace("(figures/", "(../outputs/accuracy/microwave-diagnosis-20261025-v2/figures/"), encoding="utf-8")
    # Simple HTML report preserves tables and line breaks without an extra parser.
    html_parts = []
    for part in parts:
        if part.startswith("|"):
            lines = part.splitlines()
            cells = [line.strip("|").split("|") for line in lines]
            html_parts.append("<div class='scroll'><table><thead><tr>"+"".join(f"<th>{escape(c)}</th>" for c in cells[0])+"</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{escape(c)}</td>" for c in row)+"</tr>" for row in cells[2:])+"</tbody></table></div>")
        else:
            import re
            text = escape(part)
            text = re.sub(r"!\[([^\]]+)\]\((figures/[^)]+)\)", r'<img alt="\1" src="\2">', text)
            text = re.sub(r"^###? (.+)$", r"<h2>\1</h2>", text, flags=re.M)
            html_parts.append("<section>"+text.replace("\n\n", "</p><p>").replace("\n", "<br>")+"</section>")
    (run / "report.html").write_text("<!doctype html><meta charset='utf-8'><title>微波炉能力与迁移诊断</title><style>body{font-family:system-ui,'Microsoft YaHei';max-width:1180px;margin:40px auto;padding:0 24px;line-height:1.8;color:#192e35;background:#fcfdfc}h2{margin-top:40px;color:#126477}table{border-collapse:collapse;font-size:13px;width:100%;white-space:nowrap}td,th{border-bottom:1px solid #cdd7d9;padding:7px;text-align:right}th{background:#e9f0ef}.scroll{overflow:auto}img{width:100%;max-width:900px;display:block;margin:22px auto}section{margin:20px 0}</style>"+"\n".join(html_parts), encoding="utf-8")
    output = {"experiment_id": plan["experiment_id"], "small_success_seeds": sum(r["success"] for r in small),
              "paired_comparisons": result["paired_comparisons"], "consistent_same_house_advantage": result["consistent_same_house_advantage"],
              "validation": validation, "accessed_households": plan["households"], "final_households_accessed": False,
              "outer_holdout_metrics": [r for r in rows if r["role"].endswith("outer")]}
    (ROOT / "docs/MICROWAVE_DIAGNOSIS_RESULTS_20261025.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    extra = [Path(__file__), ROOT / "scripts/run_microwave_diagnosis.py", ROOT / "scripts/run_microwave_diagnosis_snapshot.py", ROOT / "scripts/verify_microwave_diagnosis.py", ROOT / "scripts/verify_microwave_delivery.py", ROOT / "tests/test_microwave_diagnosis.py",
             ROOT / "configs/microwave_diagnosis_20261025.json", ROOT / "docs/MICROWAVE_DIAGNOSIS_REGISTRATION_20261025.md",
             *ROOT.glob("docs/MICROWAVE_*20261025.*"), *ROOT.glob("tmp/microwave-diagnosis-*.log")]
    manifest = {"run_artifacts": {p.relative_to(run).as_posix(): digest(p) for p in sorted(run.rglob("*")) if p.is_file()},
                "external_delivery_artifacts": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(extra)) if p.is_file()}}
    (run / "delivery_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "report_delivered", "report": str(run / "report.md"), "files": len(manifest["run_artifacts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
