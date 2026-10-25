"""Readable trust/quality and paired synthetic stress evidence in offline HTML."""

from html import escape
import json
from pathlib import Path

from ..trust.output import quality_summary

FLAGS = {
    "aggregate_missing": "总表缺失",
    "context_missing_or_boundary": "上下文不足或边界",
    "sum_exceeded_aggregate_scaled_down": "设备估计超总表，已向下修正",
    "power_uncalibrated": "功率区间未校准",
    "source_validation_intervals_no_target_guarantee": "仅来源验证区间，无目标家庭覆盖保证",
    "aggregate_outside_calibration_range": "总功率超出来源校准范围",
    "unexplained_load_present": "存在未解释负荷",
    "synthetic_background_injection": "合成背景注入",
}


def display(value, percent=False):
    if value is None:
        return "不可用"
    return f"{value:.1%}" if percent else f"{value:.4f}"


def table(headers, rows):
    return "<div class='scroll'><table><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr>" + "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</table></div>"


def sections(payload: dict, evaluation: dict | None = None) -> str:
    summary = quality_summary(payload)
    calibration = payload.get("calibration")
    text = "<section><h2>一致性与未解释负荷</h2>"
    text += table(["原始超限率", "修正后超限率", "平均移除 W", "可归因电量比例", "未解释电量 kWh"], [[
        display(summary["raw_exceedance_rate"], True), display(summary["corrected_exceedance_rate"], True),
        display(summary["mean_removed_power_w"]), display(summary["attributed_energy_fraction"], True), display(summary["unexplained_energy_kwh"])]])
    text += "<p>比例和电量只计有效预测区间。仅在设备估计总和超过总表时向下缩减；保留原始功率、修正幅度和剩余项。未解释负荷含未建模背景及计量、模型误差。</p></section>"
    text += "<section><h2>功率区间与可信度</h2>"
    if calibration:
        text += f"<p>来源验证残差的 {display(calibration['interval_level'], True)} 经验分位数区间。原始和修正输出分别校准。</p>"
        text += "<p>来源验证数据也用于选模型；该区间没有跨家庭覆盖保证。它是逐设备边际区间，上界不受总表截断，不代表各设备同时取上界仍满足总表约束。状态分数尚未校准为概率。</p>"
    else:
        text += "<p>未加载匹配的来源校准产物，功率区间与可信概率均不可用。</p>"
    if evaluation and evaluation.get("intervals"):
        text += table(["设备", "输出", "经验覆盖率", "平均区间宽度 W", "有效样本"], [
            [a, version, display(v["empirical_coverage"], True), display(v["mean_width_w"]), v["valid_samples"]]
            for a, versions in evaluation["intervals"].items() for version, v in versions.items()])
        text += "<p>经验覆盖率仅描述当前家庭、当前测试支持；不应由区间标称水平推断实际覆盖率。</p>"
    text += "</section><section><h2>数据质量标记</h2>"
    text += table(["标记", "样本数"], [[FLAGS.get(flag, flag), count] for flag, count in summary["quality_flag_samples"].items()])
    text += "<p>质量问题、缺数据和模型不确定性分别记录；缺失段不作为设备关闭，也不填成零电量。</p></section>"
    if evaluation and evaluation.get("attribution"):
        values = evaluation["attribution"]
        text += "<section><h2>归因比例与误差必须一起看</h2>"
        text += table(["输出", "可归因电量比例", "设备平均 MAE W", "过归因 kWh", "漏掉已知电量 kWh"], [
            [label, display(values[key]["attributed_energy_fraction"], True), display(values[key]["mean_appliance_mae_w"]),
             display(values[key]["overattributed_energy_kwh"]), display(values[key]["missed_known_energy_kwh"])]
            for key, label in (("raw", "原始"), ("corrected", "修正"), ("all_unexplained_control", "全部留作未解释对照"))])
        text += f"<p>共同有效标签支持 {values['joint_valid_samples']} 个区间，覆盖 {display(values['joint_coverage'], True)}。全部留作未解释虽没有过归因，却漏掉所有已知设备耗电，不能据此声称更准确。</p></section>"
    return text


def render_stress_report(result: dict, path: Path) -> None:
    rows = []
    for group, scores in result["paired"]["groups"].items():
        for condition in ("baseline", "injected"):
            for appliance, versions in scores[condition]["metrics"].items():
                for version, values in versions.items():
                    rows.append([group, condition, appliance, version, display(values["mae_w"]), display(values["active_mae_w"]), display(values["f1"]), display(values["energy"]["absolute_error_kwh"]), values["valid_samples"]])
    false_rows = [[group, appliance, v["actual_off_samples"], display(v["baseline_false_activation_rate"], True), display(v["injected_false_activation_rate"], True), v["new_false_activation_samples"]]
                  for group, scores in result["paired"]["groups"].items() for appliance, v in scores["false_activations"].items()]
    text = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ElectricityPlot · 背景压力测试</title>
<style>body{font:16px/1.65 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#193440;background:#f1f6f8}section{background:white;padding:24px;border-radius:14px;margin:20px 0}table{border-collapse:collapse;white-space:nowrap;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce6eb}.scroll{overflow:auto}h1{color:#087e8b}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>ElectricityPlot · 合成背景压力测试</h1><p>固定模型、校准和注入参数后完成成对预测，再读取测试标签评价。已知设备真值保持原样。</p>
<p>这是人工增加背景功率的敏感性实验，不是现场未知设备识别或真实故障。未解释项不能当作未知设备真值。</p>
<p><a href="baseline/report.html">查看基准曲线与区间</a> · <a href="injected/report.html">查看注入后曲线与区间</a> · <a href="stress.json">下载完整指标 JSON</a></p>"""
    text += f"<p>家庭 {escape(result['household_id'])}；分区 {escape(result['partition_id'])}；来源协议用途 {escape(result['source_purpose'])}。</p>"
    if result.get("data_kind") == "synthetic_contract_fixture":
        text += "<p><strong>本次基准输入、设备标签和训练数据均为合成契约样例，不是 REFIT 实测，不作为算法性能或真实用户试用证据。</strong></p>"
    for condition, label in (("baseline", "基准"), ("injected", "注入背景后")):
        score = result[condition]
        c = score["consistency"]
        text += "<section><h2>" + label + " · 总体一致性与归因</h2>"
        text += table(["原始超限率", "修正后超限率", "归因比例", "未解释 kWh", "平均移除 W"], [[display(c["raw_exceedance_rate"], True), display(c["corrected_exceedance_rate"], True), display(c["attributed_energy_fraction"], True), display(c["unexplained_energy_kwh"]), display(c["mean_removed_power_w"])]])
        attribution = score["attribution"]
        text += table(["输出", "归因比例", "设备平均 MAE W", "过归因 kWh", "漏掉已知 kWh"], [[k, display(attribution[k]["attributed_energy_fraction"], True), display(attribution[k]["mean_appliance_mae_w"]), display(attribution[k]["overattributed_energy_kwh"]), display(attribution[k]["missed_known_energy_kwh"])] for k in ("raw", "corrected", "all_unexplained_control")])
        text += "</section>"
    text += "<section><h2>有 / 无背景事件的设备误差</h2>" + table(["中心区间", "条件", "设备", "输出", "MAE W", "运行 MAE W", "F1", "电量误差 kWh", "有效样本"], rows) + "</section>"
    text += "<section><h2>真实关闭区间的错误激活</h2>" + table(["中心区间", "设备", "实际关闭样本", "基准错误激活率", "注入后错误激活率", "新增错误激活样本"], false_rows) + "<p>未来上下文会使脉冲影响延伸到事件外；这些分组按中心时刻的注入计划定义。</p></section>"
    text += "<section><h2>可追溯记录</h2><pre>" + escape(json.dumps({k: result[k] for k in ("notice", "model_id", "code_sha256", "protocol_sha256", "settings_sha256", "predictions_frozen_before_test_labels")}, ensure_ascii=False, indent=2)) + "</pre></section></html>"
    path.write_text(text, encoding="utf-8")
