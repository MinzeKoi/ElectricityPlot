"""Local file import, offline prediction, replay and report download."""

import hashlib
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from ..data.common import DataError, read_json, safe_child, sha256
from ..data.prepare import prepare
from ..models.artifacts import load_model
from ..models.prediction import predict
from ..reporting.html import NAMES
from ..trust.calibration import load_calibration
from ..trust.output import quality_summary
from ..reporting.trust import FLAGS
from .results import load_result


def main(root: Path):
    st.set_page_config(page_title="ElectricityPlot", page_icon="⚡", layout="wide")
    st.title("ElectricityPlot")
    st.caption("从家庭总功率查看设备用电估计 · 离线历史分析")
    st.info("设备用电为模型估计。未解释部分含未建模背景及模型误差；来源校准区间没有跨家庭覆盖保证。")
    with st.expander("高级：模型设置"):
        model_text = st.text_input("模型目录", "outputs/m2-20261020/seq2point")
        calibration_text = st.text_input("区间校准目录（可选）", "")
    calibration_dir = root / calibration_text if calibration_text.strip() else None
    calibration_token = None
    mode = st.radio("数据来源", ["已有家庭记录", "导入总表 CSV"], horizontal=True)
    manifest_path = root / "data/processed/refit_m1_protocol/protocol.json"
    partition_id = None
    uploaded = None
    household = "uploaded_home"
    unit = None
    context_error = None
    if mode == "已有家庭记录":
        with st.expander("高级：数据来源设置"):
            manifest_text = st.text_input("数据协议", str(manifest_path.relative_to(root)))
        manifest_path = root / manifest_text
        try:
            protocol = read_json(manifest_path)
            choices = [p["partition_id"] for p in protocol["partitions"] if p["role"] in ("frozen_test", "source_validation", "target_adaptation")]
            if not choices:
                raise DataError("没有可用的数据分区")
            default = next((i for i, item in enumerate(choices) if item.startswith("frozen_test")), 0)
            labels = {p["partition_id"]: "House " + p["household_id"] + " · " + {"frozen_test": "未见家庭测试区间", "source_validation": "来源家庭验证区间", "target_adaptation": "待适配历史区间"}.get(p["role"], p["role"]) for p in protocol["partitions"]}
            partition_id = st.selectbox("分析区间", choices, index=default, format_func=lambda value: labels[value])
            selected = next(p for p in protocol["partitions"] if p["partition_id"] == partition_id)
            artifact = selected["artifacts"]["aggregate"]
            input_token = sha256(safe_child(manifest_path.parent, artifact["path"]))
            if input_token != artifact["sha256"]:
                raise DataError("总表文件已改变，与冻结协议不符，请重新准备数据和协议")
            source_token = (str(manifest_path), sha256(manifest_path), partition_id, input_token)
        except (DataError, OSError, KeyError) as error:
            context_error = str(error)
            source_token = (str(manifest_path), "invalid")
    else:
        st.caption("CSV 字段：household_id、timestamp、aggregate_power_w；时间须带时区，空功率表示缺失。仅支持无光伏、无储能的有功功率。")
        uploaded = st.file_uploader("选择总表文件", type=["csv"])
        household = st.text_input("文件中的家庭标识", household)
        unit = st.selectbox("文件功率单位", ["请选择", "W"])
        source_token = (hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded else None, household, unit)
    try:
        metadata, _ = load_model(root / model_text)
        model_token = sha256(root / model_text / "model.json")
        if calibration_dir is not None:
            load_calibration(calibration_dir, metadata)
            calibration_token = sha256(calibration_dir / "calibration.json")
        stage = "已进行窗口弱标签状态适配，功率主干冻结" if metadata.get("adaptation") == "weak_window_state_only" else "未经目标家庭适配"
        st.caption("当前模型：" + ("轻量时序模型" if metadata["kind"] == "small_seq2point" else "简单对照") + " · " + stage)
    except (DataError, OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        model_token = "invalid"
        context_error = "模型不可用：" + str(error)
        st.warning("尚无可用模型，请先运行基线训练，或填写有效的模型目录。")
    signature = (model_text, model_token, calibration_token, mode, source_token)
    if st.session_state.get("signature") != signature or context_error:
        st.session_state.pop("prediction_directory", None)
        st.session_state.pop("evaluation_directory", None)
        from .confirmations import clear_results
        clear_results()
        st.session_state["signature"] = signature
    if context_error:
        st.error("当前输入或模型不可用：" + context_error)
    if not context_error and partition_id and partition_id.startswith("target_adaptation"):
        from .confirmations import panel
        panel(root, root / model_text, manifest_path, partition_id)
    if st.button("开始分析", type="primary"):
        st.session_state.pop("prediction_directory", None)
        st.session_state.pop("evaluation_directory", None)
        try:
            if context_error:
                raise DataError(context_error)
            directory = root / "outputs/ui" / uuid.uuid4().hex
            with st.spinner("正在检查数据并计算设备估计…"):
                if mode == "导入总表 CSV":
                    if uploaded is None or unit != "W":
                        raise DataError("请选择 CSV 并明确选择 W 单位")
                    directory.mkdir(parents=True)
                    source = directory / "uploaded.csv"
                    source.write_bytes(uploaded.getvalue())
                    prepare({household: source}, directory / "prepared", root / "configs/refit_channels.json", kind="aggregate")
                    predict(root / model_text, directory / "prediction", input_path=directory / "prepared/aggregate" / f"house_{household}.csv", calibration_dir=calibration_dir)
                else:
                    predict(root / model_text, directory / "prediction", manifest_path=manifest_path, partition_id=partition_id, calibration_dir=calibration_dir)
            st.session_state["prediction_directory"] = str(directory / "prediction")
        except (DataError, OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
            st.error("分析未完成：" + str(error))
    if "prediction_directory" not in st.session_state:
        st.caption("分析成功后将在此显示真实预测曲线和可下载报告。")
        return
    directory = Path(st.session_state["prediction_directory"])
    try:
        payload = load_result(directory)
        prediction_csv = (directory / "predictions.csv").read_bytes()
        report_html = (directory / "report.html").read_bytes()
    except (DataError, OSError, ValueError, TypeError, KeyError) as error:
        st.session_state.pop("prediction_directory", None)
        st.session_state.pop("evaluation_directory", None)
        st.error("结果不可用，请重新分析：" + str(error))
        return
    appliance = st.selectbox("查看设备", payload["appliances"], format_func=lambda a: NAMES.get(a, a))
    if payload.get("data_kind") == "synthetic_contract_fixture" or payload.get("model_data_kind") == "synthetic_contract_fixture":
        st.warning("当前数据或模型来自合成契约样例，仅用于工程演示，不代表真实 REFIT 效果。")
    st.subheader("历史回放")
    count = len(payload["rows"])
    progress = st.slider("回放进度（%）", 0, 100, 0)
    position = round((count - 1) * progress / 100)
    duration = st.selectbox("显示时长", ["1 小时", "4 小时", "全部"])
    width = {"1 小时": max(1, 3600 // payload["sample_interval_seconds"]), "4 小时": max(1, 14400 // payload["sample_interval_seconds"]), "全部": count}[duration]
    frame = pd.DataFrame([{"时间 UTC": row["timestamp"], "总表 W": row["aggregate_power_w"],
                           "设备原始估计 W": row["appliances"][appliance]["raw_power_w"],
                           "设备修正估计 W": row["appliances"][appliance]["power_w"],
                           "来源经验下界 W": row["appliances"][appliance].get("lower_w"),
                           "来源经验上界 W": row["appliances"][appliance].get("upper_w"), "未解释 W": row["unexplained_power_w"]}
                          for row in payload["rows"][position:position + width]])
    frame["时间 UTC"] = pd.to_datetime(frame["时间 UTC"], utc=True)
    from .charts import history_chart
    st.altair_chart(history_chart(frame), use_container_width=True)
    valid = sum(row["unexplained_power_w"] is not None for row in payload["rows"])
    st.metric("有效预测覆盖", f"{valid / count:.1%}")
    summary = quality_summary(payload)
    st.subheader("一致性、未解释部分与质量")
    def show(value, percent=False):
        return "不可用" if value is None else f"{value:.1%}" if percent else f"{value:.4f}"
    st.write({"可归因电量比例": show(summary["attributed_energy_fraction"], True),
              "原始超限率": show(summary["raw_exceedance_rate"], True),
              "修正后超限率": show(summary["corrected_exceedance_rate"], True),
              "平均移除功率 W": show(summary["mean_removed_power_w"]),
              "未解释电量 kWh（有效区间）": show(summary["unexplained_energy_kwh"])})
    st.dataframe(pd.DataFrame([{"质量标记": FLAGS.get(flag, flag), "样本数": number} for flag, number in summary["quality_flag_samples"].items()]))
    st.caption("来源经验区间分别校准原始与修正功率；它没有跨家庭覆盖保证。状态分数尚未校准为概率。" if payload.get("calibration") else "尚无匹配的区间校准产物；可信概率与功率区间不可用。")
    confirmations = payload.get("query_summary", {}).get("budget_used", 0)
    st.caption(f"空段表示缺失或上下文不足。已使用 {confirmations} 次确认；功率结果不通过窗口弱标签修改。")
    if mode == "已有家庭记录" and partition_id and not partition_id.startswith("target_adaptation"):
        if st.button("生成分表评价报告"):
            try:
                from ..evaluation.runner import evaluate
                target = directory.parent / ("evaluation_" + uuid.uuid4().hex)
                evaluate(directory, manifest_path, target)
                st.session_state["evaluation_directory"] = str(target)
            except (DataError, OSError, ValueError, KeyError) as error:
                st.session_state.pop("evaluation_directory", None)
                st.error("评价未完成：" + str(error))
    report_directory = directory
    if "evaluation_directory" in st.session_state:
        report_directory = Path(st.session_state["evaluation_directory"])
        try:
            evaluation = read_json(report_directory / "metrics.json")
            if evaluation["prediction_sha256"] != sha256(directory / "predictions.json"):
                raise DataError("评价与当前预测不匹配")
            report_html = (report_directory / "report.html").read_bytes()
            st.dataframe(pd.DataFrame([{"设备": NAMES.get(a, a), "输出": version, "MAE W": v["mae_w"], "运行 MAE W": v["active_mae_w"], "F1": v["f1"], "评价覆盖": v["evaluation_coverage"]} for a, versions in evaluation["metrics"].items() for version, v in versions.items()]))
            if evaluation.get("attribution"):
                st.dataframe(pd.DataFrame([{"输出": version, **evaluation["attribution"][version]} for version in ("raw", "corrected", "all_unexplained_control")]))
            if evaluation.get("intervals"):
                st.dataframe(pd.DataFrame([{"设备": a, "输出": version, **values} for a, versions in evaluation["intervals"].items() for version, values in versions.items()]))
        except (DataError, OSError, ValueError, TypeError, KeyError) as error:
            st.session_state.pop("evaluation_directory", None)
            st.error("评价结果不可用，请重新生成：" + str(error))
    st.download_button("下载分析报告 HTML", report_html, "ElectricityPlot-report.html", "text/html")
    st.download_button("下载预测 CSV", prediction_csv, "ElectricityPlot-predictions.csv", "text/csv")
    st.download_button("下载完整预测 JSON", (directory / "predictions.json").read_bytes(), "ElectricityPlot-predictions.json", "application/json")
