"""Local file import, offline prediction, replay and report download."""

import hashlib
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from ..data.common import DataError, read_json, sha256
from ..data.prepare import prepare
from ..models.artifacts import load_model
from ..models.prediction import predict
from ..reporting.html import NAMES


def main(root: Path):
    st.set_page_config(page_title="ElectricityPlot", page_icon="⚡", layout="wide")
    st.title("ElectricityPlot")
    st.caption("从家庭总功率查看设备用电估计 · 离线历史分析")
    st.info("当前为真实数据前缀上的工程试运行。未解释部分含模型误差；尚未校准可信度。")
    with st.expander("高级：模型设置"):
        model_text = st.text_input("模型目录", "outputs/m2-20261020/seq2point")
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
            source_token = (str(manifest_path), sha256(manifest_path), partition_id)
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
        st.caption("当前基线：" + ("轻量时序模型" if metadata["kind"] == "small_seq2point" else "简单对照") + " · 未经目标家庭适配")
    except (DataError, OSError, ValueError, RuntimeError, KeyError) as error:
        model_token = "invalid"
        context_error = "模型不可用：" + str(error)
        st.warning("尚无可用模型，请先运行基线训练，或填写有效的模型目录。")
    signature = (model_text, model_token, mode, source_token)
    if st.session_state.get("signature") != signature or context_error:
        st.session_state.pop("prediction_directory", None)
        st.session_state.pop("evaluation_directory", None)
        st.session_state["signature"] = signature
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
                    predict(root / model_text, directory / "prediction", input_path=directory / "prepared/aggregate" / f"house_{household}.csv")
                else:
                    predict(root / model_text, directory / "prediction", manifest_path=manifest_path, partition_id=partition_id)
            st.session_state["prediction_directory"] = str(directory / "prediction")
        except (DataError, OSError, ValueError, RuntimeError, KeyError) as error:
            st.error("分析未完成：" + str(error))
    if "prediction_directory" not in st.session_state:
        st.caption("分析成功后将在此显示真实预测曲线和可下载报告。")
        return
    directory = Path(st.session_state["prediction_directory"])
    payload = read_json(directory / "predictions.json")
    appliance = st.selectbox("查看设备", payload["appliances"], format_func=lambda a: NAMES.get(a, a))
    st.subheader("历史回放")
    count = len(payload["rows"])
    progress = st.slider("回放进度（%）", 0, 100, 0)
    position = round((count - 1) * progress / 100)
    duration = st.selectbox("显示时长", ["1 小时", "4 小时", "全部"])
    width = {"1 小时": 450, "4 小时": 1800, "全部": count}[duration]
    frame = pd.DataFrame([{"时间 UTC": row["timestamp"], "总表 W": row["aggregate_power_w"],
                           "设备估计 W": row["appliances"][appliance]["power_w"], "未解释 W": row["unexplained_power_w"]}
                          for row in payload["rows"][position:position + width]])
    frame["时间 UTC"] = pd.to_datetime(frame["时间 UTC"], utc=True)
    st.line_chart(frame.set_index("时间 UTC"))
    valid = sum(row["unexplained_power_w"] is not None for row in payload["rows"])
    st.metric("有效预测覆盖", f"{valid / count:.1%}")
    st.caption("空段表示缺失或上下文不足。确认次数为 0；设备估计尚未经目标家庭适配。")
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
        evaluation = read_json(report_directory / "metrics.json")
        st.dataframe(pd.DataFrame([{"设备": NAMES.get(a, a), "MAE W": v["corrected"]["mae_w"], "F1": v["corrected"]["f1"], "评价覆盖": v["corrected"]["evaluation_coverage"]} for a, v in evaluation["metrics"].items()]))
    st.download_button("下载分析报告 HTML", (report_directory / "report.html").read_bytes(), "ElectricityPlot-report.html", "text/html")
    st.download_button("下载预测 CSV", (directory / "predictions.csv").read_bytes(), "ElectricityPlot-predictions.csv", "text/csv")
