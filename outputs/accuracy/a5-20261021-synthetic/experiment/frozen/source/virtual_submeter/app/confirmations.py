"""Confirmation UI delegates budget, selection and fitting to service modules."""

from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from ..adaptation.session import confirm, create_session, read_session, select_next
from ..adaptation.training import adapt
from ..data.common import DataError, read_json, sha256
from ..models.prediction import predict
from ..reporting.html import NAMES

ERRORS = (DataError, OSError, ValueError, RuntimeError, KeyError, TypeError)


def clear_results():
    for key in ("m3_session_directory", "m3_model_directory", "m3_comparison"):
        st.session_state.pop(key, None)


def panel(root: Path, model_dir: Path, manifest: Path, partition_id: str):
    st.subheader("少量确认，适配当前家庭")
    st.caption("每个片段、每种电器的一次回答计 1 次；不确定和跳过也计数。只适配状态识别，功率预测冻结。")
    protocol = read_json(manifest)
    budget = st.selectbox("确认预算", protocol["query_budgets"], index=min(1, len(protocol["query_budgets"]) - 1))
    strategy = st.selectbox("选片策略", ["random", "uncertainty", "proposed"], format_func=lambda v: {"random": "随机", "uncertainty": "仅不确定性", "proposed": "不确定性 + 耗电影响 + 易回答性代理"}[v])
    seed = st.selectbox("随机种子", protocol["random_seeds"])
    signature = (str(model_dir), sha256(model_dir / "model.json"), sha256(manifest), partition_id, budget, strategy, seed)
    if st.session_state.get("m3_signature") != signature:
        clear_results()
        st.session_state["m3_signature"] = signature
    default = st.session_state.setdefault("m3_default_session", "outputs/ui/m3-" + uuid.uuid4().hex)
    session_text = st.text_input("确认会话目录（可加载已有会话）", default)
    if st.button("新建确认会话"):
        clear_results()
        try:
            directory = root / session_text
            create_session(model_dir, manifest, partition_id, directory, root / "configs/m3_adaptation.json", budget, strategy, seed)
            st.session_state["m3_session_directory"] = str(directory)
        except ERRORS as error:
            st.error("无法创建确认会话：" + str(error))
    if st.button("加载确认会话"):
        clear_results()
        try:
            directory = root / session_text
            session = read_session(directory)
            if (session["model_manifest_sha256"] != signature[1] or session["protocol_sha256"] != signature[2]
                    or session["partition_id"] != partition_id or (session["budget"], session["strategy"], session["seed"]) != (budget, strategy, seed)):
                raise DataError("会话与当前模型、家庭或预算设置不匹配")
            st.session_state["m3_session_directory"] = str(directory)
        except ERRORS as error:
            st.error("无法加载确认会话：" + str(error))
    if "m3_session_directory" not in st.session_state:
        return
    directory = Path(st.session_state["m3_session_directory"])
    try:
        session = read_session(directory)
        query = select_next(directory)
        count = len(session["records"])
        st.metric("确认次数", f"{count} / {session['budget']}")
        if query:
            st.write(f"在 {query['window_start']} 至 {query['window_end']}（UTC），是否使用过{NAMES.get(query['appliance_type'], query['appliance_type'])}？")
            rows = read_json(directory / "prediction/predictions.json")["rows"]
            frame = pd.DataFrame([{"时间 UTC": r["timestamp"], "总表 W": r["aggregate_power_w"]} for r in rows if query["window_start"] <= r["timestamp"] < query["window_end"]])
            frame["时间 UTC"] = pd.to_datetime(frame["时间 UTC"], utc=True)
            st.line_chart(frame.set_index("时间 UTC"))
            response = st.radio("你的回答", ["used", "not_used", "unsure", "skipped"], format_func=lambda v: {"used": "用过", "not_used": "没用过", "unsure": "不确定", "skipped": "跳过"}[v], key="m3_answer_" + query["query_id"])
            if st.button("提交回答", key="m3_submit_" + query["query_id"]):
                confirm(directory, query["query_id"], response)
                st.session_state.pop("m3_model_directory", None)
                st.session_state.pop("m3_comparison", None)
                st.rerun()
        else:
            st.info("预算已用完或没有完整且不重复的候选窗口。")
        st.dataframe(pd.DataFrame(session["records"]))
        st.download_button("下载确认记录 JSON", (directory / "session.json").read_bytes(), "ElectricityPlot-confirmations.json", "application/json")
        if st.button("生成适配模型", disabled=query is not None):
            st.session_state.pop("m3_model_directory", None)
            st.session_state.pop("m3_comparison", None)
            destination = directory / ("model_" + uuid.uuid4().hex)
            adapt(directory, destination)
            st.session_state["m3_model_directory"] = str(destination)
        if "m3_model_directory" in st.session_state:
            adapted = Path(st.session_state["m3_model_directory"])
            metadata = read_json(adapted / "model.json")
            st.success("适配模型已保存：" + metadata["model_id"])
            st.caption("模型版本路径：" + str(adapted))
            st.caption("窗口标签只监督窗口内最大状态分数；功率、分项电量及未解释负荷保持不变。状态分数未经可信度校准。")
            if st.button("比较冻结测试区间的适配前后结果"):
                from ..evaluation.runner import evaluate
                st.session_state.pop("m3_comparison", None)
                test = next(p for p in protocol["partitions"] if p["role"] == "frozen_test" and p["household_id"] == session["household_id"])
                comparison = directory / ("comparison_" + uuid.uuid4().hex)
                for label, model in (("before", model_dir), ("after", adapted)):
                    predict(model, comparison / label, manifest, test["partition_id"])
                    evaluate(comparison / label, manifest, comparison / (label + "_evaluation"))
                st.session_state["m3_comparison"] = str(comparison)
        if "m3_comparison" in st.session_state:
            comparison = Path(st.session_state["m3_comparison"])
            values = []
            for label in ("before", "after"):
                report = read_json(comparison / (label + "_evaluation/metrics.json"))
                for appliance, metric in report["metrics"].items():
                    item = metric["corrected"]
                    values.append({"阶段": "适配前" if label == "before" else "适配后", "电器": NAMES.get(appliance, appliance), "状态 F1": item["f1"], "MAE W": item["mae_w"], "电量误差 kWh": item["energy"]["absolute_error_kwh"]})
            st.dataframe(pd.DataFrame(values))
            st.caption("测试结果只用于最终评价，不回传选片或适配。本次比较不能证明多户效果或真实用户可回答性。")
            for label in ("before", "after"):
                st.download_button("下载" + ("适配前" if label == "before" else "适配后") + "报告", (comparison / (label + "_evaluation/report.html")).read_bytes(), label + "-report.html", "text/html")
    except ERRORS as error:
        clear_results()
        st.error("确认或适配未完成：" + str(error))
