"""Human UI for the shared A3/R2 session service."""

from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from ..accuracy import live
from ..data.common import DataError, read_json, sha256
from ..models.prediction import predict
from ..reporting.html import NAMES

ERRORS = (DataError, OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError)


def clear_results():
    for key in ("r4_session_directory", "r4_comparison", "m3_session_directory", "m3_model_directory", "m3_comparison"):
        st.session_state.pop(key, None)


def panel(root: Path, model_dir: Path, manifest: Path, partition_id: str):
    st.subheader("少量确认，适配当前家庭")
    st.info("当前为开发演示方法，尚未通过正式方法出口。每五次回答更新状态；功率和电量保持冻结。")
    st.caption("每个设备窗口的一次回答计 1 次；不确定和跳过同样计数。REFIT 历史操作演示不代表真实用户确认效果。")
    try:
        protocol = read_json(manifest)
        with st.expander("会话方法与版本"):
            learner = root / st.text_input("学习器配置", "inputs/learner.json" if (root / "inputs/learner.json").exists() else "configs/r1_learners_20261022.json")
            query = root / st.text_input("选片配置", "inputs/query.json" if (root / "inputs/query.json").exists() else "configs/r2_stability_20261023.json")
            legacy = root / st.text_input("窗口配置", "inputs/legacy.json" if (root / "inputs/legacy.json").exists() else "configs/m3_adaptation.json")
            source = st.text_input("L2 来源状态头目录（L1 无需）", "inputs/source_head" if (root / "inputs/source_head").exists() else "")
            plan = live.sessions.load_plan(query)
            st.write("实际学习器：" + plan["learner"] + "；更新间隔：5 次")
        budget = st.selectbox("确认预算", protocol["query_budgets"], index=protocol["query_budgets"].index(10) if 10 in protocol["query_budgets"] else 0)
        strategy = st.selectbox("选片策略", plan["strategies"], format_func=lambda v: {"random": "随机", "uncertainty_static": "静态不确定性", "uncertainty": "动态不确定性", "proposed": "原拟议组合（开发对照）"}.get(v, v))
        seed = st.selectbox("查询种子", protocol["random_seeds"])
        signature = (str(model_dir), sha256(model_dir / "model.json"), sha256(manifest), partition_id,
                     sha256(learner), sha256(query), sha256(legacy), source, budget, strategy, seed)
        if st.session_state.get("r4_signature") != signature:
            clear_results()
            st.session_state["r4_signature"] = signature
        workspace = root.parent / (root.name + "-workspace") if (root / "manifest.json").exists() else root / "outputs/ui"
        default = st.session_state.setdefault("r4_default_session", str(workspace / ("r4-" + uuid.uuid4().hex)))
        directory = root / st.text_input("确认会话目录（支持加载恢复）", default)
        if st.button("新建确认会话"):
            clear_results()
            live.create(directory, model_dir, manifest, partition_id, learner, query, legacy,
                        strategy, seed, budget, root / source if source.strip() else None)
            st.session_state["r4_session_directory"] = str(directory)
        if st.button("加载确认会话"):
            clear_results()
            current = live.status(directory)
            if (any(current[k] != v for k, v in (("partition_id", partition_id), ("budget", budget), ("strategy", strategy), ("seed", seed)))
                    or any(sha256(directory / name) != value for name, value in
                           (("inputs/model/model.json", signature[1]), ("inputs/protocol/protocol.json", signature[2]),
                            ("inputs/learner.json", signature[4]), ("inputs/query.json", signature[5]), ("inputs/legacy.json", signature[6])))):
                raise DataError("会话与当前模型、协议或方法设置不同")
            st.session_state["r4_session_directory"] = str(directory)
        if "r4_session_directory" not in st.session_state:
            return
        directory = Path(st.session_state["r4_session_directory"])
        current = live.status(directory)
        st.metric("确认次数", f"{current['budget_used']} / {current['budget']}")
        metadata = read_json(Path(current["model"]) / "model.json")
        st.caption(f"当前模型 {metadata['model_id']} · {current['learner']} · 已适配 {current['fitted_budget']} 次回答；未满五次的回答已保存，等待下一轮更新。")
        pending = current["pending"]
        if pending:
            st.write(f"在 {pending['window_start']} 至 {pending['window_end']}（UTC），是否使用过{NAMES.get(pending['appliance_type'], pending['appliance_type'])}？")
            rows = read_json(directory / "prediction/predictions.json")["rows"]
            frame = pd.DataFrame([{"时间 UTC": r["timestamp"], "总表 W": r["aggregate_power_w"]} for r in rows if pending["window_start"] <= r["timestamp"] < pending["window_end"]])
            frame["时间 UTC"] = pd.to_datetime(frame["时间 UTC"], utc=True)
            st.line_chart(frame.set_index("时间 UTC"))
            response = st.radio("你的回答", ["used", "not_used", "unsure", "skipped"],
                                format_func=lambda v: {"used": "用过", "not_used": "没用过", "unsure": "不确定", "skipped": "跳过"}[v], key="r4_answer_" + pending["query_id"])
            if st.button("提交回答", key="r4_submit_" + pending["query_id"]):
                st.session_state.pop("r4_comparison", None)
                live.answer(directory, pending["query_id"], response)
                st.rerun()
        else:
            st.info("确认预算已完成。")
        st.dataframe(pd.DataFrame(current["records"]))
        st.download_button("下载确认记录 JSON", (directory / "session/session.json").read_bytes(), "ElectricityPlot-confirmations.json", "application/json")
        if st.button("查看当前适配前后状态并导出"):
            st.session_state.pop("r4_comparison", None)
            comparison = directory / ("comparison_" + uuid.uuid4().hex)
            for label, model in (("before", directory / "session/models/budget_0"), ("after", Path(current["model"]))):
                predict(model, comparison / label, directory / "inputs/protocol/protocol.json", partition_id)
            st.session_state["r4_comparison"] = (str(comparison), metadata["model_id"])
        saved = st.session_state.get("r4_comparison")
        if saved and saved[1] == metadata["model_id"]:
            comparison = Path(saved[0])
            before = read_json(comparison / "before/predictions.json")["rows"]
            after = read_json(comparison / "after/predictions.json")["rows"]
            appliance = st.selectbox("状态比较设备", metadata["appliances"], format_func=lambda a: NAMES.get(a, a))
            frame = pd.DataFrame([{"时间 UTC": b["timestamp"], "确认前": b["appliances"][appliance]["state"],
                                   "确认后": a["appliances"][appliance]["state"]} for b, a in zip(before, after)])
            frame["时间 UTC"] = pd.to_datetime(frame["时间 UTC"], utc=True)
            st.line_chart(frame.set_index("时间 UTC").astype(float))
            st.caption("这里只比较总表推理状态，未读取未来分表标签；没有真值时不显示 F1。状态分数未校准为可靠概率。")
            for label, name in (("before", "适配前"), ("after", "适配后")):
                for filename, mime in (("report.html", "text/html"), ("predictions.csv", "text/csv"), ("predictions.json", "application/json")):
                    st.download_button("下载" + name + " " + filename, (comparison / label / filename).read_bytes(), label + "-" + filename, mime)
    except ERRORS as error:
        clear_results()
        st.error("确认或适配未完成，已提交回答保留；可加载原目录重试：" + str(error))
