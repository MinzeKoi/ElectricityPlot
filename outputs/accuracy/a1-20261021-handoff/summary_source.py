"""Build a compact, read-only A1 handoff from verified experiment artifacts."""

import argparse
from html import escape
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.baseline import verify
from virtual_submeter.accuracy.runner import inventory
from virtual_submeter.data.common import DataError, read_json, sha256, write_json


def summarize(run_dir, output):
    verification = verify(run_dir)
    if output.exists():
        raise DataError("Summary destination exists")
    output.mkdir(parents=True)
    state = read_json(run_dir / "run.json")
    plan = read_json(run_dir / "plan.json")
    freeze = read_json(run_dir / "baseline_freeze.json")
    selected = [c for c in state["cases"] if c["stage"] == "selected_seed" or c["stage"] == "multi" and c["candidate"] == state["selected_candidate"]]
    historical = next(c for c in state["cases"] if c["stage"] == "historical")
    per_seed, comparison = [], []
    for appliance in plan["primary_appliances"] + plan["extension_appliances"]:
        observed = []
        for case in selected:
            e = next(v for v in case["evaluations"] if v["role"] == "frozen_test" and v["appliance"] == appliance)
            m = e["raw"]
            row = {"seed": case["seed"], "appliance": appliance, **{k: m[k] for k in ("precision", "recall", "f1", "active_mae_w", "valid_samples")},
                   "energy_absolute_error_kwh": m["energy"]["absolute_error_kwh"],
                   "on_fraction": m["predicted_on_samples"] / m["valid_samples"],
                   "complete_detected": e["events"]["complete_detected"], "complete_events": e["events"]["complete_events"]}
            per_seed.append(row)
            observed.append(row)
        reference = next(v for v in historical["evaluations"] if v["role"] == "frozen_test" and v["appliance"] == appliance)
        means = {k: sum(r[k] for r in observed) / len(observed) for k in ("precision", "recall", "f1", "active_mae_w", "energy_absolute_error_kwh", "on_fraction")}
        comparison.append({"appliance": appliance, "primary": appliance in plan["primary_appliances"], "selected_mean": means,
                           "historical_raw_f1": reference["raw"]["f1"], "historical_default_f1": reference["corrected"]["f1"],
                           "historical_active_mae_w": reference["raw"]["active_mae_w"],
                           "historical_energy_absolute_error_kwh": reference["raw"]["energy"]["absolute_error_kwh"],
                           "f1_delta_vs_historical_raw": means["f1"] - reference["raw"]["f1"]})
    summary = {"date": "2026-10-21", "status": state["status"], "primary_minimum_exit_passed": state["acceptance"],
               "accuracy_superiority_established": False, "extension_exit_passed": False,
               "selected_candidate": state["selected_candidate"], "training_seeds": plan["seeds"], "trained_models": 8,
               "historical_references": 1, "source_run": run_dir.as_posix(), "verification": verification,
               "baseline_freeze": freeze, "comparison": comparison, "per_seed_development": per_seed,
               "admission": read_json(run_dir / "protocol/admission.json"),
               "limitations": ["Only inspected House 3 development evaluation; no final multi-household evidence.",
                               "Primary-device event detection passed, but pointwise F1 is below historical raw-power reference.",
                               "Microwave precision is about 2-3 percent; many false positives remain.",
                               "Washing machine is an extension and is almost always on; one source-validation seed is entirely on.",
                               "Energy error remains large; state-only improvement is not demonstrated here either.",
                               "Source validation has about 54 percent valid label coverage; development duration is four calendar days.",
                               "A1 source expansion and candidate selection are not confirmation benefits."],
               "source_evidence_sha256": sha256(run_dir / "evidence.json"),
               "source_summary_sha256": sha256(run_dir / "summary.json")}
    write_json(output / "summary.json", summary)
    shutil.copyfile(Path(__file__), output / "summary_source.py")
    names = {"kettle": "水壶", "microwave": "微波炉", "washing_machine": "洗衣机（扩展）"}
    table = '<table><tr><th>设备</th><th>三种子平均 F1</th><th>历史模型 F1</th><th>平均 Precision</th><th>平均 Recall</th><th>预测开机占比</th></tr>'
    for row in comparison:
        m = row["selected_mean"]
        table += f'<tr><td>{names[row["appliance"]]}</td><td>{m["f1"]:.3f}</td><td>{row["historical_raw_f1"]:.3f}</td><td>{m["precision"]:.3f}</td><td>{m["recall"]:.3f}</td><td>{m["on_fraction"]:.1%}</td></tr>'
    table += '</table>'
    details = '<table><tr><th>设备</th><th>种子</th><th>F1</th><th>完整阈值事件检出</th><th>运行段 MAE / W</th><th>电量绝对误差 / kWh</th></tr>'
    for row in per_seed:
        details += f'<tr><td>{names[row["appliance"]]}</td><td>{row["seed"]}</td><td>{row["f1"]:.3f}</td><td>{row["complete_detected"]}/{row["complete_events"]}</td><td>{row["active_mae_w"]:.1f}</td><td>{row["energy_absolute_error_kwh"]:.3f}</td></tr>'
    details += '</table>'
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>A1 基础模型交接</title>'
            '<style>body{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#172033;line-height:1.65}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #d7dce5;text-align:left}th{background:#edf2ff}.note{border-left:4px solid #d97706;padding:12px;background:#fff7ed}h2{margin-top:32px}</style>'
            '<h1>A1：基础模型已冻结，识别精度仍有明显问题</h1>'
            '<p>三个候选设置先跑单水壶，再跑多设备。按事先固定的规则选定 active50，再训练三个独立来源种子。主设备通过最低事件检测门槛；103 项工程测试通过。</p>'
            '<p class="note">开发逐点 F1 尚未超过历史模型。微波炉误报多，洗衣机接近持续开机，电量误差仍大。本报告不能作为“基础模型更准”或“确认有效”的证据。</p>'
            '<h2>同一开发时间轴上的结果</h2><p>均按原始功率阈值判状态，支持相同。新模型为三个种子的平均；历史模型仅一份已训练权重。种子不是独立家庭。</p>'
            + table + '<h2>逐种子结果</h2><p>事件为固定阈值段，检出只要求与真实开启段有重合，不是物理周期或事件 F1。</p>' + details
            + '<h2>数据与后续使用</h2><p>House 2 来源训练有效分表约 13.27 天，House 5 验证约 7.61 天；House 3 使用训练前按质量固定的最早 4 天开发记录。House 11 因缺乏主设备完整事件未纳入。两个失败准备目录保留。</p>'
            '<p>后续确认比较必须复用 baseline_freeze.json 中相同种子的同一权重，保留各学习器自己的零确认基线。洗衣机扩展未达质量要求，不能并入主设备收益后隐藏。</p>'
            '<p><a href="summary.json">完整交接数据与权重身份</a> · <a href="../' + escape(run_dir.name) + '/report.html">全部候选明细</a></p>'
            '<p>数据来源：<a href="https://zenodo.org/records/5063428">REFIT 清洗版公开记录</a>。当前仍为真实文件前缀，未完成完整数据或最终多户验收。</p></html>')
    (output / "report.html").write_text(page, encoding="utf-8")
    write_json(output / "evidence.json", {"artifacts_sha256": inventory(output)})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize(args.run, args.output)
    except (DataError, OSError, KeyError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
    print(result["status"], result["selected_candidate"])
