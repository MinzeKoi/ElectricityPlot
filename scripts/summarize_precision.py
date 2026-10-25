"""Build a separate, immutable precision handoff from sealed experiment evidence.

Enrich the already selected diagnostic snippets with every candidate prediction;
never reopen development labels or change the source-selected candidate.
"""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.accuracy import precision
from virtual_submeter.data.common import DataError, read_json, sha256, utc, write_json


def summarize(run, output):
    verification = precision.verify(run)
    if read_json(run / "run.json")["status"] != "evaluated":
        raise DataError("Handoff requires completed sealed evaluation")
    if output.exists():
        raise DataError("Handoff destination exists; use a new directory")
    frozen = read_json(run / "freeze.json")
    metrics = read_json(run / "metrics.json")["rows"]
    groups = frozen["groups"]
    baseline = [r for r in metrics if r["candidate"] == "W50" and r["partition"] == precision.DEVELOPMENT
                and r["appliance"] == "microwave" and r["view"] == "raw"]
    mean_fp = np.mean([r["false_positive"] for r in baseline])
    summaries = []
    for group in groups:
        entry = {"candidate": group}
        for label, partition in (("source", precision.SOURCE_VALIDATION), ("development", precision.DEVELOPMENT)):
            rows = [r for r in metrics if r["candidate"] == group and r["partition"] == partition
                    and r["appliance"] == "microwave" and r["view"] == "raw"]
            entry[label] = {k: float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None
                            for k in ("precision", "recall", "f1", "true_positive", "false_positive", "false_negative",
                                      "active_mae_w", "energy_absolute_error_kwh")}
            entry[label]["seed_recalls"] = {str(r["seed"]): r["recall"] for r in rows}
        entry["development_fp_reduction"] = 1 - entry["development"]["false_positive"] / mean_fp
        entry["post_hoc_development_checks"] = precision.development_gate(metrics, group, precision.load_plan(run / "plan.json"))["checks"]
        entry["post_hoc_checks_do_not_reselect"] = True
        summaries.append(entry)
    results = {"status": "precision_development_completed_target_not_met" if not read_json(run / "summary.json")["precision_exit_passed"] else "precision_development_target_met",
               "freeze_sha256": sha256(run / "freeze.json"), "evaluation_evidence_sha256": sha256(run / "evidence.json"),
               "execution_code_sha256": read_json(run / "registration.json")["code_sha256"],
               "verification": verification, "source_selection": read_json(run / "selection.json"),
               "decision": read_json(run / "summary.json"), "groups": summaries,
               "development_labels_reopened": False, "final_households_accessed": False,
               "diagnostic_snippets": "sealed baseline/selected error windows; all candidate predictions added without reselection"}

    with (run / "error_segments.csv").open(encoding="utf-8-sig", newline="") as f:
        segments = list(csv.DictReader(f))
    by_id = {s["segment_id"]: s for s in segments}
    predictions = {}
    for case in frozen["cases"]:
        if case["partition"] == precision.DEVELOPMENT:
            with np.load(run / case["path"], allow_pickle=False) as a:
                predictions[case["candidate"], case["seed"]] = (a["stamps"].copy(), a["raw"][:, 1].copy())
    curves = []
    with (run / "error_curves.csv").open(encoding="utf-8-sig", newline="") as f:
        for original in csv.DictReader(f):
            segment = by_id[original["segment_id"]]
            if not original["timestamp"]:
                continue
            stamp, seed = utc(original["timestamp"]), int(segment["seed"])
            row = {k: original[k] for k in ("segment_id", "timestamp", "truth_w", "aggregate_w")}
            row.update(seed=seed, error_type=segment["error_type"])
            for group in groups:
                stamps, powers = predictions[group, seed]
                i = int(np.searchsorted(stamps, stamp))
                if i >= len(stamps) or stamps[i] != stamp:
                    raise DataError("Snippet timestamp not found in a frozen candidate")
                row[group + "_w"] = float(powers[i]) if np.isfinite(powers[i]) else ""
            if row["W50_w"] != (float(original["baseline_w"]) if original["baseline_w"] else ""):
                raise DataError("Sealed baseline diagnostic curve changed")
            curves.append(row)

    output.mkdir(parents=True)
    write_json(output / "results.json", results)
    with (output / "all_candidate_error_curves.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    body = '<h1>微波炉精确率改进结果</h1><p>12 个模型、3 个训练种子；单个已查看开发家庭。最终家庭 8/9/20 保持隔离。</p>'
    body += '<p><strong>结论：' + ('预定开发出口通过。' if results['decision']['precision_exit_passed'] else '未达到 Precision≥8%、每种子 Recall≥85% 的目标，保留原模型。') + '</strong></p>'
    body += '<h2>House 17 开发区间：三种子均值</h2><table><tr><th>候选</th><th>Precision</th><th>Recall</th><th>F1</th><th>FP减少</th><th>运行段 MAE W</th><th>电量绝对误差 kWh</th></tr>'
    for entry in summaries:
        d = entry['development']
        body += f'<tr><td>{escape(entry["candidate"])}</td>'
        body += ''.join(f'<td>{v:.2%}</td>' if v is not None else '<td>不可用</td>' for v in (d['precision'], d['recall']))
        body += f'<td>{d["f1"]:.4f}</td><td>{entry["development_fp_reduction"]:.1%}</td><td>{d["active_mae_w"]:.2f}</td><td>{d["energy_absolute_error_kwh"]:.3f}</td></tr>'
    body += '</table><p>W1 的精确率提高伴随大幅漏检。W10 和困难负样本版本也未满足逐种子召回保护；不根据开发结果改选候选。电量总误差下降不能代替运行段功率或状态达标。</p>'
    body += '<h2>逐种子召回率</h2><table><tr><th>候选</th><th>42</th><th>1019</th><th>2026</th></tr>'
    for entry in summaries:
        body += '<tr><td>' + escape(entry['candidate']) + '</td>' + ''.join(f'<td>{entry["development"]["seed_recalls"][str(s)]:.2%}</td>' for s in precision.SEEDS) + '</tr>'
    body += '</table><h2>相同时段的误报与真实事件对照</h2><p>片段按原模型最长连续错误/检出段选取；下表每个时段等间隔展示五个点，完整八秒曲线见下载文件。数值为原始预测功率，单位 W；开机定义始终为≥200 W。</p>'
    selected_segments = [s for s in segments if s['candidate'] == 'W50' and s['seed'] == '42' and s['error_type'] in ('false_positive', 'true_positive') and s['segment_id'].endswith('-1')]
    for segment in selected_segments:
        points = [r for r in curves if r['segment_id'] == segment['segment_id']]
        body += '<h3>' + ('原模型误报片段' if segment['error_type'] == 'false_positive' else '原模型真实检出片段') + '：' + escape(segment['start']) + '</h3>'
        body += '<table><tr><th>时间 UTC</th><th>真实功率</th>' + ''.join('<th>' + escape(g) + '</th>' for g in groups) + '</tr>'
        for i in np.linspace(0, len(points) - 1, min(5, len(points)), dtype=int):
            row = points[i]
            body += '<tr><td>' + escape(row['timestamp']) + '</td>'
            for key in ['truth_w'] + [g + '_w' for g in groups]:
                v = row[key]
                body += '<td>' + (f'{float(v):.1f}' if v != '' else '缺失') + '</td>'
            body += '</tr>'
        body += '</table>'
    body += '<p><a href="all_candidate_error_curves.csv">全部候选同窗口曲线 CSV</a> · <a href="results.json">完整来源选择与验收证据</a></p>'
    body += '<p>水壶与洗衣机原始预测逐点不变；组合功率修正另行报告。W50 三种子与历史权重逐参数一致。正式多户结论与独立成员复现均不由本轮开发实验代替。</p>'
    (output / 'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:36px;max-width:1200px;line-height:1.6}table{border-collapse:collapse}td,th{padding:8px 12px;border:1px solid #ccc}th{background:#f3f5f7}h2{margin-top:32px}</style>' + body, encoding='utf-8')
    write_json(output / 'evidence.json', {'parent_evidence_sha256': sha256(run / 'evidence.json'),
               'summary_script_sha256': sha256(Path(__file__)),
               'artifacts': {name: sha256(output / name) for name in ('results.json', 'all_candidate_error_curves.csv', 'report.html')}})
    return {'status': 'precision_handoff_ready', 'output': str(output), 'curve_rows': len(curves),
            'development_labels_reopened': False, 'precision_exit_passed': results['decision']['precision_exit_passed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/accuracy/precision-20261024-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/accuracy/precision-20261024-handoff-v1')
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.run, args.output), ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, ValueError, KeyError) as error:
        print(json.dumps({'status': 'precision_handoff_failed', 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
