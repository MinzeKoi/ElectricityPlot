"""Self-contained error curves with exact diagnostic windows in UTC."""

from html import escape

from ..data.common import read_json, utc, write_json
from ..data.protocol import read_partition


def error_curve(before, after, labels, appliance, threshold):
    truth = {r["timestamp"]: float(r["power_w"]) if r["power_w"] != "" else None for r in labels}
    errors = [int(truth.get(r["timestamp"]) is not None and r["appliances"][appliance]["state"] is not None
                  and r["appliances"][appliance]["state"] != (truth[r["timestamp"]] >= threshold)) for r in after]
    # Select the densest 20-minute error window; deterministic, including ties.
    width = min(150, len(after))
    count = sum(errors[:width])
    best, best_count = 0, count
    for i in range(width, len(errors)):
        count += errors[i] - errors[i - width]
        if count > best_count:
            best, best_count = i - width + 1, count
    start, end = max(0, best - 75), min(len(after), best + width + 75)
    series = []
    for left, right in zip(before[start:end], after[start:end]):
        value = truth.get(right["timestamp"])
        series.append({"timestamp": right["timestamp"], "aggregate_power_w": right["aggregate_power_w"],
                       "truth_power_w": value, "predicted_power_w": right["appliances"][appliance]["power_w"],
                       "truth_state": value >= threshold if value is not None else None,
                       "baseline_state": left["appliances"][appliance]["state"],
                       "adapted_state": right["appliances"][appliance]["state"]})
    return series


def svg(series, keys, state=False):
    colors = ("#334155", "#059669", "#dc2626")
    values = [r[k] for r in series for k in keys if r[k] is not None]
    maximum = 1.0 if state else max(values, default=1) or 1
    points = []
    for key, color in zip(keys, colors):
        commands, last = [], None
        for i, row in enumerate(series):
            value = row[key]
            if value is None:
                last = None
                continue
            x = 40 + i * 910 / max(1, len(series) - 1)
            y = 125 - float(value) / maximum * 100
            contiguous = last is not None and utc(row["timestamp"]) - utc(series[last]["timestamp"]) == 8
            commands.append(f"{'L' if contiguous else 'M'}{x:.2f},{y:.2f}")
            last = i
        points.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    legend = " · ".join(f'<span style="color:{c}">{escape(k)}</span>' for k, c in zip(keys, colors))
    return (f'<p>{legend} · scale 0–{maximum:.1f}{"" if state else " W"}</p>'
            '<svg viewBox="0 0 980 160" role="img" aria-label="UTC diagnosis curve">'
            '<path d="M40 20V125H950" fill="none" stroke="#aaa"/>' + "".join(points) + '</svg>')


def render_report(output, diagnosis):
    sections, curves, table = [], {}, []
    for case in diagnosis["cases"]:
        for a, d in case["devices"].items():
            metric = d["adapted"]
            def number(v):
                return "null" if v is None else f"{v:.4f}"
            table.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in
                (case["household_id"], case["strategy"], case["seed"], case["budget"], a,
                 number(metric["precision"]), number(metric["recall"]), number(metric["f1"]),
                 number(d["f1_delta"]), number(metric["predicted_on_fraction"]), metric["state_pattern"])) + "</tr>")
    appliances = list(diagnosis["cases"][0]["devices"])
    manifest = output / "development/protocol.json"
    for appliance in appliances:
        eligible = [c for c in diagnosis["cases"] if c["budget"] > 0 and c["devices"][appliance]["f1_delta"] is not None]
        if not eligible:
            continue
        case = min(eligible, key=lambda c: (c["devices"][appliance]["f1_delta"], c["path"]))
        baseline = next(c for c in diagnosis["cases"] if c["budget"] == 0 and all(c[k] == case[k] for k in ("household_id", "strategy", "seed")))
        current = read_json(output / "comparison" / case["path"] / "prediction/predictions.json")
        before = read_json(output / "comparison" / baseline["path"] / "prediction/predictions.json")
        labels = read_partition(manifest, case["test_partition"], "development_diagnosis", True)[appliance]
        curve = error_curve(before["rows"], current["rows"], labels, appliance, current["model_settings"]["on_thresholds_w"][appliance])
        curves[appliance] = {"case": case["path"], "selection": "worst F1 delta, densest 20-minute error window; development analysis only", "rows": curve}
        sections.append(f'<h2>{escape(appliance)} · {escape(case["path"])}</h2><p>UTC {escape(curve[0]["timestamp"])} — {escape(curve[-1]["timestamp"])}</p>'
                        + svg(curve, ("aggregate_power_w", "truth_power_w", "predicted_power_w"))
                        + svg(curve, ("truth_state", "baseline_state", "adapted_state"), True))
    write_json(output / "error_curves.json", curves)
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>A0 开发退化诊断</title>'
            '<style>body{font:15px system-ui;margin:32px;color:#172033}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}svg{width:100%;max-width:1100px}h2{margin-top:40px}</style>'
            '<h1>A0 开发退化诊断</h1><p>既有 M2 权重，M3 静态选片与最大池化状态头。种子仅用于查询。全部模型和预测冻结后分析开发答案；模拟确认，非最终盲测。</p>'
            '<p>缺失值保留为空，曲线缺口断开。错误曲线按最差 F1 差值选择，全部案例见下表与 CSV。功率不随状态适配变化。</p>'
            '<p><a href="diagnosis.json">诊断 JSON</a> · <a href="query_distribution.csv">查询分布</a> · <a href="queries.csv">查询明细</a> · <a href="availability.json">数据与环境清单</a></p>'
            '<table><tr>' + ''.join(f'<th>{k}</th>' for k in ('house', 'strategy', 'seed', 'budget', 'appliance', 'precision', 'recall', 'F1', 'ΔF1', 'on fraction', 'pattern'))
            + '</tr>' + ''.join(table) + '</table>' + ''.join(sections) + '</html>')
    (output / "report.html").write_text(page, encoding="utf-8")
