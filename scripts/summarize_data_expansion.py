"""Readable data sizes, model costs and paired effects from sealed expansion evidence."""

import argparse
import csv
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.data.common import DataError, read_json, sha256, write_json
from virtual_submeter.accuracy.data_expansion import DEVICES, verify


def summarize(run, output):
    verify(run)
    if output.exists():
        raise DataError("Handoff output exists; select a fresh destination")
    output.mkdir(parents=True)
    plan = read_json(run / "plan.json")
    summary = read_json(run / "summary.json")
    models = []
    for group in ("A", "B", "C", "D"):
        for seed in plan["seeds"]:
            for device in DEVICES:
                path = run / f"models/{group}/seed_{seed}/{device}/model.json"
                if not path.exists():
                    continue
                model = read_json(path)
                models.append({"group": group, "seed": seed, "appliance": device,
                               "households": model["training_households"],
                               "valid_training_windows": sum(model["source_train_windows"].values()),
                               "windows_by_partition": model["source_train_windows"],
                               "optimizer_steps": model["optimizer_steps"],
                               "selected_checkpoint": model["selected_epoch"],
                               "fit_seconds": model["timing_seconds"]["fit"],
                               "model_sha256": sha256(path), "weights_sha256": model["weights"]["sha256"]})
    evidence = {"source_run": str(run.resolve()), "summary_sha256": sha256(run / "summary.json"),
                "metrics_refit_sha256": sha256(run / "metrics_refit.json"), "models": models,
                "reference_checks": [read_json(p) for p in sorted((run / "reference_checks").glob("*.json"))],
                "conclusions": summary}
    write_json(output / "evidence.json", evidence)
    with (output / "models.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["group", "seed", "appliance", "households", "valid_training_windows", "optimizer_steps", "selected_checkpoint", "fit_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: ("/".join(row[k]) if k == "households" else row[k]) for k in fields} for row in models)
    body = '<!doctype html><meta charset="utf-8"><title>数据扩充与重新训练</title><style>body{font:16px system-ui;margin:40px;max-width:1250px;color:#172b34}table{border-collapse:collapse}td,th{border:1px solid #ccd5da;padding:10px}pre{white-space:pre-wrap;background:#f5f7f9;padding:16px}</style>'
    body += '<h1>数据扩充与重新训练</h1><p>REFIT 来源训练与开发对照。House 8、9、20 保持隔离，界面默认模型未替换。所有扩数组保持现有模型、损失和阈值，采用相同更新步数。</p>'
    body += '<h2>实际训练数据</h2><table><tr><th>组</th><th>设备</th><th>来源家庭</th><th>有效训练窗口</th><th>每种子更新步数</th></tr>'
    for row in models:
        if row["seed"] == 42:
            body += '<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in (row['group'], row['appliance'], '/'.join(row['households']), row['valid_training_windows'], row['optimizer_steps'])) + '</tr>'
    body += '</table><p>窗口数为完整总表上下文和该设备标签同时有效的窗口，不是独立事件数。House 12 的水壶未达到预设完整事件门槛，两段均排除，未另选日期。</p>'
    body += '<h2>配对开发收益</h2><pre>' + escape(json.dumps(summary['comparisons'], ensure_ascii=False, indent=2)) + '</pre>'
    body += '<h2>逐设备识别</h2><table><tr><th>组</th><th>设备</th><th>Precision</th><th>Recall</th><th>F1</th><th>运行段 MAE W</th><th>电量绝对误差 kWh</th></tr>'
    metrics = read_json(run / 'metrics_refit.json')['metrics']
    if (run / 'metrics_external.json').exists():
        metrics += read_json(run / 'metrics_external.json')['metrics']
    for group in sorted({r['group'] for r in metrics}):
        for device in DEVICES:
            rows = [r for r in metrics if r['group'] == group and r['view'] == 'raw' and r['appliance'] == device and r['partition'].startswith('frozen_test_refit_17')]
            values = [group, device]
            for key in ('precision', 'recall', 'f1', 'active_mae_w', 'energy_absolute_error_kwh'):
                valid = [r[key] for r in rows if r[key] is not None]
                values.append(f'{sum(valid)/len(valid):.6f}' if valid else '不可用')
            body += '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in values) + '</tr>'
    body += '</table><h2>外部来源</h2><pre>' + escape(json.dumps(summary['external_status'], ensure_ascii=False, indent=2)) + '</pre>'
    body += '<p>三个种子属于同一个已查看开发家庭，不能作为三户独立泛化证据。Precision/Recall/F1、功率与电量必须分别解读。原始和一致性修正视图、逐种子结果见实验 comparison.csv；训练成本见 models.csv。历史实验及失败记录保留。</p>'
    (output / 'report.html').write_text(body, encoding='utf-8')
    return {'status': 'handoff_complete', 'report': str(output / 'report.html')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.output), ensure_ascii=False, indent=2))
