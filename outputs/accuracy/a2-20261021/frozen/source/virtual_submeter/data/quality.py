"""Self-contained, offline data quality view; no predictions or confidence claims."""

from html import escape
from pathlib import Path


def render_quality(manifest: dict, path: Path) -> None:
    rows = []
    details = []
    for house, stats in manifest["households"].items():
        for channel, item in stats["channels"].items():
            rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
                house, channel, stats["bins"], item["valid_bins"], f"{item['coverage']:.1%}", item["nonzero_bins"], f"{item['observed_energy_kwh']:.4f}"
            )) + "</tr>")
            details.append(f"<li>House {escape(house)} / {escape(channel)}: {escape(str(dict(item['flags'])))}</li>")
    notices = "".join(f"<li>{escape(item)}</li>" for item in manifest["upstream_limitations"])
    path.write_text("""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ElectricityPlot · 数据质量</title><style>body{font:16px/1.6 system-ui;margin:40px auto;max-width:1120px;padding:0 20px;color:#18333d;background:#f5f8fa}h1{color:#087f8c}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce4e8}section{background:white;padding:20px;margin:20px 0;border-radius:12px}li{overflow-wrap:anywhere}.scroll{overflow:auto}</style>
<h1>ElectricityPlot <small>数据质量审计</small></h1><p>UTC · 8 秒网格 · M1 数据检查</p>
<section><b>本页没有模型预测。</b>覆盖率只描述当前文件中可用的完整区间。清洗数据无法恢复上游填补前的原始缺失；零值不证明设备关闭。电量仅为有效区间积分，不能当作完整日用电量。</section>
<div class="scroll"><table><thead><tr><th>家庭</th><th>通道</th><th>区间数</th><th>有效区间</th><th>覆盖率</th><th>非零区间</th><th>有效区间 kWh</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table></div><section><h2>上游限制</h2><ul>" + notices + "</ul></section><section><h2>质量标记计数</h2><ul>" + "".join(details) + "</ul></section></html>", encoding="utf-8")
