"""Standalone offline report with real curves and optional ground-truth metrics."""

from html import escape
import json
from pathlib import Path
from .trust import sections

NAMES = {"kettle": "水壶", "microwave": "微波炉", "washing_machine": "洗衣机"}


def render_report(payload: dict, path: Path, evaluation: dict | None = None):
    valid = sum(row["unexplained_power_w"] is not None for row in payload["rows"])
    options = "".join(f'<option value="{escape(a, quote=True)}">{escape(NAMES.get(a, a))}</option>' for a in payload["appliances"])
    table = "<p>本次仅有总表预测，尚无分表评价指标。</p>"
    if evaluation:
        cells = []
        for appliance, versions in evaluation["metrics"].items():
            for version, values in versions.items():
                def display(value):
                    return "不可用" if value is None else f"{value:.4f}"
                cells.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
                    NAMES.get(appliance, appliance), "修正前" if version == "raw" else "修正后",
                    display(values["mae_w"]), display(values["active_mae_w"]), display(values["f1"]),
                    display(values["energy"]["absolute_error_kwh"]), f"{values['evaluation_coverage']:.1%}"
                )) + "</tr>")
        table = "<p>以下为当前真实文件前缀的工程试运行结果；电量误差只在预测和真值共同有效的区间计算。</p><div class='scroll'><table><tr><th>设备</th><th>输出</th><th>MAE W</th><th>运行片段 MAE W</th><th>F1</th><th>电量绝对误差 kWh</th><th>评价覆盖</th></tr>" + "".join(cells) + "</table></div>"
    curves = [{"t": row["timestamp"], "aggregate": row["aggregate_power_w"], "unexplained": row["unexplained_power_w"],
               "a": {a: [v["raw_power_w"], v["power_w"], v.get("lower_w"), v.get("upper_w")] for a, v in row["appliances"].items()}} for row in payload["rows"]]
    embedded = json.dumps(curves, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    limitations = "".join(f"<li>{escape(item)}</li>" for item in payload["upstream_limitations"])
    title = {"engineering_pilot_only": "工程试运行 · 未冻结正式实验", "synthetic_background_stress": "合成背景压力测试", "final_evaluation": "冻结测试评价"}.get(payload["purpose"], "用户导入 · 模型估计")
    if payload.get("data_kind") == "synthetic_contract_fixture":
        title = "合成契约样例 · 不代表 REFIT 实测效果" + (" · 背景压力测试" if payload["purpose"] == "synthetic_background_stress" else "")
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ElectricityPlot · 分项报告</title>
<style>body{font:16px/1.65 system-ui;color:#193440;background:#f1f6f8;margin:32px auto;padding:0 22px;max-width:1180px}h1{color:#087e8b;margin-bottom:0}.muted{color:#607681}section{background:white;border-radius:14px;padding:24px;margin:22px 0}select,input{font:inherit;margin:8px;padding:6px}canvas{display:block;width:100%;height:340px}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce6eb}.scroll{overflow:auto}.badge{color:#765700;background:#fff1c7;padding:4px 10px;border-radius:20px}li{overflow-wrap:anywhere}code{overflow-wrap:anywhere}</style>
<h1>ElectricityPlot</h1><p class="muted">家庭用电分项估计 · UTC</p><span class="badge">""" + escape(title) + """</span>
<section><h2>数据与模型</h2><p>模型：<code>""" + escape(payload["model_id"]) + "</code></p><p>分区：" + escape(payload.get("partition_id") or "用户导入") + f"；有效预测 {valid:,} / {len(curves):,} 个区间。</p><p>本模型未经目标家庭适配，尚未校准可信度。缺失及上下文不足的区间不显示设备关闭。</p></section>" + """
<section><h2>历史曲线</h2><label>设备<select id="device">""" + options + """</select></label><label>显示区间<select id="duration"><option value="450">1 小时</option><option value="1800">4 小时</option><option value="5400">12 小时</option><option value="all">全部</option></select></label><label>回放位置<input id="position" type="range" min="0" value="0" step="1"></label><canvas id="plot"></canvas><p id="period" class="muted"></p><p>总表 <span style="color:#667b8c">■</span>　设备原始估计 <span style="color:#bd669f">■</span>　设备修正估计 <span style="color:#087e8b">■</span>　未解释部分 <span style="color:#dba02e">■</span></p><p class="muted">曲线空段表示不可用。未解释部分包含未建模负荷、测量误差与模型误差，不能直接命名为未知电器。</p></section>
<section><h2>评价</h2>""" + table + """</section><section><h2>待确认与数据限制</h2><p>M2 未发起用户确认，当前确认预算使用量为 0。尚无可靠置信区间。</p><ul>""" + limitations + """</ul><p>有效区间电量不代表完整日账单；使用未来上下文的离线模型不应宣传为实时识别。</p></section>
<script id="series" type="application/json">""" + embedded + """</script><script>
const data=JSON.parse(document.getElementById('series').textContent), c=document.getElementById('plot'), sel=document.getElementById('device'), pos=document.getElementById('position'), dur=document.getElementById('duration');
function draw(){const count=dur.value==='all'?data.length:Number(dur.value);pos.max=Math.max(0,data.length-count);let start=Math.min(Number(pos.value),Number(pos.max));pos.value=start;const rows=data.slice(start,start+count), w=c.clientWidth,h=340,dpr=window.devicePixelRatio||1;c.width=w*dpr;c.height=h*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);const series=[r=>r.aggregate,r=>r.a[sel.value][0],r=>r.a[sel.value][1],r=>r.unexplained],colors=['#667b8c','#bd669f','#087e8b','#dba02e'];let max=1;rows.forEach(r=>series.forEach(f=>{const v=f(r);if(v!==null)max=Math.max(max,v)}));const left=58,right=w-12,top=15,bottom=h-28;ctx.font='12px system-ui';ctx.fillStyle='#607681';for(let i=0;i<=4;i++){let y=bottom-(bottom-top)*i/4;ctx.strokeStyle='#e4edf1';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillText(Math.round(max*i/4)+' W',0,y+4)}series.forEach((f,k)=>{ctx.strokeStyle=colors[k];ctx.lineWidth=k===2?2:1;ctx.beginPath();let open=false;rows.forEach((r,i)=>{let v=f(r);if(v===null){open=false;return}let x=left+(right-left)*i/Math.max(1,rows.length-1),y=bottom-(bottom-top)*v/max;if(open)ctx.lineTo(x,y);else ctx.moveTo(x,y);open=true});ctx.stroke()});document.getElementById('period').textContent=rows.length?rows[0].t+' 至 '+rows[rows.length-1].t+'（UTC）':'';}
[sel,pos,dur].forEach(el=>el.addEventListener('input',draw));window.addEventListener('resize',draw);draw();
</script></html>"""
    if payload.get("adaptation") == "weak_window_state_only":
        summary = payload["query_summary"]
        document = document.replace("本模型未经目标家庭适配，尚未校准可信度。", "本模型已进行窗口弱标签状态适配；功率主干冻结，状态分数未校准为可信概率。")
        detail = (f"确认预算使用 {summary['budget_used']} / {summary['budget_limit']}；"
                  f"策略 {summary['strategy']}，随机种子 {summary['seed']}。"
                  "仅更新状态头，功率与电量保持基线结果；窗口用过不等于每个时刻都开启。"
                  "模拟确认与真实回答分别记录。尚无可靠置信区间。")
        document = document.replace("M2 未发起用户确认，当前确认预算使用量为 0。尚无可靠置信区间。", escape(detail))
        sources = escape(json.dumps(summary["label_sources"], ensure_ascii=False))
        responses = escape(json.dumps(summary["responses"], ensure_ascii=False))
        document = document.replace("<section><h2>评价</h2>", f"<section><h2>适配与预算</h2><p>{escape(payload['state_definition'])}</p><p>标签来源：{sources}；回答计数：{responses}</p></section><section><h2>评价</h2>")
    if payload.get("adaptation") == "a4_state_power_coupled":
        document = document.replace("本模型未经目标家庭适配，尚未校准可信度。",
                                    "A4 功率联动：状态分数与来源训练的开机功率分支共同输出；状态分数未校准为可信概率。")
        detail = (f"确认预算使用 {payload['query_summary']['budget_used']} 次；"
                  f"策略 {payload['query_summary']['strategy']}。"
                  "目标窗口确认仅更新受限状态参数，开机功率幅度分支冻结；用过不等于整个窗口开启，回答不提供真实瓦数。"
                  "功率与电量收益须单独评价；当前为已查看开发家庭上的探索。尚无可靠置信区间。")
        document = document.replace("M2 未发起用户确认，当前确认预算使用量为 0。尚无可靠置信区间。", escape(detail))
    if payload.get("calibration"):
        document = document.replace("尚未校准可信度。", "已加载来源验证残差区间，尚无目标家庭覆盖保证。")
        document = document.replace("尚无可靠置信区间。", "已提供来源经验区间；状态分数尚未校准。")
    document = document.replace("<section><h2>评价</h2>", sections(payload, evaluation) + "<section><h2>评价</h2>")
    document = document.replace("当前真实文件前缀的工程试运行结果", "当前记录的评价结果")
    if payload.get("model_data_kind") == "synthetic_contract_fixture":
        document = document.replace("<h2>数据与模型</h2>", "<h2>数据与模型</h2><p>模型由合成契约数据训练，仅用于工程演示；不构成真实家庭识别效果证据。</p>")
    document = document.replace("const count=dur.value==='all'?data.length:Number(dur.value)", f"const count=dur.value==='all'?data.length:Math.max(1,Math.round(Number(dur.value)*8/{payload['sample_interval_seconds']}))")
    document = document.replace("r=>r.unexplained],colors=", "r=>r.unexplained,r=>r.a[sel.value][2],r=>r.a[sel.value][3]],colors=")
    document = document.replace("'#dba02e'];", "'#dba02e','#80babe','#80babe'];")
    document = document.replace("ctx.lineWidth=k===2?2:1;ctx.beginPath();", "ctx.lineWidth=k===2?2:1;ctx.setLineDash(k>=4?[4,3]:[]);ctx.beginPath();")
    document = document.replace("曲线空段表示不可用。", "曲线空段表示不可用；浅色虚线为修正功率的来源经验区间（已加载校准时）。")
    path.write_text(document, encoding="utf-8")
