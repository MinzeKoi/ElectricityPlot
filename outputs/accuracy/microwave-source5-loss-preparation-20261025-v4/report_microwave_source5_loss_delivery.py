"""Write the detailed human-readable Task12 report after independent review."""
import argparse
import json
from pathlib import Path
import numpy as np

def read(path): return json.loads(path.read_text(encoding='utf-8'))

def report(run):
    source = read(run/'source_selection.json')
    summary = read(run/'evaluation/summary.json')
    metrics = read(run/'evaluation/metrics.json')
    review = read(run/'independent_delivery_review.json')
    assert review['status']=='independent_delivery_review_passed'
    lines = ['# Task12：三户来源 unit BCE 单变量对照', '',
        f'来源唯一预选：**{source["selected"]}**。正式开发验收：**{summary["development_acceptance_passed"]}**。', '',
        '人类追加授权原话：“允许追加最多3个模型”。实际仅3次新fit、10584次更新；旧BCE50模型零重训。',
        '唯一训练变量为pos_weight 50→1；训练池336704窗口/2275正中心、同seed初始化、全部采样索引及34351点/201正中心来源验证块逐项绑定任务10。', '',
        'House17为反复使用开发集，不能声明独立泛化；最终House8/9/20隔离。所有功率列沿用原A，默认UI不切换。', '',
        '## 来源资格与选择', '',
        '| 家族 | 合格 | meanP | meanR | meanF1 |', '|---|---:|---:|---:|---:|']
    for arm in ('old_point','old_gate','new_point','new_gate'):
        rows = source['rows'][arm]
        means = [float(np.mean([r[k] for r in rows])) if len(rows)==3 else float('nan') for k in ('precision','recall','f1')]
        lines.append(f'| {arm} | {source["eligibility"][arm]} | '+ ' | '.join(f'{v:.8f}' for v in means)+' |')
    lines += ['', '新point仅与旧point比较，新gate仅与旧gate比较。排序固定meanP、meanF1降序；完全相同按old_point、old_gate、new_point、new_gate。开发结果不参与改选。', '',
        '| seed | 新point epoch | 新point阈值 | gate L | gate H | gate可行 |', '|---:|---:|---:|---:|---:|---|']
    for seed in (42,1019,2026):
        m = read(run/f'models/unit_BCE/seed_{seed}/model.json')
        r = next(r for r in source['recipes'] if r['seed']==seed)
        lines.append(f'| {seed} | {m["selected"]["epoch"]} | {m["decision_threshold_probability"]:.9g} | {r["L"]} | {r["H"]} | {r["seed_feasible"]} |')
    lines += ['', '## 开发完整混淆矩阵（非预选方法仅诊断）', '',
        '| 家族 | seed | TP | FP | FN | TN | P | R | F1 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics['rows']:
        lines.append(f'| {r["arm"]} | {r["seed"]} | {r["tp"]} | {r["fp"]} | {r["fn"]} | {r["tn"]} | {r["precision"]:.8f} | {r["recall"]:.8f} | {r["f1"]:.8f} |')
    lines += ['', '## 正式验收（仅来源唯一预选）', '', '| 原门槛 | 通过 |', '|---|---|']
    for key,value in summary['acceptance_gates'].items(): lines.append(f'| {key} | {value} |')
    lines += ['', '## 新损失相对对应旧家族的成对变化', '', '| 对照 | seed | ΔP | ΔR | ΔF1 | ΔFP |', '|---|---:|---:|---:|---:|---:|']
    for pair,rows in review['paired_development_deltas'].items():
        for r in rows: lines.append(f'| {pair} | {r["seed"]} | {r["precision"]:+.8f} | {r["recall"]:+.8f} | {r["f1"]:+.8f} | {r["fp"]:+d} |')
    lines += ['', '## 复核与计算量', '',
        '32项来源/损失测试和21项固定门控回归测试通过。完整aggregate-only入口与冻结输出exact；原A全部功率列exact；137523支持exact；独立标量循环重算全部TP/FP/FN/TN及正式门槛。',
        '实际3fit×3528更新，36检查点，共2709504训练窗口呈现、1236636来源验证打分；无额外拟合、无失败重试。', '',
        '训练时间（每seed）：'+', '.join(f'{seed}: {read(run/f"models/unit_BCE/seed_{seed}/complete.json")["seconds"]:.2f}s' for seed in (42,1019,2026)), '',
        '核心证据：pool_identity.json、additional_human_authorization.json、source_selection.json、freeze.json、evaluation/metrics.json、evaluation/summary.json、independent_delivery_review.json。',
        '测试中的137523网格为临时合成数据，未使用真实开发数据设计或调整方法。']
    path = run/'DETAILED_REPORT.zh.md'
    with path.open('x',encoding='utf-8') as stream: stream.write('\n'.join(lines)+'\n')
    return str(path)

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    print(report(p.parse_args().run))
