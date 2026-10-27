"""Verify and seal the completed Task21 result; never train or select a new recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]


def child(script,command,run):
    args=[str(ROOT/'.venv-m2/Scripts/python.exe'),'-B',str(script)]
    if command:args.append(command)
    args+=['--run',str(run)]
    p=subprocess.run(args,cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True,encoding='utf-8',errors='strict')
    if p.returncode:raise RuntimeError((p.stdout or '')+(p.stderr or ''))
    return p.stdout or ''


def main(run,authorization):
    run=run.resolve();authorization=authorization.resolve();sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_task21 as t
    t.require(not (run/'execution_delivery_manifest.json').exists(),'Completed execution cannot be overwritten')
    t.authorize(run,authorization);t.registered(run)
    source=t.read_json(run/'source_selection.json')
    if source['eligible']:t.require((run/'evaluation_complete.json').exists(),'Source passed; required one-shot development acceptance is incomplete')
    else:t.require(not any((run/k).exists() for k in ('freeze.json','predictions','evaluation','evaluation_complete.json')),'Source rejected: development must remain unread')
    # Both are the sealed snapshot entry points; no mutable working-copy imports.
    script=run/'run_microwave_task21.py'
    results={};sealed_failures={}
    for command in ('verify','report'):
        print('Running sealed '+command,flush=True)
        try:
            output=child(script,command,run)
        except RuntimeError as error:
            # A sealed reviewer omitted event from its validation field list.
            # Record the defect; never modify or monkeypatch the sealed source.
            t.require("KeyError: 'event'" in str(error),'Unexpected sealed verification failure: '+str(error))
            with (run/f'execution_sealed_{command}_failure.log').open('x',encoding='utf-8',newline='\n') as f:f.write(str(error))
            sealed_failures[command]=dict(error="KeyError: 'event'",training_or_source_failure=False,
                sealed_source_modified=False,independent_replacement='scripts/verify_microwave_task21_execution.py')
        else:
            with (run/f'execution_{command}.log').open('x',encoding='utf-8',newline='\n') as f:f.write(output)
            results[command]=json.loads(output)
    print('Running independent threshold/checkpoint/source audit',flush=True)
    output=child(ROOT/'scripts/verify_microwave_task21_execution.py',None,run)
    numeric=json.loads(output)
    t.require(numeric['passed'],'Independent numeric review failed')
    t.fresh(run/'independent_execution_numeric.json',numeric)
    t.fresh(run/'execution_snapshot_verification.json',dict(sealed_failures=sealed_failures,successful_results=results,
        independent_replacement_passed=numeric['passed']))
    t.require(numeric['new_fits']==3 and numeric['optimizer_updates']==10584 and numeric['source_qualified']==source['eligible'], 'Execution review matrix differs')
    control=t.read_json(run/'source_control.json')
    lines=['# Task21 固定困难负片段采样执行报告','',
        f"注册 SHA256：`{t.sha256(run/'registration.json')}`",'',
        '用户新增授权最多3个模型、10584次更新；实际完成3个模型，每个3528次更新，共10584次。没有控制模型重训、失败重试或追加候选。',
        'v2封存源码、配方、归一化、初始化和完整批次索引未改动。训练通过封存入口复用原fit_one，Adam 0.001、unitBCE、C127 TCN；唯一改动仍为每批后120个负槽位替换。',
        '', '| seed | 更新 | 原85保护所选检查点 | 原点阈值 | 来源 L95 | 来源 H95 |',
        '|---|---:|---:|---:|---:|---:|']
    for seed in t.SEEDS:
        v=numeric['seeds'][str(seed)];point=v['selected']
        lines.append(f"| {seed} | 3528 | {point['epoch']} | {point['threshold']:.9f} | {v['L95']} | {v['H95']} |")
    lines+=['','来源比较（每seed对block4/5平均；P/R/F1）：','',
        '| seed | control_guard95 P | R | F1 | 候选 P | R | F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for seed in t.SEEDS:
        old=next(r for r in control['rows']['guard95'] if r['seed']==seed)
        new=next((r for r in source['rows'] if r['seed']==seed),None)
        new_values=' | '.join(f'{new[k]:.8f}' for k in ('precision','recall','f1')) if new else '不可行 | 不可行 | 不可行'
        lines.append(f"| {seed} | {old['precision']:.8f} | {old['recall']:.8f} | {old['f1']:.8f} | {new_values} |")
    lines+=['','来源逐块：','', '| seed | block | TP | FP | FN | TN | P | R | F1 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for recipe in source['recipes']:
        for row in recipe['blocks']:
            lines.append(f"| {recipe['seed']} | {row['block']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | {row['precision']:.8f} | {row['recall']:.8f} | {row['f1']:.8f} |")
    old=control['rows']['guard95'];new=source['rows']
    gates=dict(three_seed_feasible=len(new)==3,each_block_R95=all(r['feasible'] and all(b['recall']>=.95 for b in r['blocks']) for r in source['recipes']),
        nondegenerate=all(not b['degenerate'] for r in source['recipes'] for b in r['blocks']),
        mean_F1_not_lower=len(new)==3 and sum(r['f1'] for r in new)>=sum(r['f1'] for r in old),
        each_seed_mean_R_drop_le_005=len(new)==3 and all(n['recall']-o['recall']>=-.05 for o,n in zip(old,new)),
        at_least_two_seed_P_up=sum(n['precision']>o['precision'] for o,n in zip(old,new))>=2)
    lines+=['','固定来源资格：']+[f"- {key}：{'通过' if value else '未过'}" for key,value in gates.items()]
    if not source['eligible']:
        conclusion='来源资格未通过，保留冻结control_guard95；候选不进入开发评价。'
        lines+=['',f'**{conclusion}**','House17开发读取0，未生成开发预测矩阵，未读取开发标签或计算开发成绩；最终8/9/20继续隔离。没有开发后改选、改配方或追加训练。']
    else:
        summary=t.read_json(run/'evaluation/summary.json')
        conclusion='一次开发验收通过。' if summary['development_acceptance_passed'] else '一次开发验收未通过；不得改选、修改配方或追加训练。'
        lines+=['',f'**{conclusion}**','开发前封存A/W50/候选三seed九格完整预测矩阵，公开gate.ordinary入口与配方加载结果逐点一致。原137523有效点、A原预测及其他设备状态已核对。','']
        lines += [f"- {key}：{'通过' if value else '未过'}" for key,value in summary['acceptance_gates'].items()]
        lines+=['','House17仍是反复使用的工程开发，不能称为独立测试。最终8/9/20数据继续隔离。']
    lines+=['',
        '独立核验：实际10584批日志的索引hash与封存批次逐条一致，训练实际曝光与封存干运行一致，初始化逐张量exact；36个检查点的85%保护、阈值排序及早epoch规则已另行实现复算；来源L95/H95候选集合与排序、8秒连续r15局部max及标量混淆矩阵均经独立实现核对。',
        '准备阶段已报告的曝光集中度保持不变：seed1019最大单中心hard曝光9408次（2.2222%），seed42为3136次，seed2026为4704次。这是层级片段均衡轮转的实际曝光，未据此临时调参。',
        '封存快照verify/report均已实际调用，但其执行复核代码在重选检查点时只加载t/y，遗漏原规则所需event字段，导致KeyError。该缺陷仅在结果复核阶段暴露，未影响已完成训练或来源筛选；完整错误日志已保留。v2源码没有修改，也未通过monkeypatch绕过。',
        '目录外独立复核程序另行加载原event身份，独立重算36检查点、全部实际批日志与曝光、初始化、归一化、权重、来源L95/H95/r15及混淆矩阵，通过后出具本报告与封存。重复只读检查使用该独立程序，并核对报告和全体交付hash/mtime不变。原封存verify/report缺陷仍保留，不能声称它们成功；后续若修复须单独注册，不能据此追加训练。',
        '本试验只评估分类状态；不据此推断功率、运行段MAE或电量改善。默认行为和旧训练入口未改动。预算已用满，不自动继续训练。','']
    with (run/'EXECUTION_REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
    before=t.read_json(run/'inputs_before.json');t.require(t.snapshot(before)==before,'Frozen inputs changed')
    t.fresh(run/'execution_result.json',dict(conclusion=conclusion,source_eligible=source['eligible'],source_gates=gates,
        new_fits=3,optimizer_updates=10584,development_evaluated=source['eligible'],final_reads=0,
        registration_sha256=t.sha256(run/'registration.json'),authorization_sha256=t.sha256(authorization)))
    scripts=[ROOT/'scripts'/n for n in ('execute_microwave_task21_budget.py','verify_microwave_task21_execution.py','seal_microwave_task21_execution.py')]
    t.fresh(run/'execution_delivery_manifest.json',dict(status='complete_fixed_task21',new_fits=3,optimizer_updates=10584,
        registration_sha256=t.sha256(run/'registration.json'),preparation_manifest_sha256=t.sha256(run/'preparation_manifest.json'),
        authorization_sha256=t.sha256(authorization),sealed_verify_report_succeeded=not sealed_failures,
        sealed_review_defect=sealed_failures,independent_execution_audit_passed=True,
        artifacts=t.inventory(run),execution_tools=t.snapshot(scripts)))
    all_files=[*before,*scripts,authorization,run.with_name(run.name+'.readonly-receipt.json'),*[run/k for k in t.inventory(run)]]
    sealed=t.snapshot(all_files);calls=[]
    for iteration in (1,2):
        print(f'Read-only pass {iteration}: independent execution verify and report',flush=True)
        out=child(ROOT/'scripts/verify_microwave_task21_execution.py',None,run)
        t.require(json.loads(out)==numeric,'Repeat independent execution verification changed')
        report=(run/'EXECUTION_REPORT.zh.md').read_text(encoding='utf-8')
        calls.append(dict(iteration=iteration,command='independent_verify',stdout_sha256=hashlib.sha256(out.encode()).hexdigest(),
            report_sha256=hashlib.sha256(report.encode()).hexdigest()))
        t.require(t.snapshot(sealed)==sealed,'Read-only execution verification mutated hash/mtime')
    receipt=run.with_name(run.name+'.execution-readonly-receipt.json')
    t.fresh(receipt,dict(passed=True,passes=2,all_input_delivery_hash_mtime_unchanged=True,calls=calls,
        execution_manifest_sha256=t.sha256(run/'execution_delivery_manifest.json'),snapshots=sealed))
    print(json.dumps(dict(status='complete',conclusion=conclusion,new_fits=3,updates=10584,
        manifest_sha256=t.sha256(run/'execution_delivery_manifest.json'),readonly_receipt=str(receipt)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    a=p.parse_args();main(a.run,a.authorization)
