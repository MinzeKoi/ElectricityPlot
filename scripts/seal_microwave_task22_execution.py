"""Seal Task22 v4 execution after real snapshot verify/report and independent numeric audit."""
import argparse,hashlib,json,os
from pathlib import Path
import subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
ENV=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')


def child(script,command,run):
    argv=[str(script)]+([command] if command else [])+['--run',str(run)]
    code='import runpy,sys,torch;torch.set_num_threads(4);assert torch.get_num_threads()==4;sys.argv='+repr(argv)+';runpy.run_path('+repr(str(script))+',run_name="__main__")'
    p=subprocess.run([str(ROOT/'.venv-m2/Scripts/python.exe'),'-B','-c',code],cwd=ROOT,env=ENV,capture_output=True,encoding='utf-8',errors='strict')
    out=(p.stdout or '')+(p.stderr or '')
    if p.returncode:raise RuntimeError(out)
    return p.stdout or ''


def main(run,auth):
    run=run.resolve();auth=auth.resolve();sys.path.insert(0,str(run/'source'))
    import torch
    torch.set_num_threads(4)
    from virtual_submeter.accuracy import microwave_task22 as t
    t.require(not (run/'execution_delivery_manifest.json').exists(),'Completed execution cannot be overwritten')
    t.authorize(run,auth);t.registered(run)
    source=t.read_json(run/'source_selection.json');control=t.read_json(run/'source_control.json')
    if source['eligible']:t.require((run/'evaluation_complete.json').exists(),'Source passed but development acceptance incomplete')
    else:t.require(not any((run/k).exists() for k in ('freeze.json','predictions','evaluation','evaluation_complete.json')),'Source rejected but development accessed')
    t.require((run/'diagnostics_complete.json').exists(),'Fixed diagnostics incomplete')
    successful={}
    for command in ('verify','report'):
        print('Running v4 snapshot '+command,flush=True)
        out=child(run/'run_microwave_task22.py',command,run)
        with (run/f'execution_{command}.log').open('x',encoding='utf-8',newline='\n') as f:f.write(out)
        successful[command]=json.loads(out)
    print('Running independent execution and full diagnostic probability audit',flush=True)
    out=child(ROOT/'scripts/verify_microwave_task22_execution.py',None,run);numeric=json.loads(out)
    t.require(numeric['passed'] and numeric['new_fits']==3 and numeric['optimizer_updates']==10584 and numeric['source_qualified']==source['eligible'],'Independent execution review differs')
    t.require(numeric['diagnostics']['passed'] and numeric['diagnostics']['cells']==24 and numeric['diagnostics']['caches']==12,'Diagnostic matrix review differs')
    t.fresh(run/'independent_execution_numeric.json',numeric)
    t.fresh(run/'execution_snapshot_verification.json',dict(successful_results=successful,sealed_verify_report_succeeded=True,source_paths=[str(run/'source')],independent_external_audit=str(ROOT/'scripts/verify_microwave_task22_execution.py'),external_audit_does_not_replace_snapshot=True,audit_new_fits=0,audit_optimizer_updates=0))
    old,new=control['rows']['guard95'],source['rows']
    gates=dict(three_seed_feasible=len(new)==3,each_block_R95=all(r['feasible'] and len(r['blocks'])==2 and all(b['recall']>=.95 for b in r['blocks']) for r in source['recipes']),
        nondegenerate=all(r['feasible'] and len(r['blocks'])==2 and all(not b['degenerate'] for b in r['blocks']) for r in source['recipes']),mean_F1_not_lower=len(new)==3 and sum(r['f1'] for r in new)>=sum(r['f1'] for r in old),
        each_seed_mean_R_drop_le_005=len(new)==3 and all(n['recall']-o['recall']>=-.05 for o,n in zip(old,new)),at_least_two_seed_P_up=sum(n['precision']>o['precision'] for o in old for n in new if o['seed']==n['seed'])>=2)
    t.require(source['eligible']==all(gates.values()),'Final source gate accounting differs')
    lines=['# Task22 固定数据与步数的TCN宽度对照执行报告','',f"注册 SHA256：`{t.sha256(run/'registration.json')}`",'',
        '新增预算已明确批准并绑定v4：实际3个新模型，每seed3528次更新，共10584次；预算已用满。没有重训旧控制、失败重试、追加候选或追加训练。',
        '唯一实验变量为16→32通道、5329→19809参数；原336704窗/2275正、来源block4/5 34351/201、原完整采样3528×256与16正/240均匀负、unitBCE/Adam .001/C127/CPU4保持。没有困难负替换或扩充数据。',
        '所有真实阶段通过v4封存入口执行；每个独立子进程显式设置OMP_NUM_THREADS=4、MKL_NUM_THREADS=4、torch线程4并实际断言，阶段日志记录线程数。源码/注册/原初始权重/原采样不修改。','',
        '| seed | 实际更新 | 所选epoch | 原点阈值 | 来源L95 | 来源H95 |','|---|---:|---:|---:|---:|---:|']
    for seed in t.SEEDS:
        s=numeric['seeds'][str(seed)];point=s['selected'];lines.append(f"| {seed} | 3528 | {point['epoch']} | {point['threshold']:.9f} | {s['L95']} | {s['H95']} |")
    lines+=['','来源高召回门控比较（每seed对block4/5平均）：','','| seed | control P | R | F1 | wide32 P | R | F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for seed in t.SEEDS:
        o=next(v for v in old if v['seed']==seed);n=next((v for v in new if v['seed']==seed),None)
        nv=' | '.join(f'{n[k]:.8f}' for k in ('precision','recall','f1')) if n else '不可行 | 不可行 | 不可行'
        lines.append(f"| {seed} | {o['precision']:.8f} | {o['recall']:.8f} | {o['f1']:.8f} | {nv} |")
    if len(new)==3:
        mean=lambda group,key:sum(v[key] for v in group)/3
        lines+=['',f"三seed平均：control P/R/F1={mean(old,'precision'):.8f}/{mean(old,'recall'):.8f}/{mean(old,'f1'):.8f}；wide32={mean(new,'precision'):.8f}/{mean(new,'recall'):.8f}/{mean(new,'f1'):.8f}。"]
    lines+=['','来源逐块混淆：','','| seed | block | TP | FP | FN | TN | P | R | F1 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for recipe in source['recipes']:
        for row in recipe['blocks']:lines.append(f"| {recipe['seed']} | {row['block']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | {row['precision']:.8f} | {row['recall']:.8f} | {row['f1']:.8f} |")
    lines+=['','固定来源资格：']+[f"- {key}：{'通过' if ok else '未过'}" for key,ok in gates.items()]
    if not source['eligible']:
        conclusion='来源资格未通过；保留冻结control_guard95，候选不进入开发验收。'
        lines+=['',f'**{conclusion}**','House17数值/shape/hash读取0，未生成开发九格或读取开发真值；最终8/9/20继续隔离。']
    else:
        dev=t.read_json(run/'evaluation/summary.json');conclusion='来源资格通过，一次开发验收'+('通过。' if dev['development_acceptance_passed'] else '未通过。')
        lines+=['',f'**{conclusion}**','开发前唯一候选和A/W50三seed九格已完整冻结；公开event_gate.ordinary精确回读，原137523有效点与其他设备状态不变。']
        lines += [f"- {key}：{'通过' if value else '未过'}" for key,value in dev['acceptance_gates'].items()]
        lines+=['','House17仍是反复使用的工程开发，不能称独立测试；最终8/9/20继续隔离。']
    lines+=['','固定选中模型的解释性诊断（AP、固定点混淆/FPR；不参与候选或阈值选择）：','','| family | seed | 数据 | 视图 | n | AP | P | R | F1 | FPR |','|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in numeric['diagnostics']['metrics']:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['dataset']} | {r['view']} | {r['n']} | {r['ap']:.8f} | {r['precision']:.8f} | {r['recall']:.8f} | {r['f1']:.8f} | {r['fpr']:.8f} |")
    lines+=['',
        '诊断严格保持2模型族×3seed×2数据集(train/source合并)×2视图=24格、12份概率缓存。原支持与共同strict_core31同窗对齐；旧控制和候选缓存均由独立输入转换与固定模型完整前向逐点exact回读，并逐格用另一套同分AP与标量混淆核算。缓存绑定权重、元数据、输入、时间与家庭身份。',
        '独立审计逐项复核10584条实际批次日志与封存索引、逐中心曝光、新32通道初始化exact、全部36份19809参数checkpoint、原85保护与event/日期边界、阈值排序及早epoch、固定L95/H95/r15和来源资格。',
        'v4封存verify/report已在真实执行产物上成功运行；目录外独立数值审计不替代它们。审计在禁用优化器/反传的只读环境中执行，额外fit/优化更新均为0。',
        '两遍只读检查重新执行v4 snapshot verify/report与独立审计，并核对所有输入、旧准备、权重、交付文件hash/mtime不变。完成目录与执行manifest不可覆盖；旧Task21及v2/v3记录不改动。',
        'AP与固定点诊断仅作解释，不能据此改选、调阈值、改宽度或补训。同更新数不保证不同宽度收敛程度相同；本轮结论仅限固定宽度/固定预算方案，不构成容量充分或不足的唯一因果判别。',
        '默认模型不切换；分类试验不宣称功率、电量或运行段MAE改善。预算已用尽，停止新增训练。','']
    with (run/'EXECUTION_REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
    before=t.read_json(run/'inputs_before.json');t.require(t.snapshot(before)==before,'Preparation inputs changed')
    t.fresh(run/'execution_result.json',dict(conclusion=conclusion,source_eligible=source['eligible'],source_gates=gates,new_fits=3,optimizer_updates=10584,control_retrains=0,
        diagnostic_cells=24,diagnostic_caches=12,development_evaluated=source['eligible'],final_reads=0,default_changed=False,
        registration_sha256=t.sha256(run/'registration.json'),authorization_sha256=t.sha256(auth)))
    scripts=[ROOT/'scripts'/n for n in ('execute_microwave_task22_budget.py','verify_microwave_task22_execution.py','seal_microwave_task22_execution.py')]
    t.fresh(run/'execution_delivery_manifest.json',dict(status='complete_fixed_task22_v4',new_fits=3,optimizer_updates=10584,registration_sha256=t.sha256(run/'registration.json'),
        authorization_sha256=t.sha256(auth),preparation_manifest_sha256=t.sha256(run/'preparation_manifest.json'),sealed_verify_report_succeeded=True,
        independent_execution_audit_passed=True,artifacts=t.inventory(run),execution_tools=t.snapshot(scripts)))
    protected=[*before,*scripts,auth,run.with_name(run.name+'.readonly-receipt.json'),*[run/k for k in t.inventory(run)]]
    sealed=t.snapshot(protected);calls=[]
    for iteration in (1,2):
        print(f'Readonly execution pass {iteration}',flush=True)
        for command in ('verify','report'):
            out=child(run/'run_microwave_task22.py',command,run);t.require(json.loads(out)==successful[command],'Snapshot readonly result changed')
            calls.append(dict(iteration=iteration,command='snapshot_'+command,stdout_sha256=hashlib.sha256(out.encode()).hexdigest()))
        out=child(ROOT/'scripts/verify_microwave_task22_execution.py',None,run);t.require(json.loads(out)==numeric,'Independent readonly result changed')
        calls.append(dict(iteration=iteration,command='independent_execution_and_diagnostics',stdout_sha256=hashlib.sha256(out.encode()).hexdigest()))
        t.require(t.snapshot(sealed)==sealed,'Readonly check changed input/delivery hash/mtime')
    receipt=run.with_name(run.name+'.execution-readonly-receipt.json')
    t.fresh(receipt,dict(passed=True,passes=2,all_input_delivery_hash_mtime_unchanged=True,calls=calls,execution_manifest_sha256=t.sha256(run/'execution_delivery_manifest.json'),snapshots=sealed,audit_new_fits=0,audit_optimizer_updates=0))
    print(json.dumps(dict(status='complete',conclusion=conclusion,new_fits=3,updates=10584,diagnostic_cells=24,diagnostic_caches=12,manifest_sha256=t.sha256(run/'execution_delivery_manifest.json'),readonly_receipt=str(receipt)),ensure_ascii=False,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);a=p.parse_args();main(a.run,a.authorization)
