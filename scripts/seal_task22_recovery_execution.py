"""Seal a completed Task22 seed2026 recovery; never train or change selection."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]


def main(run,authorization):
 run=run.resolve();authorization=authorization.resolve();sys.path.insert(0,str(run/'source'))
 import torch
 torch.set_num_threads(4)
 from virtual_submeter.accuracy import microwave_task22_recovery as t
 t.require(not (run/'execution_delivery_manifest.json').exists(),'Execution delivery already sealed')
 t.authorize(run,authorization);t.registered(run);complete=t.check_complete(run)
 stages=['train','source_select','diagnostics','verify','report','independent_verify','numeric_audit']
 source=t.read_json(run/'source_selection.json')
 if source['eligible']:
  t.require((run/'evaluation_complete.json').exists(),'Passed source requires completed one-shot development evaluation')
  stages+=['freeze','evaluate']
 else:t.require(not any((run/k).exists() for k in ('freeze.json','predictions','evaluation','evaluation_complete.json')),'Rejected source accessed development')
 pid_checks={}
 for stage in stages:
  finished=t.read_json(run/f'execution_{stage}_finished.json');process=t.read_json(run/f'execution_{stage}_process.json');log=run/f'execution_{stage}.log'
  t.require(finished['returncode']==0 and finished['child_pid']==process['child_pid'] and finished['log_sha256']==t.sha256(log),'Stage failed, PID or log binding differs: '+stage)
  with log.open(encoding='utf-8') as f:first=f.readline()
  runtime=json.loads(first);t.require(runtime['stage']==stage and runtime['cpu_threads']==4 and runtime['OMP_NUM_THREADS']=='4' and runtime['MKL_NUM_THREADS']=='4' and runtime['parent_pid']==process['child_pid'],'Actual runtime or PID differs')
  pid_checks[stage]=dict(actual_python_pid=runtime['pid'],launcher_pid=process['child_pid'],driver_pid=process['driver_pid'],returncode=finished['returncode'],cpu_threads=4,log_sha256=finished['log_sha256'])
 def result(stage):
  text=(run/f'execution_{stage}.log').read_text(encoding='utf-8');return json.loads(text.split('\n',1)[1])
 snapshot_verify=result('verify');snapshot_report=result('report');independent_prepare=result('independent_verify');numeric=result('numeric_audit')
 for v in (snapshot_verify,snapshot_report):
  x=v['execution'];t.require(v['passed'] and v['actual_new_fits']==1 and v['actual_new_fit_updates']==3528 and v['reused_fits']==2 and v['historic_interrupted_fits']==1
   and v['effective_matrix_updates']==10584 and v['cumulative_attempted_updates_upper_bound']==11152 and x['training_matrix']
   and x['actual_batch_logs_exact'] and x['checkpoint_rule_replayed'] and x['diagnostics']['passed'] and x['diagnostics']['cells']==24 and x['diagnostics']['caches']==12 and x['source_qualified']==source['eligible'],'Snapshot verify/report result differs')
 t.require(independent_prepare['passed'] and independent_prepare['independently_replayed_original_batches'] and independent_prepare['reused_checkpoints_independent'] and independent_prepare['event_dates_complete'] and independent_prepare['failed_attempt_binding_exact'] and independent_prepare['new_fits']==0,'Independent preparation replay failed')
 t.require(numeric['passed'] and numeric['actual_new_fits']==1 and numeric['new_fit_updates']==3528 and numeric['reused_fits']==2 and numeric['historic_interrupted_fits']==1
  and numeric['total_fit_attempts']==4 and numeric['effective_models']==3 and numeric['effective_matrix_updates']==10584 and numeric['cumulative_attempted_updates_upper_bound']==11152
  and numeric['independent_actual_ledger_exact'] and numeric['actual_initialization_exact'] and numeric['independent_all36_checkpoint_rank']
  and numeric['source_qualified']==source['eligible'] and numeric['diagnostics']['passed'] and numeric['diagnostics']['cells']==24 and numeric['diagnostics']['caches']==12,'Independent execution audit differs')
 if source['eligible']:
  t.require(snapshot_verify['execution']['independent_eight_gates'] and snapshot_verify['execution']['independent_summary_ledger_exact'] and numeric['development']['all_eight_gates_exact'] and numeric['development']['ledger_exact'],'Development review incomplete')
 old=t.read_json(run/'source_control.json')['rows']['guard95'];new=source['rows']
 gates=dict(checkpoints_original85=all(r['source_gate_passed'] for r in complete['rows']),three_seed_feasible=len(new)==3,
  each_block_R95=all(r['feasible'] and len(r['blocks'])==2 and all(b['recall']>=.95 for b in r['blocks']) for r in source['recipes']),
  nondegenerate=all(r['feasible'] and len(r['blocks'])==2 and all(not b['degenerate'] for b in r['blocks']) for r in source['recipes']),
  mean_F1_not_lower=len(new)==3 and sum(r['f1'] for r in new)>=sum(r['f1'] for r in old),
  each_seed_mean_R_drop_le_005=len(new)==3 and all(n['recall']-o['recall']>=-.05 for o,n in zip(old,new)),
  at_least_two_seed_P_up=sum(n['precision']>o['precision'] for o in old for n in new if o['seed']==n['seed'])>=2)
 t.require(source['eligible']==all(gates.values()),'Final source qualification accounting differs')
 math_test=t.read_json(run/'execution_audit_math_selftest.json')
 t.require(math_test['passed'] and math_test['static_tie_cases']==24 and math_test['independent_AP_exact'] and math_test['independent_threshold_exact'],'Independent numeric self-test failed')
 compare=numeric['interrupted_epoch1_comparison']
 lines=['# Task22 seed2026 恢复执行报告','',f"恢复注册 SHA256：`{t.sha256(run/'registration.json')}`",f"原 v4 注册 SHA256：`{t.ORIGINAL_SHA}`",'',
  '根据新的独立授权，seed2026从原initial32与原全部3528×256批索引重新运行一次，实际新增1 fit/3528更新；原seed42/1019完整模型逐文件复用2份，原model.json仍绑定旧v4注册，新模型绑定本恢复注册。未恢复旧Adam状态，也未重启原v4训练。',
  '原seed2026历史中断完成567、尝试568，第568次是否实际更新未知。实际fit尝试总计4次（2旧完整+1旧中断+1新恢复）；累计完成更新下界11151、尝试上界11152。最终有效3seed矩阵各3528步，共10584步；该数字不代表全部累计消耗。',
  '配方固定：原336704窗/2275正、来源block4/5 34351/201、16正/240均匀负、unitBCE/Adam .001/C127、32通道19809参数、12×294步、CPU4。每个阶段的PID、终态、实际线程数与完整UTF-8日志均已绑定。','',
  '| seed | 来源 | 所选epoch | 原85点阈值 | 来源L95 | 来源H95 |','|---|---|---:|---:|---:|---:|']
 for seed in t.SEEDS:
  item=numeric['seeds'][str(seed)];point=item['selected'];origin='旧完整复用' if seed in (42,1019) else '本次全新重跑'
  lines.append(f"| {seed} | {origin} | {point['epoch']} | {point['threshold']:.9f} | {item['L95']} | {item['H95']} |")
 lines+=['',f"新旧 seed2026 第1检查点（294步）逐tensor比较：{'全部exact' if compare['equal'] else '存在差异，明细见独立审计'}。此比较仅作复现诊断，不参与模型或阈值选择。",'',
  '来源高召回门控比较（每seed对block4/5取均值）：','','| seed | control P | R | F1 | wide32 P | R | F1 |','|---|---:|---:|---:|---:|---:|---:|']
 for seed in t.SEEDS:
  o=next(v for v in old if v['seed']==seed);n=next((v for v in new if v['seed']==seed),None)
  vals=' | '.join(f'{n[k]:.8f}' for k in ('precision','recall','f1')) if n else '不可行 | 不可行 | 不可行'
  lines.append(f"| {seed} | {o['precision']:.8f} | {o['recall']:.8f} | {o['f1']:.8f} | {vals} |")
 if len(new)==3:
  mean=lambda rows,k:sum(r[k] for r in rows)/3
  lines += ['',f"三seed平均 control P/R/F1：{mean(old,'precision'):.8f}/{mean(old,'recall'):.8f}/{mean(old,'f1'):.8f}；wide32：{mean(new,'precision'):.8f}/{mean(new,'recall'):.8f}/{mean(new,'f1'):.8f}。"]
 lines+=['','来源逐块混淆：','','| seed | block | TP | FP | FN | TN | P | R | F1 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for recipe in source['recipes']:
  for r in recipe['blocks']:lines.append(f"| {recipe['seed']} | {r['block']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {r['precision']:.8f} | {r['recall']:.8f} | {r['f1']:.8f} |")
 lines+=['','固定来源资格：']+[f"- {name}：{'通过' if value else '未过'}" for name,value in gates.items()]
 if not source['eligible']:
  conclusion='来源资格未通过；保留原control_guard95，不进入开发验收。'
  lines+=['',f'**{conclusion}**','House17 数值、shape、hash读取0；未冻结开发九格，未读取开发真值或计算开发成绩。最终8/9/20继续隔离。']
 else:
  summary=t.read_json(run/'evaluation/summary.json');conclusion='来源通过；唯一候选的一次开发验收'+('通过。' if summary['development_acceptance_passed'] else '未通过。')
  lines+=['',f'**{conclusion}**','开发前A/W50/唯一wide32三seed九格完整冻结，公开event_gate.ordinary逐点exact；原137523有效点及其他设备状态已核对。']
  lines += [f"- {name}：{'通过' if value else '未过'}" for name,value in summary['acceptance_gates'].items()]
  lines+=['','House17为反复使用的工程开发，不是独立测试；最终8/9/20继续隔离。']
 lines+=['','固定模型解释性诊断：点混淆和FPR使用各模型原85检查点的固定点阈值；来源资格使用L95/H95事件门控，两者口径不同，AP不依赖阈值。','','| 模型 | seed | 数据 | 视图 | n | AP | 固定点阈值 | TP | FP | FN | TN | P | R | F1 | FPR |','|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in numeric['diagnostics']['metrics']:
  lines.append(f"| {r['arm']} | {r['seed']} | {r['dataset']} | {r['view']} | {r['n']} | {r['ap']:.8f} | {r['threshold']:.9f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {r['precision']:.8f} | {r['recall']:.8f} | {r['f1']:.8f} | {r['fpr']:.8f} |")
 metrics=numeric['diagnostics']['metrics']
 ap=lambda arm,name:sum(r['ap'] for r in metrics if r['arm']==arm and r['dataset']==name and r['view']=='original')/3
 lines+=['',f"原支持三seed平均AP：训练 control16={ap('control16','train'):.8f}、wide32={ap('wide32','train'):.8f}；来源 control16={ap('control16','source'):.8f}、wide32={ap('wide32','source'):.8f}。这是解释性排序诊断，不参与checkpoint/阈值/候选选择。",'',
  '实际snapshot verify、report和independent_verify均已执行；目录外独立数值审计逐条复核有效三模型共10584条批日志、逐中心曝光、32通道初始化、双注册来源、全部36个19809参数checkpoint及原85排名、日期/event支持、L95/H95/r15与来源资格。执行审核未创建优化器或更新。',
  '诊断为固定24格、12缓存；全部固定模型概率经CPU4独立输入转换和完整前向逐点回读，AP以同分分组的独立实现及标量混淆核算，原支持与strict_core31同窗核对。该审计只读来源和训练数据，不作新选择。',
  '原v4的232个文件、旧失败目录、准备输入/源码/索引/权重与两个旧完整模型持续核对hash/mtime；新训练与执行交付另行绑定。两遍只读hash/mtime检查记录在外部回执。',
  '本实验不扩大宽度、改选、延训或追加模型；默认模型未切换。相同步数不保证不同宽度收敛程度相同，结果不构成容量是否充分的唯一因果判断；分类结果不宣称功率、电量或运行段MAE改善。','']
 with (run/'RECOVERY_EXECUTION_REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
 t.fresh(run/'execution_snapshot_verification.json',dict(snapshot_verify=snapshot_verify,snapshot_report=snapshot_report,independent_prepare=independent_prepare,
  snapshot_verify_report_independent_verify_succeeded=True,external_numeric_audit_succeeded=True,external_audit_does_not_replace_snapshot=True,audit_new_fits=0,audit_optimizer_updates=0))
 t.fresh(run/'independent_execution_numeric.json',numeric)
 t.fresh(run/'execution_result.json',dict(conclusion=conclusion,source_eligible=source['eligible'],source_gates=gates,actual_new_fits=1,new_fit_updates=3528,
  reused_fits=2,historic_interrupted_fits=1,total_fit_attempts=4,effective_models=3,effective_matrix_updates=10584,
  historical_completed_updates_lower_bound=7623,historical_attempted_updates_upper_bound=7624,cumulative_completed_updates_lower_bound=11151,cumulative_attempted_updates_upper_bound=11152,
  development_evaluated=source['eligible'],final_reads=0,default_changed=False,epoch1_exact=compare['equal'],pid_checks=pid_checks,
  registration_sha256=t.sha256(run/'registration.json'),original_registration_sha256=t.ORIGINAL_SHA,authorization_sha256=t.sha256(authorization)))
 before=t.read_json(run/'inputs_before.json');t.require(t.snapshot(before)==before,'Original data/preparation/v4 inputs changed')
 task_scripts=[ROOT/'scripts'/n for n in ('execute_task22_recovery.py','run_task22_recovery_review.py','verify_task22_recovery_execution.py','seal_task22_recovery_execution.py')]
 evidence=ROOT/'outputs/accuracy/microwave-task22-recovery-test-evidence-20261027-v1'
 t.fresh(run/'execution_delivery_manifest.json',dict(status='recovery_execution_complete',actual_new_fits=1,new_fit_updates=3528,reused_fits=2,historic_interrupted_fits=1,
  total_fit_attempts=4,effective_models=3,effective_matrix_updates=10584,cumulative_attempted_updates_upper_bound=11152,
  registration_sha256=t.sha256(run/'registration.json'),original_registration_sha256=t.ORIGINAL_SHA,authorization_sha256=t.sha256(authorization),
  snapshot_verify_report_independent_verify_succeeded=True,external_numeric_audit_passed=True,execution_tools=t.snapshot(task_scripts),artifacts=t.inventory(run)))
 tracked=[*before,*task_scripts,authorization,run.with_name(run.name+'.readonly-receipt.json'),evidence.with_name(evidence.name+'.readonly-receipt.json'),*[evidence/k for k in t.inventory(evidence)],*[run/k for k in t.inventory(run)]]
 sealed=t.snapshot(tracked);passes=[]
 for iteration in (1,2):
  print(f'Checking read-only input/delivery hash and mtime, pass {iteration}',flush=True)
  t.require(t.snapshot(sealed)==sealed,'Input/delivery hash or mtime changed')
  passes.append(dict(iteration=iteration,all_hash_mtime_unchanged=True,files=len(sealed)))
 receipt=run.with_name(run.name+'.execution-readonly-receipt.json')
 t.fresh(receipt,dict(passed=True,passes=2,all_input_delivery_hash_mtime_unchanged=True,checks=passes,execution_manifest_sha256=t.sha256(run/'execution_delivery_manifest.json'),
  snapshots=sealed,audit_new_fits=0,audit_optimizer_updates=0,original_v4_file_count=len(t.read_json(run/'old_v4_files.json'))))
 print(json.dumps(dict(status='sealed',conclusion=conclusion,actual_new_fits=1,new_fit_updates=3528,reused_fits=2,historic_interrupted_fits=1,total_fit_attempts=4,
  effective_matrix_updates=10584,cumulative_attempted_updates_upper_bound=11152,epoch1_exact=compare['equal'],manifest_sha256=t.sha256(run/'execution_delivery_manifest.json'),readonly_receipt=str(receipt)),ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);a=p.parse_args();main(a.run,a.authorization)
