"""Read-only Task22 interruption audit; no fit, optimizer or protected house input."""
import csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
V4=ROOT/'outputs/accuracy/microwave-task22-preparation-20261026-v4'
DEST=ROOT/'outputs/accuracy/microwave-task22-interruption-audit-20261027-v1'
EXPECTED='2ffd72773185d8be1372b0744729b7eb3daf8656f1dfd2097da47cb113037c53'
sys.path.insert(0,str(V4/'source'))
from virtual_submeter.accuracy import microwave_task22 as m
from virtual_submeter.accuracy.microwave_task22_review import independent_choose,numeric_equal

def main():
 if DEST.exists():raise RuntimeError('Audit attempt exists; preserve it and use new version')
 if (V4/'training_complete.json').exists() or (V4/'source_selection.json').exists() or (V4/'diagnostics').exists():raise RuntimeError('Unexpected stage; do not infer interrupted state')
 DEST.mkdir(parents=True,exist_ok=False)
 oldfiles=[p for p in V4.rglob('*') if p.is_file()]
 budget=V4.with_name(V4.name+'.human-budget.json');oldreceipt=V4.with_name(V4.name+'.readonly-receipt.json')
 paths=oldfiles+[budget,oldreceipt,m.T10/'validation/block_4.npz',m.T10/'validation/block_5.npz',Path(__file__).resolve()]
 before=m.snapshot(paths)
 with m.isolation(DEST,paths,readonly=True) as access:
  m.require(m.sha256(V4/'registration.json')==EXPECTED,'Original registration differs')
  reg=m.read_json(V4/'registration.json');m.gate.check(V4,reg['artifacts'])
  m.gate.check(V4,m.read_json(V4/'preparation_manifest.json')['artifacts'])
  receipt=m.read_json(budget);start=m.read_json(V4/'training_started.json')
  m.require(receipt==start['authorization'] and receipt['registration_sha256']==EXPECTED and receipt['authorized'] is True,'Original budget identity differs')
  plan=m.read_json(V4/'parent_training_plan.json');m.require([plan[k] for k in ('checkpoints','steps_per_checkpoint','batch_size','positive_per_batch','uniform_negative_per_batch','cpu_threads','learning_rate')]==[12,294,256,16,240,4,.001],'Original protocol differs')
  blocks={f'block_{b}':m.load(m.T10/f'validation/block_{b}.npz',['t','y','event']) for b in (4,5)}
  for b,lo,hi in m.loss.temporal.f.VALIDATION:
   d=blocks[f'block_{b}'];m.require((d['t']-120>=lo).all() and (d['t']+128<=hi).all() and np.isfinite(d['y']).all() and len(np.unique(d['event'][d['y']>=200]))>=2,'Source event/date support differs')
  complete={}
  for seed in (42,1019):
   model=V4/f'models/wide32/seed_{seed}';info=m.read_json(model/'model.json');finish=m.read_json(model/'complete.json');m.gate.check(model,finish['artifacts'])
   m.require(m.read_json(model/'progress.json')==dict(completed_updates=3528,attempted_updates=3528) and finish['updates']==3528,'Completed progress differs')
   m.require(info['registration_sha256']==EXPECTED and info['tcn_channels']==32 and info['parameter_count']==19809 and info['recipe']=='unit_BCE','Model identity differs')
   indices=m.load(V4/f'batches_{seed}.npz')['indices'];m.require(indices.shape==(3528,256),'Original matrix differs')
   with (model/'batches.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
   m.require(len(rows)==3528,'Incomplete actual log')
   for i,row in enumerate(rows):m.require(int(row['update'])==i+1 and int(row['positive'])==16 and int(row['negative'])==240 and row['indices_sha256']==hashlib.sha256(np.ascontiguousarray(indices[i]).tobytes()).hexdigest(),'Actual batch differs')
   np.testing.assert_array_equal(m.load(model/'exposure.npz')['count'],m.load(V4/f'exposure_{seed}.npz')['total'])
   first=torch.load(model/'initial.pt',map_location='cpu',weights_only=True);base=torch.load(V4/f'initial32_{seed}.pt',map_location='cpu',weights_only=True)
   m.require(first.keys()==base.keys() and all(torch.equal(v,base[k]) for k,v in first.items()),'Initialization differs')
   scores=[]
   for epoch in range(1,13):
    state=torch.load(model/f'epoch_{epoch}.pt',map_location='cpu',weights_only=True);m.require(state.keys()==base.keys() and sum(v.numel() for v in state.values())==19809,'Checkpoint differs')
    score={}
    for b in (4,5):
     z=m.load(model/f'epoch_{epoch}_block_{b}.npz');np.testing.assert_array_equal(z['t'],blocks[f'block_{b}']['t']);score[f'block_{b}']=z['scores']
    scores.append(score)
   selected=m.loss.temporal.choose(scores,blocks,plan);other=independent_choose(scores,blocks,plan)
   m.require(numeric_equal(selected,other) and selected==info['selected'] and m.sha256(model/'weights.pt')==m.sha256(model/f'epoch_{selected["epoch"]}.pt'),'Independent checkpoint replay differs')
   artifacts=m.inventory(model);digest=hashlib.sha256(json.dumps(artifacts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   complete[str(seed)]=dict(completed_updates=3528,attempted_updates=3528,original_registration_sha256=EXPECTED,selected=selected,
      all_batch_indices_exact=True,exposure_exact=True,initial_exact=True,independent_all_checkpoints=True,artifacts=artifacts,inventory_sha256=digest)
  failed=V4/'models/wide32/seed_2026';progress=m.read_json(failed/'progress.json');m.require(progress==dict(completed_updates=567,attempted_updates=568) and not (failed/'complete.json').exists(),'Interruption progress differs')
  first=torch.load(failed/'initial.pt',map_location='cpu',weights_only=True);base=torch.load(V4/'initial32_2026.pt',map_location='cpu',weights_only=True)
  m.require(first.keys()==base.keys() and all(torch.equal(v,base[k]) for k,v in first.items()),'Interrupted initialization differs')
  with (failed/'batches.csv').open(encoding='utf-8',newline='') as f:partial=list(csv.DictReader(f))
  m.require(len(partial)==294 and len(m.read_json(failed/'history.json')['rows'])==1,'Unexpected persisted checkpoint log')
  saved=m.load(V4/'batches_2026.npz')['indices']
  for i,row in enumerate(partial):m.require(int(row['update'])==i+1 and row['indices_sha256']==hashlib.sha256(np.ascontiguousarray(saved[i]).tobytes()).hexdigest(),'Persisted partial batch differs')
  files=m.inventory(failed);m.require(not any('optim' in k.lower() for k in files),'Unexpected optimizer state; investigate')
  oldstate=torch.load(failed/'epoch_1.pt',map_location='cpu',weights_only=True);m.require(oldstate.keys()==base.keys(),'Unexpected checkpoint contents')
  fail_digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  audit=dict(original_registration_sha256=EXPECTED,original_budget_sha256=m.sha256(budget),complete_models=complete,
   interrupted=dict(seed=2026,completed_updates=567,attempted_updates_upper_bound=568,step_568_applied_unknown=True,
    saved_checkpoint=1,saved_batch_log_rows=294,initial_exact=True,optimizer_state_absent=True,artifacts=files,inventory_sha256=fail_digest),
   ledger=dict(historic_started_fits=3,complete_reusable_fits=2,interrupted_fits=1,completed_updates_lower_bound=7623,attempted_updates_upper_bound=7624,
    hypothetical_recovery_fits=1,hypothetical_recovery_updates=3528,hypothetical_total_fit_attempts=4,hypothetical_effective_models=3,
    hypothetical_effective_matrix_updates=10584,hypothetical_completed_updates_lower_bound=11151,hypothetical_attempted_updates_upper_bound=11152),
   no_fit=True,no_optimizer=True,no_backward=True,development_reads=0,final_reads=0,access_reads=sorted(access['reads']),forbidden=access['forbidden'])
 m.require(m.snapshot(paths)==before,'Interruption audit modified original input')
 m.fresh(DEST/'audit.json',audit);m.fresh(DEST/'inputs_before.json',before)
 (DEST/'INTERRUPTION_REPORT.zh.md').write_text('# Task22 中断账目\n\nseed42、1019各完成3528步，原批日志、初始化、曝光及12检查点独立重放通过。seed2026完成567步、已尝试568步；第568次是否调用优化器更新无法判断。第三个模型没有完整Adam状态，不能从567步宣称精确续训。\n\n已完成更新下界7623，已尝试上界7624。若未来获新的1fit/3528授权并从原初始化与原索引重跑，历史尝试4个模型，累计完成更新下界11151、尝试上界11152；最终有效矩阵为原三seed共10584完成更新。\n\n此次复核0fit/0优化器/0反传、0开发/最终读取；原v4全部文件、预算和日志保持原样。\n',encoding='utf-8')
 m.fresh(DEST/'audit_manifest.json',dict(status='interruption_audited',original_registration_sha256=EXPECTED,artifacts=m.inventory(DEST),original_inputs=before))
 print(json.dumps(dict(passed=True,audit_manifest_sha256=m.sha256(DEST/'audit_manifest.json'),ledger=audit['ledger']),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
