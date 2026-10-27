"""Independent read-only replay of recovery preparation and original source batches."""
from pathlib import Path
import numpy as np
import torch
from ..data.common import read_json,sha256
from . import microwave_task22_recovery as task
from .microwave_task22_review import independent_batches,independent_choose,numeric_equal


def review(run):
 run=Path(run).resolve()
 with task.isolation(run,task.execution_allowed(run),readonly=True) as access:
  task.registered(run)
  reg=read_json(run/'registration.json');task.gate.check(run,reg['artifacts'])
  audit=read_json(task.AUDIT/'audit.json');ledger=audit['ledger'];identity=read_json(run/'reused_model_identity.json')
  task.require(ledger['historic_started_fits']==3 and ledger['complete_reusable_fits']==2 and ledger['interrupted_fits']==1
      and ledger['completed_updates_lower_bound']==7623 and ledger['attempted_updates_upper_bound']==7624
      and ledger['hypothetical_attempted_updates_upper_bound']==11152,'Independent interruption ledger differs')
  task.require(identity['original_registration_sha256']==task.ORIGINAL_SHA and set(identity['models'])=={'42','1019'},'Reuse matrix differs')
  for s in (42,1019):
   bound=identity['models'][str(s)];old=task.OLD/f'models/wide32/seed_{s}'
   task.require(bound==audit['complete_models'][str(s)] and task.inventory(old)==bound['artifacts'],'Reused source artifacts differ')
   meta=read_json(old/'model.json');task.require(meta['registration_sha256']==task.ORIGINAL_SHA and meta['tcn_channels']==32 and meta['parameter_count']==19809,'Reused origin differs')
  failure=task.OLD/'models/wide32/seed_2026';entry=read_json(run/'interruption_identity.json')
  task.require(read_json(failure/'progress.json')==dict(completed_updates=567,attempted_updates=568) and not (failure/'complete.json').exists(),'Failed progress differs')
  task.require(entry['interrupted']==audit['interrupted'] and entry['failed_model_inventory_sha256']==audit['interrupted']['inventory_sha256']
      and task.inventory(failure)==audit['interrupted']['artifacts'] and entry['audit_manifest_sha256']==task.AUDIT_SHA,'Failed attempt binding differs')
  d=task.load(task.T10/'train.npz',['y','house','t','event']);plan=read_json(run/'parent_training_plan.json')
  task.require(len(d['y'])==336704 and int((d['y']>=200).sum())==2275 and np.isfinite(d['y']).all(),'Original pool labels differ')
  seeds={}
  for s in task.SEEDS:
   ix=task.load(run/f'batches_{s}.npz')['indices'];oldix=task.load(task.OLD/f'batches_{s}.npz')['indices']
   task.require(ix.shape==(3528,256),'Original batch matrix incomplete');np.testing.assert_array_equal(ix,oldix)
   for row,replayed in zip(ix,independent_batches(d,s,3528)):
    np.testing.assert_array_equal(row,replayed);task.require(int((d['y'][row]>=200).sum())==16,'Sampling quota differs')
   np.testing.assert_array_equal(np.bincount(ix.ravel(),minlength=len(d['y'])),task.load(run/f'exposure_{s}.npz')['total'])
   i=torch.load(run/f'initial32_{s}.pt',map_location='cpu',weights_only=True);j=torch.load(task.OLD/f'initial32_{s}.pt',map_location='cpu',weights_only=True)
   task.require(i.keys()==j.keys() and all(torch.equal(v,j[k]) for k,v in i.items()) and sum(v.numel() for v in i.values())==19809,'Original 32-channel initialization differs')
   seeds[str(s)]=dict(all_3528_batches_exact=True,original_indices_replayed=True,exposure_exact=True,initial32_exact=True)
  blocks={f'block_{b}':task.load(task.T10/f'validation/block_{b}.npz',['t','y','event']) for b in (4,5)}
  for b,start,end in ((4,'2014-11-23','2014-11-25'),(5,'2014-11-25','2014-11-27')):
   z=blocks[f'block_{b}'];lo,hi=[int(np.datetime64(v,'s').astype(np.int64)) for v in (start,end)]
   task.require((z['t']-120>=lo).all() and (z['t']+128<=hi).all() and np.isfinite(z['y']).all() and len(np.unique(z['event'][z['y']>=200]))>=2,'Event/date boundary differs')
  for seed in (42,1019):
   model=task.OLD/f'models/wide32/seed_{seed}';scores=[]
   for epoch in range(1,13):
    score={}
    for b in (4,5):
     z=task.load(model/f'epoch_{epoch}_block_{b}.npz');np.testing.assert_array_equal(z['t'],blocks[f'block_{b}']['t']);score[f'block_{b}']=z['scores']
    scores.append(score)
   point=independent_choose(scores,blocks,plan);meta=read_json(model/'model.json')
   task.require(numeric_equal(point,meta['selected']) and sha256(model/'weights.pt')==sha256(model/f'epoch_{point["epoch"]}.pt'),'Independent old checkpoint differs')
  return dict(passed=True,registration_sha256=sha256(run/'registration.json'),original_registration_sha256=task.ORIGINAL_SHA,
      independently_replayed_original_batches=True,seeds=seeds,reused_checkpoints_independent=True,event_dates_complete=True,
      failed_attempt_binding_exact=True,new_fits=0,optimizers=0,optimizer_updates=0,backward_calls=0,development_reads=0,final_reads=0)
