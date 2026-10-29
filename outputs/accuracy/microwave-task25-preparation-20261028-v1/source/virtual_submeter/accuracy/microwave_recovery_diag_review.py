"""Read-only replay of recovery diagnostics: fixed models, fixed windows and two views."""
from pathlib import Path
import numpy as np
import torch
from .microwave_task22_metrics import independent_ap
from .microwave_task21_review import scalar_metric


def review(run,task):
 run=Path(run)
 if not (run/'diagnostics').exists() and not (run/'diagnostics_complete.json').exists():return None
 task.require((run/'diagnostics_complete.json').exists(),'Incomplete diagnostics attempt cannot be accepted')
 seal=task.read_json(run/'diagnostics_complete.json');task.gate.check(run/'diagnostics',seal['artifacts'])
 expected={f'{arm}_{seed}_{data}.npz' for arm in ('control16','wide32') for seed in task.SEEDS for data in ('train','source')}|{'metrics.json'}
 task.require(set(seal['artifacts'])==expected,'Incomplete diagnostic cache matrix')
 saved=task.read_json(run/'diagnostics/metrics.json');rows=saved['rows']
 cells={(arm,seed,data,view) for arm in ('control16','wide32') for seed in task.SEEDS for data in ('train','source') for view in ('original','strict_core31')}
 task.require(len(rows)==24 and {(r['arm'],r['seed'],r['dataset'],r['view']) for r in rows}==cells and saved['explanatory_only']
     and saved['new_thresholds']==0 and saved['new_selections']==0 and saved['development_reads']==0 and saved['cpu_threads']==4,'Diagnostic rule/matrix differs')
 data={'train':task.load(task.T10/'train.npz')}
 blocks=[task.load(task.T10/f'validation/block_{b}.npz') for b in (4,5)]
 data['source']={k:np.concatenate([v[k] for v in blocks]) for k in ('x','y','t','house')}
 torch.set_num_threads(4);supports={};bindings=[]
 for arm in ('control16','wide32'):
  for seed in task.SEEDS:
   modelpath=task.T12/f'models/unit_BCE/seed_{seed}' if arm=='control16' else run/f'models/wide32/seed_{seed}'
   meta,net=task.load_state_model(modelpath)
   for name,d in data.items():
    cached=task.load(run/f'diagnostics/{arm}_{seed}_{name}.npz')
    np.testing.assert_array_equal(cached['t'],d['t']);np.testing.assert_array_equal(cached['house'],d['house'])
    with torch.inference_mode():
     z=task.tensor_input(meta,d['x']);p=np.concatenate([torch.sigmoid(net(v)).numpy() for v in z.split(meta['batch_size'])])
    np.testing.assert_array_equal(cached['probability'],p)
    for view in ('original','strict_core31'):
     mask=np.ones(len(d['y']),bool) if view=='original' else np.isfinite(d['x'][:,48:79]).all(axis=1)
     row=next(r for r in rows if (r['arm'],r['seed'],r['dataset'],r['view'])==(arm,seed,name,view))
     key=(name,view);identity=(task.gate.hash_array(mask),task.gate.hash_array(d['t'][mask]),int(mask.sum()))
     task.require(supports.setdefault(key,identity)==identity and row['support_sha256']==identity[0] and row['n']==identity[2],'Diagnostic same-window support differs')
     predicted=(p[mask]>=meta['decision_threshold_probability']).astype(np.int8);truth=d['y'][mask]>=200
     counts=scalar_metric(predicted,truth)
     task.require(all(row[k]==v for k,v in counts.items()) and abs(row['ap']-independent_ap(p[mask],truth))<2e-12
         and row['threshold']==meta['decision_threshold_probability'],'Independent diagnostic AP/confusion differs')
    bindings.append(dict(arm=arm,seed=seed,dataset=name,cache_sha256=task.sha256(run/f'diagnostics/{arm}_{seed}_{name}.npz'),probability_replay_exact=True))
 task.require(supports[('source','original')]==supports[('source','strict_core31')],'Source strict support differs')
 return dict(passed=True,cells=24,caches=12,all_fixed_model_predictions_exact=True,independent_AP_and_confusions_exact=True,
     cpu_threads=torch.get_num_threads(),bindings=bindings,development_reads=0,audit_new_fits=0,audit_optimizer_updates=0)
