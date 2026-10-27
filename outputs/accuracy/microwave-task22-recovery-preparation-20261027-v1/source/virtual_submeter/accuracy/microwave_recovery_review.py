"""Execution review with explicit recovery context and dual model origins."""
import csv
from pathlib import Path
import numpy as np
import torch
from ..data.common import read_json,sha256
from .microwave_task22_review import scalar_metric,local_max_independent,independent_acceptance,independent_choose,numeric_equal,independent_threshold

def review_execution(run, *, task):
    """Read-only future execution review. No dev inputs until sealed evaluation.

    Uses execution's own frozen truth cache only after complete nine-cell freeze;
    independently recomputes scalar counts, temporal max and eight acceptance gates.
    """
    run=Path(run)
    with task.isolation(run,task.execution_allowed(run),readonly=True):
        task.registered(run)
        started=read_json(run/'training_started.json')
        task.require(started['seed']==2026,'Wrong recovery seed')
        complete=task.check_complete(run)
        task.require(complete['new_fits']==1 and complete['reused_fits']==2 and complete['effective_models']==3 and complete['effective_matrix_updates']==10584,'Recovery ledger differs')
        task.gate.check(run/'models',complete['artifacts'])
        for seed in task.SEEDS:
            model=run/f'models/wide32/seed_{seed}'
            task.require(read_json(model/'progress.json')==dict(completed_updates=3528,attempted_updates=3528),'Incomplete updates')
            task.require(read_json(model/'complete.json')['updates']==3528,'Per-seed budget differs')
            saved=task.load(model/'exposure.npz')['count'];expected=task.load(run/f'exposure_{seed}.npz')['total']
            np.testing.assert_array_equal(saved,expected)
            indices=task.load(run/f'batches_{seed}.npz')['indices']
            with (model/'batches.csv').open(encoding='utf-8',newline='') as f:batches=list(csv.DictReader(f))
            task.require(len(batches)==3528,'Incomplete training batch log')
            for k,row in enumerate(batches):
                task.require(int(row['update'])==k+1 and int(row['positive'])==16 and int(row['negative'])==240
                    and row['indices_sha256']==task.loss.temporal.array_hash(indices[k]),'Actual training indices differ')
            initial=torch.load(model/'initial.pt',weights_only=True,map_location='cpu')
            original=torch.load(run/f'initial32_{seed}.pt',weights_only=True,map_location='cpu')
            task.require(initial.keys()==original.keys() and all(torch.equal(initial[k],original[k]) for k in original),'Execution initialization differs')
            blocks={f'block_{b}':task.load(task.T10/f'validation/block_{b}.npz',['t','y','event']) for b in (4,5)}
            all_scores=[{f'block_{b}':task.load(model/f'epoch_{epoch}_block_{b}.npz')['scores'] for b in (4,5)} for epoch in range(1,13)]
            selected=task.loss.temporal.choose(all_scores,blocks,read_json(run/'parent_training_plan.json'))
            other=independent_choose(all_scores,blocks,read_json(run/'parent_training_plan.json'))
            task.require(numeric_equal(selected,other),'Independent checkpoint rule differs')
            for epoch in range(1,13):
                for b in (4,5):np.testing.assert_array_equal(task.load(model/f'epoch_{epoch}_block_{b}.npz')['t'],blocks[f'block_{b}']['t'])
            meta=read_json(model/'model.json')
            expected_registration=task.ORIGINAL_SHA if seed in (42,1019) else sha256(run/'registration.json')
            task.require(meta['registration_sha256']==expected_registration,'Model origin registration differs')
            task.require(sha256(model/'weights.pt')==sha256(model/f'epoch_{selected["epoch"]}.pt'),'Selected checkpoint weights differ')
            task.require(meta['selected']==selected and meta['recipe']=='unit_BCE' and meta['updates']==3528 and meta['tcn_channels']==32 and meta['parameter_count']==19809,'Original checkpoint rule differs')
        result=dict(training_matrix=True,actual_batch_logs_exact=True,checkpoint_rule_replayed=True,new_fits=1,reused_fits=2,historic_interrupted_fits=1,effective_models=3,effective_matrix_updates=10584,cumulative_attempted_updates_upper_bound=11152)
        if not (run/'source_selection.json').exists():return result
        selection=read_json(run/'source_selection.json');seal=read_json(run/'source_complete.json')
        task.require(seal['source_sha256']==sha256(run/'source_selection.json'),'Source completion differs')
        task.gate.check(run/'source_results',seal['artifacts'])
        source_rows=[]
        for recipe in selection['recipes']:
            metrics=[]
            zs=[task.load(run/f'source_results/scores_{recipe["seed"]}_{b}.npz') for b in (4,5)]
            candidates=np.unique(np.concatenate([z['p'][np.isfinite(z['p'])] for z in zs]))
            low=independent_threshold([(z['p'][z['ix']],z['y']) for z in zs],candidates)
            high=None
            if low is not None:
                hc=np.unique(np.concatenate([z['h'][np.isfinite(z['h'])&(z['h']>=low)] for z in zs]))
                high=independent_threshold([(np.where(z['p'][z['ix']]>=low,z['h'][z['ix']],-1.),z['y']) for z in zs],hc)
            task.require(recipe['L']==low and recipe['H']==high,'Independent L95/H95 selection differs')
            for b in (4,5):
                z=task.load(run/f'source_results/scores_{recipe["seed"]}_{b}.npz')
                h=local_max_independent(z['t'],z['p']);np.testing.assert_array_equal(h,z['h'])
                if recipe['feasible']:
                    state=((z['p'][z['ix']]>=recipe['L'])&(h[z['ix']]>=recipe['H'])).astype(np.int8)
                    metric=scalar_metric(state,z['y']);metrics.append(metric)
                    bound=next(v for v in recipe['blocks'] if v['block']==str(b))
                    task.require(all(bound[k]==v for k,v in metric.items()),'Independent source confusion differs')
            if metrics:source_rows.append(dict(seed=recipe['seed'],**{k:sum(v[k] for v in metrics)/2 for k in ('precision','recall','f1')},
                degenerate=any(v['degenerate'] for v in metrics),source_gate_passed=all(v['recall']>=.95 for v in metrics)))
        task.require(selection['rows']==source_rows,'Source row matrix differs')
        expected=len(source_rows)==3 and all(v['source_gate_passed'] and not v['degenerate'] for v in source_rows) and all(v['source_gate_passed'] for v in complete['rows']) and task.loss.qualified(read_json(run/'source_control.json')['rows']['guard95'],source_rows)
        task.require(selection['eligible']==expected,'Source qualification differs')
        result['source_qualified']=expected
        if not expected:
            task.require(not (run/'freeze.json').exists() and not (run/'predictions').exists() and not (run/'evaluation').exists(),'Rejected source accessed development')
            return result
        if not (run/'freeze.json').exists():return result
        task.evaluation_guard(run)
        result['complete_frozen_9_cell_matrix']=True
        if not (run/'evaluation_complete.json').exists():return result
        seal=read_json(run/'evaluation_complete.json')
        task.require(seal['freeze_sha256']==sha256(run/'freeze.json'),'Evaluation seal differs')
        task.gate.check(run/'evaluation',seal['artifacts'])
        truth=task.load(run/'evaluation/truth_once.npz');saved=read_json(run/'evaluation/metrics.json');rows=[];supports=[]
        for seed in task.SEEDS:
            a=task.load(run/f'predictions/A_{seed}.npz');w=task.load(run/f'predictions/W50_{seed}.npz');c=task.load(run/f'predictions/wide32_{seed}.npz')
            mask=np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1]);task.require(int(sum(mask))==137523,'Independent support differs')
            recipe=next(r for r in selection['recipes'] if r['seed']==seed)
            h=local_max_independent(c['t'],c['scores']);state=np.where(np.isfinite(c['scores']),(c['scores']>=recipe['L'])&(h>=recipe['H']),-1).astype(np.int8)
            np.testing.assert_array_equal(state,c['state'])
            np.testing.assert_array_equal(c['raw'],a['raw']);np.testing.assert_array_equal(c['corrected'],a['corrected'])
            np.testing.assert_array_equal(c['other_states'][:,[0,2]],a['states'][:,[0,2]])
            for arm,z in (('A',a),('W50',w),('wide32',c)):
                np.testing.assert_array_equal(z['t'],truth['t'])
                state=z['state'][mask] if arm=='wide32' else (z['raw'][mask,1]>=200).astype(np.int8)
                rows.append(dict(arm=arm,seed=seed,**scalar_metric(state,truth['y'][mask]>=200)))
            supports.append(dict(seed=seed,n=int(sum(mask)),mask_sha256=task.gate.hash_array(mask)))
        task.require(saved==dict(rows=rows,supports=supports),'Independent development confusions differ')
        gates=independent_acceptance(rows,supports,candidate="wide32");summary=read_json(run/'evaluation/summary.json')
        task.require(summary['acceptance_gates']==gates and summary['development_acceptance_passed']==all(gates.values()),'Independent eight acceptance gates differ')
        result.update(independent_development_counts=True,independent_eight_gates=True,development_acceptance_passed=all(gates.values()))
        return result
