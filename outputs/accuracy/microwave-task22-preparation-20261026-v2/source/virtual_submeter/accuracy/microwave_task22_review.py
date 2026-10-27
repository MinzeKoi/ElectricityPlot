"""Task22 read-only execution replay, complete event/date blocks required."""
import csv
from pathlib import Path
import numpy as np
import torch
from ..data.common import read_json, sha256
from . import microwave_task22 as task
from .microwave_task21_review import scalar_metric, local_max_independent, independent_acceptance

def review_execution(run):
    """Read-only future execution review. No dev inputs until sealed evaluation.

    Uses execution's own frozen truth cache only after complete nine-cell freeze;
    independently recomputes scalar counts, temporal max and eight acceptance gates.
    """
    run=Path(run)
    with task.isolation(run,task.execution_allowed(run),readonly=True):
        task.registered(run)
        started=read_json(run/'training_started.json')
        task.require(started['seeds']==task.SEEDS,'Incomplete training matrix')
        complete=read_json(run/'training_complete.json')
        task.require(complete['new_fits']==3 and complete['updates']==10584 and complete['control_retrains']==0,'Execution budget differs')
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
            task.require(sha256(model/'weights.pt')==sha256(model/f'epoch_{selected["epoch"]}.pt'),'Selected checkpoint weights differ')
            task.require(meta['selected']==selected and meta['recipe']=='unit_BCE' and meta['updates']==3528 and meta['tcn_channels']==32 and meta['parameter_count']==19809,'Original checkpoint rule differs')
        result=dict(training_matrix=True,actual_batch_logs_exact=True,checkpoint_rule_replayed=True,new_fits=3,updates=10584)
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
from collections import deque
from .microwave_task21_review import scalar_counts


def independent_batches(d,seed,count):
    rng=np.random.Generator(np.random.PCG64(seed));queues={};groups={}
    negatives=np.flatnonzero(d['y']<200)
    for i in np.flatnonzero(d['y']>=200):
        day=(int(d['house'][i]),int(d['t'][i]//86400));groups.setdefault(day,{}).setdefault(int(d['event'][i]),[]).append(int(i))
    def take(key,pool,n):
        values=[]
        for _ in range(n):
            if key not in queues or not queues[key]:queues[key]=deque(pool[int(j)] for j in rng.permutation(len(pool))[::-1])
            values.append(queues[key].popleft())
        return values
    for _ in range(count):
        values=[]
        for _ in range(16):
            day=take('date',sorted(groups),1)[0];event=take(('event',day),sorted(groups[day]),1)[0]
            values+=take(('center',day,event),groups[day][event],1)
        values+=take('negative',negatives,240)
        yield rng.permutation(np.asarray(values,np.int64))


def review_preparation(run):
    allowed=task.execution_allowed(run)
    with task.isolation(run,allowed,readonly=True):
        task.registered(run)
        if (run/'preparation_manifest.json').exists():task.gate.check(run,read_json(run/'preparation_manifest.json')['artifacts'])
        d=task.load(task.T10/'train.npz');p=read_json(run/'parent_training_plan.json');proof=read_json(run/'sampling_proof.json')
        task.require(d['x'].shape==(336704,127) and int((d['y']>=200).sum())==2275 and np.isfinite(d['y']).all(),'Independent pool differs')
        np.testing.assert_array_equal(np.isfinite(d['x']),d['mask'])
        task.require((d['event'][d['y']>=200]>=0).all(),'Missing positive event')
        for seed in task.SEEDS:
            batches=task.load(run/f'batches_{seed}.npz')['indices'];old=task.load(task.T10/f'sampling_{seed}.npz')['indices']
            np.testing.assert_array_equal(batches,old)
            for row,replayed in zip(batches,independent_batches(d,seed,3528)):np.testing.assert_array_equal(row,replayed)
            np.testing.assert_array_equal(task.load(run/f'exposure_{seed}.npz')['total'],np.bincount(batches.ravel(),minlength=336704))
            historical=torch.load(task.T10/f'initial_{seed}.pt',map_location='cpu',weights_only=True)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);oldnet=task.CenterMaskTCN()
                for key,value in oldnet.state_dict().items():task.require(torch.equal(value,historical[key]),'Independent16 initialization differs')
                torch.manual_seed(seed);wide=task.CenterMaskTCN(32)
                saved=torch.load(run/f'initial32_{seed}.pt',map_location='cpu',weights_only=True)
                for key,value in wide.state_dict().items():
                    task.require(torch.equal(value,saved[key]) and task.gate.hash_array(value.numpy())==proof[str(seed)]['initial32_tensor_hashes'][key],'Independent32 initialization differs')
            task.require(len(wide.blocks)==6 and [b.dilation for b in wide.blocks]==[(v,) for v in (1,2,4,8,16,32)]
                and all(b.kernel_size==(3,) for b in wide.blocks) and wide.head[0].out_features==32
                and sum(v.numel() for v in wide.parameters())==19809,'Width changed architecture')
        data_identity=read_json(run/'data_identity.json')
        for b,start,end in ((4,'2014-11-23','2014-11-25'),(5,'2014-11-25','2014-11-27')):
            v=task.load(task.T10/f'validation/block_{b}.npz')
            lo,hi=[int(np.datetime64(s,'s').astype(np.int64)) for s in (start,end)]
            task.require((v['t']>=lo+120).all() and (v['t']<hi-120).all() and np.isfinite(v['y']).all(),'Independent block boundary differs')
            for key,value in v.items():task.require(task.gate.hash_array(value)==data_identity['validation'][f'block_{b}'][key],'Validation identity differs')
            task.require(not set(v['t'].tolist())&set(d['t'][d['house']==2].tolist()),'Independent center leakage')
            task.require(not set(v['event'][v['y']>=200].tolist())&set(d['event'][(d['house']==2)&(d['y']>=200)].tolist()),'Independent event leakage')
        return dict(passed=True,new_fits=0,optimizers=0,optimizer_updates=0,backward_calls=0,development_reads=0,final_reads=0,
            original_10584_batches_exact=True,independent_deque_replay=True,initial16_historical_exact=True,
            initial32_seed_repeatable=True,validation_events_and_dates_complete=True,registration_sha256=sha256(run/'registration.json'))

def independent_ap(scores,truth):
    values,inverse=np.unique(scores,return_inverse=True)
    pos=np.bincount(inverse,weights=np.asarray(truth,np.int64),minlength=len(values))[::-1]
    total=np.bincount(inverse,minlength=len(values))[::-1]
    tp=np.cumsum(pos);predicted=np.cumsum(total)
    return float(np.sum(pos/max(int(np.sum(truth)),1)*tp/np.maximum(predicted,1)))


def independent_point(scores,y,threshold):
    counts=scalar_counts(scores>=threshold,y>=200);tp,fp,fn,tn=[counts[k] for k in ('tp','fp','fn','tn')]
    return dict(**counts,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
        off_false_positive_rate=fp/max(fp+tn,1),ap=independent_ap(scores,y>=200),prevalence=float((y>=200).mean()),n=len(y),threshold=float(threshold))


def independent_choose(checkpoints,blocks,p):
    task.require(len(checkpoints)==12 and set(blocks)=={'block_4','block_5'},'Independent checkpoint matrix incomplete')
    choices=[]
    for epoch,scores in enumerate(checkpoints,1):
        thresholds=np.unique(np.concatenate(list(scores.values())))
        arrays=[];eligible=np.ones(len(thresholds),bool)
        for name,d in blocks.items():
            s=scores[name];y=d['y']>=200
            task.require(len(s)==len(y) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all(),'Checkpoint alignment differs')
            bins=np.searchsorted(thresholds,s)
            positives=np.bincount(bins,weights=y.astype(np.int64),minlength=len(thresholds)).astype(np.int64)
            negatives=np.bincount(bins,weights=(~y).astype(np.int64),minlength=len(thresholds)).astype(np.int64)
            tp=positives[::-1].cumsum()[::-1];fp=negatives[::-1].cumsum()[::-1];on=tp+fp
            eligible &= (on>0)&(on<len(y))&(tp/y.sum()>=.85)
            arrays.append((tp/np.maximum(on,1),2*tp/np.maximum(on+y.sum(),1)))
        if eligible.any():
            ps=np.stack([v[0] for v in arrays]);fs=np.stack([v[1] for v in arrays])
            j=max(np.flatnonzero(eligible),key=lambda k:(ps[:,k].min(),ps[:,k].mean(),fs[:,k].mean(),thresholds[k]))
            threshold=float(thresholds[j]);metrics={k:independent_point(scores[k],d['y'],threshold) for k,d in blocks.items()}
            choices.append(dict(epoch=epoch,threshold=threshold,worst_precision=float(ps[:,j].min()),mean_precision=float(ps[:,j].mean()),
                mean_f1=float(fs[:,j].mean()),supported_blocks=list(blocks),limited_evidence=False,
                mean_recall=float(np.mean([v['recall'] for v in metrics.values()])),block_metrics=metrics,source_gate_passed=True))
    if choices:return min(choices,key=lambda r:(-r['worst_precision'],-r['mean_precision'],-r['mean_f1'],r['epoch']))
    return dict(epoch=12,threshold=.5,source_gate_passed=False,block_metrics={k:independent_point(checkpoints[-1][k],d['y'],.5) for k,d in blocks.items()})

def local_max_independent(t,p):
    """Monotonic deque over finite contiguous segments; no production max helper."""
    out=np.full(len(p),np.nan,p.dtype);i=0
    while i<len(p):
        if not np.isfinite(p[i]):i+=1;continue
        end=i+1
        while end<len(p) and np.isfinite(p[end]) and t[end]-t[end-1]==8:end+=1
        queue=deque();right=i
        for center in range(i,end):
            while right<min(end,center+16):
                while queue and p[queue[-1]]<=p[right]:queue.pop()
                queue.append(right);right+=1
            while queue[0]<center-15:queue.popleft()
            out[center]=p[queue[0]]
        i=end
    return out


def independent_threshold(blocks,candidates):
    if not len(candidates):return None
    eligible=np.ones(len(candidates),bool);ps=[];fs=[]
    for values,y in blocks:
        u,ix=np.unique(values,return_inverse=True)
        pos=np.bincount(ix,weights=y.astype(np.int64),minlength=len(u)).astype(np.int64)
        neg=np.bincount(ix,weights=(~y).astype(np.int64),minlength=len(u)).astype(np.int64)
        cut=np.searchsorted(u,candidates);tp=np.r_[pos[::-1].cumsum()[::-1],0][cut];fp=np.r_[neg[::-1].cumsum()[::-1],0][cut]
        on=tp+fp;eligible &= (on>0)&(on<len(y))&(tp/max(y.sum(),1)>=.95)
        ps.append(tp/np.maximum(on,1));fs.append(2*tp/np.maximum(on+y.sum(),1))
    ps,fs=np.array(ps),np.array(fs)
    good=np.flatnonzero(eligible)
    return float(candidates[max(good,key=lambda k:(ps[:,k].min(),ps[:,k].mean(),fs[:,k].mean(),candidates[k]))]) if len(good) else None


def numeric_equal(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(numeric_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return len(a)==len(b) and all(numeric_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return bool(np.isclose(a,b,rtol=0,atol=2e-12))
    return a==b
