"""Task21 independent preparation replay: no production sampling helpers.

Canonical segmentation is reconstructed as run lengths; queues use forward
deque consumption of reversed permutations, independently of HardStream.pop.
"""
from collections import deque
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from ..data.common import read_json, sha256, inventory
from . import microwave_task21 as task


def independent_parts(d, hard):
    parts=[]
    keys=sorted(set(zip(d['house'][hard].tolist(),d['source_role'][hard].tolist(),(d['t'][hard]//86400).tolist())))
    for h,role,day in keys:
        ids=np.flatnonzero(hard & (d['house']==h) & (d['source_role']==role) & (d['t']//86400==day))
        ids=ids[np.argsort(d['t'][ids],kind='stable')]
        boundaries=np.r_[0,np.flatnonzero(np.diff(d['t'][ids])!=8)+1,len(ids)]
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            parts.extend(ids[start:min(start+31,right)] for start in range(left,right,31))
    return parts


def independent_sequence(d, parts, seed, count):
    digest=hashlib.sha256(f'task21-hard-negative-v1|{seed}'.encode('ascii')).digest()
    rng=np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8],byteorder='big')))
    tree={}
    for k,part in enumerate(parts):
        h=int(d['house'][part[0]]);day=int(d['t'][part[0]]//86400)
        tree.setdefault(h,{}).setdefault(day,[]).append(k)
    queues={}
    def get(key,values):
        if key not in queues or len(queues[key])==0:
            queues[key]=deque([values[int(i)] for i in rng.permutation(len(values))[::-1]])
        return queues[key].popleft()
    out=np.empty(count,np.int64)
    for i in range(count):
        h=get('H',sorted(tree));day=get(('D',h),sorted(tree[h]))
        k=get(('S',h,day),tree[h][day]);out[i]=get(('C',k),parts[k])
    return out


def scalar_counts(pred,truth):
    out=dict(tp=0,fp=0,fn=0,tn=0)
    for p,y in zip(pred,truth):
        out['tp' if p and y else 'fp' if p else 'fn' if y else 'tn']+=1
    return out


def review(run):
    run=Path(run).resolve()
    reg=read_json(run/'registration.json')
    allowed=list(read_json(run/'inputs_before.json'))+[run/k for k in inventory(run)]
    with task.isolation(run,allowed,readonly=True):
        task.registered(run)
        if (run/'preparation_manifest.json').exists():
            task.gate.check(run,read_json(run/'preparation_manifest.json')['artifacts'])
        d=task.load(task.T10/'train.npz')
        d['source_role']=np.full(task.N,'original14d',dtype='U24')
        # Independent UTC interval reconstruction, not production TRAIN constants.
        dates=['2014-11-15','2014-11-17','2014-11-19','2014-11-21','2014-11-22T23:30:00']
        bounds=[int(np.datetime64(v,'s').astype(np.int64)) for v in dates]
        for block,(left,right) in enumerate(zip(bounds[:-1],bounds[1:])):
            d['source_role'][(d['house']==2)&(d['t']>=left)&(d['t']<right)]=f'house5_prefix_block_{block}'
        proof=read_json(run/'sampling_proof.json')
        results={}
        for seed in task.SEEDS:
            # Independent threshold and cache row binding checks.
            meta=read_json(task.T12/f'models/unit_BCE/seed_{seed}/model.json')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                initial_model=task.CenterMaskTCN()
                initial=torch.load(task.T10/f'initial_{seed}.pt',map_location='cpu',weights_only=True)
                unit=torch.load(task.T12/f'models/unit_BCE/seed_{seed}/initial.pt',map_location='cpu',weights_only=True)
                for key,value in initial_model.state_dict().items():
                    task.require(torch.equal(value,initial[key]) and torch.equal(value,unit[key]),'Independent initialization differs')
                    task.require(task.gate.hash_array(value.numpy())==proof[str(seed)]['initial_tensor_hashes'][key],'Initial tensor seal differs')
            with np.load(task.T19/f'predictions/control_{seed}_train.npz',allow_pickle=False) as cache:
                prob=cache['probability'][:task.N]
                np.testing.assert_array_equal(cache['row_index'][:task.N],np.arange(task.N))
            hard=np.zeros(task.N,bool)
            for i in range(task.N):
                hard[i]=bool(np.isfinite(d['y'][i]) and d['y'][i]<200 and prob[i]>=meta['decision_threshold_probability'])
            parts=independent_parts(d,hard)
            ids=np.flatnonzero(hard)
            task.require([int(sum(hard & (d['house']==h))) for h in (0,1,2)]==proof[str(seed)]['hard_counts'],'Independent hard counts differ')
            saved=task.load(run/f'hard_windows_{seed}.npz')
            for key,expected in dict(row_index=ids,probability=prob[ids],y=d['y'][ids],house=d['house'][ids],
                    t=d['t'][ids],source_role=d['source_role'][ids],utc_day=d['t'][ids]//86400,raw_index=d['raw_index'][ids]).items():
                np.testing.assert_array_equal(saved[key],expected)
            part_ids=np.full(task.N,-1,np.int64)
            for k,part in enumerate(parts):part_ids[part]=k
            np.testing.assert_array_equal(saved['subsegment'],part_ids[ids])
            with (run/f'fragments_{seed}.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
            task.require(len(rows)==len(parts),'Fragment matrix incomplete')
            for k,part in enumerate(parts):
                task.require(rows[k]=={key:str(value) for key,value in dict(subsegment=k,house=int(d['house'][part[0]]),
                    source_role=d['source_role'][part[0]],utc_day=int(d['t'][part[0]]//86400),start=int(d['t'][part[0]]),end=int(d['t'][part[-1]]),n=len(part)).items()},'Independent fragment boundary differs')
            old=task.load(task.T10/f'sampling_{seed}.npz')['indices']
            batches=task.load(run/f'batches_{seed}.npz')
            task.require(old.shape==batches['indices'].shape==(3528,256),'Incomplete batch matrix')
            sequence=independent_sequence(d,parts,seed,3528*120).reshape(3528,120)
            np.testing.assert_array_equal(sequence,batches['hard_indices'])
            counters={name:np.zeros(task.N,np.int64) for name in ('total','positive','ordinary','hard')}
            for b in range(3528):
                neg=[slot for slot,index in enumerate(old[b]) if d['y'][index]<200]
                pos=[slot for slot,index in enumerate(old[b]) if d['y'][index]>=200]
                task.require(len(neg)==240 and len(pos)==16,'Independent quota differs')
                np.testing.assert_array_equal(neg[120:],batches['replacement_slots'][b])
                np.testing.assert_array_equal(old[b,pos],batches['indices'][b,pos])
                np.testing.assert_array_equal(old[b,neg[:120]],batches['indices'][b,neg[:120]])
                np.testing.assert_array_equal(sequence[b],batches['indices'][b,neg[120:]])
                for index in batches['indices'][b]:counters['total'][index]+=1
                for index in old[b,pos]:counters['positive'][index]+=1
                for index in old[b,neg[:120]]:counters['ordinary'][index]+=1
                for index in sequence[b]:
                    task.require(hard[index],'Leak/wrong seed/missing/on replacement')
                    counters['hard'][index]+=1
            e=task.load(run/f'exposure_{seed}.npz')
            for key,value in counters.items():np.testing.assert_array_equal(value,e[key])
            np.testing.assert_array_equal(counters['total'],counters['positive']+counters['ordinary']+counters['hard'])
            np.testing.assert_array_equal(e['row_index'],np.arange(task.N))
            np.testing.assert_array_equal(e['subsegment'],part_ids)
            expected=np.zeros(task.N,np.float64)
            for part in parts:
                h=int(d['house'][part[0]]);day=int(d['t'][part[0]]//86400)
                house_count=len(set(d['house'][ids].tolist()))
                day_count=len(set((d['t'][ids[d['house'][ids]==h]]//86400).tolist()))
                segment_count=sum(d['house'][v[0]]==h and d['t'][v[0]]//86400==day for v in parts)
                for i in part:expected[i]=(3528*120)/(house_count*day_count*segment_count*len(part))
            np.testing.assert_allclose(expected,e['expected_hard'],rtol=1e-14,atol=1e-12)
            with (run/f'exposure_groups_{seed}.csv').open(encoding='utf-8',newline='') as f:groups=list(csv.DictReader(f))
            task.require(len(groups)==3+len(set(zip(d['house'].tolist(),(d['t']//86400).tolist())))+len(parts),'Exposure group matrix incomplete')
            for row in groups:
                if row['level']=='house':ix=np.flatnonzero(d['house']==int(row['key']))
                elif row['level']=='date':
                    h,day=map(int,row['key'].split('|'));ix=np.flatnonzero((d['house']==h)&(d['t']//86400==day))
                else:ix=parts[int(row['key'])]
                task.require(int(row['n'])==len(ix) and int(row['actual_total'])==sum(counters['total'][ix])
                    and int(row['actual_hard'])==sum(counters['hard'][ix])
                    and int(row['ordinary'])==sum(counters['ordinary'][ix]) and int(row['positive'])==sum(counters['positive'][ix]),'Independent group exposure differs')
                task.require(abs(float(row['expected_hard'])-sum(expected[ix]))<1e-6,'Independent expected group differs')
            results[str(seed)]=dict(hard_windows=len(ids),subsegments=len(parts),all_3528_batches_exact=True,
                positive_and_ordinary_slots_exact=True,independent_exposure_exact=True,independent_queue_exact=True)
        return dict(passed=True,new_fits=0,optimizer_updates=0,development_reads=0,final_reads=0,seeds=results,
                    registration_sha256=sha256(run/'registration.json'))


def scalar_metric(state,y):
    c=scalar_counts(state==1,y)
    tp,fp,fn,tn=(c[k] for k in ('tp','fp','fn','tn'))
    return dict(**c,n=tp+fp+fn+tn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
        f1=2*tp/max(2*tp+fp+fn,1),fpr=fp/max(fp+tn,1),predicted_on=tp+fp,degenerate=(tp+fp) in (0,tp+fp+fn+tn))


def local_max_independent(t,p):
    out=np.full(len(p),np.nan,p.dtype)
    for i in range(len(p)):
        if not np.isfinite(p[i]):continue
        left=right=i
        while left>max(0,i-15) and t[left]-t[left-1]==8 and np.isfinite(p[left-1]):left-=1
        while right<min(len(p)-1,i+15) and t[right+1]-t[right]==8 and np.isfinite(p[right+1]):right+=1
        out[i]=max(p[left:right+1])
    return out


def independent_acceptance(rows,supports,candidate="hard_negative"):
    task.require({(r['arm'],r['seed']) for r in rows}=={(a,s) for a in ('A','W50',candidate) for s in task.SEEDS}
        and len(rows)==9,'Incomplete development 9-cell matrix')
    base=[next(r for r in rows if r['arm']=='W50' and r['seed']==s) for s in task.SEEDS]
    new=[next(r for r in rows if r['arm']==candidate and r['seed']==s) for s in task.SEEDS]
    mean=lambda group,key:sum(r[key] for r in group)/3
    return dict(mean_P_at_least_008=mean(new,'precision')>=.08,each_R_at_least_085=all(r['recall']>=.85 for r in new),
        mean_F1_gain_at_least_002=mean(new,'f1')-mean(base,'f1')+1e-12>=.02,
        at_least_two_seed_P_F1_up=sum(n['precision']>b['precision'] and n['f1']>b['f1'] for b,n in zip(base,new))>=2,
        each_F1_drop_at_most_002=all(n['f1']-b['f1']>=-.02 for b,n in zip(base,new)),
        mean_FP_reduction_at_least_30_percent=mean(new,'fp')<=.7*mean(base,'fp'),
        no_new_all_on_or_off=all(not r['degenerate'] for r in new),
        support_137523_exact=len(supports)==3 and all(v['n']==137523 for v in supports))


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
            model=run/f'models/hard_negative/seed_{seed}'
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
            original=torch.load(task.T10/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
            task.require(initial.keys()==original.keys() and all(torch.equal(initial[k],original[k]) for k in original),'Execution initialization differs')
            blocks={f'block_{b}':task.load(task.T10/f'validation/block_{b}.npz',['t','y','event']) for b in (4,5)}
            all_scores=[{f'block_{b}':task.load(model/f'epoch_{epoch}_block_{b}.npz')['scores'] for b in (4,5)} for epoch in range(1,13)]
            selected=task.loss.temporal.choose(all_scores,blocks,read_json(run/'parent_training_plan.json'))
            meta=read_json(model/'model.json')
            task.require(meta['selected']==selected and meta['recipe']=='unit_BCE' and meta['updates']==3528,'Original checkpoint rule differs')
        result=dict(training_matrix=True,actual_batch_logs_exact=True,checkpoint_rule_replayed=True,new_fits=3,updates=10584)
        if not (run/'source_selection.json').exists():return result
        selection=read_json(run/'source_selection.json');seal=read_json(run/'source_complete.json')
        task.require(seal['source_sha256']==sha256(run/'source_selection.json'),'Source completion differs')
        task.gate.check(run/'source_results',seal['artifacts'])
        source_rows=[]
        for recipe in selection['recipes']:
            metrics=[]
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
        expected=all(v['source_gate_passed'] for v in complete['rows']) and task.loss.qualified(read_json(run/'source_control.json')['rows']['guard95'],source_rows)
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
            a=task.load(run/f'predictions/A_{seed}.npz');w=task.load(run/f'predictions/W50_{seed}.npz');c=task.load(run/f'predictions/hard_negative_{seed}.npz')
            mask=np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1]);task.require(int(sum(mask))==137523,'Independent support differs')
            recipe=next(r for r in selection['recipes'] if r['seed']==seed)
            h=local_max_independent(c['t'],c['scores']);state=np.where(np.isfinite(c['scores']),(c['scores']>=recipe['L'])&(h>=recipe['H']),-1).astype(np.int8)
            np.testing.assert_array_equal(state,c['state'])
            np.testing.assert_array_equal(c['raw'],a['raw']);np.testing.assert_array_equal(c['corrected'],a['corrected'])
            np.testing.assert_array_equal(c['other_states'][:,[0,2]],a['states'][:,[0,2]])
            for arm,z in (('A',a),('W50',w),('hard_negative',c)):
                np.testing.assert_array_equal(z['t'],truth['t'])
                state=z['state'][mask] if arm=='hard_negative' else (z['raw'][mask,1]>=200).astype(np.int8)
                rows.append(dict(arm=arm,seed=seed,**scalar_metric(state,truth['y'][mask]>=200)))
            supports.append(dict(seed=seed,n=int(sum(mask)),mask_sha256=task.gate.hash_array(mask)))
        task.require(saved==dict(rows=rows,supports=supports),'Independent development confusions differ')
        gates=independent_acceptance(rows,supports);summary=read_json(run/'evaluation/summary.json')
        task.require(summary['acceptance_gates']==gates and summary['development_acceptance_passed']==all(gates.values()),'Independent eight acceptance gates differ')
        result.update(independent_development_counts=True,independent_eight_gates=True,development_acceptance_passed=all(gates.values()))
        return result
