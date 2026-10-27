"""Independent numeric Task22 execution audit; source-only unless already evaluated."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]


def threshold(blocks,candidates,recall):
    """Descending prefix counts, independent of the production curve helpers."""
    if not len(candidates):return None
    precision=[];f1=[];valid=np.ones(len(candidates),bool)
    for score,truth in blocks:
        order=np.argsort(-score,kind='stable');ordered=score[order]
        number=np.searchsorted(-ordered,-candidates,side='right')
        positives=np.r_[0,np.cumsum(truth[order],dtype=np.int64)][number]
        p=positives/np.maximum(number,1)
        f=2*positives/np.maximum(number+int(sum(truth)),1)
        valid&=(number>0)&(number<len(truth))&(positives/max(int(sum(truth)),1)>=recall)
        precision.append(p);f1.append(f)
    ids=np.flatnonzero(valid)
    if not len(ids):return None
    ps=np.asarray(precision);fs=np.asarray(f1)
    k=max(ids,key=lambda i:(float(min(ps[:,i])),float(sum(ps[:,i])/len(ps)),float(sum(fs[:,i])/len(fs)),float(candidates[i])))
    return dict(threshold=float(candidates[k]),worst_precision=float(min(ps[:,k])),
        mean_precision=float(sum(ps[:,k])/len(ps)),mean_f1=float(sum(fs[:,k])/len(fs)))


def scalar_confusion(state,truth):
    counts=[0,0,0,0]
    for pred,y in zip(state,truth):
        counts[0 if pred and y else 1 if pred else 2 if y else 3]+=1
    tp,fp,fn,tn=counts
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,n=sum(counts),precision=tp/max(tp+fp,1),
        recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),degenerate=tp+fp in (0,sum(counts)))


def main(run):
    run=run.resolve();sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_task22 as t
    from virtual_submeter.accuracy.microwave_task22_review import local_max_independent
    from virtual_submeter.models.center_mask_tcn import load_state_model
    result=dict(passed=True,seeds={},new_fits=3,optimizer_updates=10584,development_reads=0,final_reads=0)
    with t.isolation(run,t.execution_allowed(run)+[Path(__file__).resolve()],readonly=True) as access:
        t.registered(run)
        torch.set_num_threads(4)
        t.require(torch.get_num_threads()==4,'Audit CPU threads differ')
        p=t.read_json(run/'parent_training_plan.json')
        t.require(p['minimum_validation_recall']==.85,'Original recall85 changed')
        complete=t.read_json(run/'training_complete.json');t.gate.check(run/'models',complete['artifacts'])
        t.require(complete['new_fits']==3 and complete['updates']==10584,'Training matrix/budget differs')
        labels={b:t.load(t.T10/f'validation/block_{b}.npz',['t','y','event']) for b in (4,5)}
        for b,z in labels.items():
            dates={4:('2014-11-23','2014-11-25'),5:('2014-11-25','2014-11-27')}
            low,high=(int(np.datetime64(v,'s').astype(np.int64)) for v in dates[b])
            t.require(len(z['t'])=={4:19127,5:15224}[b] and int(sum(z['y']>=200))=={4:136,5:65}[b]
                and np.all(z['t']-120>=low) and np.all(z['t']+128<=high)
                and np.all(np.diff(z['t'])>0) and np.all(z['t']%8==0)
                and z['event'].shape==z['y'].shape==z['t'].shape, 'Original source UTC date/core/support boundary differs')
            t.require(np.isfinite(z['y']).all() and int(sum(z['y']>=200))>=p['minimum_block_positive_windows']
                and len(set(z['event'][z['y']>=200].tolist()))>=p['minimum_block_events'] and (z['y']<200).any(), 'Original supported-block event protection differs')
        for seed in t.SEEDS:
            model=run/f'models/wide32/seed_{seed}';meta=t.read_json(model/'model.json')
            t.require(t.read_json(model/'progress.json')==dict(completed_updates=3528,attempted_updates=3528)
                and t.read_json(model/'complete.json')['updates']==3528,'Per-seed actual update budget differs')
            trained_initial=torch.load(model/'initial.pt',weights_only=True,map_location='cpu')
            frozen_initial=torch.load(run/f'initial32_{seed}.pt',weights_only=True,map_location='cpu')
            t.require(trained_initial.keys()==frozen_initial.keys() and all(torch.equal(trained_initial[k],frozen_initial[k]) for k in frozen_initial), 'Actual initialization differs')
            norm=t.read_json(t.T10/'normalization.json')
            t.require(all(meta[k]==v for k,v in norm.items()) and meta['recipe']=='unit_BCE'
                and meta['registration_sha256']==t.sha256(run/'registration.json') and meta['updates']==3528
                and meta['arm']=='wide32' and meta['window_samples']==127 and meta['tcn_channels']==32 and meta['parameter_count']==19809,'Actual model/normalization/recipe identity differs')
            sealed=t.load(run/f'batches_{seed}.npz')['indices']
            with (model/'batches.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
            t.require(len(rows)==3528,'Incomplete actual batch log')
            for step,row in enumerate(rows):
                t.require(int(row['update'])==step+1 and int(row['positive'])==16 and int(row['negative'])==240
                    and row['indices_sha256']==hashlib.sha256(np.ascontiguousarray(sealed[step]).tobytes()).hexdigest(),'Actual batches differ from sealed indices')
            np.testing.assert_array_equal(t.load(model/'exposure.npz')['count'],np.bincount(sealed.ravel(),minlength=336704))
            points=[];history=t.read_json(model/'history.json')['rows']
            t.require(len(history)==12,'Incomplete checkpoint history')
            for epoch in range(1,13):
                state=torch.load(model/f'epoch_{epoch}.pt',weights_only=True,map_location='cpu')
                t.require(state.keys()==frozen_initial.keys() and all(state[k].shape==frozen_initial[k].shape for k in state) and sum(v.numel() for v in state.values())==19809,'Checkpoint width/parameter identity differs')
                blocks=[]
                for b in (4,5):
                    z=t.load(model/f'epoch_{epoch}_block_{b}.npz')
                    np.testing.assert_array_equal(z['t'],labels[b]['t'])
                    t.require(z['scores'].shape==labels[b]['y'].shape and np.isfinite(z['scores']).all()
                        and np.all((z['scores']>=0)&(z['scores']<=1)),'Checkpoint support/probability range differs')
                    blocks.append((z['scores'],labels[b]['y']>=200))
                point=threshold(blocks,np.unique(np.concatenate([v[0] for v in blocks])),.85)
                actual=history[epoch-1]['selection']
                t.require((point is None)==(actual is None),'Checkpoint eligibility differs')
                if point:
                    for k,v in point.items():t.require(abs(v-actual[k])<1e-15,'Independent checkpoint threshold/rank differs')
                    points.append(dict(epoch=epoch,**point))
            selected=min(points,key=lambda v:(-v['worst_precision'],-v['mean_precision'],-v['mean_f1'],v['epoch'])) if points else None
            if selected is None:
                t.require(meta['selected']['epoch']==12 and meta['selected']['threshold']==.5 and meta['selected']['source_gate_passed'] is False,'Independent fallback checkpoint differs')
                selected=dict(epoch=12,threshold=.5,source_gate_passed=False)
            else:t.require(all(abs(meta['selected'][k]-v)<1e-15 for k,v in selected.items()),'Independent chosen checkpoint differs')
            t.require(t.sha256(model/'weights.pt')==t.sha256(model/f'epoch_{selected["epoch"]}.pt')==meta['weights_sha256'],'Chosen checkpoint weight identity differs')
            result['seeds'][str(seed)]=dict(checkpoints_recomputed=12,selected=selected,all_3528_actual_index_hashes_exact=True,actual_exposure_exact=True)
        if (run/'source_selection.json').exists():
            source=t.read_json(run/'source_selection.json');control=t.read_json(run/'source_control.json')
            source_seal=t.read_json(run/'source_complete.json')
            t.require(source_seal['source_sha256']==t.sha256(run/'source_selection.json')
                and source_seal['registration_sha256']==t.sha256(run/'registration.json')
                and source['training_sha256']==t.sha256(run/'training_complete.json'),'Source execution seal differs')
            t.gate.check(run/'source_results',source_seal['artifacts'])
            t.require([r['seed'] for r in source['recipes']]==t.SEEDS,'Incomplete source seed matrix')
            candidate_rows=[]
            for recipe in source['recipes']:
                seed=recipe['seed'];blocks=[];saved={}
                for b in (4,5):
                    z=t.load(run/f'source_results/scores_{seed}_{b}.npz');saved[b]=z
                    np.testing.assert_array_equal(z['t'][z['ix']],labels[b]['t'])
                    np.testing.assert_array_equal(z['y'],labels[b]['y']>=200)
                    np.testing.assert_array_equal(z['h'],local_max_independent(z['t'],z['p']))
                    blocks.append((z['p'][z['ix']],z['y']))
                candidates=np.unique(np.concatenate([v['p'][np.isfinite(v['p'])] for v in saved.values()]))
                low_point=threshold(blocks,candidates,.95);low=low_point['threshold'] if low_point else None
                t.require(low==recipe['L'],'Independent L95 differs')
                high=None
                if low is not None:
                    hc=np.unique(np.concatenate([v['h'][np.isfinite(v['h'])&(v['h']>=low)] for v in saved.values()]))
                    hb=[(np.where(v['p'][v['ix']]>=low,v['h'][v['ix']],-1.),v['y']) for v in saved.values()]
                    high_point=threshold(hb,hc,.95);high=high_point['threshold'] if high_point else None
                t.require(high==recipe['H'] and recipe['feasible']==(high is not None),'Independent H95/feasibility differs')
                metrics=[]
                if high is not None:
                    for b,z in saved.items():
                        state=(z['p'][z['ix']]>=low)&(z['h'][z['ix']]>=high)
                        c=scalar_confusion(state,z['y']);metrics.append(c)
                        actual=next(v for v in recipe['blocks'] if v['block']==str(b))
                        t.require(all(actual[k]==v for k,v in c.items()),'Independent source scalar confusion differs')
                    candidate_rows.append(dict(seed=seed,**{k:sum(v[k] for v in metrics)/2 for k in ('precision','recall','f1')},
                        degenerate=any(v['degenerate'] for v in metrics),source_gate_passed=all(v['recall']>=.95 for v in metrics)))
                result['seeds'][str(seed)].update(L95=low,H95=high,source_blocks=metrics)
            old=control['rows']['guard95']
            for r in control['recipes']:
                for b in r['blocks']:
                    t.require(b['precision']==b['tp']/max(b['tp']+b['fp'],1)
                        and b['recall']==b['tp']/max(b['tp']+b['fn'],1)
                        and b['f1']==2*b['tp']/max(2*b['tp']+b['fp']+b['fn'],1),'Control scalar source metric differs')
            qualified=(len(candidate_rows)==3 and all(v['source_gate_passed'] and not v['degenerate'] for v in candidate_rows)
                and sum(v['f1'] for v in candidate_rows)>=sum(v['f1'] for v in old)
                and all(n['recall']-o['recall']>=-.05 for o,n in zip(old,candidate_rows))
                and sum(n['precision']>o['precision'] for o,n in zip(old,candidate_rows))>=2
                and all(v['source_gate_passed'] for v in complete['rows']))
            t.require(source['rows']==candidate_rows and source['eligible']==qualified,'Independent source qualification differs')
            result.update(source_qualified=qualified,independent_L95_H95_rank=True,independent_source_scalars=True)
            if not qualified:
                t.require(not (run/'freeze.json').exists() and not (run/'predictions').exists() and not (run/'evaluation').exists(),'Rejected source opened development')
            elif (run/'evaluation_complete.json').exists():
                result['development_reads']='already_sealed_evaluation_only'
        result['diagnostics']=audit_diagnostics(run,t)
        result['numerical_reads']=sorted(access['reads'])
        result.update(actual_initialization_exact=True,actual_model_normalization_exact=True,
            original85_event_support_exact=True,original_date_core_boundaries_exact=True,
            original_min_positive_and_event_support_exact=True,independent_all36_checkpoint_rank=True)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))



def tied_ap_scalar(probability,truth):
    groups={}
    for p,y in zip(probability,truth):
        pos,total=groups.get(float(p),(0,0));groups[float(p)]=(pos+int(y),total+1)
    tp=n=0;out=0.;positive=int(sum(truth))
    for key in sorted(groups,reverse=True):
        pos,total=groups[key];tp+=pos;n+=total;out+=(pos/max(positive,1))*(tp/n)
    return out


def audit_diagnostics(run,t):
    seal=t.read_json(run/'diagnostics_complete.json');t.gate.check(run/'diagnostics',seal['artifacts'])
    saved=t.read_json(run/'diagnostics/metrics.json')
    t.require(saved['explanatory_only'] and saved['new_thresholds']==saved['new_selections']==saved['development_reads']==0,'Diagnostic selection changed')
    expected={(arm,s,name,view) for arm in ('control16','wide32') for s in t.SEEDS for name in ('train','source') for view in ('original','strict_core31')}
    t.require(len(saved['rows'])==24 and {(r['arm'],r['seed'],r['dataset'],r['view']) for r in saved['rows']}==expected,'Diagnostic24 matrix differs')
    t.require(len(list((run/'diagnostics').glob('*.npz')))==12,'Diagnostic12 cache matrix differs')
    train=t.load(t.T10/'train.npz');vals=[t.load(t.T10/f'validation/block_{b}.npz') for b in (4,5)]
    datasets={'train':train,'source':{k:np.concatenate([v[k] for v in vals]) for k in ('x','y','t','house')}}
    input_hashes=dict(train=t.sha256(t.T10/'train.npz'),block4=t.sha256(t.T10/'validation/block_4.npz'),block5=t.sha256(t.T10/'validation/block_5.npz'))
    from virtual_submeter.models.center_mask_tcn import load_state_model
    results=[];bindings=[];support={}
    for arm in ('control16','wide32'):
        for seed in t.SEEDS:
            path=t.T12/f'models/unit_BCE/seed_{seed}' if arm=='control16' else run/f'models/wide32/seed_{seed}'
            meta,model=load_state_model(path)
            t.require(meta.get('tcn_channels',16)==(16 if arm=='control16' else 32),'Diagnostic width differs')
            for name,d in datasets.items():
                cachepath=run/f'diagnostics/{arm}_{seed}_{name}.npz';z=t.load(cachepath)
                np.testing.assert_array_equal(z['t'],d['t']);np.testing.assert_array_equal(z['house'],d['house'])
                observed=np.isfinite(d['x']);normalized=np.zeros(d['x'].shape,np.float32)
                normalized[observed]=(np.log1p(d['x'][observed])-meta['input_log_mean'])/meta['input_log_std']
                channels=torch.from_numpy(np.stack([normalized,observed.astype(np.float32)],axis=1))
                with torch.inference_mode():
                    probability=np.concatenate([torch.sigmoid(model(v)).numpy() for v in channels.split(meta['batch_size'])])
                np.testing.assert_array_equal(z['probability'],probability)
                for view in ('original','strict_core31'):
                    mask=np.ones(len(d['y']),bool) if view=='original' else observed[:,48:79].all(1)
                    row=next(r for r in saved['rows'] if (r['arm'],r['seed'],r['dataset'],r['view'])==(arm,seed,name,view))
                    t.require(row['n']==int(mask.sum()) and row['support_sha256']==t.gate.hash_array(mask),'Diagnostic support differs')
                    expected_mask=(t.gate.hash_array(mask),t.gate.hash_array(d['t'][mask]),row['n'])
                    t.require(support.setdefault((name,view),expected_mask)==expected_mask,'Diagnostic arms not same-window aligned')
                    p=probability[mask];truth=d['y'][mask]>=200;metric=scalar_confusion(p>=meta['decision_threshold_probability'],truth)
                    metric.update(fpr=metric['fp']/max(metric['fp']+metric['tn'],1),predicted_on=metric['tp']+metric['fp'],ap=tied_ap_scalar(p,truth),threshold=meta['decision_threshold_probability'])
                    for key,value in metric.items():
                        t.require(abs(value-row[key])<2e-12 if isinstance(value,float) else value==row[key],'Independent diagnostic AP/scalar differs: '+key)
                    results.append(dict(arm=arm,seed=seed,dataset=name,view=view,**metric))
                bindings.append(dict(arm=arm,seed=seed,dataset=name,cache_sha256=t.sha256(cachepath),model_json_sha256=t.sha256(path/'model.json'),weights_sha256=t.sha256(path/'weights.pt'),probability_replay_exact=True))
    t.require(support[('source','original')]==support[('source','strict_core31')],'Source strict support differs')
    return dict(passed=True,cells=24,caches=12,metrics=results,cache_bindings=bindings,input_hashes=input_hashes,cpu_threads=torch.get_num_threads(),
        support={f'{a}/{b}':dict(mask_sha256=v[0],timestamps_sha256=v[1],n=v[2]) for (a,b),v in support.items()},
        model_probability_replay_exact=True,independent_scalar_AP_exact=True,selection_or_threshold_changes=0,development_reads=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
