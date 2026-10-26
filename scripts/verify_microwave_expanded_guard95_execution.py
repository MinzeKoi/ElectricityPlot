"""Independent Task18 source arithmetic and training ledger audit; no dev reads.

This adds evidence only. It does not change the frozen training implementation,
threshold rule, source winner, budget, or original preparation manifest.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import torch


def confusion(state,y):
    counts=dict(tp=0,fp=0,fn=0,tn=0)
    for p,target in zip(state,y):
        assert p in (0,1)
        counts['tp' if p and target else 'fp' if p else 'fn' if target else 'tn']+=1
    tp,fp,fn,tn=(counts[k] for k in ('tp','fp','fn','tn'));n=tp+fp+fn+tn
    return dict(**counts,n=n,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),fpr=fp/max(fp+tn,1),predicted_on=tp+fp,degenerate=tp+fp in (0,n))


def threshold(blocks,candidates,recall):
    """Stable global order + reverse cumulative positives; no production curve helper."""
    candidates=np.asarray(candidates);ps=[];fs=[];good=np.ones(len(candidates),bool)
    for values,y in blocks:
        order=np.argsort(values,kind='stable');sorted_values=values[order]
        suffix=np.r_[np.cumsum(y[order][::-1],dtype=np.int64)[::-1],0]
        cut=np.searchsorted(sorted_values,candidates,side='left')
        tp=suffix[cut];on=len(y)-cut
        good &= (on>0)&(on<len(y))&(tp/int(y.sum())>=recall)
        ps.append(tp/np.maximum(on,1));fs.append(2*tp/np.maximum(on+int(y.sum()),1))
    valid=np.flatnonzero(good)
    if not len(valid):return None
    ps=np.stack(ps);fs=np.stack(fs)
    rank=np.lexsort((candidates[valid],fs.mean(0)[valid],ps.mean(0)[valid],ps.min(0)[valid]))
    i=int(valid[rank[-1]])
    return dict(threshold=float(candidates[i]),worst_precision=float(ps[:,i].min()),mean_precision=float(ps[:,i].mean()),mean_f1=float(fs[:,i].mean()))


def maximum(t,p):
    """Padded sliding max, independently cut at all missing/time boundaries."""
    out=np.full(p.shape,np.nan,p.dtype);finite=np.isfinite(p)
    starts=np.flatnonzero(finite&np.r_[True,~finite[:-1]|(np.diff(t)!=8)])
    for start in starts:
        end=int(start)+1
        while end<len(p) and finite[end] and t[end]-t[end-1]==8:end+=1
        segment=p[start:end]
        out[start:end]=np.lib.stride_tricks.sliding_window_view(np.pad(segment,(15,15),constant_values=-np.inf),31).max(1)
    return out


def main(run,authorization,output):
    run=run.resolve();sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_expanded_guard95 as e
    e.authorize(run,authorization) # first budget gate, before identity/model/data
    allowed=e.allowed_files(run,'source')
    with e.isolation('source',allowed) as access:
        cfg,identity,selection=e.source_guard(run)
        e.require(not (run/'evaluation').exists(),'Source independent audit must precede development')
        val={str(b):e.npz(run/f'validation/block_{b}.npz',['t','y']) for b in (4,5)}
        assert sum(len(v['y']) for v in val.values())==34351 and sum(int((v['y']>=200).sum()) for v in val.values())==201
        ytrain=e.npz(run/'train.npz',['y'])['y'];training=[];sources=[];derived=[]
        with Path(identity['source']['aggregate']['path']).open(encoding='utf-8',newline='') as f:raw=list(csv.DictReader(f))
        assert all(r['household_id']=='refit_5' for r in raw)
        times=np.array([e.utc(r['timestamp']) for r in raw])
        source_rows={}
        for seed in e.SEEDS:
            dest=run/f'models/expanded/seed_{seed}'
            progress=e.read_json(dest/'progress.json');assert progress==dict(completed_updates=6468,attempted_updates=6468)
            assert e.read_json(dest/'complete.json')['updates']==6468 and e.read_json(dest/'started.json')['maximum_updates']==6468
            init=torch.load(dest/'initial.pt',weights_only=True,map_location='cpu');frozen=torch.load(run/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
            assert init.keys()==frozen.keys() and all(torch.equal(init[k],frozen[k]) for k in init)
            indices=e.npz(run/f'sampling_{seed}.npz',['indices'])['indices']
            with (dest/'batches.csv').open(encoding='utf-8',newline='') as f:batches=list(csv.DictReader(f))
            assert len(batches)==6468 and indices.shape==(6468,256)
            count=np.zeros(len(ytrain),np.int64)
            for number,(row,ix) in enumerate(zip(batches,indices),1):
                assert int(row['update'])==number and int(row['positive'])==16 and int(row['negative'])==240 and (ytrain[ix]>=200).sum()==16
                assert row['indices_sha256']==hashlib.sha256(np.ascontiguousarray(ix).tobytes()).hexdigest()
                assert all(np.isfinite(float(row[k])) for k in ('loss','positive_loss_sum','negative_loss_sum'))
                np.add.at(count,ix,1)
            actual=e.npz(dest/'exposure.npz',['count'])['count'];expected=e.npz(run/f'exposure_{seed}.npz',['count'])['count']
            np.testing.assert_array_equal(count,actual);np.testing.assert_array_equal(count,expected)
            choices=[]
            for epoch in range(1,13):
                caches={b:e.npz(dest/f'epoch_{epoch}_block_{b}.npz',['t','scores']) for b in ('4','5')}
                for b in caches:np.testing.assert_array_equal(caches[b]['t'],val[b]['t'])
                point=threshold([(caches[b]['scores'],val[b]['y']>=200) for b in ('4','5')],np.unique(np.concatenate([v['scores'] for v in caches.values()])),.85)
                if point is not None: choices.append(point|dict(epoch=epoch))
            chosen=min(choices,key=lambda p:(-p['worst_precision'],-p['mean_precision'],-p['mean_f1'],p['epoch'])) if choices else None
            meta=e.read_json(dest/'model.json')
            assert chosen is not None and all(meta['selected'][k]==v for k,v in chosen.items())
            assert e.sha256(dest/'weights.pt')==e.sha256(dest/f'epoch_{chosen["epoch"]}.pt')
            training.append(dict(seed=seed,actual_updates=6468,complete_12_checkpoint_selection_exact=True,selected=chosen,initial_tensors_exact=True,all_6468_batch_hashes_and_quotas_exact=True,center_exposure_exact=True))
            for arm in ('control_guard95','expanded_guard95'):
                parent=dest if arm=='expanded_guard95' else Path(identity['parent'])/f'models/unit_BCE/seed_{seed}'
                scores={}
                for block,lo,hi in e.loss.temporal.f.VALIDATION:
                    rows=[raw[i] for i in np.flatnonzero((times>=lo)&(times<hi))]
                    t,p,_=e.gate.infer(parent,rows);b=str(block);ix=np.searchsorted(t,val[b]['t']);np.testing.assert_array_equal(t[ix],val[b]['t'])
                    scores[b]=dict(t=t,p=p,h=maximum(t,p),ix=ix,y=val[b]['y']>=200)
                recipe=next(r for r in selection['recipes' if arm=='expanded_guard95' else 'control_recipes'] if r['seed']==seed)
                if arm=='expanded_guard95':
                    low=threshold([(d['p'][d['ix']],d['y']) for d in scores.values()],np.unique(np.concatenate([d['p'][np.isfinite(d['p'])] for d in scores.values()])),.95)
                    L=low['threshold'] if low else None
                    high=threshold([(np.where(d['p'][d['ix']]>=L,d['h'][d['ix']],-1.),d['y']) for d in scores.values()],np.unique(np.concatenate([d['h'][np.isfinite(d['h'])&(d['h']>=L)] for d in scores.values()])),.95) if L is not None else None
                    H=high['threshold'] if high else None
                    assert recipe['L']==L and recipe['H']==H and recipe['feasible']==(H is not None)
                if not recipe['feasible']:
                    source_rows.setdefault(arm,[]);continue
                blocks=[]
                for b,d in scores.items():
                    state=((d['p'][d['ix']]>=recipe['L'])&(d['h'][d['ix']]>=recipe['H'])).astype(np.int8)
                    row=confusion(state,d['y']);recorded=next(r for r in recipe['blocks'] if str(r['block'])==b)
                    assert all(recorded[k]==v for k,v in row.items()),(arm,seed,b,row,recorded)
                    blocks.append(dict(block=b,**row))
                mean={k:sum(r[k] for r in blocks)/2 for k in ('precision','recall','f1')}
                row=dict(seed=seed,**mean,degenerate=any(r['degenerate'] for r in blocks),source_gate_passed=all(r['recall']>=.95 for r in blocks))
                assert row==next(r for r in selection['rows'][arm] if r['seed']==seed)
                source_rows.setdefault(arm,[]).append(row)
                sources.append(dict(seed=seed,arm=arm,L=recipe['L'],H=recipe['H'],blocks=blocks,mean=mean))
        old=source_rows['control_guard95'];new=source_rows.get('expanded_guard95',[])
        eligible=len(new)==3 and all(r['source_gate_passed'] and not r['degenerate'] for r in new) and sum(r['f1'] for r in new)>=sum(r['f1'] for r in old) and all(b['recall']-a['recall']>=-.05 for a,b in zip(old,new)) and sum(b['precision']>a['precision'] for a,b in zip(old,new))>=2
        winner='control_guard95'
        if eligible and (sum(r['precision'] for r in new)/3,sum(r['f1'] for r in new)/3)>(sum(r['precision'] for r in old)/3,sum(r['f1'] for r in old)/3):winner='expanded_guard95'
        assert bool(eligible)==selection['expanded_eligible'] and winner==selection['selected']
        result=dict(status='independent_source_and_training_passed',registration_sha256=e.sha256(run/'registration.json'),source_selection_sha256=e.sha256(run/'source_selection.json'),training_complete_sha256=e.sha256(run/'training_complete.json'),
            actual_new_fits=3,actual_updates=sum(r['actual_updates'] for r in training),control_retrains=0,source_eligible=bool(eligible),source_selected=winner,training=training,source_rows=sources,
            development_reads=0,final_house_reads=0,reads=sorted(access['reads']),
            method='Scalar confusions; independently sorted reverse-cumulative curves at fixed .85/.95; independent sliding-window radius15 max; all checkpoint ranks/batch hashes/init/exposure audited')
    e.fresh(output,result)
    return {k:v for k,v in result.items() if k not in ('reads','training','source_rows')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(main(a.run,a.authorization.resolve(),a.output.resolve()),ensure_ascii=False,indent=2))
