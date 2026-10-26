"""Independent Task18 preparation arithmetic. Readonly; no optimizer or development."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
import numpy as np
from virtual_submeter.accuracy import microwave_expanded_guard95 as e


def independent_context(t,a,c,low,high):
    offsets=np.arange(-63,64)
    indices=c[:,None]+offsets
    safe=np.clip(indices,0,len(t)-1)
    values=a[safe].copy()
    valid=(indices>=0)&(indices<len(t))&(t[safe]>=low)&(t[safe]<high)
    for direction in (-1,1):
        cols=63+direction*np.arange(1,64)
        idx=safe[:,cols];prev=safe[:,cols-direction]
        obs=np.isfinite(a[idx]);previous_obs=np.c_[np.isfinite(a[c]),obs[:,:-1]]
        stop=(~valid[:,cols])|(t[idx]-t[prev]!=8*direction)|(~obs&~previous_obs)
        valid[:,cols] &= np.cumsum(stop,axis=1)==0
    values[~valid]=np.nan
    return values


def main(run):
    run=Path(run)
    with e.isolation('verify_preparation',e.allowed_files(run,'verify_preparation')) as access:
        e.registered(run)
        # Independent interval intersections and convolution from immutable Task17
        # verifier, not production resample/core selection.
        independent=e.load_module(e.AUDIT/'verify.py','task18_independent_intervals')
        cfg=e.read_json(run/'config.json')
        with np.load(run/'train.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
        old=e.npz(e.TEMPORAL/'train.npz',e.loss.TRAIN_FIELDS)
        for key,v in old.items(): assert d[key][:len(v)].dtype==v.dtype and d[key][:len(v)].tobytes()==v.tobytes()
        results=[];start=len(old['y'])
        for hi,h in enumerate(('2','19')):
            low,high=cfg['candidate_bounds'][h]
            t,a,y=independent.raw(h,low,high,'Appliance5' if h=='2' else 'Appliance4')
            frozen=e.npz(run/f'extra/house_{h}.npz',['t','aggregate','y'])
            for actual,wanted in ((t,frozen['t']),(a,frozen['aggregate']),(y,frozen['y'])): np.testing.assert_array_equal(actual,wanted)
            obs=np.isfinite(a);c=np.arange(15,len(t)-15)
            observed=np.convolve(obs.astype(np.int32),np.ones(31,np.int32),'valid')
            pairs=np.convolve((~obs[:-1]&~obs[1:]).astype(np.int32),np.ones(30,np.int32),'valid')
            gaps=np.convolve((np.diff(t)!=8).astype(np.int32),np.ones(30,np.int32),'valid')
            keep=(observed>=24)&(pairs==0)&(gaps==0)&obs[c]&np.isfinite(y[c])&(t[c]-120>=low)&(t[c]+128<=high)
            c=c[keep]
            n=len(c);sl=slice(start,start+n)
            np.testing.assert_array_equal(d['t'][sl],t[c]);np.testing.assert_array_equal(d['raw_index'][sl],c)
            np.testing.assert_array_equal(d['y'][sl],y[c]);assert (d['house'][sl]==hi).all() and (d['source_role'][sl]=='extended14d').all()
            active=np.isfinite(y)&(y>=200);starts=active&np.r_[True,~active[:-1]|(np.diff(t)!=8)]
            ids=np.where(active,np.cumsum(starts)-1,-1)
            off=int(old['event'][old['house']==hi].max())+1
            np.testing.assert_array_equal(d['event'][sl],np.where(ids[c]>=0,ids[c]+off,-1))
            x=independent_context(t,a,c,low,high)
            np.testing.assert_array_equal(d['x'][sl],x);np.testing.assert_array_equal(d['mask'][sl],np.isfinite(x))
            np.testing.assert_array_equal(d['mask_identity'][sl],np.packbits(np.isfinite(x),axis=1))
            assert [n,int((y[c]>=200).sum()),len(set(ids[c[y[c]>=200]].tolist()))]==cfg['added_support'][h]
            results.append(dict(house=h,centers=n,positive=int((y[c]>=200).sum()),native_values_and_context_exact=True));start+=n
        assert start==615922 and sum(bool(v>=200) for v in d['y'])==2994
        groups={}
        for i in np.flatnonzero(d['y']>=200):
            groups.setdefault((int(d['house'][i]),int(d['t'][i]//86400)),{}).setdefault(int(d['event'][i]),[]).append(int(i))
        expected=np.full(len(d['y']),6468*240/612928,np.float64)
        for events in groups.values():
            for centers in events.values(): expected[centers]=103488/(len(groups)*len(events)*len(centers))
        rows=[]
        for seed in e.SEEDS:
            ix=e.npz(run/f'sampling_{seed}.npz',['indices'])['indices']
            # Different summation path from production bincount.
            count=np.zeros(len(d['y']),np.int64)
            for batch in ix:
                assert len(batch)==256 and sum(d['y'][batch]>=200)==16
                np.add.at(count,batch,1)
            ex=e.npz(run/f'exposure_{seed}.npz',['count','expected'])
            np.testing.assert_array_equal(count,ex['count']);np.testing.assert_array_equal(expected,ex['expected'])
            with (run/f'exposure_{seed}.csv').open(encoding='utf-8',newline='') as f: ledger=list(csv.DictReader(f))
            assert sum(int(r['actual_presentations']) for r in ledger)==6468*256
            for r in ledger:
                h=['refit_2','refit_19','refit_5'].index(r['house'])
                m=(d['house']==h)&(d['source_role']==r['source_role'])&(d['t']//86400==e.utc(r['utc_date']+'T00:00:00Z')//86400)&(d['event']==int(r['event']))&((d['y']>=200)==(r['positive']=='True'))
                assert int(m.sum())==int(r['support']) and sum(map(int,count[m]))==int(r['actual_presentations'])
                assert abs(float(expected[m].sum())-float(r['expected_presentations']))<1e-7
            h5=(d['house']==2)&(d['y']>=200)
            unseen=(d['y']>=200)&(count==0)
            added=np.arange(len(count))>=len(old['y'])
            assert not (unseen&added).any() and not (unseen&(d['house']!=2)).any()
            rows.append(dict(seed=seed,house5_positive_actual=int(count[h5].sum()),house5_positive_expected=float(expected[h5].sum()),unique_positive=int((count[d['y']>=200]>0).sum()),unique_negative=int((count[d['y']<200]>0).sum()),
                unseen_positive_total=int(unseen.sum()),unseen_positive_by_house=[int((unseen&(d['house']==hi)).sum()) for hi in (0,1,2)],unseen_added_positive=int((unseen&added).sum()),added_positive_seen=int(((d['y']>=200)&added&(count>0)).sum())))
        result=dict(status='independent_preparation_passed',additional=results,total=[615922,2994,612928],
            registration_sha256=e.sha256(run/'registration.json'),
            positive_day_groups_by_house=[sum(h==hi for h,day in groups) for hi in (0,1,2)],positive_day_groups=len(groups),
            house5_old_positive_expected=3528*16*8/31,house5_new_positive_expected=6468*16*8/57,
            old_date_positive_exposure_ratio=(6468/57)/(3528/31),seeds=rows,new_fits=0,optimizer_updates=0,development_reads=0,final_house_reads=0,
            method='Independent interval intersection, convolution core mask, cumulative stop context, scalar positive count and per-batch add.at exposure; all ledger rows recomputed')
    result['protected_reads']=len(access['reads']);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();r=main(a.run.resolve())
    if a.output: e.fresh(a.output,r)
    print(json.dumps(r,ensure_ascii=False,indent=2))
