"""Independent fixed-output checks; pretruth mode cannot open any truth cache."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np


def main(run, receipt, mode):
    sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_recall_guard95 as e
    e.authorize(run,receipt)
    identity=json.loads((run/'identity.json').read_text(encoding='utf-8'))
    source=json.loads((run/'source_selection.json').read_text(encoding='utf-8'))
    frozen=json.loads((run/'freeze.json').read_text(encoding='utf-8'))
    allowed=list(identity['source_inputs'])+list(identity['development_unlabeled'])+[run/n for n in frozen['artifacts']]
    ordinary=[run/f'ordinary_guard95_seed{s}.csv' for s in e.SEEDS]
    allowed+=ordinary
    if mode=='pretruth':
        assert not (run/'evaluation').exists()
        with e.isolation('freeze',allowed) as access:
            e.evaluation_guard(run)
            rows=[]
            for seed,path in zip(e.SEEDS,ordinary):
                saved=e.npz(run/f'predictions/guard95_{seed}.npz',['t','scores','state'])
                with path.open(encoding='utf-8',newline='') as f:csvrows=list(csv.DictReader(f))
                t=np.array([e.utc(r['timestamp']) for r in csvrows],np.int64)
                p=np.array([float(r['probability_on']) if r['probability_on'] else np.nan for r in csvrows])
                state=np.array([-1 if not r['state'] else 1 if r['state']=='True' else 0 for r in csvrows],np.int8)
                for key,actual in [('t',t),('scores',p),('state',state)]:np.testing.assert_array_equal(actual,saved[key])
                rows.append(dict(seed=seed,rows=len(t),timestamp_probability_state_exact=True,csv_sha256=e.sha256(path),prediction_sha256=e.sha256(run/f'predictions/guard95_{seed}.npz')))
            result=dict(status='all_three_public_CLI_predictions_exact_before_truth',freeze_sha256=e.sha256(run/'freeze.json'),checks=rows,truth_opened=False,new_fits=0,optimizer_updates=0,reads=sorted(access['reads']))
            e.fresh(run/'ordinary_pretruth_seal.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='reads'},indent=2));return
    seal=json.loads((run/'ordinary_pretruth_seal.json').read_text(encoding='utf-8'))
    assert seal['truth_opened'] is False and seal['freeze_sha256']==e.sha256(run/'freeze.json')
    for r in seal['checks']:
        assert r['csv_sha256']==e.sha256(run/f"ordinary_guard95_seed{r['seed']}.csv")
        assert r['prediction_sha256']==e.sha256(run/f"predictions/guard95_{r['seed']}.npz")
    parent=Path(identity['parent'])
    manifest=e.read_json(parent/'final_delivery_manifest.json')['artifacts']
    olds=[parent/f'predictions/new_gate_{s}.npz' for s in e.SEEDS]
    allowed += olds+[identity['truth_cache']['path'],run/'evaluation/confusions.csv']
    with e.isolation('verify',allowed) as access:
        e.evaluation_guard(run)  # Complete matrix before truth hash/open.
        assert e.sha256(Path(identity['truth_cache']['path']))==identity['truth_cache']['sha256']
        truth=e.npz(Path(identity['truth_cache']['path']),['t','y'])
        recorded=e.read_json(run/'evaluation/metrics.json')['rows']
        independent=[];paired=[]
        for seed,oldpath in zip(e.SEEDS,olds):
            a=e.npz(run/f'predictions/A_{seed}.npz',['t','raw','corrected','states'])
            w=e.npz(run/f'predictions/W50_{seed}.npz',['t','raw'])
            c=e.npz(run/f'predictions/guard95_{seed}.npz',['t','raw','corrected','other_states','scores','state'])
            assert e.sha256(oldpath)==manifest[f'predictions/new_gate_{seed}.npz']
            old=e.npz(oldpath,['t','state','raw'])
            for z in (a,w,c,old):np.testing.assert_array_equal(z['t'],truth['t'])
            mask=np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1]);assert mask.sum()==137523
            np.testing.assert_array_equal(mask,np.isfinite(truth['y'])&np.isfinite(a['raw'][:,1]))
            np.testing.assert_array_equal(np.isfinite(c['scores']),np.isfinite(a['raw'][:,1]))
            np.testing.assert_array_equal(c['raw'],a['raw']);np.testing.assert_array_equal(old['raw'],a['raw'])
            np.testing.assert_array_equal(c['corrected'],a['corrected']);np.testing.assert_array_equal(c['other_states'][:,[0,2]],a['states'][:,[0,2]])
            y=truth['y'][mask]>=200
            preds=dict(A=a['raw'][mask,1]>=200,W50=w['raw'][mask,1]>=200,guard95=c['state'][mask],Task12_H85=old['state'][mask])
            for arm,p in preds.items():
                assert np.isin(p,[0,1]).all()
                counts=dict(tp=0,fp=0,fn=0,tn=0)
                for pred,target in zip(p,y):counts['tp' if pred and target else 'fp' if pred else 'fn' if target else 'tn']+=1
                tp,fp,fn,tn=(counts[k] for k in ('tp','fp','fn','tn'))
                row=dict(seed=seed,arm=arm,**counts,n=len(y),precision=tp/(tp+fp),recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn))
                if arm!='Task12_H85':
                    saved=next(r for r in recorded if r['seed']==seed and r['arm']==arm)
                    for k,v in row.items():assert saved[k]==v,(seed,arm,k,saved[k],v)
                independent.append(row)
            new=np.asarray(preds['guard95'],bool)
            for control in ('W50','Task12_H85'):
                before=np.asarray(preds[control],bool)
                nr=next(r for r in independent if r['seed']==seed and r['arm']=='guard95')
                br=next(r for r in independent if r['seed']==seed and r['arm']==control)
                paired.append(dict(seed=seed,reference=control,restored_tp=int((~before&new&y).sum()),new_fn=int((before&~new&y).sum()),removed_fp=int((before&~new&~y).sum()),added_fp=int((~before&new&~y).sum()),
                    delta_tp=nr['tp']-br['tp'],delta_fp=nr['fp']-br['fp'],delta_fn=nr['fn']-br['fn'],delta_tn=nr['tn']-br['tn'],delta_P=nr['precision']-br['precision'],delta_R=nr['recall']-br['recall'],delta_F1=nr['f1']-br['f1']))
        groups={arm:[r for r in independent if r['arm']==arm] for arm in ('A','W50','guard95','Task12_H85')}
        means={arm:{k:float(np.mean([r[k] for r in rs])) for k in ('precision','recall','f1','fp')} for arm,rs in groups.items()}
        w,c=groups['W50'],groups['guard95'];wm,cm=means['W50'],means['guard95']
        gates=dict(mean_P_at_least_008=cm['precision']>=.08,each_R_at_least_085=all(r['recall']>=.85 for r in c),mean_F1_gain_at_least_002=cm['f1']-wm['f1']+1e-12>=.02,
            at_least_two_seed_P_F1_up=sum(b['precision']>a['precision'] and b['f1']>a['f1'] for a,b in zip(w,c))>=2,each_F1_drop_at_most_002=all(b['f1']-a['f1']>=-.02 for a,b in zip(w,c)),mean_FP_reduction_at_least_30_percent=cm['fp']<=.7*wm['fp'],
            no_new_all_on_or_off=all(0<r['tp']+r['fp']<r['n'] for r in c),support_137523_exact=all(r['n']==137523 for r in c))
        formal=e.read_json(run/'evaluation/summary.json');assert gates==formal['acceptance_gates'] and all(gates.values())==formal['development_acceptance_passed']
        # Derive source qualification independently from registered block counts.
        sm=[]
        for recipe in source['recipes']:
            ss=recipe['seed'];blocks=recipe['blocks'];ms=[]
            assert recipe['L']==identity['original_L'][str(ss)] and recipe['H']>=recipe['L']
            for b in blocks:
                tp,fp,fn,tn=(b[k] for k in ('tp','fp','fn','tn'))
                assert tp/(tp+fn)>=.95 and 0<tp+fp<tp+fp+fn+tn
                ms.append(dict(precision=tp/(tp+fp),recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn)))
            sm.append(dict(seed=ss,**{k:float(np.mean([m[k] for m in ms])) for k in ('precision','recall','f1')}))
        ws=source['rows']['W50'];source_ok=np.mean([r['f1'] for r in sm])>=np.mean([r['f1'] for r in ws]) and all(b['recall']-a['recall']>=-.05 for a,b in zip(ws,sm)) and sum(b['precision']>a['precision'] for a,b in zip(ws,sm))>=2
        assert source_ok and source['guard95_eligible'] and source['selected']=='guard95'
        for phase in ('source_select','freeze','evaluate'):
            aa=e.read_json(run/(phase+'_access.json'));assert not aa['forbidden'] and aa['new_fits']==aa['optimizer_updates']==0
        result=dict(status='independent_scalar_numeric_and_paired_verification_passed',rows=independent,means=means,paired=paired,source_qualified=True,formal_gates=gates,formal_passed=all(gates.values()),
            mean_FP_reduction=1-cm['fp']/wm['fp'],mean_F1_gain=cm['f1']-wm['f1'],registration_sha256=e.sha256(run/'registration.json'),source_selection_sha256=e.sha256(run/'source_selection.json'),freeze_sha256=e.sha256(run/'freeze.json'),
            ordinary_pretruth_seal_sha256=e.sha256(run/'ordinary_pretruth_seal.json'),original_truth_csv_reads=0,new_fits=0,optimizer_updates=0,default_changed=False,independent_new_house_validation=False,reads=sorted(access['reads']))
        e.fresh(run/'independent_execution_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('reads','rows','paired')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['pretruth','post']);p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.authorization.resolve(),a.mode)
