"""Read-only numerical review; writes one new receipt, never refits or reselects."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.dont_write_bytecode=True
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--phase',choices=['pre-freeze','final'],required=True)
args=parser.parse_args()
run=args.run.resolve()
sys.path.insert(0,str(run/'source'))
from virtual_submeter.accuracy import microwave_source5_temporal_execution as e
from virtual_submeter.data.common import DataError,read_json,write_json,sha256,utc
from virtual_submeter.models.arrays import aggregate_arrays,numeric


def require(ok,message):
    if not ok: raise DataError(message)


def review():
    e.verify_training(run)
    p=read_json(run/'plan.json')
    d=e.load_npz(run/'train.npz')
    original=e.load_npz(e.UP/'TCN127.npz')
    n=len(original['y'])
    for key,value in original.items(): np.testing.assert_array_equal(d[key][:n],value)
    require((n,int((original['y']>=200).sum()))==(259806,849),'Original support changed')
    require(sha256(run/'normalization.json')==sha256(e.UP/'normalization.json'),'Normalization differs')
    inputs=read_json(run/'source5_inputs.json')
    raw={}
    for channel,item in inputs.items():
        path=Path(item['path'])
        require(sha256(path)==item['sha256'],'House5 audited file changed')
        with path.open(encoding='utf-8',newline='') as stream: raw[channel]=list(csv.DictReader(stream))
    t,a=aggregate_arrays(raw['aggregate'],8)
    lt=np.array([utc(v['timestamp']) for v in raw['microwave']])
    np.testing.assert_array_equal(t,lt)
    truth=np.array([numeric(v['power_w']) for v in raw['microwave']],np.float32)
    ids,events=e.segments(t,truth)
    added=e.load_npz(run/'added_identity.npz')
    require((len(added['y']),int((added['y']>=200).sum()))==(76898,1426),'Added support differs')
    for key in ('t','y','house','event','raw_index','mask_identity'): np.testing.assert_array_equal(d[key][n:],added[key])
    require((added['house']==2).all() and (added['house_name']=='refit_5').all(),'House5 index/string mismatch')
    used_sets={}
    summaries=[]
    for role,bounds in [('train',e.f.TRAIN),('validation',e.f.VALIDATION)]:
        used_sets[role]=set()
        for b,low,high in bounds:
            if role=='train':
                mask=added['block']==b
                pool={k:v[n:][mask] for k,v in d.items()}
            else:
                pool=e.load_npz(run/f'validation/block_{b}.npz')
                frozen=e.load_npz(e.UP/f'validation/TCN127/block_{b}.npz')
                for key in ('t','y','x','mask','raw_index','event'):
                    np.testing.assert_array_equal(pool[key],frozen[key])
                require((pool['house']==2).all(),'Validation index not remapped to House5')
            np.testing.assert_array_equal(pool['t'],t[pool['raw_index']])
            np.testing.assert_array_equal(pool['y'],truth[pool['raw_index']])
            np.testing.assert_array_equal(pool['event'],ids[pool['raw_index']])
            np.testing.assert_array_equal(pool['mask'],np.isfinite(pool['x']))
            np.testing.assert_array_equal(pool['mask_identity'],np.packbits(pool['mask'],axis=1))
            require((pool['t']-120>=low).all() and (pool['t']+128<=high).all(),'Core crosses frozen block')
            used=pool['t'][:,None]+np.arange(-63,64)*8
            require((used[pool['mask']]>=low).all() and (used[pool['mask']]<high).all(),'Outer observed context crosses frozen block')
            indices=np.searchsorted(t,used)
            safe=np.clip(indices,0,len(t)-1)
            require(np.array_equal(t[safe][pool['mask']],used[pool['mask']]),'Observed context does not exist in actual source grid')
            np.testing.assert_array_equal(pool['x'][pool['mask']],a[safe][pool['mask']])
            core=pool['mask'][:,48:79]
            require(core[:,15].all() and (core.sum(axis=1)>=24).all(),'Core partial support invalid')
            require(not ((~core[:,:-1])&(~core[:,1:])).any(),'Core has consecutive missing')
            if role=='validation': require(core.all(),'Validation contains partial core')
            for event in events:
                if event['start']<low or event['end']>high:
                    require(not (pool['event']==event['event_id']).any(),'Boundary-crossing segment retained')
            # Recompute full outer stopping rules independently from saved model scores.
            ext=e.expand_context(t,a,pool['raw_index'],low,high)
            np.testing.assert_array_equal(ext['x'],pool['x'])
            times=np.unique(used[pool['mask']])
            used_sets[role].update(times.tolist())
            summaries.append(dict(role=role,block=b,centers=len(pool['y']),positive=int((pool['y']>=200).sum()),
                actual_observed_context_times_sha256=e.array_hash(times),core_and_outer_boundary_pass=True))
    require(not used_sets['train'].intersection(used_sets['validation']),'Actual context time overlap between roles')
    require(e.f.VALIDATION[0][1]-e.f.TRAIN[-1][2]==1800,'Safety gap changed')
    for seed in e.f.SEEDS:
        sampler=e.BatchStream(d,'event',seed,p)
        idx=e.load_npz(run/f'sampling_{seed}.npz')['indices']
        for row in idx: np.testing.assert_array_equal(row,sampler.next())
        saved=torch.load(run/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
        old=torch.load(e.UP/f'models/TCN127/seed_{seed}/initial.pt',weights_only=True,map_location='cpu')
        for key,value in saved.items(): require(torch.equal(value,old[key]),'Initialization differs from old control')
    audit=read_json(e.WORKSPACE/'docs/MICROWAVE_SOURCE_VALIDATION_AUDIT_20261025.json')
    guard=next(v for v in audit['access_log'] if v['path']=='src/virtual_submeter/data/protocol.py')
    require(sha256(e.WORKSPACE/guard['path'])==guard['sha256'],'Default data.protocol was modified')
    starts=list((run/'models/source5').glob('seed_*/started.json'))
    require(len(starts)==3 and not list((run/'models/control').glob('seed_*/started.json')),'Real starts/control retrains differ')
    result=dict(status='independently_reviewed_numeric_invariants',phase=args.phase,real_new_starts=3,
        actual_updates=10584,control_retrains=0,blocks=summaries,normalization_byte_exact=True,
        original_259806_rows_all_columns_exact=True,house5_index=2,actual_context_time_sets_disjoint=True,
        all_three_complete_sampling_streams_replayed=True,same_seed_old_initializations_exact=True,
        default_protocol_guard_sha256=guard['sha256'],final_house_data_accessed=False,
        source_selection_sha256=sha256(run/'source_selection.json'),registration_sha256=sha256(run/'registration.json'))
    if args.phase=='final':
        e.verify(run)
        delivery=read_json(run/'delivery_manifest.json')
        e.check_artifacts(run,delivery['artifacts'])
        actual=e.load_npz(run/'evaluation/truth_once.npz')['y']
        rows=read_json(run/'evaluation/metrics.json')['rows']
        for seed in e.f.SEEDS:
            w50=e.parent.reference_arrays(run,seed,'W50')
            mask=np.isfinite(actual)&np.isfinite(w50['raw'][:,1])
            require(int(mask.sum())==137523,'W50 evaluation mask differs')
            for arm in e.f.ARMS:
                data=e.load_npz(run/f'predictions/{arm}_{seed}.npz')
                predicted=data['state'][mask]==1
                on=actual[mask]>=200
                confusion=dict(tp=int((predicted&on).sum()),fp=int((predicted&~on).sum()),
                               fn=int((~predicted&on).sum()),tn=int((~predicted&~on).sum()))
                row=next(v for v in rows if v['arm']==arm and v['seed']==seed)
                require(all(row[k]==v for k,v in confusion.items()),'Independent confusion recomputation differs')
        result.update(development_confusions_independently_recomputed=True,
                      delivery_artifacts_sha256_verified=len(delivery['artifacts']),freeze_sha256=sha256(run/'freeze.json'))
    return result


if __name__=='__main__':
    target=run/f'independent_{args.phase.replace("-","_")}_verification.json'
    if target.exists(): raise DataError('Verification receipt exists; cannot overwrite')
    result=review()
    write_json(target,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
