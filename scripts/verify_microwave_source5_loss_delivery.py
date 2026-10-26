"""Independent scalar/numeric audit; preparation mode never opens development."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def fields(path,keys):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in keys}
def check(root,artifacts):
    for k,v in artifacts.items(): assert digest(root/k)==v,k


def audit(run,preparation=False):
    reg,identity,pool = [read(run/name) for name in ('registration.json','identity.json','pool_identity.json')]
    check(run,reg['artifacts'])
    for path,value in identity['source_inputs'].items(): assert digest(Path(path))==value,path
    access = read(run/'prepare_access.json')
    assert access['house17_reads']==access['new_fits']==access['optimizer_updates']==0 and not access['forbidden']
    assert not any('refit_17' in p or 'truth_once' in p or 'evaluation' in p for p in access['reads'])
    parent = Path(identity['parent'])
    d = fields(parent/'train.npz',list(pool['fields']))
    assert len(d['y'])==336704 and (d['y']>=200).sum()==2275
    for key,value in pool['fields'].items(): assert array_hash(d[key])==value,key
    assert digest(parent/'normalization.json')==pool['normalization_sha256']
    for seed in (42,1019,2026):
        ix = fields(parent/f'sampling_{seed}.npz',['indices'])['indices']
        assert ix.shape==(3528,256) and ((d['y'][ix]>=200).sum(1)==16).all()
        assert array_hash(ix)==pool['seeds'][str(seed)]['indices_sha256']
    if preparation:
        assert not (run/'training_started.json').exists() and not (run/'models').exists()
        return dict(status='independent_preparation_review_passed',new_fits=0,updates=0,source_pool_exact=True,all_sampling_indices_exact=True,house17_reads=0)
    complete = read(run/'training_complete.json')
    check(run/'models',complete['artifacts'])
    assert complete['new_fits']==3 and complete['updates']==10584 and complete['control_retrains']==0
    assert len(list((run/'models').glob('*/seed_*/started.json')))==3
    authorization = read(run/'training_started.json')['authorization']
    assert authorization['approval_origin']=='human' and authorization['human_authorization_quote']=='允许追加最多3个模型'
    assert authorization['registration_sha256']==digest(run/'registration.json') and authorization['additional_new_fits']==3
    for seed in (42,1019,2026): assert read(run/f'models/unit_BCE/seed_{seed}/progress.json')==dict(completed_updates=3528,attempted_updates=3528)
    freeze = read(run/'freeze.json')
    check(run,freeze['artifacts'])
    assert freeze['source_selection_sha256']==digest(run/'source_selection.json')
    ec = read(run/'evaluation_complete.json')
    check(run/'evaluation',ec['artifacts'])
    assert ec['truth_csv_parses']==1 and ec['freeze_sha256']==digest(run/'freeze.json')
    truth = fields(run/'evaluation/truth_once.npz',['t','y'])
    rows = read(run/'evaluation/metrics.json')['rows']
    for seed in (42,1019,2026):
        a,w = [fields(run/f'predictions/{arm}_{seed}.npz',['t','raw']) for arm in ('A','W50')]
        mask = np.isfinite(truth['y']) & np.isfinite(w['raw'][:,1])
        assert mask.sum()==137523
        np.testing.assert_array_equal(mask,np.isfinite(truth['y'])&np.isfinite(a['raw'][:,1]))
        for row in [r for r in rows if r['seed']==seed]:
            arm = row['arm']
            z = fields(run/f'predictions/{arm}_{seed}.npz',['t','raw']+([] if arm in ('A','W50') else ['state','scores']))
            np.testing.assert_array_equal(z['t'],truth['t'])
            if arm not in ('A','W50'):
                np.testing.assert_array_equal(z['raw'],a['raw'])
                pred = z['state'][mask]
                assert np.isin(pred,[0,1]).all()
                np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
            else: pred = z['raw'][mask,1]>=200
            counts = dict(tp=0,fp=0,fn=0,tn=0)
            for p,y in zip(pred,truth['y'][mask]>=200): counts['tp' if p and y else 'fp' if p else 'fn' if y else 'tn'] += 1
            assert all(row[k]==v for k,v in counts.items())
            tp,fp,fn = [counts[k] for k in ('tp','fp','fn')]
            for k,value in dict(precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1)).items(): assert row[k]==value
    selection = read(run/'source_selection.json')
    order = ['old_point','old_gate','new_point','new_gate']
    def mean(group,k): return float(np.mean([r[k] for r in group]))
    def qualify(old,new):
        return len(new)==3 and all(not r['degenerate'] and r['source_gate_passed'] for r in new) and mean(new,'f1')>=mean(old,'f1') and all(b['recall']-a['recall']>=-.05 for a,b in zip(old,new)) and sum(b['precision']>a['precision'] for a,b in zip(old,new))>=2
    assert selection['eligibility']==dict(old_point=True,old_gate=True,
        new_point=qualify(selection['rows']['old_point'],selection['rows']['new_point']),new_gate=qualify(selection['rows']['old_gate'],selection['rows']['new_gate']))
    selected = max([a for a in order if selection['eligibility'][a]],key=lambda a:(mean(selection['rows'][a],'precision'),mean(selection['rows'][a],'f1'),-order.index(a)))
    assert selected==selection['selected']
    groups = {a:[r for r in rows if r['arm']==a] for a in ['A','W50']+order}
    base,new = groups['W50'],groups[selected]
    gates = dict(mean_P_at_least_008=mean(new,'precision')>=.08,each_R_at_least_085=all(r['recall']>=.85 for r in new),
        mean_F1_gain_at_least_002=mean(new,'f1')-mean(base,'f1')+1e-12>=.02,
        at_least_two_seed_P_F1_up=sum(b['precision']>a['precision'] and b['f1']>a['f1'] for a,b in zip(base,new))>=2,
        each_F1_drop_at_most_002=all(b['f1']-a['f1']>=-.02 for a,b in zip(base,new)),
        mean_FP_reduction_at_least_30_percent=mean(new,'fp')<=.7*mean(base,'fp'),
        no_new_all_on_or_off=all(not r['degenerate'] for r in new),support_137523_exact=True)
    summary = read(run/'evaluation/summary.json')
    assert summary['acceptance_gates']==gates and summary['development_acceptance_passed']==all(gates.values())
    paired = {}
    for old,new in [('old_point','new_point'),('old_gate','new_gate')]:
        old_by_seed = {r['seed']:r for r in groups[old]}
        paired[new+'_minus_'+old] = [dict(seed=b['seed'],**{k:b[k]-old_by_seed[b['seed']][k] for k in ('precision','recall','f1','fp','tp','fn','tn')}) for b in groups[new]]
    return dict(status='independent_delivery_review_passed',real_new_fits=3,updates=10584,control_retrains=0,
        original_137523_support_exact=True,all_confusions_scalar_exact=True,raw_A_all_columns_exact=True,
        source_selection_sha256=digest(run/'source_selection.json'),freeze_sha256=digest(run/'freeze.json'),
        source_unique_selected=selected,acceptance_gates_independently_recomputed=gates,
        paired_development_deltas=paired)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--preparation',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = audit(args.run,args.preparation)
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))
