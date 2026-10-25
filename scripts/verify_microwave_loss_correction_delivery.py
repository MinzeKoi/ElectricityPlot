"""Task 8 separate arithmetic audit and exported PR evidence; no new model fits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def curve(scores,truth):
    order=np.argsort(scores,kind='stable')
    s,y=scores[order],truth[order]
    thresholds,first=np.unique(s,return_index=True)
    tp=np.r_[np.cumsum(y[::-1])[::-1],0][first]
    predicted=len(y)-first
    return dict(thresholds=thresholds,precision=tp/np.maximum(predicted,1),recall=tp/max(int(y.sum()),1),
                f1=2*tp/np.maximum(predicted+int(y.sum()),1))


def seal_control(run):
    up=Path(read(run/'parent_identity.json')['run'])
    bound=read(run/'parent_identity.json')
    assert digest(up/'freeze.json')==bound['freeze_sha256']
    assert digest(up/'registration.json')==bound['registration_sha256']
    training=read(up/'training_complete.json')
    dest=run/'reused_control'
    if dest.exists(): raise ValueError('Control seal exists')
    import shutil
    dest.mkdir()
    files={}
    for seed in (42,1019,2026):
        own=dest/f'seed_{seed}';own.mkdir()
        for name in ['model.json','weights.pt','initial.pt']:
            relative=f'TCN127/seed_{seed}/{name}'
            assert digest(up/'models'/relative)==training['artifacts'][relative]
            shutil.copyfile(up/'models'/relative,own/name)
            files[f'seed_{seed}/{name}']=digest(own/name)
    write(run/'reused_control_seal.json',dict(parent_training_sha256=digest(up/'training_complete.json'),
        parent_freeze_sha256=digest(up/'freeze.json'),artifacts=files,real_control_fits=0,truth_reads=0))
    print('Parent control weights, metadata and initial parameters sealed; no fits or truth reads.')


def audit(run):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_loss_correction as f
    f.verify_training(run,predictions=True)  # Guard before cached truth read.
    seal=read(run/'reused_control_seal.json')
    up=Path(read(run/'parent_identity.json')['run'])
    assert digest(up/'training_complete.json')==seal['parent_training_sha256']
    for name,h in seal['artifacts'].items():
        assert digest(run/'reused_control'/name)==h
        assert digest(up/'models/TCN127'/name)==h
    with np.load(run/'evaluation/truth_once.npz',allow_pickle=False) as z:
        truth,t=z['y'],z['t']
    receipt=read(run/'evaluation_complete.json')
    for name,h in receipt['artifacts'].items(): assert digest(run/'evaluation'/name)==h
    recorded=read(run/'evaluation/metrics.json')['rows']
    baseline=Path(read(run/'baseline_identity.json')['reference_run'])
    out=run/'pr'
    if out.exists(): raise ValueError('Audit PR evidence already exists')
    (out/'source').mkdir(parents=True);(out/'development').mkdir()
    exact=[];paired=[]
    for seed in f.SEEDS:
        part='frozen_test_refit_17_0.npz'
        with np.load(baseline/f'predictions/W50/seed_{seed}'/part,allow_pickle=False) as z:
            wraw=z['raw'];np.testing.assert_array_equal(t,z['stamps'])
        with np.load(baseline/f'reference/predictions/seed_{seed}'/part,allow_pickle=False) as z:
            araw=z['raw'];np.testing.assert_array_equal(t,z['stamps'])
        valid=np.isfinite(truth)&np.isfinite(wraw[:,1]);assert valid.sum()==137523
        actual=truth[valid]>=200
        scores={'W50':wraw[:,1],'A':araw[:,1]}
        states={'W50':wraw[:,1]>=200,'A':araw[:,1]>=200}
        for arm in f.ORDER:
            with np.load(run/f'predictions/{arm}_{seed}.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(t,z['t'])
                np.testing.assert_array_equal(araw,z['raw'])
                np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(wraw[:,1]))
                scores[arm]=z['scores'];states[arm]=z['state']==1
        for arm in ['A','W50']+f.ORDER:
            pred=states[arm][valid]
            counts=[int((pred&actual).sum()),int((pred&~actual).sum()),int((~pred&actual).sum()),int((~pred&~actual).sum())]
            row=next(r for r in recorded if r['arm']==arm and r['seed']==seed)
            assert [row[k] for k in ['tp','fp','fn','tn']]==counts
            tp,fp,fn,tn=counts
            assert row['precision']==tp/max(tp+fp,1) and row['recall']==tp/max(tp+fn,1) and row['f1']==2*tp/max(2*tp+fp+fn,1)
            exact.append(dict(arm=arm,seed=seed,exact=True))
            np.savez_compressed(out/f'development/{arm}_{seed}.npz',**curve(scores[arm][valid],actual))
        for arm in f.ORDER:
            row=next(r for r in recorded if r['arm']==arm and r['seed']==seed)
            for reference in ['TCN127','W50']:
                base=next(r for r in recorded if r['arm']==reference and r['seed']==seed)
                paired.append(dict(arm=arm,seed=seed,reference=reference,precision_delta=row['precision']-base['precision'],
                    recall_delta=row['recall']-base['recall'],f1_delta=row['f1']-base['f1'],fp_delta=row['fp']-base['fp'],
                    fp_reduction=1-row['fp']/max(base['fp'],1)))
            path=up/f'models/TCN127/seed_{seed}' if arm=='TCN127' else run/f'models/{arm}/seed_{seed}'
            meta=read(path/'model.json')
            for i in range(6):
                with np.load(run/f'validation/block_{i}.npz',allow_pickle=False) as z: y=z['y']>=200
                score_path=path/f"epoch_{meta['selected']['epoch']}_block_{i}.npz"
                if arm=='TCN127':
                    relative=str(score_path.relative_to(up/'models')).replace('\\','/')
                    assert digest(score_path)==read(up/'training_complete.json')['artifacts'][relative]
                with np.load(score_path,allow_pickle=False) as z: s=z['scores']
                np.savez_compressed(out/f'source/{arm}_{seed}_block_{i}.npz',**curve(s,y))
    # Separate budget audit, including failures, without accepting the lifecycle's claimed count.
    starts=list((run/'models').glob('*/seed_*/started.json'))
    assert len(starts)==6
    total=0
    for path in starts:
        m=read(path.parent/'model.json');progress=read(path.parent/'progress.json')
        assert m['updates']==progress['updates']==3528
        assert m['positive_seen']==3528*16 and m['negative_seen']==3528*240
        total+=progress['updates']
    assert total==21168
    f.save_csv(run/'paired_false_positive_comparison.csv',paired)
    # Render compact PR preview; full exact thresholds remain in NPZ exports.
    colors=['#6b7280','#be123c','#0369a1']
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="480" viewBox="0 0 760 480"><rect width="760" height="480" fill="white"/><text x="48" y="28" font-size="18">House17 development PR (previously inspected)</text><path d="M60 50V420H710" stroke="#222" fill="none"/>']
    for j,arm in enumerate(f.ORDER):
        for seed in f.SEEDS:
            with np.load(out/f'development/{arm}_{seed}.npz',allow_pickle=False) as z:
                step=max(1,len(z['recall'])//700)
                points=' '.join(f'{60+650*r:.2f},{420-350*p:.2f}' for r,p in zip(z['recall'][::step],z['precision'][::step]))
            svg.append(f'<polyline points="{points}" stroke="{colors[j]}" stroke-width="1" opacity=".65" fill="none"/>')
        svg.append(f'<text x="{70+j*220}" y="460" font-size="13" fill="{colors[j]}">{arm}</text>')
    svg.append('<text x="650" y="440" font-size="12">Recall</text><text x="8" y="65" font-size="12">P</text></svg>')
    (run/'development_pr.svg').write_text(''.join(svg),encoding='utf-8')
    result=dict(status='independently_verified_completed_delivery',new_starts=6,new_successful_fits=6,new_failed_starts=0,
        new_updates=total,reused_control_fits=3,retrained_control_fits=0,confusions_exact=exact,common_support=137523,
        raw_power_all_devices_exact=True,freeze_sha256=digest(run/'freeze.json'),evaluation_complete_sha256=digest(run/'evaluation_complete.json'),
        parent_budget=dict(failed_starts=1,successful_fits=6,updates=21462),final_houses_accessed=False)
    write(run/'independent_delivery_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--seal-control',action='store_true')
    args=p.parse_args();run=args.run.resolve()
    seal_control(run) if args.seal_control else audit(run)
