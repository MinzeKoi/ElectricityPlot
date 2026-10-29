import csv, json, hashlib
from pathlib import Path
import numpy as np
base=Path.cwd()/'outputs/accuracy'; target=base/'microwave-task25-source-diagnostic-20261029-v1'
dirs={24:base/'microwave-task24-recovery-preparation-20261027-v1',25:base/'microwave-task25-preparation-20261028-v2'}
load=lambda p:json.loads(p.read_text(encoding='utf8'))
sel={k:load(v/'source_selection.json') for k,v in dirs.items()}
with (target/'point_ap.csv').open(newline='',encoding='utf8') as f: aps={(r['model'],int(r['seed']),int(r['block'])):r for r in csv.DictReader(f)}
with (target/'gate_fp_overlap.csv').open(newline='',encoding='utf8') as f: overlaps={(int(r['seed']),int(r['block'])):r for r in csv.DictReader(f)}
for k,d in dirs.items():
    m=load(d/'source_complete.json')
    assert hashlib.sha256((d/'source_selection.json').read_bytes()).hexdigest()==m['source_sha256']
    for name,h in m['artifacts'].items(): assert hashlib.sha256((d/'source_results'/name).read_bytes()).hexdigest()==h
rows=[]
for seed in (42,1019,2026):
    control=np.load(base/f'microwave-task22-recovery-preparation-20261027-v3/diagnostics/control16_{seed}_source.npz')
    offset=0
    for block in (4,5):
        values={}
        for k,d in dirs.items():
            with np.load(d/f'source_results/scores_{seed}_{block}.npz') as z: values[k]={n:z[n] for n in ('t','p','h','ix','y')}
        a,b=values[24],values[25]
        assert all(np.array_equal(a[x],b[x]) for x in ('t','ix','y'))
        truth=a['y']; n=len(truth); t=a['t'][a['ix']]
        assert np.array_equal(t,control['t'][offset:offset+n]) and np.all(control['house'][offset:offset+n]==2)
        pvals={'control':control['probability'][offset:offset+n],'task24':a['p'][a['ix']],'task25':b['p'][b['ix']]}; offset+=n
        for name,p in pvals.items():
            assert np.isfinite(p).all()
            _,inv=np.unique(p,return_inverse=True)
            positives=np.bincount(inv,weights=truth,minlength=inv.max()+1)[::-1]
            counts=np.bincount(inv,minlength=inv.max()+1)[::-1]
            cumpos=np.cumsum(positives); cumcount=np.cumsum(counts)
            ap=float(np.sum((positives/truth.sum())*(cumpos/cumcount)))
            r=aps[(name,seed,block)]
            assert (int(r['n']),int(r['positives']))==(n,int(truth.sum())) and abs(ap-float(r['ap']))<1e-12
        fps={}
        for k in (24,25):
            z=values[k]; recipe=next(x for x in sel[k]['recipes'] if x['seed']==seed)
            pred=(z['p'][z['ix']]>=recipe['L']) & (z['h'][z['ix']]>=recipe['H'])
            tp=int((pred&truth).sum()); fp=int((pred&~truth).sum()); fn=int((~pred&truth).sum()); tn=int((~pred&~truth).sum())
            expected=next(x for x in recipe['blocks'] if int(x['block'])==block)
            assert (tp,fp,fn,tn)==tuple(expected[x] for x in ('tp','fp','fn','tn'))
            fps[k]=pred&~truth
        x,y=fps[24],fps[25]; actual={'n':n,'positives':int(truth.sum()),'task24_fp':int(x.sum()),'task25_fp':int(y.sum()),'shared_fp':int((x&y).sum()),'task24_only':int((x&~y).sum()),'task25_only':int((y&~x).sum())}
        assert all(int(overlaps[(seed,block)][k])==v for k,v in actual.items())
        rows.append({'seed':seed,'block':block,**actual})
    assert offset==len(control['t']); control.close()
print(json.dumps({'passed':True,'ap_rows':len(aps),'overlap_rows':rows,'tie_method':'group exact equal scores then sum precision times positive recall increment','control_gate_overlap':'unavailable: h/full timeline not in saved control source scores'},indent=2,sort_keys=True))
