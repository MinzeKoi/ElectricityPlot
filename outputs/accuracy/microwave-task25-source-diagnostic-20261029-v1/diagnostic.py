import csv, hashlib, json
from pathlib import Path
import numpy as np
root=Path.cwd(); base=root/'outputs/accuracy'; out=base/'microwave-task25-source-diagnostic-20261029-v1'
dirs={'task24':base/'microwave-task24-recovery-preparation-20261027-v1','task25':base/'microwave-task25-preparation-20261028-v2'}
control=base/'microwave-task22-recovery-preparation-20261027-v3/diagnostics'
read=lambda f: json.loads(f.read_text(encoding='utf8'))
selections={k:read(v/'source_selection.json') for k,v in dirs.items()}
for name,d in dirs.items():
 manifest=read(d/'source_complete.json')
 for filename,expected in manifest['artifacts'].items():
  assert hashlib.sha256((d/'source_results'/filename).read_bytes()).hexdigest()==expected
 assert hashlib.sha256((d/'source_selection.json').read_bytes()).hexdigest()==manifest['source_sha256']
arrays={}; reference={}; metrics=[]; pairs=[]
for seed in (42,1019,2026):
 for block in (4,5):
  for name,d in dirs.items():
   with np.load(d/f'source_results/scores_{seed}_{block}.npz') as z:
    a={k:z[k].copy() for k in ('t','p','h','ix','y')}
   assert np.isfinite(a['p'][a['ix']]).all() and np.isfinite(a['h'][a['ix']]).all()
   if block not in reference: reference[block]={k:a[k].copy() for k in ('t','ix','y')}
   for k in ('t','ix','y'): assert np.array_equal(a[k],reference[block][k])
   recipe=next(r for r in selections[name]['recipes'] if r['seed']==seed)
   pred=(a['p'][a['ix']]>=recipe['L']) & (a['h'][a['ix']]>=recipe['H'])
   truth=a['y']; tp=int((pred & truth).sum()); fp=int((pred & ~truth).sum()); fn=int((~pred & truth).sum()); tn=int((~pred & ~truth).sum())
   claimed=next(r for r in recipe['blocks'] if int(r['block'])==block)
   assert (tp,fp,fn,tn)==tuple(claimed[k] for k in ('tp','fp','fn','tn'))
   arrays[(name,seed,block)]={'p':a['p'][a['ix']], 'fp':pred & ~truth}
  x=arrays[('task24',seed,block)]['fp']; y=arrays[('task25',seed,block)]['fp']
  pairs.append(dict(seed=seed,block=block,n=len(x),positives=int(reference[block]['y'].sum()),task24_fp=int(x.sum()),task25_fp=int(y.sum()),shared_fp=int((x & y).sum()),task24_only=int((x & ~y).sum()),task25_only=int((y & ~x).sum())))
for seed in (42,1019,2026):
 with np.load(control/f'control16_{seed}_source.npz') as z:
  t=z['t']; scores=z['probability']; house=z['house']
 assert np.array_equal(t,np.concatenate([reference[b]['t'][reference[b]['ix']] for b in (4,5)]))
 assert np.all(house==2) and np.isfinite(scores).all()
 offset=0
 for block in (4,5):
  truth=reference[block]['y']; n=len(truth); arrays[('control',seed,block)]={'p':scores[offset:offset+n]}; offset+=n
  for name in ('control','task24','task25'):
   p=arrays[(name,seed,block)]['p']; order=np.argsort(-p,kind='stable'); s=p[order]; truth_sorted=truth[order]; ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),n-1]; tp=np.cumsum(truth_sorted)[ends]; increments=np.diff(np.r_[0,tp]); ap=float(np.sum(increments*tp/(ends+1))/truth.sum())
   metrics.append(dict(model=name,seed=seed,block=block,n=n,positives=int(truth.sum()),ap=ap))
 assert offset==len(scores)
for filename,rows in [('point_ap.csv',metrics),('gate_fp_overlap.csv',pairs)]:
 with (out/filename).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('AP',metrics); print('overlap',pairs)

