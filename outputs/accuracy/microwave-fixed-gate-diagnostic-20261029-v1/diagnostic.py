import csv, hashlib, json
from pathlib import Path
import numpy as np
root=Path.cwd(); base=root/'outputs/accuracy'; out=base/'microwave-fixed-gate-diagnostic-20261029-v1'
dirs={'task24':base/'microwave-task24-recovery-preparation-20261027-v1','task25':base/'microwave-task25-preparation-20261028-v2'}
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
read=lambda f:json.loads(f.read_text(encoding='utf8'))
assert sha(out/'分析定义.md')=='650a25fdd9acae17bb0b57ae4b6922e093bc4800880c98bf351867c89bb0ae54'
identity={'definition_sha256':sha(out/'分析定义.md'),'inputs':{}}
selections={}
for name,d in dirs.items():
 manifest=read(d/'source_complete.json'); selection=d/'source_selection.json'; selections[name]=read(selection)
 assert sha(selection)==manifest['source_sha256']
 identity['inputs'][str(d.relative_to(root))]={'source_complete_sha256':sha(d/'source_complete.json'),'source_selection_sha256':sha(selection),'scores':{}}
 for seed in (42,1019,2026):
  for block in (4,5):
   n=f'scores_{seed}_{block}.npz'; path=d/'source_results'/n; digest=sha(path); assert digest==manifest['artifacts'][n]; identity['inputs'][str(d.relative_to(root))]['scores'][n]=digest
reference={}; quadrant=[]; summary=[]; transitions=[]
for seed in (42,1019,2026):
 for block in (4,5):
  states={}
  for name,d in dirs.items():
   with np.load(d/f'source_results/scores_{seed}_{block}.npz') as z: t,p,h,ix,y=(z[k].copy() for k in ('t','p','h','ix','y'))
   if block not in reference: reference[block]=(t,ix,y)
   assert all(np.array_equal(x,z) for x,z in zip((t,ix,y),reference[block]))
   assert np.isfinite(p[ix]).all() and np.isfinite(h[ix]).all()
   recipe=next(x for x in selections[name]['recipes'] if x['seed']==seed)
   a=p[ix]>=recipe['L']; b=h[ix]>=recipe['H']; state=2*a.astype(int)+b.astype(int); gate=a&b; states[name]=state
   tp=int((gate&y).sum()); fp=int((gate&~y).sum()); fn=int((~gate&y).sum()); tn=int((~gate&~y).sum())
   claimed=next(x for x in recipe['blocks'] if int(x['block'])==block)
   assert (tp,fp,fn,tn)==tuple(claimed[k] for k in ('tp','fp','fn','tn'))
   for truth in (0,1):
    for value in range(4): quadrant.append(dict(model=name,seed=seed,block=block,truth=truth,state=f'{value:02b}',count=int(((y==truth)&(state==value)).sum())))
   summary.append(dict(model=name,seed=seed,block=block,n=len(y),positives=int(y.sum()),L=recipe['L'],H=recipe['H'],l_only_tp=int((a&y).sum()),h_filtered_tp=int((a&~b&y).sum()),retained_tp=tp,l_only_fp=int((a&~y).sum()),h_filtered_fp=int((a&~b&~y).sum()),retained_fp=fp,fn=fn,tn=tn))
  for old in range(4):
   for new in range(4): transitions.append(dict(seed=seed,block=block,old_state=f'{old:02b}',new_state=f'{new:02b}',negative_points=int(((~y)&(states['task24']==old)&(states['task25']==new)).sum())))
for filename,rows in [('quadrants.csv',quadrant),('gate_summary.csv',summary),('fp_transitions.csv',transitions)]:
 with (out/filename).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
(out/'输入身份.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False)); print(json.dumps([x for x in transitions if (x['old_state']=='11') != (x['new_state']=='11')],ensure_ascii=False))
