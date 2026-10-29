import csv, json, hashlib
from pathlib import Path
import numpy as np
base=Path.cwd()/'outputs/accuracy'; target=base/'microwave-fixed-gate-diagnostic-20261029-v1'
dirs={'task24':base/'microwave-task24-recovery-preparation-20261027-v1','task25':base/'microwave-task25-preparation-20261028-v2'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding='utf8'))
assert sha(target/'分析定义.md')=='650a25fdd9acae17bb0b57ae4b6922e093bc4800880c98bf351867c89bb0ae54'
sel={k:load(v/'source_selection.json') for k,v in dirs.items()}
for k,d in dirs.items():
 m=load(d/'source_complete.json'); assert sha(d/'source_selection.json')==m['source_sha256']
 for name,h in m['artifacts'].items(): assert sha(d/'source_results'/name)==h
with (target/'quadrants.csv').open(newline='') as f: quad={(r['model'],int(r['seed']),int(r['block']),int(r['truth']),r['state']):int(r['count']) for r in csv.DictReader(f)}
with (target/'gate_summary.csv').open(newline='') as f: summary={(r['model'],int(r['seed']),int(r['block'])):r for r in csv.DictReader(f)}
with (target/'fp_transitions.csv').open(newline='') as f: transitions={(int(r['seed']),int(r['block']),r['old_state'],r['new_state']):int(r['negative_points']) for r in csv.DictReader(f)}
assert (len(quad),len(summary),len(transitions))==(96,12,96)
reference={}; totals={'filtered_tp':0,'filtered_fp':0,'added_fp':0,'removed_fp':0}
for seed in (42,1019,2026):
 for block in (4,5):
  masks={}
  for name,d in dirs.items():
   with np.load(d/f'source_results/scores_{seed}_{block}.npz') as z: t,ix,y,p,h=(z[k] for k in ('t','ix','y','p','h'))
   if block not in reference: reference[block]=(t,ix,y)
   assert all(np.array_equal(a,b) for a,b in zip((t,ix,y),reference[block]))
   recipe=next(x for x in sel[name]['recipes'] if x['seed']==seed); a=p[ix]>=recipe['L']; b=h[ix]>=recipe['H']
   assert np.isfinite(p[ix]).all() and np.isfinite(h[ix]).all()
   state=2*a.astype(int)+b.astype(int); masks[name]=state
   for truth in (0,1):
    for code in range(4): assert quad[(name,seed,block,truth,f'{code:02b}')]==int(((y==bool(truth))&(state==code)).sum())
   c=next(x for x in recipe['blocks'] if int(x['block'])==block)
   tp=int(((state==3)&y).sum()); fp=int(((state==3)&~y).sum()); fn=int(((state!=3)&y).sum()); tn=int(((state!=3)&~y).sum())
   assert (tp,fp,fn,tn)==tuple(c[k] for k in ('tp','fp','fn','tn'))
   s=summary[(name,seed,block)]; expected={'n':len(y),'positives':int(y.sum()),'l_only_tp':int((a&y).sum()),'h_filtered_tp':int((a&~b&y).sum()),'retained_tp':tp,'l_only_fp':int((a&~y).sum()),'h_filtered_fp':int((a&~b&~y).sum()),'retained_fp':fp,'fn':fn,'tn':tn}
   assert all(int(s[k])==v for k,v in expected.items()) and float(s['L'])==recipe['L'] and float(s['H'])==recipe['H']
   totals['filtered_tp']+=expected['h_filtered_tp']; totals['filtered_fp']+=expected['h_filtered_fp']
  old,new=masks['task24'],masks['task25']
  for i in range(4):
   for j in range(4): assert transitions[(seed,block,f'{i:02b}',f'{j:02b}')]==int(((old==i)&(new==j)&~y).sum())
  totals['added_fp']+=int(((old!=3)&(new==3)&~y).sum()); totals['removed_fp']+=int(((old==3)&(new!=3)&~y).sum())
print(json.dumps({'passed':True,'quadrant_rows':len(quad),'summary_rows':len(summary),'transition_rows':len(transitions),'totals_over_six_blocks_both_models':totals,'definition_sha256':sha(target/'分析定义.md'),'prior_source_report_sha256':sha(base/'microwave-task25-source-diagnostic-20261029-v1/报告.md'),'diagnostic_csv_sha256':{n:sha(target/n) for n in ('quadrants.csv','gate_summary.csv','fp_transitions.csv')}},indent=2,sort_keys=True))
