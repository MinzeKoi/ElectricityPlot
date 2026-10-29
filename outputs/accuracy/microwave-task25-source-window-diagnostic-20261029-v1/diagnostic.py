import csv, hashlib, json
from pathlib import Path
import numpy as np
root=Path.cwd(); base=root/'outputs/accuracy'; out=base/'microwave-task25-source-window-diagnostic-20261029-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
assert sha(out/'分析定义.md')=='1d3858bf7513191a9bf187fb8dadabe96bfe8808a80f5a9d022531ca6b623d1b'
dirs={'task24':base/'microwave-task24-recovery-preparation-20261027-v1','task25':base/'microwave-task25-preparation-20261028-v2'}
regpath=dirs['task24']/'registration.json'; reg=read(regpath); identity={'definition':sha(out/'分析定义.md'),'registration':sha(regpath),'inputs':{}}
selections={}
for name,d in dirs.items():
 manifest=read(d/'source_complete.json'); selection=d/'source_selection.json'; selections[name]=read(selection); assert sha(selection)==manifest['source_sha256']
 identity['inputs'][name]={'source_complete':sha(d/'source_complete.json'),'source_selection':sha(selection),'scores':{}}
 for seed in (42,1019,2026):
  for block in (4,5):
   file=f'scores_{seed}_{block}.npz'; value=sha(d/'source_results'/file); assert value==manifest['artifacts'][file]; identity['inputs'][name]['scores'][file]=value
rows=[]; groups=[]
for block in (4,5):
 path=Path(reg['training_sources']['validation'][f'block_{block}']); digest=sha(path); assert digest==reg['inputs'][str(path)]; identity['inputs'][f'block_{block}']={'path':str(path),'sha256':digest}
 with np.load(path) as z: x,mask,y,t,house,blocks=(z[k].copy() for k in ('x','mask','y','t','house','block'))
 assert np.array_equal(mask,np.isfinite(x)) and np.all(house==2) and np.all(blocks==block) and np.isfinite(y).all() and np.all(y>=0)
 assert x.shape==(len(y),127) and np.all(x[mask]>=0)
 core=mask[:,48:79]; outer=mask.sum(axis=1)-core.sum(axis=1)
 vals={'center':x[:,63], 'outer_observed':outer.astype(float)}
 for label,sl in (('core31',slice(48,79)),('full127',slice(None))):
  v=x[:,sl].astype(float); m=mask[:,sl]; hi=np.where(m,v,-np.inf).max(axis=1); lo=np.where(m,v,np.inf).min(axis=1); vals[label+'_range']=np.where(m.any(axis=1),hi-lo,np.nan)
  valid=m[:,:-1]&m[:,1:]; delta=np.where(valid,np.abs(v[:,1:]-v[:,:-1]),-np.inf).max(axis=1); vals[label+'_max_adjacent_change']=np.where(valid.any(axis=1),delta,np.nan)
 reference=None
 for seed in (42,1019,2026):
  gate={}
  for name,d in dirs.items():
   with np.load(d/f'source_results/scores_{seed}_{block}.npz') as z: st,p,h,ix,truth=(z[k].copy() for k in ('t','p','h','ix','y'))
   if reference is None: reference=(st,ix,truth)
   assert all(np.array_equal(a,b) for a,b in zip((st,ix,truth),reference))
   assert np.array_equal(t,st[ix]) and np.array_equal(truth,y>=200)
   recipe=next(r for r in selections[name]['recipes'] if r['seed']==seed); gate[name]=(p[ix]>=recipe['L'])&(h[ix]>=recipe['H']); pred=gate[name]
   result=(int((pred&truth).sum()),int((pred&~truth).sum()),int((~pred&truth).sum()),int((~pred&~truth).sum()))
   claimed=next(v for v in recipe['blocks'] if int(v['block'])==block); assert result==tuple(claimed[k] for k in ('tp','fp','fn','tn'))
  a,b=gate['task24'],gate['task25']; positive=y>=200
  categories={'shared_fp':~positive&a&b,'only24_fp':~positive&a&~b,'only25_fp':~positive&~a&b,'shared_tn':~positive&~a&~b,'shared_tp':positive&a&b}
  for group,sel in categories.items():
   count=int(sel.sum()); yy=y[sel]
   groups.append(dict(seed=seed,block=block,group=group,n=count,block_support=len(y),block_positive=int(positive.sum()),core31_all=int(core[sel].all(axis=1).sum()),full127_all=int(mask[sel].all(axis=1).sum()),label_zero=int((yy==0).sum()),label_between=int(((yy>0)&(yy<200)).sum()),label_ge200=int((yy>=200).sum()),label_unknown=int((~np.isfinite(yy)|(yy<0)).sum())))
   for name,v in vals.items():
    z=v[sel]; z=z[np.isfinite(z)]; q=np.quantile(z,[.1,.5,.9]) if len(z) else [float('nan')]*3
    rows.append(dict(seed=seed,block=block,group=group,metric=name,n_total=count,n_valid=len(z),n_missing=count-len(z),p10=q[0],p50=q[1],p90=q[2]))
for name,data in [('groups.csv',groups),('quantiles.csv',rows)]:
 with (out/name).open('w',encoding='utf8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=data[0].keys()); writer.writeheader(); writer.writerows(data)
(out/'输入身份.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(groups,ensure_ascii=False))
