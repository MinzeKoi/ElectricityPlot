import csv, json, hashlib
from pathlib import Path
import numpy as np
base=Path.cwd()/'outputs/accuracy'; target=base/'microwave-task25-source-window-diagnostic-20261029-v1'
dirs={24:base/'microwave-task24-recovery-preparation-20261027-v1',25:base/'microwave-task25-preparation-20261028-v2'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); read=lambda p:json.loads(p.read_text(encoding='utf8'))
assert sha(target/'分析定义.md')=='1d3858bf7513191a9bf187fb8dadabe96bfe8808a80f5a9d022531ca6b623d1b'
reg=read(dirs[24]/'registration.json'); selection={k:read(d/'source_selection.json') for k,d in dirs.items()}
for k,d in dirs.items():
 m=read(d/'source_complete.json'); assert sha(d/'source_selection.json')==m['source_sha256']
 for seed in (42,1019,2026):
  for block in (4,5):
   name=f'scores_{seed}_{block}.npz'; assert sha(d/'source_results'/name)==m['artifacts'][name]
with (target/'groups.csv').open(newline='') as f: groups={(int(r['seed']),int(r['block']),r['group']):r for r in csv.DictReader(f)}
with (target/'quantiles.csv').open(newline='') as f: qs={(int(r['seed']),int(r['block']),r['group'],r['metric']):r for r in csv.DictReader(f)}
assert len(groups)==30 and len(qs)==180
reference={}
for block in (4,5):
 path=Path(reg['training_sources']['validation'][f'block_{block}']); assert sha(path)==reg['inputs'][str(path)]
 with np.load(path) as z: x,mask,y,t,house,blocks=(z[k] for k in ('x','mask','y','t','house','block'))
 assert np.array_equal(mask,np.isfinite(x)) and np.all(house==2) and np.all(blocks==block) and x.shape==(len(y),127)
 for seed in (42,1019,2026):
  pred={}
  for k,d in dirs.items():
   with np.load(d/f'source_results/scores_{seed}_{block}.npz') as z: st,ix,truth,p,h=(z[n] for n in ('t','ix','y','p','h'))
   if block not in reference: reference[block]=(st,ix,truth)
   assert all(np.array_equal(a,b) for a,b in zip((st,ix,truth),reference[block])) and np.array_equal(t,st[ix]) and np.array_equal(y>=200,truth)
   recipe=next(r for r in selection[k]['recipes'] if r['seed']==seed); pred[k]=(p[ix]>=recipe['L'])&(h[ix]>=recipe['H'])
   c=next(r for r in recipe['blocks'] if int(r['block'])==block)
   assert tuple(int(v.sum()) for v in (pred[k]&truth,pred[k]&~truth,~pred[k]&truth,~pred[k]&~truth))==tuple(c[n] for n in ('tp','fp','fn','tn'))
  on=y>=200; a,b=pred[24],pred[25]
  cats={'shared_fp':~on&a&b,'only24_fp':~on&a&~b,'only25_fp':~on&~a&b,'shared_tn':~on&~a&~b,'shared_tp':on&a&b}
  for group,chosen in cats.items():
   yy=y[chosen]; xx=x[chosen]; mm=mask[chosen]; n=len(yy)
   expect={'n':n,'block_support':len(y),'block_positive':int(on.sum()),'core31_all':int(mm[:,48:79].all(axis=1).sum()),'full127_all':int(mm.all(axis=1).sum()),'label_zero':int((yy==0).sum()),'label_between':int(((yy>0)&(yy<200)).sum()),'label_ge200':int((yy>=200).sum()),'label_unknown':int((~np.isfinite(yy)|(yy<0)).sum())}
   assert all(int(groups[(seed,block,group)][k])==v for k,v in expect.items())
   measures={'center':[],'outer_observed':[],'core31_range':[],'core31_max_adjacent_change':[],'full127_range':[],'full127_max_adjacent_change':[]}
   for row,m in zip(xx,mm):
    measures['center'].append(float(row[63]) if m[63] else np.nan); measures['outer_observed'].append(int(m[:48].sum()+m[79:].sum()))
    for name,sl in (('core31',slice(48,79)),('full127',slice(None))):
     v=row[sl]; ok=m[sl]; finite=v[ok].astype(float); diffs=np.abs(np.diff(v.astype(float)))[ok[:-1]&ok[1:]]
     measures[name+'_range'].append(float(np.max(finite)-np.min(finite)) if len(finite) else np.nan)
     measures[name+'_max_adjacent_change'].append(float(np.max(diffs)) if len(diffs) else np.nan)
   for metric,arr in measures.items():
    arr=np.asarray(arr); valid=arr[np.isfinite(arr)]; r=qs[(seed,block,group,metric)]
    assert (int(r['n_total']),int(r['n_valid']),int(r['n_missing']))==(n,len(valid),n-len(valid))
    if len(valid): assert np.allclose(np.percentile(valid,[10,50,90]),[float(r[k]) for k in ('p10','p50','p90')],rtol=1e-12,atol=1e-9)
    else: assert all(np.isnan(float(r[k])) for k in ('p10','p50','p90'))
print(json.dumps({'passed':True,'group_rows':len(groups),'quantile_rows':len(qs),'definition_sha256':sha(target/'分析定义.md'),'groups_sha256':sha(target/'groups.csv'),'quantiles_sha256':sha(target/'quantiles.csv'),'interpretation':'source validation points and overlapping windows, not independent events'},indent=2))
