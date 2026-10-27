import csv, json, hashlib, math, sys, datetime, os, socket
from pathlib import Path
import numpy as np
ROOT=Path.cwd().resolve(); RUN=ROOT/'outputs/accuracy/microwave-task23-guarded-audit-20261027-v3'
H={'10':'Appliance8','12':'Appliance5','15':'Appliance7','18':'Appliance9'}
P=ROOT/'data/processed/refit_expansion_additional_20261024_v1'; S=ROOT/'data/raw/refit_expansion_additional_20261024_v1_slices'; R=ROOT/'data/raw/refit_complete_20261022/households'; OLD=ROOT/'outputs/accuracy/data-expansion-20261024-v1'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def info(p):
 s=p.stat(); return {'path':str(p.relative_to(ROOT)).replace('\\','/'),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)}
def iso(t): return datetime.datetime.fromtimestamp(int(t),datetime.timezone.utc).isoformat().replace('+00:00','Z')
def ts(t): return int(datetime.datetime.fromisoformat(t.replace('Z','+00:00')).timestamp())
def safe(p,root):
 p=(root/p).resolve(); assert p.is_relative_to(root.resolve()); return p
def dump(p,o): p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvout(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def seal():
 assert not (RUN/'input_seal.json').exists()
 manifest=read(P/'manifest.json'); selection=read(S/'selection.json'); catalog=read(ROOT/'configs/refit_channels_complete_20261022.json'); adm=read(OLD/'admission.json')['C']; proto=read(OLD/'protocols/C/protocol.json'); reg=read(OLD/'registration.json')
 inputs=[P/'manifest.json',S/'selection.json',ROOT/'configs/refit_channels_complete_20261022.json',ROOT/'configs/r0_roles_20261022_v2.json',OLD/'admission.json',OLD/'protocols/C/protocol.json',OLD/'registration.json',OLD/'plan.json',ROOT/'src/virtual_submeter/accuracy/data_expansion.py',ROOT/'src/virtual_submeter/data/prepare.py',ROOT/'src/virtual_submeter/accuracy/microwave_mask.py',ROOT/'src/virtual_submeter/models/center_mask_tcn.py',RUN/'audit.py',RUN/'guard.py',RUN/'independent_check.py',RUN/'guard_tests.py',RUN/'report_builder.py',RUN/'verify.py']
 out={}
 for h,ch in H.items():
  m=manifest['households'][h]; sel=selection['households'][h]; receipt=read(R/f'CLEAN_House{h}.receipt.json'); sl=read(S/f'CLEAN_House{h}.slice.json')
  assert catalog['households'][h]['microwave']==[ch] and m['channel_mapping']['microwave']==[ch]
  assert receipt['sha256']==m['source_sha256']==sl['original_sha256']; assert sl['slice_sha256']==sel['source_sha256']==m['exact_slice_sha256']
  parts=[x for x in proto['partitions'] if x['household_id']=='refit_'+h and x['role']=='source_train']; checks=[x for x in adm if x['household_id']=='refit_'+h and x['role']=='source_train']; assert len(parts)==len(checks)==2
  assert parts[0]['start']==sel['start'] and parts[1]['end']==sel['end'] and ts(parts[1]['start'])-ts(parts[0]['end'])==3600
  pp=[]
  for part in parts:
   check=next(x for x in checks if x['partition_id']==part['partition_id']); assert check['start']==part['start'] and check['end']==part['end']
   pp.append({'id':part['partition_id'],'start':part['start'],'end':part['end'],'old_microwave_admission':check['devices']['microwave']})
  paths={'raw':R/f'CLEAN_House{h}.csv','receipt':R/f'CLEAN_House{h}.receipt.json','slice':S/f'CLEAN_House{h}.csv','slice_receipt':S/f'CLEAN_House{h}.slice.json'}
  for typ in ['aggregate','microwave']:
   art=m['artifacts'][typ]; p=safe(art['path'],P); assert (typ=='aggregate' and p==P/f'aggregate/house_{h}.csv') or (typ=='microwave' and p==P/f'labels/house_{h}/microwave.csv'); paths[typ]=p; assert sha(p)==art['sha256']
  for k,p in paths.items(): assert p.is_file(); inputs.append(p)
  assert sha(paths['raw'])==receipt['sha256']; assert sha(paths['slice'])==sl['slice_sha256']
  out[h]={'channel':ch,'selection':sel,'partitions':pp,'paths':{k:str(v.relative_to(ROOT)).replace('\\','/') for k,v in paths.items()},'provenance':receipt}
 assert sha(OLD/'plan.json')==reg['inputs']['plan.json']
 dump(RUN/'input_seal.json',{'scope':'four houses, source_train only; no fit, prediction, download, normalization or dataset merge','houses':out,'inputs':[info(p) for p in inputs]})
def loadnorm(path,col,h,ch=None):
 t=[];v=[]
 with path.open(encoding='utf-8',newline='') as f:
  for r in csv.DictReader(f):
   assert r['household_id']==h
   if ch: assert r['appliance_id']=='microwave:'+ch and r['appliance_type']=='microwave'
   t.append(ts(r['timestamp'])); v.append(float(r[col]) if r[col] else np.nan)
 return np.array(t,dtype=np.int64),np.array(v)
def rawgrid(path,ch,start,end):
 n=(end-start)//8; cov=np.zeros((2,n),np.int16); sums=np.zeros((2,n),np.float64); prev=None; prevt=None
 with path.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   t=int(row['Unix']); assert prevt is None or t>prevt
   if prev is not None:
    endhold=min(t,prevt+8,end); begin=max(prevt,start)
    while begin<endhold:
     ix=(begin-start)//8; stop=min(endhold,start+(ix+1)*8); d=stop-begin
     for j,val in enumerate(prev):
      if val is not None: cov[j,ix]+=d; sums[j,ix]+=d*val
     begin=stop
   def val(x):
    try: z=float(x); return z if math.isfinite(z) and z>=0 else None
    except (ValueError,TypeError): return None
   assert row['Issues'] in ('0','1'); prev=[val(row['Aggregate']),val(row[ch]) if row['Issues']=='0' else None]; prevt=t
 result=np.where(cov==8,sums/8,np.nan); return result,cov

def main():
 assert not any((RUN/name).exists() for name in ['window_ids.csv','partitions.csv','daily.csv','fragments.csv','numerical_checks.json']); seal=read(RUN/'input_seal.json'); daily=[]; partsout=[]; fragments=[]; seen=set(); windows=RUN/'window_ids.csv'
 with windows.open('w',encoding='utf-8',newline='') as wf:
  w=csv.writer(wf); w.writerow(['house','source_role','center_utc','grid_index','fragment_id','on','partial_core31','strict_core31','full127','left_stop','right_stop'])
  for h,ch in H.items():
   item=seal['houses'][h]; paths={k:ROOT/v for k,v in item['paths'].items()}; start=ts(item['selection']['start']); end=ts(item['selection']['end']); t,a=loadnorm(paths['aggregate'],'aggregate_power_w',h); ty,y=loadnorm(paths['microwave'],'power_w',h,ch); assert np.array_equal(t,ty) and np.array_equal(t,np.arange(start,end,8))
   raw,cov=rawgrid(paths['slice'],ch,start,end)
   for j,arr in enumerate((a,y)):
    assert np.array_equal(np.isfinite(raw[j]),np.isfinite(arr)); assert np.allclose(raw[j],arr,equal_nan=True,rtol=1e-8,atol=1e-7)
   for part in item['partitions']:
    lo=ts(part['start']); hi=ts(part['end']); sl=(t>=lo)&(t<hi); tt=t[sl]; aa=a[sl]; yy=y[sl]; n=len(tt); assert n==(hi-lo)//8 and np.all(np.diff(tt)==8)
    valid=np.isfinite(yy); obs=np.isfinite(aa); on=valid&(yy>=200); ids=np.full(n,-1,dtype=int); change=np.diff(np.r_[False,on,False].astype(int)); starts=np.flatnonzero(change==1); ends=np.flatnonzero(change==-1); complete=[]
    for k,(b,e) in enumerate(zip(starts,ends)):
     ids[b:e]=k; c=bool(b>0 and e<n and valid[b-1] and valid[e] and yy[b-1]<200 and yy[e]<200); complete.append(c)
    core=np.lib.stride_tricks.sliding_window_view(obs,31); c=np.arange(15,n-15); coregood=(core.sum(axis=1)>=24)&~((~core[:,:-1])&(~core[:,1:])).any(axis=1); legal=coregood&obs[c]&valid[c]; partial=np.zeros(n,bool);partial[c]=legal; strict=np.zeros(n,bool); strict[c]=legal&core.all(axis=1)
    full=np.zeros(n,bool); full[63:n-63]=np.lib.stride_tricks.sliding_window_view(obs,127).all(axis=1)&partial[63:n-63]
    reasons=np.full(n,'core31_boundary',dtype='U32'); reasons[c]='eligible'; reasons[c[~obs[c]]]='center_aggregate_missing'; reasons[c[obs[c]&~valid[c]]]='center_label_missing'; reasons[c[obs[c]&valid[c]&~coregood]]='core31_missing_rule'; reasons[partial]='partial'
    # Vectorized directional stopping, identical to the registered stop rule, no power windows stored.
    centers=np.flatnonzero(partial); stop=[]
    for direction in (-1,1):
     alive=np.ones(len(centers),bool); missing=np.zeros(len(centers),np.int8); reason=np.full(len(centers),'radius',dtype='U16')
     for dist in range(1,64):
      ix=centers+direction*dist; boundary=(ix<0)|(ix>=n); safeix=np.clip(ix,0,n-1); reason[alive&boundary]='boundary'; alive &= ~boundary; missing=np.where(obs[safeix],0,missing+1); broken=alive&(missing>=2);reason[broken]='missing_run';alive &= ~broken
     stop.append(reason)
    for z,i in enumerate(centers):
     key=(h,part['id'],int(tt[i])); assert key not in seen;seen.add(key)
     w.writerow([h,part['id'],iso(tt[i]),int((tt[i]-start)//8),f'{part["id"]}:{ids[i]}' if ids[i]>=0 else '',int(on[i]),1,int(strict[i]),int(full[i]),stop[0][z],stop[1][z]])
    def counts(mask):
     return {'grid':int(mask.sum()),'aggregate_valid':int((mask&obs).sum()),'label_valid':int((mask&valid).sum()),'on_centers':int((mask&on).sum()),'partial_on':int((mask&partial&on).sum()),'partial_off':int((mask&partial&valid&~on).sum()),'strict_on':int((mask&strict&on).sum()),'strict_off':int((mask&strict&valid&~on).sum()),'full_on':int((mask&full&on).sum()),'full_off':int((mask&full&valid&~on).sum()),'edge_excluded':int((mask&(reasons=='core31_boundary')).sum()),'center_aggregate_missing':int((mask&(reasons=='center_aggregate_missing')).sum()),'center_label_missing':int((mask&(reasons=='center_label_missing')).sum()),'core_missing_excluded':int((mask&(reasons=='core31_missing_rule')).sum())}
    q=counts(np.ones(n,bool)); old=part['old_microwave_admission']; assert q['label_valid']==old['valid_samples'] and q['on_centers']==old['active_samples'] and sum(complete)==old['complete_threshold_events']; q.update(house=h,partition=part['id'],start=part['start'],end=part['end'],old_admitted=old['included'],fragments=len(starts),complete_fragments=sum(complete));partsout.append(q)
    for k,(b,e) in enumerate(zip(starts,ends)):
     fragments.append({'house':h,'partition':part['id'],'fragment_id':f'{part["id"]}:{k}','start_utc':iso(tt[b]),'end_utc_exclusive':iso(tt[e-1]+8),'centers':int(e-b),'complete':int(complete[k]),'partial_centers':int(partial[b:e].sum()),'strict_centers':int(strict[b:e].sum()),'full127_centers':int(full[b:e].sum())})
    for day in np.unique(tt//86400):
     mask=tt//86400==day; q=counts(mask); q.update(house=h,partition=part['id'],utc_date=iso(day*86400)[:10],fragments_started=int(sum(tt[b]//86400==day for b in starts)),complete_fragments_started=int(sum(tt[b]//86400==day and complete[k] for k,b in enumerate(starts))));daily.append(q)
   print('verified',h,flush=True)
 csvout(RUN/'partitions.csv',partsout);csvout(RUN/'daily.csv',daily);csvout(RUN/'fragments.csv',fragments)
 dump(RUN/'numerical_checks.json',{'raw_slice_vs_normalized':'all four houses, all selected 8-second bins and both channels exactly match within 1e-7 W','timestamp_alignment':'same house, exact UTC grid','duplicate_window_ids':0,'old_admission_counts':'valid, active and complete fragment counts matched in all eight partitions','grid_index_definition':'zero-based row in selected 28-day normalized grid; not an original CSV row. A time-weighted bin may have several raw observations; no one-to-one raw row is claimed.','fit':0,'optimizer':0,'backpropagation':0,'inference':0,'downloads':0})
if __name__=='__main__':
 if sys.argv[1]=='seal':seal()
 elif sys.argv[1]=='compute':main()
