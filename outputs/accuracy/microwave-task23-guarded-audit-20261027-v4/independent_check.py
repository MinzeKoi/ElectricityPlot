import csv, json, math, datetime, sys
from pathlib import Path
import numpy as np
ROOT=Path.cwd().resolve(); RUN=ROOT/'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4'
def ts(x):return int(datetime.datetime.fromisoformat(x.replace('Z','+00:00')).timestamp())
def norm(p,field,h,channel=None):
 t=[];v=[]
 with p.open(encoding='utf-8',newline='') as f:
  for r in csv.DictReader(f):
   assert r['household_id']==h
   if channel: assert r['appliance_id']=='microwave:'+channel and r['appliance_type']=='microwave'
   t.append(ts(r['timestamp']));v.append(float(r[field]) if r[field] else np.nan)
 return np.array(t,dtype=np.int64),np.array(v)
def recompute(p,ch,lo,hi):
 times=[]; x=[]; y=[]
 def number(s):
  try:
   z=float(s);return z if math.isfinite(z) and z>=0 else np.nan
  except (ValueError,TypeError):return np.nan
 with p.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   assert r['Issues'] in ('0','1');times.append(int(r['Unix']));x.append(number(r['Aggregate']));y.append(number(r[ch]) if r['Issues']=='0' else np.nan)
 t=np.array(times,np.int64);assert np.all(np.diff(t)>0); x=np.array(x);y=np.array(y); n=(hi-lo)//8
 # Each observation contributes to at most two bins because its hold is at most 8 seconds.
 left=np.maximum(t[:-1],lo);right=np.minimum(np.minimum(t[1:],t[:-1]+8),hi); ok=right>left; left=left[ok];right=right[ok];xx=x[:-1][ok];yy=y[:-1][ok]; idx=(left-lo)//8; cut=np.minimum(right,lo+(idx+1)*8)
 out=[]
 for values in (xx,yy):
  cov=np.zeros(n,np.int64); energy=np.zeros(n); finite=np.isfinite(values)
  for indices,duration in ((idx,cut-left),(idx+1,right-cut)):
   m=finite&(duration>0)&(indices>=0)&(indices<n);np.add.at(cov,indices[m],duration[m]);np.add.at(energy,indices[m],duration[m]*values[m])
  out.append(np.where(cov==8,energy/8,np.nan))
 return out

def support(t,a,y,lo,hi):
 n=len(t); assert n==(hi-lo)//8 and np.array_equal(t,np.arange(lo,hi,8));obs=np.isfinite(a);valid=np.isfinite(y); c=np.arange(n)
 # Prefix sums independently calculate rolling count and adjacency; no sliding_window_view.
 bad=(~obs).astype(np.int64); adj=np.r_[False,(~obs[:-1])&(~obs[1:])].astype(np.int64); pb=np.r_[0,np.cumsum(bad)];pa=np.r_[0,np.cumsum(adj)]; interior=(c>=15)&(c<n-15); ix=c[interior];missing=pb[ix+16]-pb[ix-15]; pairs=pa[ix+16]-pa[ix-14]
 partial=np.zeros(n,bool);partial[ix]=obs[ix]&valid[ix]&(missing<=7)&(pairs==0)
 strict=np.zeros(n,bool);strict[ix]=partial[ix]&(missing==0)
 full=np.zeros(n,bool);cc=c[(c>=63)&(c<n-63)];full[cc]=partial[cc]&((pb[cc+64]-pb[cc-63])==0)
 on=valid&(y>=200); ids=np.full(n,-1,np.int32); starts=np.flatnonzero(on & ~np.r_[False,on[:-1]]);ends=np.flatnonzero(on & ~np.r_[on[1:],False])+1
 complete=[]
 for k,(b,e) in enumerate(zip(starts,ends)):
  ids[b:e]=k;complete.append(b>0 and e<n and valid[b-1] and valid[e] and not on[b-1] and not on[e])
 # Directional independent scan, vectorised across centers.
 centers=np.flatnonzero(partial); stops=[]
 for direction in (-1,1):
  active=np.ones(len(centers),bool);lastmissing=np.zeros(len(centers),bool); reason=np.full(len(centers),'radius',dtype='U16')
  for d in range(1,64):
   j=centers+direction*d; boundary=(j<0)|(j>=n);reason[active&boundary]='boundary';active &= ~boundary; obsj=obs[np.clip(j,0,n-1)]; fail=active&lastmissing&~obsj;reason[fail]='missing_run';active &= ~fail;lastmissing=~obsj
  stops.append(reason)
 return partial,strict,full,on,ids,starts,ends,complete,centers,stops

def verify_all():
 seal=json.loads((RUN/'input_seal.json').read_text(encoding='utf-8')); partition_rows={r['partition']:r for r in csv.DictReader((RUN/'partitions.csv').open(encoding='utf-8',newline=''))}; daily=list(csv.DictReader((RUN/'daily.csv').open(encoding='utf-8',newline=''))); fragments=list(csv.DictReader((RUN/'fragments.csv').open(encoding='utf-8',newline=''))); checked=0; rows_checked=0
 with (RUN/'window_ids.csv').open(encoding='utf-8',newline='') as wf:
  windows=iter(csv.DictReader(wf))
  for h,info in seal['houses'].items():
   paths={k:ROOT/v for k,v in info['paths'].items()};lo=ts(info['selection']['start']);hi=ts(info['selection']['end']);t,a=norm(paths['aggregate'],'aggregate_power_w',h);ty,y=norm(paths['microwave'],'power_w',h,info['channel']);assert np.array_equal(t,ty) and np.array_equal(t,np.arange(lo,hi,8)); aa,yy=recompute(paths['slice'],info['channel'],lo,hi)
   for first,second in [(a,aa),(y,yy)]:assert np.array_equal(np.isfinite(first),np.isfinite(second)) and np.allclose(first,second,equal_nan=True,atol=1e-7,rtol=1e-8)
   assert len(info['partitions'])==2 and ts(info['partitions'][0]['start'])==lo and ts(info['partitions'][1]['end'])==hi and ts(info['partitions'][1]['start'])-ts(info['partitions'][0]['end'])==3600
   for p in info['partitions']:
    low=ts(p['start']);high=ts(p['end']);mask=(t>=low)&(t<high);tt=t[mask]; x=a[mask];z=y[mask]; partial,strict,full,on,ids,starts,ends,complete,centers,stops=support(tt,x,z,low,high); rr=partition_rows[p['id']];checked+=1
    keys={'grid':len(tt),'aggregate_valid':np.isfinite(x).sum(),'label_valid':np.isfinite(z).sum(),'on_centers':on.sum(),'partial_on':(partial&on).sum(),'partial_off':(partial&~on).sum(),'strict_on':(strict&on).sum(),'strict_off':(strict&~on).sum(),'full_on':(full&on).sum(),'full_off':(full&~on).sum(),'fragments':len(starts),'complete_fragments':sum(complete)}
    for k,v in keys.items():assert int(rr[k])==int(v),(p['id'],k,rr[k],v)
    for k in list(keys)[:10]: assert sum(int(d[k]) for d in daily if d['partition']==p['id'])==int(keys[k])
    # Independently recompute every UTC-day support and first-applicable exclusion count.
    obs=np.isfinite(x);valid=np.isfinite(z);ix=np.arange(len(tt));edge=(ix<15)|(ix>=len(tt)-15);center_a=~edge&~obs;center_y=~edge&obs&~valid;corebad=~edge&obs&valid&~partial
    for dr in (d for d in daily if d['partition']==p['id']):
     day=ts(dr['utc_date']+'T00:00:00Z')//86400; m=tt//86400==day
     dc={'grid':m.sum(),'aggregate_valid':(m&obs).sum(),'label_valid':(m&valid).sum(),'on_centers':(m&on).sum(),'partial_on':(m&partial&on).sum(),'partial_off':(m&partial&~on).sum(),'strict_on':(m&strict&on).sum(),'strict_off':(m&strict&~on).sum(),'full_on':(m&full&on).sum(),'full_off':(m&full&~on).sum(),'edge_excluded':(m&edge).sum(),'center_aggregate_missing':(m&center_a).sum(),'center_label_missing':(m&center_y).sum(),'core_missing_excluded':(m&corebad).sum(),'fragments_started':sum(tt[b]//86400==day for b in starts),'complete_fragments_started':sum(tt[b]//86400==day and complete[k] for k,b in enumerate(starts))}
     for k,v in dc.items():assert int(dr[k])==int(v),(p['id'],dr['utc_date'],k)
    ff=[f for f in fragments if f['partition']==p['id']];assert len(ff)==len(starts)
    for k,(b,e) in enumerate(zip(starts,ends)):
     f=ff[k];assert f['fragment_id']==p['id']+':'+str(k) and ts(f['start_utc'])==tt[b] and ts(f['end_utc_exclusive'])==tt[e-1]+8 and int(f['centers'])==e-b and int(f['complete'])==complete[k]
     for key,view in [('partial_centers',partial),('strict_centers',strict),('full127_centers',full)]:assert int(f[key])==view[b:e].sum()
    for k,i in enumerate(centers):
     r=next(windows);idx=int((tt[i]-lo)//8);assert r['house']==h and r['source_role']==p['id'] and ts(r['center_utc'])==tt[i] and int(r['grid_index'])==idx
     assert r['fragment_id']==(p['id']+':'+str(ids[i]) if ids[i]>=0 else '');assert int(r['on'])==on[i] and int(r['partial_core31'])==1 and int(r['strict_core31'])==strict[i] and int(r['full127'])==full[i] and r['left_stop']==stops[0][k] and r['right_stop']==stops[1][k];rows_checked+=1
  assert next(windows,None) is None
 print('Independent full replay OK:',checked,'partitions;',rows_checked,'window identities; all slices/grid values, masks, labels, stops, fragments and partition totals.')
if __name__=='__main__':verify_all()
