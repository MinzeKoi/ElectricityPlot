"""Independent counts: interval intersections and convolution, no audit resampler."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from datetime import datetime, timezone

RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[2]
def utc(s): return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())
def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def raw(h,low,high,col):
    path=ROOT/f'data/raw/refit_complete_20261022/households/CLEAN_House{h}.csv'
    rows=[];prior=None
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            t=int(row['Unix'])
            if t<low:prior=row;continue
            if not rows and prior is not None:rows.append(prior)
            rows.append(row)
            if t>=high:break
    stamps=np.array([int(r['Unix']) for r in rows]); grid=np.arange(low,high,8)
    start=stamps[:-1];end=np.minimum(stamps[1:],start+8)
    result=[]
    for column in ('Aggregate',col):
        values=[]
        for r in rows[:-1]:
            try:v=float(r[column])
            except ValueError:v=np.nan
            if not np.isfinite(v) or v<0 or (column!= 'Aggregate' and r['Issues']=='1'):v=np.nan
            values.append(v)
        values=np.array(values);coverage=np.zeros(len(grid));weighted=np.zeros(len(grid))
        # A <=8s hold intersects at most two 8s bins; never bridge 9s or long gaps.
        base=(start-low)//8
        for offset in (0,1):
            index=base+offset
            left=low+index*8;right=left+8
            duration=np.maximum(0,np.minimum(end,right)-np.maximum(start,left))
            legal=(index>=0)&(index<len(grid))&np.isfinite(values)&(duration>0)
            np.add.at(coverage,index[legal],duration[legal]);np.add.at(weighted,index[legal],duration[legal]*values[legal])
        result.append(np.where(coverage==8,weighted/8,np.nan).astype(np.float32))
    return grid,*result

def source(h):
    arrays=[]
    for ch,field in [('aggregate','aggregate_power_w'),('microwave','power_w')]:
        p=ROOT/f'outputs/accuracy/microwave-data-audit-20261025-v2/inputs/refit_{h}/{ch}.csv'
        with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        arrays.append((np.array([utc(r['timestamp']) for r in rows]),np.array([float(r[field]) if r[field] else np.nan for r in rows],np.float32)))
    assert np.array_equal(arrays[0][0],arrays[1][0]);return arrays[0][0],arrays[0][1],arrays[1][1]

def count(t,a,y):
    active=np.isfinite(y)&(y>=200)
    starts=active&np.r_[True,(~active[:-1])|(np.diff(t)!=8)]
    ids=np.where(active,np.cumsum(starts)-1,-1)
    obs=np.isfinite(a)
    n=np.convolve(obs.astype(int),np.ones(31,dtype=int),'valid')
    double=(~obs[:-1])&(~obs[1:])
    no_double=np.convolve(double.astype(int),np.ones(30,dtype=int),'valid')==0
    c=np.arange(15,len(t)-15)
    legal=(n>=24)&no_double&obs[c]&np.isfinite(y[c])
    legal &= t[c]-120>=t[0]; legal &= t[c]+128<=t[-1]+8
    for i in np.flatnonzero(starts):
        eid=ids[i]; inds=np.flatnonzero(ids==eid)
        if t[inds[0]]<t[0] or t[inds[-1]]+8>t[-1]+8:legal &= ids[c]!=eid
    c=c[legal]
    return dict(aggregate_known=int(obs.sum()),label_known=int(np.isfinite(y).sum()),positive_label_bins=int(active.sum()),
                observable_segments=int(starts.sum()),eligible_centers=len(c),eligible_positive_centers=int(active[c].sum()),
                supported_segments=len(set(ids[c[active[c]]].tolist())))

def main():
    summary=load_json(RUN/'summary.json');cfg=load_json(RUN/'config.json');results=[]
    for s in summary['houses']:
        h=s['house'];low=utc(s['start_utc']);high=utc(s['end_utc'])
        arrays=source(h) if s['role']=='frozen_train_original14d' else raw(h,low,high,'Appliance5' if h=='2' else 'Appliance4')
        counts=count(*arrays)
        for key,value in counts.items():assert value==s[key],(h,s['role'],key,value,s[key])
        results.append(dict(house=h,role=s['role'],independent_counts=counts,passed=True))
    for item in load_json(RUN/'access_evidence.json')['reads']:
        assert sha(item['path'])==item['sha256'],item['path']
    assert cfg['training_interfaces']==[] and cfg['allowed_numeric_houses']==[2,19]
    assert all(summary[k]==0 for k in ['models_fit','optimizer_steps','inference','calibration','development_evaluations','house17_reads','final_house_reads'])
    out=dict(status='passed',method='Independent raw interval intersection (<=8s cap), event start cumsum and convolution core eligibility; no audit functions imported',
             checks=results,input_hashes_rechecked=True,model_code_imported=False)
    with (RUN/'independent_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
