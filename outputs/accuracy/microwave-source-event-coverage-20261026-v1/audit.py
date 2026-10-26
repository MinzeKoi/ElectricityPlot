"""Task17: fixed source-only count audit. No model dependency or training route."""
import argparse, csv, hashlib, json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT/'src'))
from virtual_submeter.data.prepare import Sample, resample, power

def iso(t):
    return datetime.fromtimestamp(int(t), timezone.utc).isoformat().replace('+00:00','Z')

def utc(s):
    return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())

def save(name, obj):
    with (RUN/name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

READS = []
def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def record(p, scope):
    p = Path(p).resolve()
    READS.append(dict(path=str(p),scope=scope,sha256=digest(p)))
    return p

def metadata(rel):
    p = record(ROOT/rel,'source identity/protocol metadata only')
    return json.loads(p.read_text(encoding='utf-8'))

def permit(h, low, high, config):
    if h not in ('2','19') or (low,high) != tuple(config['candidate_bounds'][h]):
        raise ValueError('Only exact preselected House2/19 candidate dates are permitted')

def segments(t, y):
    ids=np.full(len(t),-1,np.int64); events=[]
    for i in np.flatnonzero(np.isfinite(y)&(y>=200)):
        if i==0 or ids[i-1]<0 or t[i]-t[i-1]!=8:
            events.append(dict(event_id=len(events),start=int(t[i]),end=int(t[i]+8),samples=0,complete=False))
        e=events[-1]; ids[i]=e['event_id']; e['end']=int(t[i]+8); e['samples']+=1
    for e in events:
        a,b=np.searchsorted(t,[e['start'],e['end']])
        e['complete']=bool(a>0 and b<len(t) and t[a]-t[a-1]==8 and t[b]==e['end']
                           and np.isfinite(y[[a-1,b]]).all() and (y[[a-1,b]]<200).all())
    return ids,events

def legal_centers(t,a,y,low,high):
    ids,events=segments(t,y)
    c=np.arange(15,len(t)-15)
    x=np.lib.stride_tricks.sliding_window_view(a,31)
    times=np.lib.stride_tricks.sliding_window_view(t,31)
    obs=np.isfinite(x)
    keep=(t[c]-120>=low)&(t[c]+128<=high)&(np.diff(times,axis=1)==8).all(1)
    keep &= np.isfinite(a[c]) & np.isfinite(y[c]) & (obs.sum(1)>=24)
    keep &= ~((~obs[:,:-1])&(~obs[:,1:])).any(1)
    for e in events:
        if e['start']<low or e['end']>high:
            keep &= ids[c]!=e['event_id']
    return c[keep],ids,events

def quant(v):
    v=np.asarray(v); v=v[np.isfinite(v)]
    return dict(n=len(v),q0_q10_q50_q90_q100=np.quantile(v,[0,.1,.5,.9,1]).tolist()) if len(v) else dict(n=0,q0_q10_q50_q90_q100=[])

def load_frozen(h, identity):
    arrays=[]
    for channel,field in [('aggregate','aggregate_power_w'),('microwave','power_w')]:
        item=identity['source_inputs']['refit_'+h][channel]
        p=record(item['path'],'original14d House'+h+' '+channel)
        assert READS[-1]['sha256']==item['sha256']
        t=[]; v=[]; coverage=[]
        with p.open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                assert row['household_id']=='refit_'+h
                t.append(utc(row['timestamp'])); v.append(float(row[field]) if row[field] else np.nan)
                coverage.append(float(row['coverage']))
        arrays.append((np.array(t),np.array(v,np.float32),np.array(coverage)))
    assert np.array_equal(arrays[0][0],arrays[1][0])
    return arrays[0][0],arrays[0][1],arrays[1][1],arrays[0][2],arrays[1][2]

def raw_selected(h,low,high,col,config):
    permit(h,low,high,config)
    path=ROOT/f'data/raw/refit_complete_20261022/households/CLEAN_House{h}.csv'
    receipt=metadata(str(path.relative_to(ROOT).with_suffix('.receipt.json')))
    record(path,'SHA256 whole allowed house; numeric parsing candidate range plus one predecessor/closing observation only')
    assert READS[-1]['sha256']==receipt['sha256']
    rows=[]; prev=None; prev_t=None; parsed_outside=[]
    # CSV tokenization passes old rows, but only Unix is interpreted before the fixed date range.
    # One predecessor and first closing observation preserve native hold-age semantics.
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            t=int(row['Unix'])
            if t<low:
                prev=row; continue
            if not rows and prev is not None:
                rows.append(prev)
            rows.append(row)
            if t>=high: break
    samples=[]; stamps=[]
    for row in rows:
        t=int(row['Unix']); stamps.append(t)
        if not low<=t<high: parsed_outside.append(t)
        av,af=power(row['Aggregate']); yv,yf=power(row[col])
        if row['Issues']=='1': yv=None; yf.add('upstream_aggregate_discrepancy')
        elif row['Issues']!='0': raise ValueError('Unknown Issues')
        samples.append(Sample(t,{'aggregate':av,'microwave':yv},{'aggregate':af,'microwave':yf}))
    assert np.all(np.diff(stamps)>0)
    assert len(parsed_outside)<=2
    counters={'raw_intervals_seconds':Counter()}
    bins=[(t,v) for t,v in resample(samples,8,counters) if low<=t<high]
    t=np.array([b[0] for b in bins],np.int64)
    def vals(ch,key):
        return np.array([np.nan if b[1][ch][key] is None else b[1][ch][key] for b in bins],np.float32)
    evidence=dict(house=h,range_start=iso(low),range_end=iso(high),raw_rows=len(rows),
                  first_raw=iso(stamps[0]),last_raw=iso(stamps[-1]),boundary_context_unix=parsed_outside,
                  delta_histogram=dict(Counter(map(int,np.diff(stamps)))),
                  issues1_rows=sum(r['Issues']=='1' for r in rows),selected_rows_sha256=hashlib.sha256(
                      '\n'.join(','.join(r[k] for k in ('Unix','Aggregate',col,'Issues')) for r in rows).encode()).hexdigest())
    return (t,vals('aggregate','power_w'),vals('microwave','power_w'),vals('aggregate','coverage'),vals('microwave','coverage')),evidence

def summarize(h,role,arrays,low,high,centers=None,frozen_ids=None):
    t,a,y,ac,yc=arrays
    c,ids,events=legal_centers(t,a,y,low,high)
    if centers is not None:
        assert np.array_equal(t[c],centers)
        assert np.array_equal(ids[c],frozen_ids)
    pos=np.isfinite(y)&(y>=200); known=np.isfinite(y)&(y>=0)
    cp=pos[c]; supported=set(ids[c[cp]].tolist())
    evrows=[]
    for e in events:
        m=ids==e['event_id']; keep=ids[c]==e['event_id']
        evrows.append(dict(house=h,role=role,event_id=e['event_id'],start_utc=iso(e['start']),end_utc=iso(e['end']),
                           duration_observed_seconds=e['samples']*8,samples=e['samples'],complete_off_boundaries=e['complete'],
                           eligible_positive_centers=int(keep.sum()),full_eligible=bool(e['complete'] and keep.sum()==e['samples']),
                           power_min_w=float(y[m].min()),power_median_w=float(np.median(y[m])),power_max_w=float(y[m].max())))
    daily=[]
    for day in range(low//86400,(high-1)//86400+1):
        lo=max(low,day*86400); hi=min(high,(day+1)*86400)
        m=(t>=lo)&(t<hi); cm=(t[c]>=lo)&(t[c]<hi)
        active_events=set(ids[m&pos].tolist())
        daily.append(dict(house=h,role=role,utc_date=iso(day*86400)[:10],start_utc=iso(lo),end_utc=iso(hi),
                          full_utc_date=hi-lo==86400,expected_grid_bins=(hi-lo)//8,present_bins=int(m.sum()),
                          aggregate_known=int((m&np.isfinite(a)).sum()),label_known=int((m&known).sum()),
                          aggregate_covered_seconds=float(ac[m].sum()*8),label_covered_seconds=float(yc[m].sum()*8),
                          label_missing=int((m&~known).sum()),positive_label_bins=int((m&pos).sum()),
                          observable_segments_touching_date=len(active_events),
                          observable_segments_starting_date=sum(lo<=e['start']<hi for e in events),
                          eligible_centers=int(cm.sum()),eligible_positive_centers=int((cm&cp).sum()),
                          eligible_segment_ids=len(set(ids[c[cm&cp]].tolist()))))
    background=np.isfinite(a)&known&(y<200)
    residual=a[background]-y[background]
    s=dict(house=h,role=role,start_utc=iso(low),end_utc=iso(high),grid_bins=len(t),expected_grid_bins=(high-low)//8,
           utc_dates=len(daily),full_utc_dates=sum(d['full_utc_date'] for d in daily),
           aggregate_known=int(np.isfinite(a).sum()),label_known=int(known.sum()),
           aggregate_covered_seconds=float(ac.sum()*8),label_covered_seconds=float(yc.sum()*8),
           aggregate_missing_bins=int((~np.isfinite(a)).sum()),label_missing_bins=int((~known).sum()),
           positive_label_bins=int(pos.sum()),observable_segments=len(events),complete_off_boundary_segments=sum(e['complete'] for e in events),
           eligible_centers=len(c),eligible_positive_centers=int(cp.sum()),eligible_negative_centers=int((~cp).sum()),
           supported_segments=len(supported),fully_supported_complete_segments=sum(e['full_eligible'] for e in evrows),
           positive_power_w=quant(y[pos]),observed_segment_seconds=quant([e['samples']*8 for e in events]),
           off_background_residual_w=quant(residual),negative_residual_bins=int((residual<0).sum()),
           centers_are_audit_counts_only=True)
    return s,daily,evrows

def write_csv(name,rows):
    with (RUN/name).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def audit():
    if (RUN/'summary.json').exists() or (RUN/'config.json').exists():
        raise ValueError('Completed or started audit cannot be overwritten; use new version')
    identity=metadata('outputs/accuracy/microwave-source3-background-preparation-20261025-v4/identity.json')
    manifest=metadata('data/processed/refit_r1_sources_20261022/manifest.json')
    catalog=metadata('configs/refit_channels_complete_20261022.json')
    config=dict(task=17,authorized_audit_only=True,allowed_numeric_houses=[2,19],house5='existing frozen aggregate counts only',
                forbidden_houses=[17,8,9,20],date_selection='first complete UTC date after original14d end, then 14 consecutive UTC dates; no label ranking',
                candidate_bounds={h:[(identity['roles']['refit_'+h][0][2]+86399)//86400*86400,
                                     (identity['roles']['refit_'+h][0][2]+86399)//86400*86400+14*86400] for h in ('2','19')},
                threshold_w=200,native_seconds=8,core=31,context=127,min_core_real=24,two_consecutive_missing='reject',
                training_interfaces=[],model_fits=0,optimizer_steps=0,inference=0,calibration=0,development_evaluations=0)
    save('config.json',config) # Persist date selection before any candidate power interpretation.
    save('source_metadata.json',dict(manifest={h:{k:manifest['households'][h][k] for k in ('start','end','source_sha256','artifacts')} for h in ('2','19')},
                                     channels={h:catalog['households'][h]['microwave'] for h in ('2','19')},
                                     agents='No applicable AGENTS.md found in workspace ancestors and accessed directory ancestors'))
    train=ROOT/'outputs/accuracy/microwave-source5-temporal-20261025-v2/train.npz'
    record(train,'frozen house/t/y/event fields only; H5 aggregate identity count only')
    assert READS[-1]['sha256']==identity['train_sha256']
    with np.load(train,allow_pickle=False) as z:
        d={k:z[k] for k in ('house','t','y','event')}
    assert len(d['y'])==336704 and (d['y']>=200).sum()==2275
    save('frozen_identity.json',dict(sha256=READS[-1]['sha256'],fields_read=list(d),house5_aggregate_only=dict(windows=int((d['house']==2).sum()),positive_centers=int(((d['house']==2)&(d['y']>=200)).sum()))))
    pool=metadata('outputs/accuracy/microwave-source3-background-preparation-20261025-v4/source_pool_audit.json')
    pool_groups={};
    p=record(ROOT/'outputs/accuracy/microwave-source3-background-preparation-20261025-v4/pool_manifest.csv','source-only pool metadata; H5 aggregate counts only')
    with p.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['kind']=='template':
                key=(r['house'],int(r['event_id']));pool_groups[key]=pool_groups.get(key,0)+1
    save('sealed_pool_metadata.json',dict(total_templates=pool['templates'],template_segments=len(pool_groups),
        per_house=[dict(house='refit_'+h,templates=sum(v for (hh,e),v in pool_groups.items() if hh=='refit_'+h),
                        segments=sum(hh=='refit_'+h for hh,e in pool_groups),
                        detailed_metadata=next(x for x in pool['houses'] if x['house']=='refit_'+h) if h!='5' else 'no new House5 parsing; aggregate counts only') for h in ('2','19','5')]))
    summaries=[];daily=[];events=[];evidence=[];roles=[]
    for hi,h in enumerate(('2','19')):
        low,high=identity['roles']['refit_'+h][0][1:]
        arrays=load_frozen(h,identity); ix=d['house']==hi
        s,day,ev=summarize(h,'frozen_train_original14d',arrays,low,high,d['t'][ix],d['event'][ix])
        np.testing.assert_array_equal(arrays[2][np.searchsorted(arrays[0],d['t'][ix])],d['y'][ix])
        summaries.append(s);daily+=day;events+=ev
        lo,hi2=config['candidate_bounds'][h]
        roles.append(dict(house=h,role='excluded_tail_partial_date',start_utc=iso(high),end_utc=iso(lo),used_for_training=False))
        arrays,evd=raw_selected(h,lo,hi2,catalog['households'][h]['microwave'][0],config)
        s,day,ev=summarize(h,'candidate_audit_only_not_training',arrays,lo,hi2)
        summaries.append(s);daily+=day;events+=ev;evidence.append(evd)
        for dayrow in day:
            roles.append(dict(house=h,role=dayrow['role'],start_utc=dayrow['start_utc'],end_utc=dayrow['end_utc'],used_for_training=False))
        print('House',h,'candidate centers/positive/segments',s['eligible_centers'],s['eligible_positive_centers'],s['supported_segments'],flush=True)
    write_csv('daily_counts.csv',daily);write_csv('segment_details.csv',events)
    save('date_roles.json',dict(original_training=[dict(house=s['house'],role=s['role'],start_utc=s['start_utc'],end_utc=s['end_utc'],used_for_training=True) for s in summaries if s['role']=='frozen_train_original14d'],
                               unused_dates=roles,house5_prefix='existing 2014-11-15 through 2014-11-22T23:30Z unchanged; suffix block4/5 validation only; not new training'))
    save('raw_range_evidence.json',evidence)
    save('summary.json',dict(status='audit_complete_no_training',houses=summaries,models_fit=0,optimizer_steps=0,inference=0,calibration=0,development_evaluations=0,house17_reads=0,final_house_reads=0))
    save('access_evidence.json',dict(reads=READS,scope='explicit named metadata and numeric House2/19 only; no recursive data search',
        earlier_discovery='Initial broad filename-only rg listing under outputs/accuracy was overbroad; no listed data contents opened. Further discovery explicitly scoped. One nonrecursive listing of refit_complete_28d cache denied; not retried.',
        raw_power_boundary_exception='one predecessor and first closing observation only; necessary to reproduce native 8-second hold age; not candidate centers'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit']);args=parser.parse_args();audit()
