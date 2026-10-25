"""Source-only reproducible microwave availability audit; never fits a model."""
import sys, csv, json, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from virtual_submeter.data.common import utc, iso, sha256
from virtual_submeter.data.prepare import Sample, resample
from virtual_submeter.models.arrays import windows

def raw_slice(path, low, high, column):
    pieces=[]; prev=None
    for chunk in pd.read_csv(path,usecols=['Unix','Aggregate',column,'Issues'],dtype=str,keep_default_na=False,chunksize=200000):
        t=chunk.Unix.astype('int64'); before=chunk[t<low]
        if len(before): prev=before.tail(1)
        inside=chunk[(t>=low)&(t<high)]
        if len(inside):
            if not pieces and prev is not None: pieces.append(prev)
            pieces.append(inside)
        after=chunk[t>=high]
        if len(after): pieces.append(after.head(1)); break
    return pd.concat(pieces,ignore_index=True)

def probe():
    house='19'; old=ROOT/'outputs/accuracy/microwave-diagnosis-20261025-v2'
    meta=json.loads((old/'source_protocol.json').read_text())['sources']['refit_19']
    low=utc(meta['start']); high=low+86400
    raw=raw_slice(ROOT/'data/raw/refit_complete_20261022/households/CLEAN_House19.csv',low,high,'Appliance4')
    ts=raw.Unix.astype('int64').to_numpy(); assert np.all(np.diff(ts)>0)
    samples=[Sample(int(t),{'aggregate':float(v)},{'aggregate':set()}) for t,v in zip(ts,raw.Aggregate)]
    bins=[(t,v['aggregate']) for t,v in resample(samples,8,{'raw_intervals_seconds':Counter()}) if low<=t<high]
    t=np.array([r[0] for r in bins]); v=np.array([np.nan if r[1]['power_w'] is None else r[1]['power_w'] for r in bins])
    _,c=windows(t,v,31,8)
    print(json.dumps({'house':house,'day':0,'raw_rows':len(raw),'delta_histogram':dict(Counter(map(int,np.diff(ts)))),'coverage_histogram':dict(Counter(str(r[1]['coverage']) for r in bins)),'complete_c31':len(c)}))
    assert len(c)==0, 'Exact reported symptom no longer reproduced'
    # Red-capable contract: a continuous nine-second cadence must NOT gain fabricated coverage.
    s=[Sample(t,{'aggregate':100.},{'aggregate':set()}) for t in range(0,361,9)]
    b=list(resample(s,8,{'raw_intervals_seconds':Counter()}))
    assert any(v['aggregate']['coverage']<1 for _,v in b)
    print('PASS: real first-day zero-window symptom reproduced; nine-second cap counterexample preserved')
def event_runs(t, y):
    """Threshold observation segments, not inferred physical cycles."""
    active=np.isfinite(y)&(y>=200)
    starts=np.flatnonzero(active & ~np.r_[False,active[:-1]])
    ends=np.flatnonzero(active & ~np.r_[active[1:],False])+1
    return [{'id':i,'a':int(a),'b':int(b),'start':iso(int(t[a])),
             'end':iso(int(t[b-1])+8),'positive_centers':int(b-a),
             'observed_off_boundaries':bool(a>0 and b<len(y) and
                 np.isfinite(y[[a-1,b]]).all() and (y[[a-1,b]]<200).all())}
            for i,(a,b) in enumerate(zip(starts,ends))]

def window_masks(t, aggregate, y):
    _,centers=windows(t,aggregate,31,8)
    context=np.zeros(len(t),bool);context[centers]=True
    edge=np.zeros(len(t),bool);edge[:15]=True;edge[-15:]=True
    # Mutually exclusive losses in the stated order.
    center_missing=~edge & ~np.isfinite(aggregate)
    context_missing=~edge & np.isfinite(aggregate) & ~context
    label_missing=context & ~np.isfinite(y)
    usable=context & np.isfinite(y)
    assert np.all(edge.astype(int)+center_missing+context_missing+label_missing+usable==1)
    return context,usable,dict(edge=edge,aggregate_center_missing=center_missing,
        aggregate_context_missing=context_missing,label_center_missing=label_missing,usable=usable)

def block_support(t,y,usable,events,low,high):
    keep=usable & (t-120>=low)&(t+128<=high)
    crossing=[]
    for e in events:
        a,b=e['a'],e['b']
        if t[a]<low or t[b-1]+8>high:
            if keep[a:b].any(): crossing.append(e['id'])
            keep[a:b]=False
    eligible=[e for e in events if keep[e['a']:e['b']].any()]
    full=[e for e in eligible if e['observed_off_boundaries'] and keep[e['a']:e['b']].all()
          and t[e['a']]-8>=low and t[e['b']-1]+16<=high]
    return keep,{'start':iso(low),'end':iso(high),'windows':int(keep.sum()),
        'effective_hours':float(keep.sum()*8/3600),'positive_windows':int((keep&(y>=200)).sum()),
        'supported_threshold_segments':len(eligible),'complete_observed_segments':len(full),
        'complete_event_ids':[e['id'] for e in full], 'cross_boundary_event_ids_excluded':crossing}

def run_audit(dest):
    import shutil
    from virtual_submeter.data.prepare import read_samples
    from virtual_submeter.data.common import write_json,inventory
    if dest.exists(): raise ValueError('Output exists; choose a new version')
    dest.mkdir(parents=True)
    old=ROOT/'outputs/accuracy/microwave-diagnosis-20261025-v2'
    protocol=ROOT/'outputs/accuracy/state-20261024-v2/protocol/protocol.json'
    parts={p['household_id']:p for p in json.loads(protocol.read_text())['partitions'] if p['role']=='source_train'}
    summary={'schema_version':1,'audit_id':'microwave-data-audit-20261025-v2','status':'running',
        'accessed_actual_data_households':['2','19'],'model_fits':0,'real_fix_applied':False,
        'data_version':'strict8-replayed-identical-v1','shared_code_changed':False,
        'default_behavior_changed':False,'old_run_hashes':{},'households':{}}
    for name in ['registration.json','freeze.json','daily_support.json','delivery_manifest.json','source_protocol.json']:
        summary['old_run_hashes'][name]=sha256(old/name)
    daily_all=[];all_events=[];examples=[];checks=[]
    for h,col in [('2','Appliance5'),('19','Appliance4')]:
        print('Audit raw source House'+h,flush=True)
        p=parts['refit_'+h];low,high=utc(p['start']),utc(p['end'])
        rawpath=ROOT/f'data/raw/refit_complete_20261022/households/CLEAN_House{h}.csv'
        digest=sha256(rawpath);assert digest==p['origin']['source_sha256']
        raw=raw_slice(rawpath,low,high,col)
        sliced=dest/f'raw_slices/House{h}.csv';sliced.parent.mkdir(exist_ok=True);raw.to_csv(sliced,index=False)
        rt=raw.Unix.astype('int64').to_numpy();dt=np.diff(rt)
        assert (dt>0).all()
        samples=list(read_samples(sliced,h,{'microwave':[col]},'refit'))
        # Audit-only branch isolates Issues label masking; it is never a model input.
        from virtual_submeter.data.prepare import power
        label_unmasked=[Sample(s.timestamp,{'microwave':power(v)[0]}, {'microwave':set()})
                        for s,v in zip(samples,raw[col])]
        unmasked_bins={stamp:v['microwave'] for stamp,v in resample(iter(label_unmasked),8,{'raw_intervals_seconds':Counter()}) if low<=stamp<high}
        counters={'raw_intervals_seconds':Counter()}
        bins=[(t,v) for t,v in resample(iter(samples),8,counters) if low<=t<high]
        t=np.array([t for t,v in bins],dtype=np.int64)
        assert np.array_equal(t,np.arange(low,high,8))
        arrays={};replay_checks={}
        for channel,key in [('aggregate','aggregate_power_w'),('microwave','power_w')]:
            prior=old/f'inputs/refit_{h}/{channel}.csv'
            assert sha256(prior)==p['artifacts'][channel]['sha256']
            rows=list(csv.DictReader(prior.open(encoding='utf-8',newline='')))
            values=np.array([np.nan if v[channel]['power_w'] is None else v[channel]['power_w'] for _,v in bins])
            prior_values=np.array([float(r[key]) if r[key] else np.nan for r in rows])
            assert len(rows)==len(t)
            assert all(utc(r['timestamp'])==int(stamp) for r,stamp in zip(rows,t))
            np.testing.assert_allclose(values,prior_values,rtol=1e-10,atol=1e-7,equal_nan=True)
            assert all(float(r['coverage'])==v[channel]['coverage'] and json.loads(r['quality_flags'])==v[channel]['quality_flags'] for r,(_,v) in zip(rows,bins))
            target=dest/f'inputs/refit_{h}/{channel}.csv';target.parent.mkdir(parents=True,exist_ok=True)
            # Byte identical audited source inputs; replay proves their derivation.
            shutil.copyfile(prior,target)
            replay_checks[channel]={'missing_before':int(np.isnan(prior_values).sum()),'missing_after':int(np.isnan(values).sum()),
                'missing_mask_changes':0,'power_changes':0,'coverage_flag_timestamp_changes':0,
                'input_path':target.relative_to(ROOT).as_posix(),'sha256':sha256(target),'old_path':prior.relative_to(ROOT).as_posix()}
            arrays[channel]=values
        agg,y=arrays['aggregate'],arrays['microwave']
        preissue=np.array([np.nan if unmasked_bins[int(stamp)]['power_w'] is None else unmasked_bins[int(stamp)]['power_w'] for stamp in t])
        context,usable,loss=window_masks(t,agg,y);events=event_runs(t,y)
        # Attribute every uncovered second to exact raw delta categories, without filling it.
        gap_seconds={k:np.zeros(len(t),dtype=np.int64) for k in ['delta9','delta10_16','delta_gt16','raw_aggregate_invalid']}
        def add_span(target,a,b):
            a=max(int(a),low);b=min(int(b),high)
            if a>=b:return
            for i in range((a-low)//8,(b-1-low)//8+1):
                target[i]+=max(0,min(b,int(t[i])+8)-max(a,int(t[i])))
        for left,right in zip(samples,samples[1:]):
            d=right.timestamp-left.timestamp
            if d>8:
                k='delta9' if d==9 else 'delta10_16' if d<=16 else 'delta_gt16'
                add_span(gap_seconds[k],left.timestamp+8,right.timestamp)
            if left.values['aggregate'] is None:
                add_span(gap_seconds['raw_aggregate_invalid'],left.timestamp,min(left.timestamp+8,right.timestamp))
        # There is a predecessor for these source ranges, so no unassigned left edge.
        missing_seconds=np.array([round((1-v['aggregate']['coverage'])*8) for _,v in bins])
        np.testing.assert_array_equal(sum(gap_seconds.values()),missing_seconds)
        for e in events:
            a,b=e['a'],e['b'];e.update(household='refit_'+h,
                usable_positive_centers=int(usable[a:b].sum()),
                complete_c31_support=bool(e['observed_off_boundaries'] and usable[a:b].all()),
                loss_counts={k:int(v[a:b].sum()) for k,v in loss.items() if k!='usable'})
        all_events.extend(events)
        for day in range(14):
            a=low+day*86400;b=a+86400;mask=(t>=a)&(t<b);on=mask&(y>=200)
            es=[e for e in events if a<=utc(e['start'])<b]
            rr=(rt>=a)&(rt<b)
            row={'household':'refit_'+h,'day_from_zero':day,'start':iso(a),'end':iso(b),
                'raw_records':int(rr.sum()),'raw_issues1':int((rr&(raw.Issues.to_numpy()=='1')).sum()),
                'raw_on_records_before_issues':int((rr&(raw[col].astype(float).to_numpy()>=200)).sum()),
                'label_valid_bins_before_issues':int((mask&np.isfinite(preissue)).sum()),
                'label_bins_lost_to_issues_after_resampling':int((mask&np.isfinite(preissue)&~np.isfinite(y)).sum()),
                'on_grid_centers_before_issues':int((mask&(preissue>=200)).sum()),
                'on_grid_centers_lost_to_issues':int((mask&(preissue>=200)&~np.isfinite(y)).sum()),
                'grid_bins':int(mask.sum()),'aggregate_valid_bins':int((mask&np.isfinite(agg)).sum()),
                'aggregate_full_bin_hours':float((mask&np.isfinite(agg)).sum()*8/3600),
                'aggregate_supported_seconds':int(sum(v['aggregate']['coverage']*8 for stamp,v in bins if a<=stamp<b)),
                'label_valid_bins':int((mask&np.isfinite(y)).sum()),'observed_on_centers':int(on.sum()),
                'complete_aggregate_windows':int((mask&context).sum()),'usable_windows':int((mask&usable).sum()),
                'usable_on_centers':int((on&usable).sum()),'usable_hours':float((mask&usable).sum()*8/3600),
                'threshold_segments_starting':len(es),'complete_observed_segments':sum(e['complete_c31_support'] for e in es),
                'raw_delta_histogram':dict(Counter(map(str,dt[rr[:-1]]))),
                'aggregate_coverage_histogram':dict(Counter(str(v['aggregate']['coverage']) for stamp,v in bins if a<=stamp<b)),
                'uncovered_seconds_by_raw_cause':{k:int(v[mask].sum()) for k,v in gap_seconds.items()},
                'window_loss_exclusive':{k:int((v&mask).sum()) for k,v in loss.items()},
                'on_window_loss_exclusive':{k:int((v&on).sum()) for k,v in loss.items()}}
            daily_all.append(row)
        olddaily=json.loads((old/'daily_support.json').read_text())['rows']
        for row in daily_all[-14:]:
            prev=next(r for r in olddaily if r['household']==row['household'] and r['day_from_zero']==row['day_from_zero'])
            for k in ['complete_aggregate_windows','usable_on_centers']:assert row[k]==prev[k]
        blocks={};keeps={}
        # Multiple alternatives checked for availability only; no model outputs loaded.
        definitions={'historical_train':(8*86400,11*86400-1800),
            'historical_validation':(11*86400+1800,12*86400-1800),
            'historical_outer':(12*86400+1800,14*86400),
            'recommended_train':(0,10*86400-1800),
            'recommended_validation':(10*86400+1800,11*86400-1800),
            'recommended_outer':(11*86400+1800,14*86400),
            'early_0_7':(0,7*86400-1800),
            **{f'day_{d}':(d*86400+1800,(d+1)*86400-1800) for d in range(8,14)}}
        for name,(a,b) in definitions.items():
            keep,blocks[name]=block_support(t,y,usable,events,low+a,low+b);keeps[name]=keep
        for family in ['historical','recommended']:
            names=[family+'_'+r for r in ['train','validation','outer']]
            split_sets=[set(np.flatnonzero(keeps[n])) for n in names]
            assert not any(a&b for i,a in enumerate(split_sets) for b in split_sets[i+1:])
            for first,second in zip(names,names[1:]):
                a=np.flatnonzero(keeps[first]);b=np.flatnonzero(keeps[second]);assert t[a[-1]]+128<=t[b[0]]-120
                ea={e['id'] for e in events if keeps[first][e['a']:e['b']].any()}; eb={e['id'] for e in events if keeps[second][e['a']:e['b']].any()};assert not ea&eb
        np.savez_compressed(dest/f'refit_{h}_support.npz',t=t,aggregate_valid=np.isfinite(agg),label_valid=np.isfinite(y),
            complete_context=context,usable=usable,**{'mask_'+k:v for k,v in keeps.items()})
        # Small raw/binned counterexamples retain exact times and unchanged powers.
        gap_indices=np.flatnonzero((dt==9)&(rt[:-1]>=low)&(rt[:-1]<low+86400))[:3]
        examples.append({'household':h,'nine_second_examples':[{'left':iso(int(rt[i])),
            'right':iso(int(rt[i+1])),'raw_aggregate_left':raw.Aggregate.iloc[i],
            'unsupported_interval':[iso(int(rt[i])+8),iso(int(rt[i+1]))]} for i in gap_indices]})
        summary['households']['refit_'+h]={'original_path':rawpath.relative_to(ROOT).as_posix(),'original_sha256':digest,
            'slice_path':sliced.relative_to(ROOT).as_posix(),'slice_sha256':sha256(sliced),'start':iso(low),'end':iso(high),
            'raw_selected_records_with_bracketing':len(raw),'timestamp_duplicates_or_decreases':0,
            'non_8_second_raw_intervals':int((dt!=8).sum()),'unix_used_local_Time_ignored':True,
            'replay':replay_checks,'total_usable_windows':int(usable.sum()),'total_positive_windows':int((usable&(y>=200)).sum()),
            'complete_observed_segments':sum(e['complete_c31_support'] for e in events),
            'usable_window_delta':0,'usable_positive_delta':0,'complete_event_delta':0,'candidate_blocks':blocks}
        checks.append({'household':h,'raw_hash_verified':True,'exact_grid':True,'all_channels_replayed':True,
            'uncovered_seconds_fully_attributed':True,'daily_old_symptom_reproduced':True,'split_context_event_isolation':True})
    summary.update(status='complete',root_causes=[
        'Strict 8-second observation hold cap plus frequent 9-second raw intervals creates genuine unsupported seconds under the frozen conservative contract.',
        'Occasional longer raw gaps further fragment context; a 31-bin window requires all 248 seconds, so small scattered holes remove many windows.',
        'Issues=1 invalidates labels only; additional label loss is reproducible, and does not gate aggregate validity.',
        'No import, timestamp, label alignment or window implementation defect was found in the audited source chain.'],
        contract_verdict='No correctness repair justified. Allowing a nine-second hold would be a new observational assumption, not recovered measurements; not applied.',
        unrecoverable_limits=['Original pre-cleaning missingness and upstream zero/forward-filled values are unavailable.',
            'Upstream IAM >4000W clipping cannot be reversed.','Physical cycles cannot be identified independently from threshold segments.',
            'Real inter-record gaps remain unsupported; strict8 audit does not recover early House19 positive windows.'],
        recommendation={'selection_basis':'availability only; no model scores read by this audit',
            'primary_blocks':['recommended_train','recommended_validation','recommended_outer'],
            'primary_relative_days':{'train':[0,10],'validation':[10,11],'outer':[11,14],'boundary_half_gap_seconds':1800},
            'multiple_validation_day_candidates_with_complete_support_both_houses':['day_8','day_9','day_10','day_13'],
            'unsuitable_single_day_complete_event_validation':{'day_11':'House2 has zero complete observed segments','day_12':'House19 has zero observed positive centers'},
            'use_condition':'Inspect per-house complete_observed_segments; reserve validation days before fitting and retain >=1800s each side of boundaries. Recompute windows and whole event exclusion for the chosen split; day masks above are diagnostic alternatives, not simultaneously disjoint roles.',
            'historical_contamination':'Days 12-14 have already been evaluated in the old diagnosis; they cannot be called fresh unseen evaluation. Availability does not establish independent test evidence.',
            'fresh_holdout_available_within_audited_range':False},
        downstream={'input_paths':{h:{c:v['input_path'] for c,v in info['replay'].items()} for h,info in summary['households'].items()},
            'support_masks':{h:(dest/f'{h}_support.npz').relative_to(ROOT).as_posix() for h in summary['households']},
            'input_contract':'Identical CSV schema and bytes as old diagnostic source inputs, 8s UTC, missing preserved, microwave on >=200W; opt in explicitly to these paths.',
            'normalization':'Estimate only on selected training windows; this audit creates no normalization or model.',
            'events_file':(dest/'events.json').relative_to(ROOT).as_posix()},
        rebuild_command='.venv-m2/Scripts/python.exe scripts/audit_microwave_data.py --run outputs/accuracy/microwave-data-audit-20261025-vNEXT',
        verification=checks)
    write_json(dest/'daily_support.json',{'rows':daily_all})
    write_json(dest/'events.json',{'definition':'Observed >=200W segments, not independent physical cycles; complete requires observed off boundaries and every positive center C31 usable','events':all_events})
    write_json(dest/'counterexamples.json',{'examples':examples})
    write_json(dest/'downstream_plan.json',{'schema_version':1,'audit_id':summary['audit_id'],
        'purpose':'availability-approved source development comparison; not fresh final evaluation',
        'sample_seconds':8,'window_samples':31,'on_threshold_w':200,
        'inputs':summary['downstream']['input_paths'],'support_masks':summary['downstream']['support_masks'],
        'blocks':{h:{role:info['candidate_blocks']['recommended_'+role] for role in ['train','validation','outer']} for h,info in summary['households'].items()},
        'boundary_half_gap_seconds':1800,'historical_outer_already_used':True,
        'require_refreeze_before_model_comparison':True})
    # Save exact execution dependencies; no copy of model outputs or other household data.
    for rel in ['scripts/audit_microwave_data.py','src/virtual_submeter/data/prepare.py',
                'src/virtual_submeter/data/common.py','src/virtual_submeter/models/arrays.py']:
        target=dest/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    write_json(dest/'audit.json',summary)
    write_json(dest/'complete.json',{'status':'complete','artifacts':inventory(dest)})
    print(json.dumps({'status':'complete','run':str(dest),'houses':{h:{k:v[k] for k in ['total_usable_windows','total_positive_windows','complete_observed_segments']} for h,v in summary['households'].items()}}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.run: run_audit(args.run.resolve())
    else: probe()
