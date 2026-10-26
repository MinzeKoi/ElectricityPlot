"""Read-only frozen gate audit. No fitting, selection, or threshold search."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from virtual_submeter.accuracy import microwave_event_gate as gate
from virtual_submeter.accuracy import microwave_source5_temporal as temporal
from virtual_submeter.models.arrays import aggregate_arrays
from virtual_submeter.data.common import utc, iso
from virtual_submeter.models.center_mask_tcn import expand_context

RUN = ROOT/'outputs/accuracy/microwave-source5-loss-preparation-20261025-v4'
OUT = ROOT/'outputs/accuracy/microwave-gate-recall-audit-20261025-v1'
SEEDS = [42, 1019, 2026]
READS = {}
NPZ_KEYS = {}
NUMERIC_READS = set()
READ_ONLY = True

def install_read_boundary(identity, manifest):
    allowed={str(Path(p).resolve()).casefold() for p in [*identity['source_inputs'],*identity['development_unlabeled']]}
    allowed.update(str((RUN/p).resolve()).casefold() for p in manifest['artifacts'])
    own=str(OUT.resolve()).casefold()+'\\'
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):
            return
        path=Path(args[0]).resolve(); key=str(path).casefold()
        mode=args[1]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if READ_ONLY and writing:
            raise RuntimeError('Read-only replay forbids file writes: '+str(path))
        if path.suffix.lower() not in ('.npz','.pt','.csv'):
            return
        if writing:
            if not key.startswith(own): raise RuntimeError('Audit forbids numerical writes outside its output')
        else:
            if key not in allowed and not key.startswith(own): raise RuntimeError('Numerical read outside explicit parent whitelist: '+str(path))
            NUMERIC_READS.add(str(path))
    sys.addaudithook(audit)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def js(p):
    READS[str(Path(p).resolve())] = sha(p)
    return json.loads(Path(p).read_text(encoding='utf-8'))

def load(p, keys):
    READS[str(Path(p).resolve())] = sha(p)
    NPZ_KEYS[str(Path(p).resolve())] = keys
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k] for k in keys}

def write(name, data):
    if READ_ONLY:
        return
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def csvout(name, rows):
    if READ_ONLY or not rows:
        return
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def independent_max(t, p):
    # Sliding-window view, segmented independently from the production shift implementation.
    h = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    breaks = np.r_[True, (np.diff(t)!=8) | ~finite[:-1] | ~finite[1:], True]
    cuts = np.flatnonzero(breaks)
    for a,b in zip(cuts[:-1],cuts[1:]):
        if not finite[a]:
            continue
        v = np.pad(p[a:b], (15,15), constant_values=-np.inf)
        h[a:b] = np.lib.stride_tricks.sliding_window_view(v,31).max(axis=1)
    return h

def observable_fragments(t, positive):
    ix=np.flatnonzero(positive)
    return [g for g in np.split(ix,np.flatnonzero((np.diff(ix)!=1)|(np.diff(t[ix])!=8))+1) if len(g)]

def synth():
    t = np.arange(90)*8
    t[60:] += 8
    p = np.zeros(90,dtype=np.float32)+.2
    p[15]=.9
    p[40]=np.nan
    p[59]=.95
    p[80]=.8
    h = gate.local_max(t,p)
    np.testing.assert_array_equal(h,independent_max(t,p))
    assert np.isnan(h[40]) and h[60]==.2 and h[31]==.2
    low,high=.1,.85
    s=gate.gate_state(p,h,low,high)
    # low-fail remains FN even with nearby peak; added FN and recovered TP kept separate.
    pp=np.array([.05,.2,.7,.3]); hh=np.array([.9,.9,.7,.9]); y=np.ones(4,bool)
    old=pp>=.5; new=gate.gate_state(pp,hh,.1,.85)==1
    assert int((y&old&~new).sum())==1 and int((y&~old&new).sum())==2
    assert not new[0]
    # Label unavailability does not remove finite p from the max input.
    valid=np.ones(90,bool); valid[15]=False
    assert h[16]==p[15] and not valid[15] and s[16]==1
    tt=np.array([0,8,16,32,40]); yy=np.array([300,np.nan,300,300,300])
    assert [g.tolist() for g in observable_fragments(tt,np.isfinite(yy)&(yy>=200))]==[[0],[2],[3,4]]
    return dict(passed=True, cases=['NaN barrier','8-second gap','one centered max','non-evaluation finite peak','low-fail FN','added FN versus recovered TP','truth fragments split at missing labels and time gaps'])

def build_source(identity):
    parent=Path(identity['parent'])
    path=Path(identity['source']['aggregate']['path'])
    assert str(path) in identity['source_inputs'] and sha(path)==identity['source_inputs'][str(path)]
    READS[str(path)]=sha(path)
    raw=gate.csv_rows(path,'refit_5')
    times=np.array([utc(r['timestamp']) for r in raw])
    for seed in SEEDS:
        model=RUN/f'models/unit_BCE/seed_{seed}'
        meta=js(model/'model.json')
        for block,lo,hi in temporal.VALIDATION:
            target=OUT/f'source_{seed}_{block}.npz'
            truthpath=parent/f'validation/block_{block}.npz'
            assert str(truthpath) in identity['source_inputs']
            assert sha(truthpath)==identity['source_inputs'][str(truthpath)]
            truth=load(truthpath,['t','y'])
            rr=[raw[j] for j in np.flatnonzero((times>=lo)&(times<hi))]
            t,p,_=gate.infer(model,rr)
            at,agg=aggregate_arrays(rr,8)
            np.testing.assert_array_equal(t,at)
            ix=gate.align(t,truth['t'])
            selected=load(model/f'epoch_{meta["selected"]["epoch"]}_block_{block}.npz',['t','scores'])
            np.testing.assert_array_equal(selected['t'],truth['t'])
            np.testing.assert_allclose(p[ix],selected['scores'],rtol=1e-5,atol=1e-7)
            np.testing.assert_array_equal(p[ix]>=meta['decision_threshold_probability'],selected['scores']>=meta['decision_threshold_probability'])
            y=np.full(len(t),np.nan,np.float32); y[ix]=truth['y']
            np.savez_compressed(target,t=t,p=p,y=y,aggregate=agg)
    write('source_cache_provenance.json',dict(inputs=READS.copy(),caches={p.name:sha(p) for p in sorted(OUT.glob('source_*.npz'))},algorithm=sha(Path(gate.__file__)),context_algorithm=sha(ROOT/'src/virtual_submeter/models/center_mask_tcn.py')))

def deeper(data, fragments, identity):
    context_rows=[]; observations={}
    for part in ['source_4','source_5','development']:
        first=data[(42,part)]; t,agg=first['t'],first['agg']
        centers=np.flatnonzero(first['mask']&(first['y']>=200))
        ctx=expand_context(t,agg,centers,int(t[0]),int(t[-1])+8)
        for k,j in enumerate(centers):
            observations[(part,int(j))]=float(ctx['outer_fraction'][k])
        for seed in SEEDS:
            d=data[(seed,part)]; on=d['on'][centers]; add=d['added'][centers]
            for k,j in enumerate(centers):
                left=right=0
                while left<15 and j-left>0 and t[j-left]-t[j-left-1]==8 and np.isfinite(d['p'][j-left-1]): left+=1
                while right<15 and j+right+1<len(t) and t[j+right+1]-t[j+right]==8 and np.isfinite(d['p'][j+right+1]): right+=1
                context_rows.append(dict(seed=seed,part=part,index=int(j),utc=iso(int(t[j])),outer_fraction=float(ctx['outer_fraction'][k]),left_stop=str(ctx['left_stop'][k]),right_stop=str(ctx['right_stop'][k]),gate_peak_left_samples=left,gate_peak_right_samples=right,gate_tp=bool(on[k]),added_fn=bool(add[k])))
    csvout('positive_context.csv',context_rows)
    # Source-only quartiles define descriptive groups; no decision threshold is changed.
    source=np.concatenate([np.c_[data[(42,p)]['y'][data[(42,p)]['mask']&(data[(42,p)]['y']>=200)],(data[(42,p)]['agg']-data[(42,p)]['y'])[data[(42,p)]['mask']&(data[(42,p)]['y']>=200)]] for p in ['source_4','source_5']])
    boundaries={field:np.quantile(source[:,k],[.25,.5,.75]).tolist() for k,field in enumerate(['power_w','background_w'])}
    write('group_definitions.json',dict(origin='source blocks 4/5 true-on evaluation points only',quartiles=boundaries,bin_rule='searchsorted(cuts,value,side=right); Q1..Q4',outer_rule='full outer_fraction==1; partial outer_fraction<1',fragment_rule='maximal observed y>=200 points with adjacent row and UTC delta 8, within evaluation support; not physical cycle'))
    grouped=[]
    for (seed,part),d in data.items():
        pos=d['mask']&(d['y']>=200)
        for field,v in [('power_w',d['y']),('background_w',d['agg']-d['y'])]:
            bins=np.searchsorted(boundaries[field],v,side='right')
            for b in range(4):
                m=pos&(bins==b)
                grouped.append(dict(seed=seed,part=part,field=field,group=f'Q{b+1}',n=int(m.sum()),gate_tp=int((m&d['on']).sum()),added_fn=int((m&d['added']).sum())))
        for label,predicate in [('full',lambda v:v==1),('partial',lambda v:v<1)]:
            ix=np.array([j for j in np.flatnonzero(pos) if predicate(observations[(part,int(j))])],dtype=int)
            grouped.append(dict(seed=seed,part=part,field='outer_context',group=label,n=len(ix),gate_tp=int(d['on'][ix].sum()),added_fn=int(d['added'][ix].sum())))
    csvout('source_defined_groups.csv',grouped)
    # Match observable fragment profiles using source-only fragment quartiles.
    sf=[r for r in fragments if r['seed']==42 and r['part'].startswith('source')]
    cuts={f:np.quantile([r[f] for r in sf],[.25,.5,.75]).tolist() for f in ['n','power_mean','background_mean']}
    def profile(r): return tuple(int(np.searchsorted(cuts[f],r[f],side='right')) for f in cuts)
    contrasts=[]
    lost=[r for r in fragments if r['seed']==1019 and r['part']=='development' and r['added_fn']]
    for r in lost:
        same=[s for s in sf if profile(s)==profile(r)]
        match={f'{s}_gate_tp':next(v['gate_tp'] for v in fragments if v['seed']==s and v['part']=='development' and v['start_utc']==r['start_utc']) for s in [42,2026]}
        contrasts.append(dict(**r,**match,source_profile=list(profile(r)),source_similar_fragments=len(same),source_similar_positive_points=sum(s['n'] for s in same)))
    write('lost_fragment_contrasts.json',dict(source_only_fragment_quartiles=cuts,source_fragments=len(sf),lost_fragments=contrasts,caution='Sparse joint bins describe support, not causal matching or independent physical cycles.'))
    # Authorized train metadata only, no x tensors or optimizer state.
    parent=Path(identity['parent']); tp=parent/'train.npz'
    assert sha(tp)==identity['source_inputs'][str(tp)]
    train=load(tp,['t','y','event','house','outer_fraction'])
    coverage=[]
    for house in np.unique(train['house']):
        ix=np.flatnonzero((train['house']==house)&(train['y']>=200)); ix=ix[np.argsort(train['t'][ix])]
        groups=np.split(ix,np.flatnonzero(np.diff(train['t'][ix])!=8)+1)
        coverage.append(dict(house_code=int(house),positive_centers=len(ix),observable_threshold_fragments=sum(bool(len(g)) for g in groups),distinct_event_ids=int(len(np.unique(train['event'][ix]))),power=quant(train['y'][ix]),outer_fraction=quant(train['outer_fraction'][ix])))
    write('training_coverage.json',dict(train_sha256=sha(tp),fields=['t','y','event','house','outer_fraction'],house_code_note='Encoded parent cache identities; event IDs not asserted to be independent physical runs.',coverage=coverage))
    structural=[]
    selection=js(RUN/'source_selection.json')
    for (seed,part),d in data.items():
        recipe=next(r for r in selection['recipes'] if r['seed']==seed)
        t,p,h=d['t'],d['p'],d['h']; L,H=recipe['L'],recipe['H']
        finite=np.isfinite(p); above=finite&(p>=L)
        # Segment identities describe existing scores only; do not produce component-gate predictions.
        full_start=np.r_[True,(np.diff(t)!=8)|~finite[:-1]|~finite[1:]]
        low_start=np.r_[True,(np.diff(t)!=8)|~above[:-1]|~above[1:]]
        full_ids=np.cumsum(full_start); low_ids=np.cumsum(low_start)
        high_indices=np.flatnonzero(finite&(p>=H))
        by_full={}; by_low={}
        for j in high_indices:
            by_full.setdefault(int(full_ids[j]),[]).append(int(j))
            by_low.setdefault(int(low_ids[j]),[]).append(int(j))
        low_bounds={}
        for a,b in zip(np.flatnonzero(low_start),np.r_[np.flatnonzero(low_start)[1:],len(t)]):
            if above[a]: low_bounds[int(low_ids[a])]=(int(a),int(b-1))
        for j in np.flatnonzero(d['mask']&(d['y']>=200)&(p>=L)&(h<H)):
            candidates=by_full.get(int(full_ids[j]),[])
            same=by_low.get(int(low_ids[j]),[])
            nearest=min(candidates,key=lambda k:abs(int(t[k])-int(t[j]))) if candidates else None
            nearest_low=min(same,key=lambda k:abs(int(t[k])-int(t[j]))) if same else None
            a,b=low_bounds[int(low_ids[j])]
            r=dict(seed=seed,part=part,index=int(j),utc=iso(int(t[j])),p=float(p[j]),h=float(h[j]),added_fn=bool(d['added'][j]),low_segment_start_utc=iso(int(t[a])),low_segment_end_utc=iso(int(t[b])),low_segment_samples=b-a+1,low_segment_observed_seconds=(b-a+1)*8,low_segment_elapsed_seconds=int(t[b]-t[a]),low_segment_max_p=float(p[a:b+1].max()),full_segment_has_high=bool(candidates),same_low_segment_has_high=bool(same),nearest_full_high_distance_seconds=None if nearest is None else abs(int(t[nearest])-int(t[j])),nearest_full_high_utc=None if nearest is None else iso(int(t[nearest])),nearest_full_high_same_low_segment=False if nearest is None else bool(low_ids[nearest]==low_ids[j]),nearest_same_low_high_distance_seconds=None if nearest_low is None else abs(int(t[nearest_low])-int(t[j])))
            if nearest_low is not None: assert r['nearest_same_low_high_distance_seconds']>120
            for other in SEEDS:
                z=data[(other,part)]
                r[f'{other}_point_tp']=bool(z['ps'][j]); r[f'{other}_gate_tp']=bool(z['on'][j])
                r[f'{other}_point_on_in_low_segment']=int(z['ps'][a:b+1].sum())
                r[f'{other}_gate_on_in_low_segment']=int(z['on'][a:b+1].sum())
            structural.append(r)
    csvout('high_fail_peak_distances.csv',structural)
    structural_summary=[]
    for seed in SEEDS:
        for part in ['source_4','source_5','development']:
            rr=[r for r in structural if r['seed']==seed and r['part']==part]
            structural_summary.append(dict(seed=seed,part=part,high_fail_points=len(rr),unique_low_segments=len({r['low_segment_start_utc'] for r in rr}),same_low_segment_has_high=sum(r['same_low_segment_has_high'] for r in rr),no_same_low_segment_high=sum(not r['same_low_segment_has_high'] for r in rr),no_high_in_full_finite_segment=sum(not r['full_segment_has_high'] for r in rr),nearest_full_high_distances_seconds=quant([r['nearest_full_high_distance_seconds'] for r in rr if r['nearest_full_high_distance_seconds'] is not None]),same_low_high_distances_seconds=quant([r['nearest_same_low_high_distance_seconds'] for r in rr if r['nearest_same_low_high_distance_seconds'] is not None]),low_segment_seconds=quant([r['low_segment_observed_seconds'] for r in rr])))
    write('peak_distance_summary.json',structural_summary)
    return dict(lost_fragments=len(lost),added_fn_total=sum(r['added_fn'] for r in lost),full_outer_added_fn=sum(r['added_fn'] for r in context_rows if r['part']=='development' and r['seed']==1019 and r['outer_fraction']==1),cross_seed_correct={str(s):int(sum(data[(s,'development')]['on'][data[(1019,'development')]['added']])) for s in [42,2026]})

def quant(v):
    v=np.asarray(v); v=v[np.isfinite(v)]
    return dict(n=len(v),q05=float(np.quantile(v,.05)),q25=float(np.quantile(v,.25)),q50=float(np.quantile(v,.5)),q75=float(np.quantile(v,.75)),q95=float(np.quantile(v,.95))) if len(v) else dict(n=0)

def main():
    global OUT, READ_ONLY
    ap=argparse.ArgumentParser(); ap.add_argument('--build-source',action='store_true'); ap.add_argument('--assert-acceptance',action='store_true'); ap.add_argument('--details',action='store_true'); ap.add_argument('--verify-ordinary',action='store_true'); ap.add_argument('--output',type=Path,help='Full audit only: an explicit, nonexistent sibling directory under outputs/accuracy'); args=ap.parse_args()
    start=time.perf_counter()
    full=args.build_source or args.details or args.verify_ordinary
    if full and args.output is None:
        ap.error('Full audit requires --output NEW_DIRECTORY; existing audits cannot be overwritten')
    if args.output is not None:
        target=args.output.resolve()
        if target.parent != (ROOT/'outputs/accuracy').resolve():
            ap.error('--output must be a new sibling directory directly under outputs/accuracy')
        if target.exists():
            ap.error('--output already exists; refusing to overwrite')
        if not all((args.build_source,args.details,args.verify_ordinary)):
            ap.error('--output requires --build-source --details --verify-ordinary for a complete audit')
        target.mkdir(exist_ok=False)
        OUT=target
        READ_ONLY=False
    selection=js(RUN/'source_selection.json'); identity=js(RUN/'identity.json')
    assert selection['selected']=='new_gate'
    manifest=js(RUN/'final_delivery_manifest.json')
    install_read_boundary(identity,manifest)
    # Manifest verification reads bytes only; never interpret unapproved datasets.
    for rel,expected in manifest['artifacts'].items():
        assert sha(RUN/rel)==expected,rel
    if args.build_source:
        import torch
        torch.set_num_threads(4)
        build_source(identity)
    cache_provenance=js(OUT/'source_cache_provenance.json')
    for rel,expected in cache_provenance['caches'].items(): assert sha(OUT/rel)==expected
    for path,expected in cache_provenance['inputs'].items(): assert sha(path)==expected
    assert cache_provenance['algorithm']==sha(Path(gate.__file__))
    assert cache_provenance['context_algorithm']==sha(ROOT/'src/virtual_submeter/models/center_mask_tcn.py')
    truth=load(RUN/'evaluation/truth_once.npz',['t','y'])
    devpath=next(Path(p) for p in identity['development_unlabeled'] if p.endswith('aggregate.csv'))
    assert sha(devpath)==identity['development_unlabeled'][str(devpath)]
    READS[str(devpath)]=sha(devpath)
    dt,da=aggregate_arrays(gate.csv_rows(devpath,'refit_17'),8)
    np.testing.assert_array_equal(dt,truth['t'])
    rows=[]; margins=[]; fnrows=[]; fragments=[]; data={}; checks=[]
    for seed in SEEDS:
        recipe=next(r for r in selection['recipes'] if r['seed']==seed)
        meta=js(RUN/f'models/unit_BCE/seed_{seed}/model.json')
        assert recipe['algorithm_sha256']==sha(Path(gate.__file__))
        for name,expected in recipe['parent_hashes'].items():
            assert sha(RUN/f'models/unit_BCE/seed_{seed}'/name)==expected
        for part in ['development','source_4','source_5']:
            if part=='development':
                z=load(RUN/f'predictions/new_gate_{seed}.npz',['t','scores','state'])
                point=load(RUN/f'predictions/new_point_{seed}.npz',['t','scores','state'])
                w=load(RUN/f'predictions/W50_{seed}.npz',['t','raw'])
                np.testing.assert_array_equal(z['t'],truth['t']); np.testing.assert_array_equal(z['t'],w['t'])
                np.testing.assert_array_equal(z['scores'],point['scores']); np.testing.assert_array_equal(point['t'],z['t'])
                t,p,y,agg=z['t'],z['scores'],truth['y'],da
                mask=np.isfinite(y)&np.isfinite(w['raw'][:,1])
                assert mask.sum()==137523 and (y[mask]>=200).sum()==383
            else:
                block=part[-1]; z=load(OUT/f'source_{seed}_{block}.npz',['t','p','y','aggregate'])
                t,p,y,agg=(z[k] for k in ['t','p','y','aggregate'])
                mask=np.isfinite(y)
            assert np.all(t%8==0) and np.all(np.diff(t)>0)
            h=gate.local_max(t,p); h2=independent_max(t,p)
            np.testing.assert_array_equal(h,h2)
            L,H=recipe['L'],recipe['H']; state=gate.gate_state(p,h,L,H)
            ps=(p>=meta['decision_threshold_probability']) & np.isfinite(p)
            if part=='development':
                np.testing.assert_array_equal(state,z['state']); np.testing.assert_array_equal(ps[mask],point['state'][mask])
            else:
                expected=next(r for r in selection['block_metrics'] if r['seed']==seed and r['block']==part[-1])
                got=gate.confusion(state[mask],y[mask]>=200)
                assert all(got[k]==expected[k] for k in ['tp','fp','fn','tn'])
            positive=mask&(y>=200); negative=mask&(y<200); on=state==1
            added=positive&ps&~on; recovered=positive&~ps&on
            decomposition=dict(nonfinite=int((positive&~np.isfinite(p)).sum()),low_fail=int((positive&np.isfinite(p)&(p<L)).sum()),high_fail=int((positive&(p>=L)&(h<H)).sum()))
            metrics=gate.confusion(state[mask],y[mask]>=200)
            assert sum(decomposition.values())==metrics['fn']
            assert np.isfinite(p[mask]).all()
            rows.append(dict(seed=seed,part=part,L=L,H=H,point_threshold=meta['decision_threshold_probability'],epoch=meta['selected']['epoch'],**metrics,**decomposition,point_tp=int((positive&ps).sum()),added_fn=int(added.sum()),recovered_tp=int(recovered.sum())))
            checks.append(dict(seed=seed,part=part,independent_max_exact=True,utc_8s_aligned=True,nonfinite_in_mask=0,finite_non_evaluation=int((~mask&np.isfinite(p)).sum())))
            for group,g in [('true_on',positive),('false_positive',negative&on),('true_off',negative),('high_rejected_true_on',positive&(p>=L)&(h<H))]:
                for field,values in [('p_minus_L',p-L),('h_minus_H',h-H),('power_w',y),('aggregate_w',agg),('background_w',agg-y)]:
                    margins.append(dict(seed=seed,part=part,group=group,field=field,**quant(values[g])))
            # Observable truth-threshold fragments: missing labels, support holes and time gaps split them.
            groups=observable_fragments(t,positive)
            for event,ix in enumerate(groups):
                if not len(ix): continue
                fragments.append(dict(seed=seed,part=part,fragment=event,start_utc=iso(int(t[ix[0]])),end_utc=iso(int(t[ix[-1]])),n=len(ix),power_mean=float(y[ix].mean()),aggregate_mean=float(agg[ix].mean()),background_mean=float((agg[ix]-y[ix]).mean()),p_max=float(p[ix].max()),h_max=float(h[ix].max()),gate_tp=int(on[ix].sum()),point_tp=int(ps[ix].sum()),added_fn=int(added[ix].sum()),recovered_tp=int(recovered[ix].sum()),low_fail=int((p[ix]<L).sum()),high_fail=int(((p[ix]>=L)&(h[ix]<H)).sum())))
            for j in np.flatnonzero(positive&~on):
                fnrows.append(dict(seed=seed,part=part,index=int(j),utc=iso(int(t[j])),t=int(t[j]),power_w=float(y[j]),aggregate_w=float(agg[j]),p=float(p[j]),h=float(h[j]),p_minus_L=float(p[j]-L),h_minus_H=float(h[j]-H),reason='low_fail' if p[j]<L else 'high_fail',point_tp=bool(ps[j]),added_fn=bool(added[j])))
            data[(seed,part)]=dict(t=t,p=p,h=h,y=y,mask=mask,on=on,ps=ps,added=added,agg=agg)
    write('replay_checks.json',checks); csvout('decomposition.csv',rows); csvout('margins.csv',margins); csvout('false_negatives.csv',fnrows); csvout('fragments.csv',fragments)
    # Cross-seed comparison at every newly lost 1019 point, without changing a rule.
    d=data[(1019,'development')]; cross=[]
    for j in np.flatnonzero(d['added']):
        r=dict(t=int(d['t'][j]),utc=iso(int(d['t'][j])),power_w=float(d['y'][j]))
        for seed in SEEDS:
            z=data[(seed,'development')]
            r.update({f'{seed}_p':float(z['p'][j]),f'{seed}_h':float(z['h'][j]),f'{seed}_gate_tp':bool(z['on'][j]),f'{seed}_point_tp':bool(z['ps'][j])})
        cross.append(r)
    csvout('seed1019_added_fn_cross_seed.csv',cross)
    detail=deeper(data,fragments,identity) if args.details else None
    if args.verify_ordinary:
        z=data[(1019,'development')]
        ot,op,_=gate.infer(RUN/'models/unit_BCE/seed_1019',gate.csv_rows(devpath,'refit_17'))
        np.testing.assert_array_equal(ot,z['t']); np.testing.assert_array_equal(op,z['p'])
        write('ordinary_independent_recheck.json',dict(seed=1019,ordinary_infer_full_aggregate_scores_exact=True,source_all_seeds_selected_epoch_scores_checked=True,source_ordinary_to_frozen_cache=True))
    result=dict(selected='new_gate',eachseedR85=all(r['recall']>=.85 for r in rows if r['part']=='development'),development=[r for r in rows if r['part']=='development'],source=[r for r in rows if r['part']!='development'],synthetic=synth(),parent_artifact_hashes_checked=len(manifest['artifacts']),elapsed_seconds=time.perf_counter()-start,training_fits=0,optimizer_updates=0,threshold_searches=0)
    write('summary.json',result); write('input_hashes.json',READS)
    if detail: write('detail_summary.json',detail)
    write('full_input_hashes.json' if args.details else 'replay_input_hashes.json',READS)
    write('full_read_boundary.json' if args.details else 'replay_read_boundary.json',dict(numerical_reads=sorted(NUMERIC_READS),npz_explicit_keys=NPZ_KEYS,original_development_label_csv_read=False,final_house_data_read=False,policy='Only sealed parent manifest, explicit identity source/unlabeled whitelist and audit outputs; numerical writes restricted to audit output.'))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.assert_acceptance and not result['eachseedR85']:
        sys.exit(2)

if __name__=='__main__':
    main()
