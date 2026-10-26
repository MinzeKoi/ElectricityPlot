"""Independent histogram/max-window review; never fits or changes selection."""
import argparse
from pathlib import Path
import sys
import json
import numpy as np

sys.dont_write_bytecode = True
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--phase',choices=['source','final','delivery'],required=True)
a = p.parse_args()
run = a.run.resolve()
sys.path.insert(0,str(run/'source'))
from virtual_submeter.accuracy import microwave_event_gate as e


def reference_max(t, scores):
    """Different implementation: padded original windows, no shifted updates."""
    result = np.full(len(scores),np.nan,np.float32)
    valid = np.flatnonzero(np.isfinite(scores))
    chunks = np.split(valid,np.flatnonzero((np.diff(valid)!=1) | (np.diff(t[valid])!=8))+1)
    for ix in chunks:
        if len(ix):
            padded = np.pad(scores[ix],(15,15),constant_values=-np.inf)
            result[ix] = np.lib.stride_tricks.sliding_window_view(padded,31).max(axis=1)
    return result


def histogram_metrics(values, labels, thresholds):
    n = len(thresholds)
    index = np.searchsorted(thresholds,values,side='right')-1
    def counts(mask):
        hist = np.bincount(index[mask & (index>=0)],minlength=n)
        return np.cumsum(hist[::-1])[::-1]
    tp,fp = counts(labels),counts(~labels)
    fn = int(labels.sum())-tp
    predicted = tp+fp
    return dict(precision=tp/np.maximum(predicted,1),recall=tp/max(int(labels.sum()),1),
        f1=2*tp/np.maximum(2*tp+fp+fn,1),nondegenerate=(predicted>0) & (predicted<len(values)))


def independent_threshold(blocks, thresholds, minimum_recall):
    if not len(thresholds):
        return None
    m = [histogram_metrics(v,y,thresholds) for v,y in blocks]
    eligible = np.logical_and.reduce([r['nondegenerate'] & (r['recall']>=minimum_recall) for r in m])
    ixs = np.flatnonzero(eligible)
    if not len(ixs):
        return None
    # Plain tuple comparison provides independent deterministic ordering.
    rank = [(min(r['precision'][i] for r in m),float(np.mean([r['precision'][i] for r in m])),
        float(np.mean([r['f1'][i] for r in m])),float(thresholds[i])) for i in ixs]
    return max(rank)[-1]


def scalar_metrics(state, y):
    tp=fp=fn=tn=0
    for pred,on in zip(state,y):
        e.require(pred in (0,1),'Unknown prediction in source support')
        if pred:
            if on: tp+=1
            else: fp+=1
        elif on: fn+=1
        else: tn+=1
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1))


def review_source():
    identity,config,s = e.source_guard(run)
    parent = Path(identity['parent'])
    with (run/'source_results/block_metrics.csv').open(encoding='utf-8',newline='') as stream:
        import csv
        recorded = list(csv.DictReader(stream))
    summaries, roundoff, parent_rows, gate_rows = [],[],[],[]
    for seed in e.SEEDS:
        r = next(r for r in s['recipes'] if r['seed']==seed)
        all_scores, low_blocks, high_blocks = [],[],[]
        pm,gm = [],[]
        for block in ('4','5'):
            d = e.npz(run/f'source_results/{seed}_block_{block}.npz',['t','scores','h','state','eval_t'])
            v = e.npz(parent/f'validation/block_{block}.npz',['t','y','house'])
            ix = e.align(d['t'],v['t'])
            e.require((v['house']==2).all(),'Household identity changed')
            np.testing.assert_array_equal(d['eval_t'],v['t'])
            h = reference_max(d['t'],d['scores'])
            np.testing.assert_array_equal(h,d['h'])
            # Appliance-label availability affects only the evaluation mapping.
            # Features are reconstructed from every original finite score.
            all_scores.append(d)
            low_blocks.append((d['scores'][ix],v['y']>=200))
            old = e.npz(parent/f'models/source5/seed_{seed}/epoch_{r["parent_epoch"]}_block_{block}.npz',['scores'])['scores']
            roundoff.append(dict(seed=seed,block=block,max_abs_difference=float(np.max(np.abs(old-d['scores'][ix]))),
                parent_point_states_exact=bool(np.array_equal(old>=r['parent_point_threshold'],d['scores'][ix]>=r['parent_point_threshold']))))
            e.require(roundoff[-1]['parent_point_states_exact'],'Parent point state differs')
            parent_state = (d['scores'][ix]>=r['parent_point_threshold']).astype(np.int8)
            for arm,state in [('parent',parent_state)]+([('gate',d['state'][ix])] if r['seed_feasible'] else []):
                m = scalar_metrics(state,v['y']>=200)
                rec = next(z for z in recorded if int(z['seed'])==seed and z['arm']==arm and z['block']==block)
                e.require(all(float(rec[k])==value for k,value in m.items()),'Source independent confusion differs')
                (pm if arm=='parent' else gm).append(m)
            summaries.append(dict(seed=seed,block=block,total_finite_unlabeled_feature_centers=int(np.isfinite(d['scores']).sum()),
                evaluation_centers=len(ix),additional_non_evaluation_finite_centers=int(np.isfinite(d['scores']).sum())-len(ix),
                h_independently_exact=True,low_fail_count=int((d['scores'][ix]<r['L']).sum()) if r['L'] is not None else None,
                low_fail_positive_count=int(((d['scores'][ix]<r['L']) & (v['y']>=200)).sum()) if r['L'] is not None else None))
            if r['L'] is not None:
                high_blocks.append((np.where(d['scores'][ix]>=r['L'],h[ix],-1.),v['y']>=200))
        lc = np.unique(np.concatenate([d['scores'][np.isfinite(d['scores'])] for d in all_scores]))
        L = independent_threshold(low_blocks,lc,.95)
        e.require(L==r['L'],'Independent L optimum differs')
        H = None
        if L is not None:
            hc = np.unique(np.concatenate([d['h'][np.isfinite(d['h']) & (d['h']>=L)] for d in all_scores]))
            H = independent_threshold(high_blocks,hc,.85)
        e.require(H==r['H'],'Independent H optimum differs')
        parent_rows.append(e.mean_rows(pm))
        if gm:
            gate_rows.append(e.mean_rows(gm))
    feasible = len(gate_rows)==3
    qualified = (feasible and np.mean([r['f1'] for r in gate_rows])>=np.mean([r['f1'] for r in parent_rows])
        and all(b['recall']-a['recall']>=-.05 for a,b in zip(parent_rows,gate_rows))
        and sum(b['precision']>a['precision'] for a,b in zip(parent_rows,gate_rows))>=2)
    parent_mean = e.mean_rows(parent_rows)
    gate_mean = e.mean_rows(gate_rows) if feasible else None
    choice = 'gate' if qualified and (gate_mean['precision'],gate_mean['f1'])>(parent_mean['precision'],parent_mean['f1']) else 'parent'
    e.require(bool(qualified)==s['gate_qualified'] and choice==s['selected'],'Source group/preselection differs')
    return dict(status='independently_verified_source',source_selected=choice,blocks=summaries,
        independent_histogram_thresholds_exact=True,independent_padded_max_exact=True,
        independent_scalar_source_confusions_exact=True,parent_full_aggregate_roundoff=roundoff,
        source_selection_sha256=e.sha256(run/'source_selection.json'),new_fits=0,optimizer_updates=0)


if __name__ == '__main__':
    target = run/f'independent_{a.phase}_review.json'
    e.require(not target.exists(),'Review receipt exists')
    identity = e.read_json(run/'identity.json')
    with e.isolation(run,'source_select' if a.phase=='source' else 'verify',identity['inputs']) as access:
        result = review_source()
        if a.phase in ('final','delivery'):
            result['final_verification'] = e.verify(run)
            result['status'] = 'independently_verified_source_and_development'
            result['freeze_sha256'] = e.sha256(run/'freeze.json')
        if a.phase=='delivery':
            import csv
            with (run/'ordinary_gate_seed42.csv').open(encoding='utf-8',newline='') as stream:
                rows = list(csv.DictReader(stream))
            frozen = e.npz(run/'predictions/gate_42.npz',['t','scores','h','state'])
            actual = dict(t=np.array([e.utc(r['timestamp']) for r in rows],np.int64),
                scores=np.array([float(r['score']) if r['score'] else np.nan for r in rows],np.float32),
                h=np.array([float(r['local_max_score']) if r['local_max_score'] else np.nan for r in rows],np.float32),
                state=np.array([-1 if not r['state'] else 1 if r['state']=='True' else 0 for r in rows],np.int8))
            for k,v in actual.items():
                np.testing.assert_array_equal(v,frozen[k])
            result['ordinary_cli_all_150750_rows_exact'] = len(rows)==150750
            e.require(result['ordinary_cli_all_150750_rows_exact'],'Ordinary CLI not full partition')
            metrics = e.read_json(run/'evaluation/metrics.json')['rows']
            s = e.read_json(run/'source_selection.json')
            summary = e.read_json(run/'evaluation/summary.json')
            base = [next(r for r in metrics if r['seed']==seed and r['arm']=='W50') for seed in e.SEEDS]
            new = [next(r for r in metrics if r['seed']==seed and r['arm']==s['selected']) for seed in e.SEEDS]
            independent_gates = [sum(r['precision'] for r in new)/3>=.08,all(r['recall']>=.85 for r in new),
                sum(b['f1']-a['f1'] for a,b in zip(base,new))/3+1e-12>=.02,
                sum(b['precision']>a['precision'] and b['f1']>a['f1'] for a,b in zip(base,new))>=2,
                all(b['f1']-a['f1']>=-.02 for a,b in zip(base,new)),sum(r['fp'] for r in new)<=.7*sum(r['fp'] for r in base),
                all(not r['degenerate'] for r in new)]
            e.require(list(summary['acceptance_gates'].values())==independent_gates,'Independent acceptance differs')
            with (run/'evaluation/false_positive_changes.csv').open(encoding='utf-8',newline='') as stream:
                changes = list(csv.DictReader(stream))
            comparisons = []
            for seed in e.SEEDS:
                gate = next(r for r in metrics if r['seed']==seed and r['arm']=='gate')
                for ref in ('W50','parent'):
                    previous = next(r for r in metrics if r['seed']==seed and r['arm']==ref)
                    added = sum(r['seed']==str(seed) and r['reference']==ref and r['change']=='added' for r in changes)
                    removed = sum(r['seed']==str(seed) and r['reference']==ref and r['change']=='removed' for r in changes)
                    e.require(gate['fp']-previous['fp']==added-removed,'FP added/removed table differs')
                    comparisons.append(dict(seed=seed,reference=ref,FP_added=added,FP_removed=removed,FP_delta=added-removed))
            result['independent_acceptance_gates_exact'] = True
            result['false_positive_change_counts'] = comparisons
        result['read_audit'] = dict(reads=sorted(access['reads']),forbidden=access['forbidden'])
    result['verification_script_sha256'] = e.sha256(Path(__file__))
    e.fresh_json(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('blocks','read_audit','parent_full_aggregate_roundoff')},ensure_ascii=False,indent=2))
