"""Single preregistered max gate over three frozen parents; inference only."""
import contextlib
import csv
import functools
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from ..data.common import DataError, read_json, write_json, sha256, inventory, utc, iso
from ..models.arrays import aggregate_arrays
from ..models.center_mask_tcn import load_state_model, context_at_complete_centers, predict_windows

ROOT = Path(__file__).resolve().parents[3]
SEEDS = [42, 1019, 2026]
PARTITION = 'frozen_test_refit_17_0.npz'


def require(ok, message):
    if not ok:
        raise DataError(message)


def fresh_json(path, value):
    require(not path.exists(), 'Refusing overwrite: '+str(path))
    write_json(path, value)


def check(root, artifacts):
    for name, digest in artifacts.items():
        require(sha256(Path(root)/name) == digest, 'Hash differs: '+name)


def npz(path, fields):
    # Never iterate/decompress unrequested fields, including any future truth.
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in fields}


def csv_rows(path, house):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require({r['household_id'] for r in rows} == {house}, 'Household alignment mismatch')
    t = np.array([utc(r['timestamp']) for r in rows], np.int64)
    require(np.all(np.diff(t) > 0) and np.all(t % 8 == 0), 'UTC grid mismatch')
    return rows


def csv_save(path, rows):
    rows = list(rows)
    require(not path.exists(), 'CSV exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


@contextlib.contextmanager
def isolation(run, phase, allowed=()):
    """Deny unregistered data reads, source/development leakage, and all fitting."""
    state = dict(active=True, reads=set(), forbidden=[])
    local = str(Path(run).resolve()).lower()+os.sep
    permit = {str(Path(v).resolve()).lower() for v in allowed}
    def audit(event, args):
        if event != 'open' or not state['active'] or not isinstance(args[0], str):
            return
        name = str(Path(args[0]).resolve()).lower()
        mode = args[1]
        if mode is not None and any(c in mode for c in 'wax+'):
            return
        if mode is None and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            return
        numerical = name.endswith(('.csv','.npz','.pt'))
        protected = numerical or (not name.endswith(('.py','.md')) and any(v in name for v in ('truth_once','evaluation/','evaluation\\','details_v2')))
        forbidden = protected and name not in permit and not name.startswith(local)
        if protected and phase in ('prepare','source_select') and any(v in name for v in ('truth_once','evaluation','frozen_test_refit_17','predictions/source5_')):
            forbidden = True
        if forbidden:
            state['forbidden'].append(name)
            raise DataError('Phase forbids data read: '+name)
        if protected:
            state['reads'].add(name)
    def no_fit(*args, **kwargs):
        raise DataError('Zero-fit experiment forbids optimizer creation and backward')
    old_backward = torch.Tensor.backward
    old_add = torch.optim.Optimizer.add_param_group
    torch.Tensor.backward = no_fit
    torch.optim.Optimizer.add_param_group = no_fit
    sys.addaudithook(audit)
    try:
        yield state
    finally:
        state['active'] = False
        torch.Tensor.backward = old_backward
        torch.optim.Optimizer.add_param_group = old_add


def phase(name):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(run):
            run = Path(run)
            identity = read_json(run/'identity.json') if (run/'identity.json').exists() else None
            allowed = []
            if identity:
                allowed = list(identity['inputs'])
                if name in ('freeze','evaluate','verify','report'):
                    allowed += list(identity['development_unlabeled'])
                if name in ('evaluate','verify','report'):
                    allowed.append(identity['truth_cache']['path'])
            with isolation(run, name, allowed) as access:
                result = fn(run)
            if name != 'verify':
                fresh_json(run/(name+'_access.json'), dict(reads=sorted(access['reads']), forbidden=access['forbidden'],
                    new_fits=0, optimizer_updates=0, optimizer_and_backward_blocked=True))
            return result
        return wrapped
    return decorate


def local_max(t, p, radius=15, block=None):
    """One symmetric max of ORIGINAL p, within finite contiguous segments only."""
    t, p = np.asarray(t), np.asarray(p)
    require(t.ndim == p.ndim == 1 and len(t) == len(p), 'Invalid score grid')
    require(radius == 15, 'Only the preregistered radius 15 is permitted')
    require(np.all(np.diff(t) > 0), 'Nonincreasing time')
    block = np.zeros(len(t), np.int64) if block is None else np.asarray(block)
    require(len(block) == len(t), 'Invalid block grid')
    finite = np.isfinite(p)
    h = np.full(len(t), np.nan, p.dtype if p.dtype.kind == 'f' else np.float64)
    start = 0
    while start < len(t):
        if not finite[start]:
            start += 1
            continue
        end = start+1
        while end < len(t) and finite[end] and t[end]-t[end-1] == 8 and block[end] == block[end-1]:
            end += 1
        # Slice views refer only to p; max values never feed subsequent maxima.
        values = p[start:end]
        out = h[start:end]
        out[:] = values
        for d in range(1, 16):
            if d >= len(values):
                break
            out[d:] = np.maximum(out[d:], values[:-d])
            out[:-d] = np.maximum(out[:-d], values[d:])
        start = end
    return h


def gate_state(p, h, low, high):
    require(0 <= low <= high <= 1, 'Invalid thresholds')
    p, h = np.asarray(p), np.asarray(h)
    require(np.array_equal(np.isfinite(p), np.isfinite(h)), 'Max support differs')
    return np.where(np.isfinite(p), (p >= low) & (h >= high), -1).astype(np.int8)


def confusion(state, y, t=None):
    state, y = np.asarray(state), np.asarray(y, bool)
    require(len(state) == len(y) and np.isin(state, [0,1]).all(), 'Lost evaluation support')
    on = state == 1
    tp, fp = int((on & y).sum()), int((on & ~y).sum())
    fn, tn = int((~on & y).sum()), int((~on & ~y).sum())
    r = dict(tp=tp, fp=fp, fn=fn, tn=tn, n=len(y), precision=tp/max(tp+fp,1),
        recall=tp/max(tp+fn,1), f1=2*tp/max(2*tp+fp+fn,1), fpr=fp/max(fp+tn,1),
        predicted_on=int(on.sum()), degenerate=int(on.sum()) in (0,len(y)))
    if t is not None:
        r['predicted_on_segments'] = int((on & np.r_[True, (~on[:-1]) | (np.diff(t) != 8)]).sum())
    return r


def exact_curve(values, y, candidates):
    """Includes ALL original evaluation points; low failures use finite -1."""
    require(np.isfinite(values).all(), 'Nonfinite evaluation score forbidden')
    y = np.asarray(y, bool)
    pos, neg = np.sort(values[y]), np.sort(values[~y])
    tp = len(pos)-np.searchsorted(pos, candidates, side='left')
    fp = len(neg)-np.searchsorted(neg, candidates, side='left')
    fn, tn = len(pos)-tp, len(neg)-fp
    precision = tp/np.maximum(tp+fp,1)
    recall = tp/max(len(pos),1)
    f1 = 2*tp/np.maximum(2*tp+fp+fn,1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1,
        nondegenerate=(tp+fp > 0) & (tp+fp < len(y)))


def select_threshold(blocks, candidates, minimum_recall):
    candidates = np.asarray(candidates)
    curves = {k: exact_curve(d['values'],d['y'],candidates) for k,d in blocks.items()}
    if not len(candidates):
        return None, curves
    good = np.logical_and.reduce([v['nondegenerate'] & (v['recall'] >= minimum_recall) for v in curves.values()])
    if not good.any():
        return None, curves
    ps = np.stack([v['precision'] for v in curves.values()])
    fs = np.stack([v['f1'] for v in curves.values()])
    ixs = np.flatnonzero(good)
    order = np.lexsort((candidates[ixs], fs.mean(0)[ixs], ps.mean(0)[ixs], ps.min(0)[ixs]))
    return float(candidates[ixs[order[-1]]]), curves


def mean_rows(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ('precision','recall','f1')}


def infer(parent, rows):
    meta, model = load_state_model(parent)
    t, a = aggregate_arrays(rows,8)
    x, centers = context_at_complete_centers(meta,t,a)
    p = np.full(len(t), np.nan,np.float32)
    p[centers] = predict_windows(meta,model,x)
    return t, p, meta


def load_recipe(path):
    r = read_json(path)
    require(r.get('schema') == 'event_gate_recipe_v1' and r.get('eligible') is True,
        'Unqualified recipe cannot be loaded')
    require(r['radius_samples'] == 15 and r['interval_seconds'] == 8 and r['score_semantics'] == 'uncalibrated sigmoid state score', 'Rule identity differs')
    require(0 <= r['L'] <= r['H'] <= 1, 'Invalid gate thresholds')
    parent = Path(r['parent'])
    check(parent, r['parent_hashes'])
    require(r['algorithm_sha256'] == sha256(Path(__file__)), 'Rule source changed')
    return r


def ordinary(recipe, rows):
    r = load_recipe(recipe)
    # This public inference API consumes aggregate rows only.
    t, p, meta = infer(Path(r['parent']), rows)
    h = local_max(t,p)
    return dict(t=t, scores=p, h=h, state=gate_state(p,h,r['L'],r['H']))


def predict(recipe, aggregate, output):
    require(recipe is not None and aggregate is not None and output is not None, 'Recipe, aggregate and output required')
    require(not Path(output).exists(), 'Output exists')
    with Path(aggregate).open(encoding='utf-8',newline='') as stream:
        rows = list(csv.DictReader(stream))
    with isolation(Path(output).parent, 'predict', [Path(aggregate), *[Path(load_recipe(recipe)['parent'])/k for k in ('model.json','weights.pt')]]):
        z = ordinary(recipe, rows)
    csv_save(Path(output), [dict(timestamp=iso(int(t)), score=float(p) if np.isfinite(p) else None,
        local_max_score=float(h) if np.isfinite(h) else None, state=None if s<0 else bool(s))
        for t,p,h,s in zip(z['t'],z['scores'],z['h'],z['state'])])
    return dict(output=str(output), rows=len(z['t']), new_fits=0)


def prepare(run):
    require(not run.exists(), 'New run directory required')
    config = read_json(ROOT/'configs/microwave_event_gate_20261025.json')
    parent = (ROOT/config['parent_run']).resolve()
    manifest = read_json(parent/'final_delivery_manifest.json')['artifacts']
    review = read_json(parent/'independent_final_verification.json')
    selection = read_json(parent/'source_selection.json')
    require(selection['source_only'] and selection['selected_arm'] == 'source5', 'Parent must be source-preselected source5')
    require(review['status'] == 'independently_reviewed_numeric_invariants' and review['ordinary_cli_timestamp_probability_state_exact']
        and review['development_confusions_independently_recomputed'] and review['real_new_starts'] == 3 and review['actual_updates'] == 10584, 'Task10 terminal review not passed')
    require(review['source_selection_sha256'] == sha256(parent/'source_selection.json') and review['freeze_sha256'] == sha256(parent/'freeze.json'), 'Parent final receipt binding differs')
    bound_names = ['independent_final_verification.json','source_selection.json','registration.json','freeze.json',
        'normalization.json','baseline_identity.json','source5_inputs.json','validation/block_4.npz','validation/block_5.npz']
    for seed in SEEDS:
        bound_names += [f'models/source5/seed_{seed}/{name}' for name in ('model.json','weights.pt')]
        meta = read_json(parent/f'models/source5/seed_{seed}/model.json')
        bound_names += [f'models/source5/seed_{seed}/epoch_{meta["selected"]["epoch"]}_block_{i}.npz' for i in (4,5)]
    bound = {n: manifest[n] for n in bound_names}
    source = read_json(parent/'source5_inputs.json')
    baseline = read_json(parent/'baseline_identity.json')
    b = Path(baseline['reference_run'])
    dev = {str(b/'protocol'/baseline['development_partition']['artifacts']['aggregate']['path']): baseline['development_partition']['artifacts']['aggregate']['sha256']}
    for seed in SEEDS:
        key = f'seed_{seed}/{PARTITION}'
        for group in ('reference/predictions','predictions/W50'):
            dev[str(b/group/key)] = baseline['artifacts'][group][key]
        n = f'predictions/source5_{seed}.npz'
        dev[str(parent/n)] = manifest[n]
    truth_path = parent/'evaluation/truth_once.npz'
    allowed = [parent/n for n in bound_names]+[Path(v['path']) for v in source.values()]
    run.mkdir(parents=True)
    with isolation(run,'prepare',allowed) as access:
        check(parent,bound)
        for v in source.values():
            require(sha256(Path(v['path'])) == v['sha256'], 'Source data changed')
        # Copy a self-contained executable snapshot. Existing directories never modified.
        shutil.copytree(ROOT/'src/virtual_submeter',run/'source/virtual_submeter',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(ROOT/'scripts/run_microwave_event_gate.py',run/'run_microwave_event_gate.py')
        shutil.copyfile(ROOT/'tests/test_microwave_event_gate.py',run/'test_microwave_event_gate.py')
        shutil.copyfile(ROOT/'docs/MICROWAVE_EVENT_GATE_PLAN_20261025.md',run/'protocol.md')
        fresh_json(run/'config.json', config)
        identity = dict(parent=str(parent), parent_artifacts=bound,
            parent_manifest_sha256=sha256(parent/'final_delivery_manifest.json'),
            inputs={str(parent/n):v for n,v in bound.items()} | {v['path']:v['sha256'] for v in source.values()},
            source=source, development_unlabeled=dev,
            truth_cache=dict(path=str(truth_path), sha256=manifest['evaluation/truth_once.npz']),
            source_parent_selection_sha256=sha256(parent/'source_selection.json'),
            plan_sha256=sha256(run/'protocol.md'), config_sha256=sha256(run/'config.json'),
            task10_terminal_review_passed=True, new_fits=0, optimizer_updates=0)
        fresh_json(run/'identity.json', identity)
        fresh_json(run/'prepare_access.json',dict(reads=sorted(access['reads']),forbidden=access['forbidden'],house17_numeric_reads=0,new_fits=0))
        fresh_json(run/'registration.json', dict(artifacts=inventory(run),new_fits=0,optimizer_updates=0))
    return dict(status='prepared_parent_and_algorithm_frozen',new_fits=0)


def registered(run):
    check(run,read_json(run/'registration.json')['artifacts'])
    identity = read_json(run/'identity.json')
    config = read_json(run/'config.json')
    require(config['seeds'] == SEEDS and config['radius_samples'] == 15 and config['new_fits'] == config['optimizer_updates'] == 0,
        'Protocol changed')
    check(Path(identity['parent']),identity['parent_artifacts'])
    for item in identity['source'].values():
        require(sha256(Path(item['path'])) == item['sha256'], 'Source data changed')
    return identity, config


def curve_save(path, candidates, curves, minimum_recall):
    csv_save(path, (dict(block=block, threshold=float(v), eligible=bool(c['nondegenerate'][i] and c['recall'][i]>=minimum_recall),
        **{k:float(c[k][i]) if k in ('precision','recall','f1') else int(c[k][i]) for k in ('tp','fp','fn','tn','precision','recall','f1')})
        for block,c in curves.items() for i,v in enumerate(candidates)))


def align(t, desired):
    t, desired = np.asarray(t), np.asarray(desired)
    idx = np.searchsorted(t,desired)
    require(np.all(idx < len(t)) and np.array_equal(t[np.minimum(idx,len(t)-1)],desired), 'Household/UTC support mismatch')
    return idx


@phase('source_select')
def source_select(run):
    identity, config = registered(run)
    require(not (run/'source').joinpath('selection_complete.json').exists() and not (run/'source_selection.json').exists(), 'Source already selected')
    parent = Path(identity['parent'])
    rows = csv_rows(Path(identity['source']['aggregate']['path']), 'refit_5')
    all_t, all_a = aggregate_arrays(rows,8)
    blocks = {}
    for i,lo,hi in config['blocks']:
        block = npz(parent/f'validation/block_{i}.npz',['t','y','house'])
        require((block['house']==2).all(), 'Source household index differs')
        rr = [rows[j] for j in np.flatnonzero((all_t>=utc(lo)) & (all_t<utc(hi)))]
        blocks[str(i)] = dict(eval=block, rows=rr)
    require([sum(len(d['eval']['y']) for d in blocks.values()),sum(int((d['eval']['y']>=200).sum()) for d in blocks.values())] == config['source_support'], 'Source support differs')
    dest = run/'source_results'
    dest.mkdir(exist_ok=False)
    parent_rows, gate_rows, recipes, all_metrics = [],[],[],[]
    for seed in SEEDS:
        model_path = parent/f'models/source5/seed_{seed}'
        low_blocks, high_blocks, scores = {},{},{}
        pm = read_json(model_path/'model.json')
        for key,d in blocks.items():
            t,p,meta = infer(model_path,d['rows'])
            ix = align(t,d['eval']['t'])
            require(np.isfinite(p[ix]).all(), 'Parent misses source support')
            old = npz(model_path/f'epoch_{meta["selected"]["epoch"]}_block_{key}.npz',['t','scores'])
            # These are fixed-epoch source predictions, not a checkpoint search.
            np.testing.assert_array_equal(old['t'],d['eval']['t'])
            # Full aggregate includes extra unlabeled centers; a different final
            # CPU batch size can produce tiny roundoff versus label-only batches.
            # Parent point decisions must nevertheless remain EXACTLY identical.
            np.testing.assert_allclose(old['scores'],p[ix],rtol=1e-5,atol=1e-7)
            np.testing.assert_array_equal(old['scores']>=meta['decision_threshold_probability'],p[ix]>=meta['decision_threshold_probability'])
            h = local_max(t,p)
            y = d['eval']['y'] >= 200
            scores[key] = dict(t=t,p=p,h=h,ix=ix,y=y)
            low_blocks[key] = dict(values=p[ix],y=y)
            m = confusion((p[ix]>=meta['decision_threshold_probability']).astype(np.int8),y,t[ix])
            all_metrics.append(dict(seed=seed,arm='parent',block=key,**m))
        candidates = np.unique(np.concatenate([d['p'][np.isfinite(d['p'])] for d in scores.values()]))
        low, curves = select_threshold(low_blocks,candidates,.95)
        curve_save(dest/f'L_curve_{seed}.csv',candidates,curves,.95)
        high = None
        if low is not None:
            hc = np.unique(np.concatenate([d['h'][np.isfinite(d['h']) & (d['h']>=low)] for d in scores.values()]))
            for key,d in scores.items():
                high_blocks[key] = dict(values=np.where(d['p'][d['ix']]>=low,d['h'][d['ix']],-1.),y=d['y'])
            high, curves = select_threshold(high_blocks,hc,.85)
            curve_save(dest/f'H_curve_{seed}.csv',hc,curves,.85)
        eligible = low is not None and high is not None
        recipe = dict(schema='event_gate_recipe_v1',seed=seed,eligible=eligible,L=low,H=high,radius_samples=15,interval_seconds=8,
            parent=str(model_path),parent_hashes={k:sha256(model_path/k) for k in ('model.json','weights.pt')},
            parent_epoch=pm['selected']['epoch'],parent_point_threshold=pm['decision_threshold_probability'],
            score_semantics=config['score_semantics'],maximum_effective_context_samples=157,offline_future_context=True,
            algorithm_sha256=sha256(Path(__file__)), registration_sha256=sha256(run/'registration.json'),
            source_data_sha256={k:v['sha256'] for k,v in identity['source'].items()},
            missing='None; low failure remains off',segments='finite original score, continuous 8 seconds, within household and partition',
            reason=None if eligible else 'No feasible L with per-block recall >=0.95' if low is None else 'No feasible H with per-block recall >=0.85')
        for key,d in scores.items():
            state = gate_state(d['p'],d['h'],low,high) if eligible else np.full(len(d['p']),-1,np.int8)
            np.savez_compressed(dest/f'{seed}_block_{key}.npz',t=d['t'],scores=d['p'],h=d['h'],state=state,eval_t=d['t'][d['ix']])
            if eligible:
                all_metrics.append(dict(seed=seed,arm='gate',block=key,**confusion(state[d['ix']],d['y'],d['t'][d['ix']])))
        parent_rows.append(dict(seed=seed,**mean_rows([m for m in all_metrics if m['seed']==seed and m['arm']=='parent'])))
        if eligible:
            gate_rows.append(dict(seed=seed,**mean_rows([m for m in all_metrics if m['seed']==seed and m['arm']=='gate'])))
        recipes.append(recipe)
    all_eligible = all(r['eligible'] for r in recipes)
    protection = dict(all_seeds_feasible=all_eligible,
        mean_f1_not_lower=all_eligible and np.mean([r['f1'] for r in gate_rows])>=np.mean([r['f1'] for r in parent_rows]),
        each_seed_recall_drop_at_most_005=all_eligible and all(b['recall']-a['recall']>=-.05 for a,b in zip(parent_rows,gate_rows)),
        at_least_two_seed_precision_up=all_eligible and sum(b['precision']>a['precision'] for a,b in zip(parent_rows,gate_rows))>=2)
    protection = {k:bool(v) for k,v in protection.items()}
    qualified = bool(all(protection.values()))
    parent_mean = mean_rows(parent_rows)
    gate_mean = mean_rows(gate_rows) if all_eligible else None
    selected = 'gate' if qualified and (gate_mean['precision'],gate_mean['f1'])>(parent_mean['precision'],parent_mean['f1']) else 'parent'
    for r in recipes:
        r['seed_feasible'] = r['eligible']
        r['eligible'] = bool(r['eligible'] and qualified)
        r['source_group_qualified'] = qualified
        if not qualified and r['reason'] is None:
            r['reason'] = 'Three-seed source protection failed; diagnostic candidate only'
        fresh_json(dest/f'recipe_{r["seed"]}.json',r)
    csv_save(dest/'block_metrics.csv',all_metrics)
    result = dict(selected=selected,source_only=True,gate_qualified=qualified,protection=protection,
        parent_rows=parent_rows,gate_rows=gate_rows,parent_mean=parent_mean,gate_mean=gate_mean,recipes=recipes,
        artifacts=inventory(dest),new_fits=0,optimizer_updates=0)
    fresh_json(run/'source_selection.json',result)
    return {k:result[k] for k in ('selected','gate_qualified','protection','parent_mean','gate_mean')}


def source_guard(run):
    identity, config = registered(run)
    s = read_json(run/'source_selection.json')
    check(run/'source_results',s['artifacts'])
    require(s['source_only'] and s['selected'] in ('parent','gate') and (s['selected']!='gate' or s['gate_qualified']), 'Invalid source preselection')
    return identity,config,s


@phase('freeze')
def freeze(run):
    identity,config,s = source_guard(run)
    dest = run/'predictions'
    dest.mkdir(exist_ok=False)
    parent = Path(identity['parent'])
    baseline = read_json(parent/'baseline_identity.json')
    base = Path(baseline['reference_run'])
    # No truth cache open or shape inspection occurs here.
    for path,digest in identity['development_unlabeled'].items():
        require(sha256(Path(path)) == digest, 'Development unlabeled input changed')
    aggregate = base/'protocol'/baseline['development_partition']['artifacts']['aggregate']['path']
    rows = csv_rows(aggregate,'refit_17')
    checks, matrix = [],[]
    for seed in SEEDS:
        r = next(r for r in s['recipes'] if r['seed']==seed)
        model = Path(r['parent'])
        t,p,meta = infer(model,rows)
        saved = npz(parent/f'predictions/source5_{seed}.npz',['t','scores','state','raw'])
        for key, value in [('t',t),('scores',p)]:
            np.testing.assert_array_equal(value,saved[key])
        parent_state = np.where(np.isfinite(p),p>=r['parent_point_threshold'],-1).astype(np.int8)
        np.testing.assert_array_equal(parent_state,saved['state'])
        h = local_max(t,p)
        np.savez_compressed(dest/f'parent_{seed}.npz',t=t,scores=p,h=h,state=parent_state,raw=saved['raw'])
        matrix.append(f'parent/{seed}')
        for group,folder in [('A','reference/predictions'),('W50','predictions/W50')]:
            ref = npz(base/folder/f'seed_{seed}'/PARTITION,['stamps','raw'])
            np.testing.assert_array_equal(t,ref['stamps'])
            np.testing.assert_array_equal(np.isfinite(p),np.isfinite(ref['raw'][:,1]))
            if group=='A':
                np.testing.assert_array_equal(saved['raw'],ref['raw'])
            np.savez_compressed(dest/f'{group}_{seed}.npz',t=t,raw=ref['raw'])
        # Full replay through the aggregate-only public loader when deployable;
        # infeasible candidates remain unavailable, never receive fake thresholds.
        if r['seed_feasible']:
            gate = gate_state(p,h,r['L'],r['H'])
            if r['eligible']:
                replay = ordinary(run/f'source_results/recipe_{seed}.json',rows)
            else:
                tt,pp,_ = infer(model,rows)
                replay = dict(t=tt,scores=pp,h=local_max(tt,pp),state=gate_state(pp,local_max(tt,pp),r['L'],r['H']))
            for key,value in [('t',t),('scores',p),('h',h),('state',gate)]:
                np.testing.assert_array_equal(value,replay[key])
            np.savez_compressed(dest/f'gate_{seed}.npz',t=t,scores=p,h=h,state=gate,raw=saved['raw'])
            matrix.append(f'gate/{seed}')
        else:
            # Public ordinary ruleloader must reject missing/unqualified recipe.
            try:
                load_recipe(run/f'source_results/recipe_{seed}.json')
            except DataError:
                pass
            else:
                raise DataError('Unqualified recipe accepted')
        # Replay parent with a separate complete aggregate inference as well.
        tt,pp,_ = infer(model,rows)
        np.testing.assert_array_equal(t,tt)
        np.testing.assert_array_equal(p,pp)
        checks.append(dict(seed=seed,parent_ordinary_exact=True,gate_ordinary_exact=bool(r['seed_feasible']),
            gate_available=bool(r['seed_feasible']),public_recipe_eligible=r['eligible'],raw_A_exact=True,
            original_W50_support_exact=True,parent_weights_sha256=sha256(model/'weights.pt'),
            finite_score_count=int(np.isfinite(p).sum())))
    fresh_json(run/'inference_checks.json',dict(checks=checks,house17_truth_opened=False))
    fresh_json(run/'freeze.json',dict(matrix=matrix,selected=s['selected'],
        source_selection_sha256=sha256(run/'source_selection.json'),registration_sha256=sha256(run/'registration.json'),
        identity_sha256=sha256(run/'identity.json'),house17_truth_opened=False,
        artifacts={'predictions/'+k:v for k,v in inventory(dest).items()} | {'inference_checks.json':sha256(run/'inference_checks.json')},
        new_fits=0,optimizer_updates=0))
    evaluation_guard(run)
    return dict(status='all_available_predictions_and_ordinary_replays_frozen',matrix=matrix,selected=s['selected'],new_fits=0)


def evaluation_guard(run):
    identity,config,s = source_guard(run)
    require((run/'freeze.json').exists(), 'Prediction matrix must be sealed before evaluation')
    f = read_json(run/'freeze.json')
    expected = [v for seed in SEEDS for v in ([f'parent/{seed}']+([f'gate/{seed}'] if next(r for r in s['recipes'] if r['seed']==seed)['seed_feasible'] else []))]
    require(f['matrix'] == expected and f['selected'] == s['selected'] and not f['house17_truth_opened'], 'Prediction matrix incomplete')
    require(f['source_selection_sha256'] == sha256(run/'source_selection.json') and f['registration_sha256'] == sha256(run/'registration.json')
        and f['identity_sha256'] == sha256(run/'identity.json'), 'Freeze binding changed')
    require(all(f'predictions/{arm}_{seed}.npz' in f['artifacts'] for arm,seed in [v.split('/') for v in expected]),'Unbound predictions')
    check(run,f['artifacts'])
    c = read_json(run/'inference_checks.json')['checks']
    require([v['seed'] for v in c] == SEEDS and all(v['parent_ordinary_exact'] and v['raw_A_exact'] and v['original_W50_support_exact']
        and (not v['gate_available'] or v['gate_ordinary_exact']) for v in c), 'Ordinary replay incomplete')
    return identity,config,s


def hash_array(a):
    import hashlib
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def score_frozen(run, truth):
    identity,config,s = evaluation_guard(run)
    rows,supports,changes,segment_rows = [],[],[],[]
    for seed in SEEDS:
        p = npz(run/f'predictions/parent_{seed}.npz',['t','state','scores','raw'])
        w = npz(run/f'predictions/W50_{seed}.npz',['t','raw'])
        a = npz(run/f'predictions/A_{seed}.npz',['t','raw'])
        np.testing.assert_array_equal(truth['t'],p['t'])
        for ref in (w,a):
            np.testing.assert_array_equal(p['t'],ref['t'])
        mask = np.isfinite(truth['y']) & np.isfinite(w['raw'][:,1])
        require(int(mask.sum()) == config['development_support'], 'Original 137523 support differs')
        np.testing.assert_array_equal(mask,np.isfinite(truth['y']) & np.isfinite(a['raw'][:,1]))
        require(np.isfinite(p['scores'][mask]).all() and (p['state'][mask]>=0).all(), 'Lost parent support')
        supports.append(dict(seed=seed,n=int(mask.sum()),mask_sha256=hash_array(mask),timestamps_sha256=hash_array(p['t'][mask])))
        y = truth['y'][mask]>=200
        states = dict(A=(a['raw'][mask,1]>=200).astype(np.int8),W50=(w['raw'][mask,1]>=200).astype(np.int8),parent=p['state'][mask])
        full = dict(A=np.where(np.isfinite(a['raw'][:,1]),a['raw'][:,1]>=200,-1).astype(np.int8),
            W50=np.where(np.isfinite(w['raw'][:,1]),w['raw'][:,1]>=200,-1).astype(np.int8),parent=p['state'])
        if (run/f'predictions/gate_{seed}.npz').exists():
            g = npz(run/f'predictions/gate_{seed}.npz',['t','state','raw'])
            np.testing.assert_array_equal(g['t'],p['t'])
            np.testing.assert_array_equal(g['raw'],a['raw'])
            require((g['state'][mask]>=0).all(), 'Gate loses original support')
            states['gate'],full['gate'] = g['state'][mask],g['state']
        for arm,state in states.items():
            rows.append(dict(seed=seed,arm=arm,**confusion(state,y,p['t'][mask])))
            on = full[arm] == 1
            indices = np.flatnonzero(on)
            runs = np.split(indices,np.flatnonzero((np.diff(indices)!=1) | (np.diff(p['t'][indices])!=8))+1)
            for ix in runs:
                if not len(ix):
                    continue
                evaluable = ix[mask[ix]]
                segment_rows.append(dict(seed=seed,arm=arm,start=iso(int(p['t'][ix[0]])),end_exclusive=iso(int(p['t'][ix[-1]]+8)),
                    samples=len(ix),evaluation_points=len(evaluable),false_positive_points=int((truth['y'][evaluable]<200).sum())))
        if 'gate' in states:
            ix = np.flatnonzero(mask)
            for reference in ('parent','W50'):
                off = ~y
                added = off & (states['gate']==1) & (states[reference]==0)
                removed = off & (states['gate']==0) & (states[reference]==1)
                for j in np.flatnonzero(added|removed):
                    changes.append(dict(seed=seed,reference=reference,timestamp=iso(int(p['t'][ix[j]])),
                        change='added' if added[j] else 'removed',parent_score=float(p['scores'][ix[j]])))
    return rows,supports,changes,segment_rows


def summarize(rows, selected):
    arms = sorted({r['arm'] for r in rows})
    groups = {a:[next(r for r in rows if r['arm']==a and r['seed']==seed) for seed in SEEDS] for a in arms
        if all(any(r['arm']==a and r['seed']==seed for r in rows) for seed in SEEDS)}
    require(selected in groups, 'Selected matrix incomplete')
    means = {a:mean_rows(v) | {'fp':float(np.mean([r['fp'] for r in v]))} for a,v in groups.items()}
    base,new = groups['W50'],groups[selected]
    gates = dict(mean_P_at_least_008=means[selected]['precision']>=.08,
        each_R_at_least_085=all(r['recall']>=.85 for r in new),
        mean_F1_gain_at_least_002=means[selected]['f1']-means['W50']['f1']+1e-12>=.02,
        at_least_two_seed_P_F1_up=sum(b['precision']>a['precision'] and b['f1']>a['f1'] for a,b in zip(base,new))>=2,
        each_F1_drop_at_most_002=all(b['f1']-a['f1']>=-.02 for a,b in zip(base,new)),
        FP_reduction_at_least_30_percent=sum(r['fp'] for r in new)<=.7*sum(r['fp'] for r in base),nondegenerate=all(not r['degenerate'] for r in new))
    deltas = {a:{ref:{k:means[a][k]-means[ref][k] for k in means[a]} for ref in ('W50','parent')} for a in means}
    return dict(selected_before_development=selected,development_acceptance_passed=bool(all(gates.values())),
        acceptance_gates=gates,means=means,deltas=deltas,new_fits=0,optimizer_updates=0,
        only_source_preselection_accepted=True,power_improvement_claim=False,default_changed=False,
        final_houses_accessed=False,independent_new_house_validation=False)


@phase('evaluate')
def evaluate(run):
    identity,config,s = evaluation_guard(run)  # BEFORE hashing/opening truth.
    dest = run/'evaluation'
    dest.mkdir(exist_ok=False)
    fresh_json(dest/'started.json',dict(freeze_sha256=sha256(run/'freeze.json')))
    cache = Path(identity['truth_cache']['path'])
    require(sha256(cache) == identity['truth_cache']['sha256'], 'Previously verified truth cache changed')
    truth = npz(cache,['t','y'])  # One development truth read after complete freeze.
    np.savez_compressed(dest/'truth_once.npz',**truth)
    rows,supports,changes,segments = score_frozen(run,truth)
    fresh_json(dest/'metrics.json',dict(rows=rows,supports=supports))
    summary = summarize(rows,s['selected'])
    fresh_json(dest/'summary.json',summary)
    csv_save(dest/'development_confusions.csv',rows)
    csv_save(dest/'false_positive_changes.csv',changes)
    csv_save(dest/'predicted_on_segments.csv',segments)
    fresh_json(run/'evaluation_complete.json',dict(freeze_sha256=sha256(run/'freeze.json'),source_selection_sha256=sha256(run/'source_selection.json'),
        truth_cache_parses=1,truth_cache_sha256=identity['truth_cache']['sha256'],artifacts=inventory(dest)))
    return summary


@phase('verify')
def verify(run):
    identity,config,s = evaluation_guard(run)
    c = read_json(run/'evaluation_complete.json')
    require(c['freeze_sha256']==sha256(run/'freeze.json') and c['source_selection_sha256']==sha256(run/'source_selection.json') and c['truth_cache_parses']==1, 'Evaluation binding differs')
    check(run/'evaluation',c['artifacts'])
    truth = npz(run/'evaluation/truth_once.npz',['t','y'])
    rows,supports,changes,segments = score_frozen(run,truth)
    require(dict(rows=rows,supports=supports) == read_json(run/'evaluation/metrics.json'), 'Confusion replay differs')
    require(summarize(rows,s['selected']) == read_json(run/'evaluation/summary.json'), 'Acceptance replay differs')
    # Independent scalar loop verifies every count, including low failures.
    for seed in SEEDS:
        w = npz(run/f'predictions/W50_{seed}.npz',['raw'])
        mask = np.isfinite(truth['y']) & np.isfinite(w['raw'][:,1])
        for arm in ('parent','gate'):
            path = run/f'predictions/{arm}_{seed}.npz'
            if not path.exists():
                continue
            z = npz(path,['state'])
            counts = dict(tp=0,fp=0,fn=0,tn=0)
            for pred,y in zip(z['state'][mask],truth['y'][mask]>=200):
                require(pred in (0,1),'Missing evaluation point')
                counts['tp' if pred and y else 'fp' if pred else 'fn' if y else 'tn'] += 1
            row = next(r for r in rows if r['seed']==seed and r['arm']==arm)
            require(all(row[k]==v for k,v in counts.items()), 'Scalar count differs')
    for name in ('prepare','source_select','freeze','evaluate'):
        access = read_json(run/(name+'_access.json'))
        require(not access['forbidden'] and access['new_fits']==0, 'Access audit failure')
    if (run/'delivery_manifest.json').exists():
        check(run,read_json(run/'delivery_manifest.json')['artifacts'])
    return dict(status='verified',new_fits=0,optimizer_updates=0,task10_models_retrained=0,
        original_137523_support_exact=True,scalar_confusions_exact=True,
        hashes_and_ordinary_replays_bound=True,source_preselection_unchanged=True)


@phase('report')
def report(run):
    # Avoid overwriting any prior report or completed artifact directory.
    require(not (run/'REPORT.zh.md').exists(), 'Report exists')
    receipt = verify(run)
    fresh_json(run/'verification.json',receipt)
    s = read_json(run/'source_selection.json')
    summary = read_json(run/'evaluation/summary.json')
    lines = ['# 微波炉固定事件门控试验（任务11）','',
        f'来源唯一预选：{s["selected"]}；门控来源资格：{s["gate_qualified"]}；相对W50整体开发验收：{summary["development_acceptance_passed"]}。',
        '','本任务0新拟合、0新训练、0优化更新；任务10的3个House5新增模型未被重训。只加载原冻结权重、epoch、归一化和point阈值。',
        '','|seed|L|H|seed可行|全组可部署资格|父epoch|父point阈值|','|---|---:|---:|---|---|---:|---:|']
    for r in s['recipes']:
        lines.append(f'|{r["seed"]}|{r["L"]}|{r["H"]}|{r["seed_feasible"]}|{r["eligible"]}|{r["parent_epoch"]}|{r["parent_point_threshold"]}|')
        if r['reason']:
            lines.append(f'\nseed {r["seed"]}：{r["reason"]}。')
    lines += ['','来源保护：'+str(s['protection']),
        '','|来源组|均P|均R|均F1|','|---|---:|---:|---:|']
    for arm,m in [('parent',s['parent_mean']),('gate',s['gate_mean'])]:
        if m:
            lines.append(f'|{arm}|{m["precision"]:.6f}|{m["recall"]:.6f}|{m["f1"]:.6f}|')
    lines += ['','|开发组|均P|均R|均F1|均FP|','|---|---:|---:|---:|---:|']
    for arm,m in summary['means'].items():
        lines.append(f'|{arm}|{m["precision"]:.6f}|{m["recall"]:.6f}|{m["f1"]:.6f}|{m["fp"]:.2f}|')
    lines += ['','验收只针对事前source预选，不按开发诊断切换。各项W50验收：'+str(summary['acceptance_gates']),
        '','开发对父及W50的逐项差值见evaluation/summary.json的deltas；逐seed P/R/F1、TP/FP/FN/TN、FPR和开机预测段数见evaluation/development_confusions.csv。',
        '来源逐块同样指标见source_results/block_metrics.csv；全部精确L/H阈值曲线在source_results。逐段开机预测见evaluation/predicted_on_segments.csv，误报增加/移除对父及W50逐点对照见evaluation/false_positive_changes.csv。',
        '','严格原C31来源34351/201点与开发每seed137523点；低于L的有效点始终为off、计入FN，标签未知中心仅作无标签max特征。原分数有限段连续8秒、单次半径15 max，不跨缺失、断裂、House5验证块；不插补、不递归传播。',
        '普通aggregate推理已完整重放与冻结父/门控逐点一致，原A功率及水壶/洗衣机输出逐点保持；无功率改善声明、无默认切换。未校准sigmoid状态分数不称置信概率。',
        '最大有效上下文157样本，约21分钟，仍含离线未来上下文；House5最后4日是历史复用来源验证，House17是已有开发工程验收，不是独立新户验证。最终8/9/20数据未访问。',
        '','候选无来源资格时，公共ruleloader明确拒绝recipe；其冻结开发结果只供诊断。不得部署失败recipe或依据开发结果重新选阈值。',
        '','复核命令（只读完整哈希、封存缓存与独立标量计数）：','```powershell',
        f'.venv-m2/Scripts/python.exe -B "{run}/run_microwave_event_gate.py" verify --run "{run}"','```',
        '','合格recipe普通预测命令（输入只有aggregate；输出路径必须未存在）：','```powershell',
        f'.venv-m2/Scripts/python.exe -B "{run}/run_microwave_event_gate.py" predict --run "{run}" --recipe "{run}/source_results/recipe_42.json" --aggregate "<aggregate.csv>" --output "<新输出.csv>"','```',
        '','本试验的算法和阈值未根据House17改变。终态确认工具意外返回任务10已公开汇总开发指标的审计记录见protocol.md；本任务source阶段数值输入与真值隔离由访问审计记录。']
    (run/'REPORT.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Access receipt is written by phase wrapper after this function returns.
    return dict(report=str(run/'REPORT.zh.md'),**receipt,acceptance_passed=summary['development_acceptance_passed'])
