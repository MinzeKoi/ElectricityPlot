"""Independent, disabled H95 preregistration. Inference only; never fits."""
import contextlib
import csv
import functools
import os
from pathlib import Path
import re
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch

from ..data.common import DataError, read_json, sha256, inventory, utc, iso
from ..models.arrays import aggregate_arrays
from . import microwave_event_gate as gate
from . import microwave_source5_loss as loss

ROOT = loss.ROOT
SEEDS = [42, 1019, 2026]
CONFIG = 'configs/microwave_recall_guard95_20261025.disabled.json'
PLAN = 'docs/MICROWAVE_RECALL_GUARD95_PLAN_20261025.md'
SCRIPT = 'scripts/run_microwave_recall_guard95.py'
TEST = 'tests/test_microwave_recall_guard95.py'
SV = 'source_validation_refit_5_0.npz'
FAMILIES = ['W50', 'guard95']
require, fresh, npz = gate.require, gate.fresh_json, gate.npz


@contextlib.contextmanager
def isolation(phase, allowed):
    """Exact numerical allowlist, including local files; no truth before freeze."""
    state = dict(active=True, reads=set(), forbidden=[])
    permit = {str(Path(p).resolve()).lower() for p in allowed}
    def audit(event, args):
        if event != 'open' or not state['active'] or not isinstance(args[0], str): return
        mode = args[1]
        if mode is not None and any(c in mode for c in 'wax+'): return
        if mode is None and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT): return
        name = str(Path(args[0]).resolve()).lower()
        numerical = name.endswith(('.npz','.csv','.pt','.npy','.parquet','.pkl','.h5','.hdf5','.arrow'))
        truth = 'truth_once' in name
        forbidden = (numerical or truth) and name not in permit
        if re.search(r'(?:house|refit)[_-](?:8|9|20)(?:[_/\\.]|$)', name): forbidden = True
        if numerical and phase in ('prepare','source_select','verify_preparation') and any(v in name for v in ('truth_once','frozen_test','refit_17','house_17','predictions/guard95_','predictions\\guard95_')): forbidden = True
        if numerical and 'frozen_test' in name and name.endswith('.csv') and 'aggregate' not in name: forbidden = True
        if forbidden:
            state['forbidden'].append(name)
            raise DataError('Phase forbids numerical read: '+name)
        if numerical or truth: state['reads'].add(name)
    def no_fit(*args, **kwargs): raise DataError('0fit/0update: optimizer and backward forbidden')
    sys.addaudithook(audit)
    with patch.object(torch.Tensor,'backward',no_fit), patch.object(torch.optim.Optimizer,'add_param_group',no_fit), patch.object(torch.autograd,'backward',no_fit), patch.object(torch.autograd,'grad',no_fit):
        try: yield state
        finally: state['active'] = False


def authorize(run, receipt):
    # Check before identity, numerical hashes, model loading or output creation.
    require(receipt is not None, 'Independent H95 rule needs explicit human approval; old fit approval/config is insufficient')
    a = read_json(Path(receipt))
    require(a.get('authorized') is True and a.get('approval_origin') == 'human'
        and a.get('experiment') == 'microwave_recall_guard95_v1'
        and a.get('registration_sha256') == sha256(Path(run)/'registration.json')
        and a.get('new_fits') == a.get('optimizer_updates') == 0
        and a.get('human_authorization_quote','').strip()
        and a.get('human_authorization_reference','').strip()
        and a.get('approved_rule') == 'fixed_L95_H95_radius15_W50_source_guard'
        and a.get('seeds') == SEEDS, 'Missing human H95 authorization bound to this registration')
    return a


def execution(name):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(run, authorization=None, *args, **kwargs):
            run = Path(run)
            a = authorize(run, authorization)
            identity = read_json(run/'identity.json')
            allowed = list(identity['source_inputs'])
            allowed += [run/p for p in read_json(run/'registration.json')['artifacts'] if p.endswith(('.npz','.pt','.csv'))]
            if name in ('freeze','evaluate','verify','report'):
                allowed += list(identity['development_unlabeled'])
            if name == 'freeze':
                allowed += [run/f'predictions/{arm}_{seed}.npz' for arm in ('A','W50','guard95') for seed in SEEDS]
            if name == 'evaluate': allowed.append(run/'evaluation/confusions.csv')
            if (run/'freeze.json').exists() and name in ('evaluate','verify','report'):
                allowed += [run/p for p in read_json(run/'freeze.json')['artifacts'] if p.endswith('.npz')]
                # No truth hash/open until matrix and all source seals verify.
                evaluation_guard(run)
                allowed.append(identity['truth_cache']['path'])
            if name == 'predict': allowed.append(kwargs['aggregate'])
            if name in ('verify','report') and (run/'evaluation_complete.json').exists():
                allowed += [run/'evaluation'/p for p in read_json(run/'evaluation_complete.json')['artifacts']]
            with isolation(name, allowed) as access:
                result = fn(run, *args, **kwargs)
            if name not in ('verify','report','predict'):
                fresh(run/(name+'_access.json'),dict(reads=sorted(access['reads']),forbidden=access['forbidden'],new_fits=0,optimizer_updates=0,authorization=a))
            return result
        return wrapped
    return decorate


def bindings(config):
    parent = (ROOT/config['parent_run']).resolve()
    identity = read_json(parent/'identity.json')
    temporal = Path(identity['parent'])
    baseline = read_json(temporal/'baseline_identity.json')
    base = Path(baseline['reference_run'])
    manifest = read_json(parent/'final_delivery_manifest.json')['artifacts']
    pmanifest = read_json(temporal/'final_delivery_manifest.json')['artifacts']
    selected = read_json(parent/'source_selection.json')
    require(selected['source_only'] and len(selected['recipes']) == 3, 'Task12 source matrix differs')
    bound = {}
    def bind(p, digest): bound[str(p.resolve())] = digest
    for n in ('source_selection.json','registration.json','training_complete.json','identity.json'):
        bind(parent/n,manifest[n])
    require(sha256(temporal/'final_delivery_manifest.json') == identity['parent_manifest_sha256'],'Task10 manifest differs from Task12 binding')
    for n in ('normalization.json','source5_inputs.json','baseline_identity.json'):
        bind(temporal/n,pmanifest[n])
    for i in (4,5): bind(temporal/f'validation/block_{i}.npz',pmanifest[f'validation/block_{i}.npz'])
    for item in identity['source'].values(): bind(Path(item['path']),item['sha256'])
    for seed in SEEDS:
        recipe = next(r for r in selected['recipes'] if r['seed'] == seed)
        model = parent/f'models/unit_BCE/seed_{seed}'
        meta = read_json(model/'model.json')
        require(recipe['seed_feasible'] and recipe['algorithm_sha256'] == sha256(Path(gate.__file__)), 'Original gate algorithm differs')
        require((ROOT/recipe['parent']).resolve() == model and meta['recipe'] == 'unit_BCE', 'Fixed unit_BCE parent differs')
        for n in ('model.json','weights.pt',f'epoch_{meta["selected"]["epoch"]}_block_4.npz',f'epoch_{meta["selected"]["epoch"]}_block_5.npz'):
            rel = f'models/unit_BCE/seed_{seed}/{n}'
            bind(model/n,manifest[rel])
        for n in ('model.json','weights.pt'):
            require(recipe['parent_hashes'][n] == manifest[f'models/unit_BCE/seed_{seed}/{n}'], 'Task12 recipe weights changed')
            bind(base/f'models/W50/seed_{seed}/{n}',baseline['artifacts']['models/W50'][f'seed_{seed}/{n}'])
        bind(base/f'predictions/W50/seed_{seed}/{SV}',baseline['artifacts']['predictions/W50'][f'seed_{seed}/{SV}'])
    dev = {p:h for p,h in identity['development_unlabeled'].items()
           if str(base).lower() in p.lower()}  # only A, W50 and aggregate
    require(len(dev) == 7, 'Development allowlist must be aggregate plus 3seed A/W50')
    # Read JSON identities only. No House17 cache is hashed or inspected here.
    return dict(parent=str(parent),temporal=str(temporal),baseline=str(base),source=identity['source'],source_inputs=bound,
        development_unlabeled=dev,truth_cache=dict(path=str(parent/'evaluation/truth_once.npz'),sha256=manifest['evaluation/truth_once.npz']),
        gate_algorithm_sha256=sha256(Path(gate.__file__)),parent_manifest_sha256=sha256(parent/'final_delivery_manifest.json'),
        original_L={str(r['seed']):r['L'] for r in selected['recipes']})


def source_blocks(identity):
    out = {}
    for i,lo,hi in loss.temporal.f.VALIDATION:
        z = npz(Path(identity['temporal'])/f'validation/block_{i}.npz',['t','y','house','house_name'])
        require(np.isfinite(z['y']).all() and (z['house'] == 2).all() and (z['house_name'] == 'refit_5').all(), 'Source label/house identity differs')
        require(((z['t'] >= lo) & (z['t'] < hi)).all() and (np.diff(z['t']) > 0).all(), 'Source UTC block differs')
        out[str(i)] = z
    require([sum(len(z['y']) for z in out.values()),sum(int((z['y']>=200).sum()) for z in out.values())] == [34351,201], 'Source support differs')
    return out


def aligned_w50(z, blocks, aggregate_t):
    np.testing.assert_array_equal(z['stamps'],aggregate_t)
    require(z['raw'].shape == (len(aggregate_t),3), 'Original W50 raw column identity differs')
    out = {}
    for key,b in blocks.items():
        ix = gate.align(z['stamps'],b['t'])
        p = z['raw'][ix,1]
        require(np.isfinite(p).all(), 'W50 misses original source center')
        out[key] = gate.confusion((p>=200).astype(np.int8),b['y']>=200)
    return out


def replay_source(identity):
    """Old source replay only: L95 and H85; NEVER calibrate real H95 here."""
    blocks = source_blocks(identity)
    rows = gate.csv_rows(Path(identity['source']['aggregate']['path']),'refit_5')
    times = np.array([utc(r['timestamp']) for r in rows])
    label_rows = gate.csv_rows(Path(identity['source']['microwave']['path']),'refit_5')
    from ..models.arrays import numeric
    labels = {utc(r['timestamp']):numeric(r['power_w']) for r in label_rows}
    for b in blocks.values(): np.testing.assert_array_equal(b['y'],[labels[int(t)] for t in b['t']])
    selected = read_json(Path(identity['parent'])/'source_selection.json')
    proof = dict(blocks={k:dict(n=len(b['y']),positive=int((b['y']>=200).sum()),t_sha256=gate.hash_array(b['t']),y_sha256=gate.hash_array(b['y']),house='refit_5') for k,b in blocks.items()},seeds={})
    norm = read_json(Path(identity['temporal'])/'normalization.json')
    for seed in SEEDS:
        model = Path(identity['parent'])/f'models/unit_BCE/seed_{seed}'
        meta = read_json(model/'model.json')
        require(all(meta[k] == norm[k] for k in norm),'Frozen normalization changed')
        scores = {}
        for i,lo,hi in loss.temporal.f.VALIDATION:
            b = blocks[str(i)]
            rr = [rows[j] for j in np.flatnonzero((times>=lo)&(times<hi))]
            t,p,_ = gate.infer(model,rr)
            ix = gate.align(t,b['t'])
            saved = npz(model/f'epoch_{meta["selected"]["epoch"]}_block_{i}.npz',['scores','t'])
            np.testing.assert_array_equal(saved['t'],b['t'])
            np.testing.assert_allclose(p[ix],saved['scores'],rtol=1e-5,atol=1e-7)
            np.testing.assert_array_equal(p[ix]>=meta['decision_threshold_probability'],saved['scores']>=meta['decision_threshold_probability'])
            scores[str(i)] = dict(t=t,p=p,h=gate.local_max(t,p),ix=ix,y=b['y']>=200)
        oldL,oldH = loss.calibrate(scores)
        old = next(r for r in selected['recipes'] if r['seed'] == seed)
        require(oldL == old['L'] and oldH == old['H'], 'Old L/H85 replay differs')
        wpath = Path(identity['baseline'])/f'predictions/W50/seed_{seed}/{SV}'
        # Explicitly never read truth or state fields from prediction cache.
        w = npz(wpath,['stamps','raw'])
        wm = read_json(Path(identity['baseline'])/f'models/W50/seed_{seed}/model.json')
        require(wm['checkpoint_metric'] == 'normalized_mae' and wm['settings']['active_loss_weight'] == 50. and wm['settings']['on_thresholds_w']['microwave'] == 200., 'Original W50 model identity differs')
        wb = aligned_w50(w,blocks,times)
        proof['seeds'][str(seed)] = dict(L=oldL,H85=oldH,old_thresholds_exact=True,epoch=meta['selected']['epoch'],
            W50_threshold_w=200,W50_epoch=wm['selected_epoch'],W50_checkpoint_metric=wm['checkpoint_metric'],W50_fields_read=['stamps','raw'],
            W50_source_sha256=sha256(wpath),W50_blocks=wb,W50_mean=gate.mean_rows(list(wb.values())),
            exact_source_centers=34351,scores={k:{f:gate.hash_array(d[f]) for f in ('t','p','h','ix','y')} for k,d in scores.items()})
    return proof


def prepare(run):
    run = Path(run)
    require(not run.exists(),'New preparation directory required')
    config = read_json(ROOT/CONFIG)
    require(config['authorized'] is False and config['new_fits'] == config['optimizer_updates'] == 0,'Preparation must be disabled/0fit')
    identity = bindings(config)
    with isolation('prepare',identity['source_inputs']) as access:
        loss.check_paths(identity['source_inputs'])
        evidence = replay_source(identity)
        run.mkdir(parents=True)
        shutil.copytree(ROOT/'src/virtual_submeter',run/'source/virtual_submeter',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for src,dst in ((SCRIPT,'run_microwave_recall_guard95.py'),(PLAN,'protocol.md'),(TEST,'test_microwave_recall_guard95.py')): shutil.copyfile(ROOT/src,run/dst)
        fresh(run/'workspace_identity.json',dict(root=str(ROOT)))
        fresh(run/'config.json',config)
        fresh(run/'identity.json',identity)
        fresh(run/'source_identity_replay.json',evidence)
        fresh(run/'authorization_template.disabled.json',dict(authorized=False,approval_origin='human',experiment=config['experiment'],registration_sha256=None,
            approved_rule='fixed_L95_H95_radius15_W50_source_guard',human_authorization_quote=None,human_authorization_reference=None,seeds=SEEDS,new_fits=0,optimizer_updates=0))
        fresh(run/'prepare_access.json',dict(reads=sorted(access['reads']),forbidden=access['forbidden'],house17_numerical_reads=0,new_H95_calibrations=0,new_fits=0,optimizer_updates=0))
        fresh(run/'registration.json',dict(artifacts=inventory(run),authorized=False,new_fits=0,optimizer_updates=0))
    return dict(status='prepared_waiting_for_explicit_human_H95_rule_approval',new_fits=0,optimizer_updates=0,registration_sha256=sha256(run/'registration.json'))


def registered(run):
    gate.check(run,read_json(run/'registration.json')['artifacts'])
    def py_files(root): return {k:v for k,v in inventory(root).items() if k.endswith('.py')}
    require(py_files(Path(__file__).resolve().parents[1]) == py_files(run/'source/virtual_submeter'),'Use registered executable source snapshot; live package changed')
    config,identity = read_json(run/'config.json'),read_json(run/'identity.json')
    require(config == read_json(ROOT/CONFIG) and config['authorized'] is False,'Disabled fixed config changed')
    require(identity['gate_algorithm_sha256'] == sha256(Path(gate.__file__)),'Original gate algorithm changed')
    loss.check_paths(identity['source_inputs'])
    return config,identity


def verify_preparation(run):
    run = Path(run)
    identity = read_json(run/'identity.json')
    with isolation('verify_preparation',identity['source_inputs']): registered(run)
    require(not any((run/p).exists() for p in ('source_selection.json','predictions','evaluation','admission.json')), 'Preparation-only audit refuses executed run')
    return dict(status='prepared_only_verified',new_fits=0,optimizer_updates=0,authorized=False,registration_sha256=sha256(run/'registration.json'))


def calibrate95(blocks, original_low):
    low_blocks = {k:dict(values=d['p'][d['ix']],y=d['y']) for k,d in blocks.items()}
    candidates = np.unique(np.concatenate([d['p'][np.isfinite(d['p'])] for d in blocks.values()]))
    low,_ = gate.select_threshold(low_blocks,candidates,.95)
    require(low == original_low,'Replayed L95 changed; cannot recalibrate a different L')
    if low is None: return None
    hc = np.unique(np.concatenate([d['h'][np.isfinite(d['h']) & (d['h']>=low)] for d in blocks.values()]))
    hb = {k:dict(values=np.where(d['p'][d['ix']]>=low,d['h'][d['ix']],-1.),y=d['y']) for k,d in blocks.items()}
    high,_ = gate.select_threshold(hb,hc,.95)
    return high


def choose(w50, candidate):
    eligible = loss.qualified(w50,candidate)
    # W50 wins exact ties. Task12 H85 never competes here.
    selected = 'W50'
    if eligible:
        w,c = gate.mean_rows(w50),gate.mean_rows(candidate)
        if (c['precision'],c['f1']) > (w['precision'],w['f1']): selected = 'guard95'
    return dict(selected=selected,guard95_eligible=eligible,source_rejection=None if eligible else 'source95/nondegenerate or fixed original W50 relative guard failed')


@execution('source_select')
def source_select(run):
    _,identity = registered(run)
    require(not (run/'source_selection.json').exists() and not (run/'admission.json').exists(),'Source selection is one-shot; no retry')
    fresh(run/'admission.json',dict(registration_sha256=sha256(run/'registration.json'),new_fits=0,optimizer_updates=0))
    blocks = source_blocks(identity)
    raw = gate.csv_rows(Path(identity['source']['aggregate']['path']),'refit_5')
    times = np.array([utc(r['timestamp']) for r in raw])
    proof = read_json(run/'source_identity_replay.json')
    recipes,wr,cr = [],[],[]
    for seed in SEEDS:
        scores = {}
        for i,lo,hi in loss.temporal.f.VALIDATION:
            b = blocks[str(i)]
            rr = [raw[j] for j in np.flatnonzero((times>=lo)&(times<hi))]
            t,p,_ = gate.infer(Path(identity['parent'])/f'models/unit_BCE/seed_{seed}',rr)
            scores[str(i)] = dict(t=t,p=p,h=gate.local_max(t,p),ix=gate.align(t,b['t']),y=b['y']>=200)
            require({f:gate.hash_array(scores[str(i)][f]) for f in ('t','p','h','ix','y')} == proof['seeds'][str(seed)]['scores'][str(i)],'Old source replay changed')
        low = identity['original_L'][str(seed)]
        high = calibrate95(scores,low)
        metrics = [dict(block=k,**gate.confusion(gate.gate_state(d['p'],d['h'],low,high)[d['ix']],d['y'])) for k,d in scores.items()] if high is not None else []
        recipes.append(dict(seed=seed,L=low,H=high,feasible=high is not None,blocks=metrics))
        wr.append(dict(seed=seed,**proof['seeds'][str(seed)]['W50_mean']))
        if metrics: cr.append(dict(seed=seed,**gate.mean_rows(metrics),degenerate=any(m['degenerate'] for m in metrics),source_gate_passed=all(m['recall']>=.95 for m in metrics)))
    choice = choose(wr,cr)
    fresh(run/'source_selection.json',dict(source_only=True,**choice,rows=dict(W50=wr,guard95=cr),recipes=recipes,
        registration_sha256=sha256(run/'registration.json'),algorithm_sha256=sha256(Path(__file__))))
    fresh(run/'source_complete.json',dict(source_selection_sha256=sha256(run/'source_selection.json'),registration_sha256=sha256(run/'registration.json')))
    return choice


def source_guard(run):
    config,identity = registered(run)
    s = read_json(run/'source_selection.json')
    complete = read_json(run/'source_complete.json')
    require(complete == dict(source_selection_sha256=sha256(run/'source_selection.json'),registration_sha256=sha256(run/'registration.json')),'Source completion seal differs')
    require(s['source_only'] and s['registration_sha256'] == sha256(run/'registration.json') and s['algorithm_sha256'] == sha256(Path(__file__)),'Source seal differs')
    require({k:s[k] for k in ('selected','guard95_eligible','source_rejection')} == choose(s['rows']['W50'],s['rows']['guard95']),'Source preselection differs')
    require([r['seed'] for r in s['recipes']] == SEEDS,'Source matrix incomplete')
    return config,identity,s


def ordinary(run, seed, rows):
    _,identity,s = source_guard(run)
    recipe = next(r for r in s['recipes'] if r['seed']==seed)
    require(recipe['feasible'],'No feasible H95; cannot fabricate candidate prediction')
    t,p,_ = gate.infer(Path(identity['parent'])/f'models/unit_BCE/seed_{seed}',rows)
    return dict(t=t,scores=p,state=gate.gate_state(p,gate.local_max(t,p),recipe['L'],recipe['H']))


def matrix(s):
    require(all(r['feasible'] for r in s['recipes']),'Infeasible candidate: W50 source fallback retained; no incomplete development evaluation')
    return [(a,seed) for a in ['A','W50','guard95'] for seed in SEEDS]


@execution('freeze')
def freeze(run):
    _,identity,s = source_guard(run)
    matrix(s)  # Must reject before reading development input.
    require(not (run/'predictions').exists(),'Prediction matrix is one-shot')
    loss.check_paths(identity['development_unlabeled'])
    base = Path(identity['baseline'])
    agg = next(Path(p) for p in identity['development_unlabeled'] if p.endswith('aggregate.csv'))
    rows = gate.csv_rows(agg,'refit_17')
    dest = run/'predictions'
    dest.mkdir()
    checks = []
    for seed in SEEDS:
        a = npz(base/f'reference/predictions/seed_{seed}/{gate.PARTITION}',['stamps','raw','corrected'])
        w = npz(base/f'predictions/W50/seed_{seed}/{gate.PARTITION}',['stamps','raw'])
        np.testing.assert_array_equal(a['stamps'],w['stamps'])
        np.testing.assert_array_equal(np.isfinite(a['raw']),np.isfinite(w['raw']))
        astates = np.where(np.isfinite(a['corrected']),a['corrected']>=np.array([2000,200,20]),-1).astype(np.int8)
        np.savez_compressed(dest/f'A_{seed}.npz',t=a['stamps'],raw=a['raw'],corrected=a['corrected'],states=astates)
        np.savez_compressed(dest/f'W50_{seed}.npz',t=w['stamps'],raw=w['raw'])
        for arm,expected in (('A',a),('W50',w)):
            saved = npz(dest/f'{arm}_{seed}.npz',['t','raw'])
            np.testing.assert_array_equal(saved['t'],expected['stamps'])
            np.testing.assert_array_equal(saved['raw'],expected['raw'])
        if all(r['feasible'] for r in s['recipes']):
            z = ordinary(run,seed,rows)
            replay = ordinary(run,seed,rows)
            for k in z: np.testing.assert_array_equal(z[k],replay[k])
            np.testing.assert_array_equal(z['t'],a['stamps'])
            np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
            other = astates.copy()
            other[:,1] = z['state']
            np.savez_compressed(dest/f'guard95_{seed}.npz',**z,raw=a['raw'],corrected=a['corrected'],other_states=other)
            saved = npz(dest/f'guard95_{seed}.npz',['t','scores','state','raw','corrected','other_states'])
            for k in z: np.testing.assert_array_equal(saved[k],replay[k])
            np.testing.assert_array_equal(saved['raw'],a['raw'])
            np.testing.assert_array_equal(saved['corrected'],a['corrected'])
            np.testing.assert_array_equal(saved['other_states'][:,[0,2]],astates[:,[0,2]])
            checks.append(dict(seed=seed,ordinary_exact=True,raw_A_exact=True,other_states_A_exact=True,support_exact=True))
    fresh(run/'inference_checks.json',dict(checks=checks))
    fresh(run/'freeze.json',dict(matrix=matrix(s),source_selection_sha256=sha256(run/'source_selection.json'),registration_sha256=sha256(run/'registration.json'),identity_sha256=sha256(run/'identity.json'),
        artifacts={'predictions/'+k:v for k,v in inventory(dest).items()}|{'inference_checks.json':sha256(run/'inference_checks.json')},truth_opened=False))
    return dict(status='frozen',matrix=matrix(s),source_selected=s['selected'])


def evaluation_guard(run):
    _,identity,s = source_guard(run)
    f = read_json(run/'freeze.json')
    require(f['matrix']==[list(v) for v in matrix(s)] and not f['truth_opened'],'Incomplete freeze matrix')
    for name in ('source_selection','registration','identity'): require(f[name+'_sha256']==sha256(run/(name+'.json')),'Freeze binding differs')
    expected = {f'predictions/{a}_{seed}.npz' for a,seed in matrix(s)}|{'inference_checks.json'}
    require(set(f['artifacts'])==expected,'Incomplete/extra frozen matrix')
    gate.check(run,f['artifacts'])
    checks = read_json(run/'inference_checks.json')['checks']
    candidate = all(r['feasible'] for r in s['recipes'])
    require(len(checks)==(3 if candidate else 0) and all(all(c[k] for k in ('ordinary_exact','raw_A_exact','other_states_A_exact','support_exact')) for c in checks),'Ordinary readback matrix incomplete')
    return identity,s


def score(run,truth):
    identity,s = evaluation_guard(run)
    result,supports = [],[]
    for seed in SEEDS:
        a = npz(run/f'predictions/A_{seed}.npz',['t','raw','corrected','states'])
        w = npz(run/f'predictions/W50_{seed}.npz',['t','raw'])
        np.testing.assert_array_equal(a['t'],truth['t'])
        np.testing.assert_array_equal(w['t'],a['t'])
        mask = np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1])
        np.testing.assert_array_equal(mask,np.isfinite(truth['y'])&np.isfinite(a['raw'][:,1]))
        require(mask.sum()==137523,'Original development support changed')
        supports.append(dict(seed=seed,n=int(mask.sum()),mask_sha256=gate.hash_array(mask),t_sha256=gate.hash_array(a['t'][mask])))
        for arm,z in (('A',a),('W50',w)): result.append(dict(seed=seed,arm=arm,**gate.confusion((z['raw'][mask,1]>=200).astype(np.int8),truth['y'][mask]>=200)))
        if ('guard95',seed) in matrix(s):
            z = npz(run/f'predictions/guard95_{seed}.npz',['t','raw','scores','state','corrected','other_states'])
            np.testing.assert_array_equal(z['t'],a['t'])
            np.testing.assert_array_equal(z['raw'],a['raw'])
            np.testing.assert_array_equal(z['corrected'],a['corrected'])
            np.testing.assert_array_equal(z['other_states'][:,[0,2]],a['states'][:,[0,2]])
            np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
            result.append(dict(seed=seed,arm='guard95',**gate.confusion(z['state'][mask],truth['y'][mask]>=200)))
    return dict(rows=result,supports=supports)


def summary(run,metrics):
    # Exact original formal gates, including the original W50 reference.
    out = loss.summary(run,metrics)
    out.update(new_fits=0,updates=0,source95_is_cross_house_guarantee=False)
    return out


@execution('evaluate')
def evaluate(run):
    identity,s = evaluation_guard(run)
    dest = run/'evaluation'
    dest.mkdir(exist_ok=False)
    fresh(dest/'started.json',dict(freeze_sha256=sha256(run/'freeze.json')))
    info = identity['truth_cache']
    require(sha256(Path(info['path']))==info['sha256'],'Verified Task12 truth cache changed')
    truth = npz(Path(info['path']),['t','y'])
    metrics = score(run,truth)
    fresh(dest/'metrics.json',metrics)
    fresh(dest/'summary.json',summary(run,metrics))
    gate.csv_save(dest/'confusions.csv',metrics['rows'])
    fresh(run/'evaluation_complete.json',dict(freeze_sha256=sha256(run/'freeze.json'),truth_cache_sha256=info['sha256'],original_truth_csv_reads=0,artifacts=inventory(dest)))
    return summary(run,metrics)


@execution('verify')
def verify(run):
    identity,s = evaluation_guard(run)
    complete = read_json(run/'evaluation_complete.json')
    require(complete['freeze_sha256']==sha256(run/'freeze.json') and complete['truth_cache_sha256']==identity['truth_cache']['sha256'] and complete['original_truth_csv_reads']==0,'Evaluation seal differs')
    # Evaluate files contain no per-point truth, only summaries/confusions.
    gate.check(run/'evaluation',complete['artifacts'])
    truth = npz(Path(identity['truth_cache']['path']),['t','y'])
    metrics = score(run,truth)
    require(metrics==read_json(run/'evaluation/metrics.json') and summary(run,metrics)==read_json(run/'evaluation/summary.json'),'Metric replay differs')
    return dict(status='verified',new_fits=0,optimizer_updates=0,default_changed=False)


@execution('report')
def report(run):
    _,_,selection = source_guard(run)
    if not all(r['feasible'] for r in selection['recipes']):
        require(selection['selected']=='W50' and selection['source_rejection'],'Infeasible source must retain W50 fallback')
        result = dict(status='source_only_rejected_no_development_evaluation',selected='W50',source_rejection=selection['source_rejection'],new_fits=0,optimizer_updates=0,development_labels_read=False)
        fresh(run/'source_only_report.json',result)
        return result
    verify.__wrapped__(run)
    s = read_json(run/'evaluation/summary.json')
    require(not (run/'REPORT.zh.md').exists(),'Report exists')
    fresh(run/'report.json',s)
    (run/'REPORT.zh.md').write_text('House5 H95 独立预登记试验\n\n来源预选：'+s['selected_before_development']+'；正式开发验收：'+str(s['development_acceptance_passed'])+'。\n\n0 fit / 0 update；House17反复用工程开发，不能声称盲测独立泛化或来源95保证跨户召回。原H85失败保留，默认UI不切换。\n',encoding='utf-8')
    return s


@execution('predict')
def predict(run, *, seed, aggregate, output):
    require(seed in SEEDS and not Path(output).exists(),'Registered seed/new output required')
    _,identity,s = source_guard(run)
    with Path(aggregate).open(encoding='utf-8',newline='') as stream: rows=list(csv.DictReader(stream))
    require(not any(r['household_id'] in ('refit_8','refit_9','refit_20') for r in rows),'Final houses forbidden')
    if s['selected'] == 'guard95':
        z = ordinary(run,seed,rows)
        output_rows = [dict(timestamp=iso(int(t)),probability_on=None if not np.isfinite(p) else float(p),state=None if st<0 else bool(st)) for t,p,st in zip(z['t'],z['scores'],z['state'])]
    else:
        from ..models.artifacts import load_model
        from ..models.prediction import predict_rows
        meta,model = load_model(Path(identity['baseline'])/f'models/W50/seed_{seed}')
        pred = predict_rows(meta,model,rows)
        output_rows = [dict(timestamp=r['timestamp'],raw_power_w=r['appliances']['microwave']['raw_power_w'],state=r['appliances']['microwave']['raw_state']) for r in pred]
    gate.csv_save(Path(output),output_rows)
    return dict(output=str(output),rows=len(output_rows),selected=s['selected'],new_fits=0)
