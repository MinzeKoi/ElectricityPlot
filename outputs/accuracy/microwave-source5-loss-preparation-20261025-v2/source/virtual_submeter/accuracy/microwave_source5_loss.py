"""Task12: one loss variable, frozen source pool, explicit additional-fit budget."""
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
from ..models.arrays import numeric
from ..models.center_mask_tcn import CenterMaskTCN, load_state_model, predict_windows
from . import microwave_source5_temporal_execution as temporal
from . import microwave_event_gate as gate
from .microwave_sampling import BatchStream

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(read_json(PACKAGE_ROOT/'workspace_identity.json')['root']) if (PACKAGE_ROOT/'workspace_identity.json').exists() else PACKAGE_ROOT
SEEDS = [42,1019,2026]
FAMILIES = ['old_point','old_gate','new_point','new_gate']
TRAIN_FIELDS = ['y','t','raw_index','event','house','x','mask','outer_observed','outer_fraction','left_stop','right_stop','mask_identity']
VAL_FIELDS = ['x','mask','outer_observed','outer_fraction','left_stop','right_stop','mask_identity','y','t','raw_index','house','event','house_name','block']
require, fresh, npz = gate.require, gate.fresh_json, gate.npz


@contextlib.contextmanager
def access_guard(run, phase, allowed=()):
    """Exact data allowlist, no final-house access; prepare cannot build a model."""
    state = dict(active=True,reads=set(),forbidden=[])
    permit = {str(Path(p).resolve()).lower() for p in allowed}
    local = str(Path(run).resolve()).lower()+os.sep
    def audit(event,args):
        if event != 'open' or not state['active'] or not isinstance(args[0],str): return
        mode = args[1]
        if mode is not None and any(c in mode for c in 'wax+'): return
        if mode is None and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT): return
        name = str(Path(args[0]).resolve()).lower()
        protected = name.endswith(('.npz','.csv','.pt')) or (not name.endswith(('.py','.md')) and any(s in name for s in ('evaluation/','evaluation\\','truth_once','details_v2')))
        forbidden = protected and name not in permit and not name.startswith(local)
        if phase in ('prepare','train','source_select') and protected and any(s in name for s in ('truth_once','evaluation','refit_17','predictions/source5_')):
            forbidden = True
        if forbidden:
            state['forbidden'].append(name)
            raise DataError('Phase forbids read: '+name)
        if protected: state['reads'].add(name)
    def blocked(*args,**kwargs): raise DataError('This phase forbids model construction, optimizers or backward')
    old_back,old_add,old_init = torch.Tensor.backward,torch.optim.Optimizer.add_param_group,CenterMaskTCN.__init__
    if phase != 'train':
        torch.Tensor.backward,torch.optim.Optimizer.add_param_group = blocked,blocked
    if phase == 'prepare': CenterMaskTCN.__init__ = blocked
    sys.addaudithook(audit)
    try: yield state
    finally:
        state['active'] = False
        torch.Tensor.backward,torch.optim.Optimizer.add_param_group,CenterMaskTCN.__init__ = old_back,old_add,old_init


def phase(name):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(run,*args,**kwargs):
            run = Path(run)
            identity = read_json(run/'identity.json')
            allowed = list(identity['source_inputs'])
            if name in ('freeze','evaluate','verify','report'):
                allowed += list(identity['development_unlabeled'])
            if name in ('evaluate','verify','report') and (run/'freeze.json').exists():
                allowed.append(identity['truth_original']['path'])
            with access_guard(run,name,allowed) as state:
                result = fn(run,*args,**kwargs)
            if name not in ('verify','report'):
                fresh(run/(name+'_access.json'),dict(reads=sorted(state['reads']),forbidden=state['forbidden']))
            return result
        return wrapped
    return decorate


def parent_bindings(config):
    parent,old_gate = ((ROOT/config[k]).resolve() for k in ('parent_run','gate_run'))
    registration = read_json(parent/'registration.json')['artifacts']
    manifest = read_json(parent/'final_delivery_manifest.json')['artifacts']
    selected = read_json(parent/'source_selection.json')
    gs = read_json(old_gate/'source_selection.json')
    require(selected['source_only'] and selected['selected_arm']=='source5' and next(a for a in selected['arms'] if a['arm']=='source5')['eligible'],'Task10 source identity differs')
    require(gs['source_only'] and gs['gate_qualified'] and gs['selected']=='gate' and all(gs['protection'].values()),'Task11 source gate unqualified')
    bound = {}
    def bind(path,digest): bound[str(path.resolve())] = digest
    for name in ['train.npz','normalization.json','plan.json','support.json','source5_inputs.json','baseline_identity.json','baseline_settings.json']:
        bind(parent/name,registration[name])
    for name in ['source_selection.json','source_rows.json','freeze.json','registration.json','training_complete.json']:
        bind(parent/name,manifest[name])
    for i in (4,5): bind(parent/f'validation/block_{i}.npz',registration[f'validation/block_{i}.npz'])
    for seed in SEEDS:
        for name in [f'sampling_{seed}.npz',f'dry_exposure_{seed}.npz',f'initial_{seed}.pt']:
            bind(parent/name,registration[name])
        model = parent/f'models/source5/seed_{seed}'
        meta = read_json(model/'model.json')
        for name in ['model.json','weights.pt','initial.pt']+[f'epoch_{meta["selected"]["epoch"]}_block_{i}.npz' for i in (4,5)]:
            rel = f'models/source5/seed_{seed}/{name}'
            bind(parent/rel,manifest[rel])
        recipe = old_gate/f'source_results/recipe_{seed}.json'
        require(sha256(recipe)==gs['artifacts'][f'recipe_{seed}.json'],'Task11 recipe binding differs')
        r = read_json(recipe)
        require(r == next(r for r in gs['recipes'] if r['seed']==seed),'Task11 recipe/selection differ')
        require(r['eligible'] and r['radius_samples']==15 and r['parent_point_threshold']==meta['decision_threshold_probability'] and Path(r['parent']).resolve()==model.resolve(),'Old gate parent differs')
        require(r['algorithm_sha256']==sha256(Path(gate.__file__)),'Task11 algorithm changed')
        for name,digest in r['parent_hashes'].items(): require(digest==bound[str((model/name).resolve())],'Task11 weights differ')
        bind(recipe,gs['artifacts'][f'recipe_{seed}.json'])
    bind(old_gate/'source_selection.json',sha256(old_gate/'source_selection.json'))
    # Read frozen identities only, never development predictions/truth or scores.
    gb = read_json(old_gate/'freeze.json')
    require(gb['source_selection_sha256']==sha256(old_gate/'source_selection.json'),'Task11 source is not sealed')
    bind(old_gate/'freeze.json',sha256(old_gate/'freeze.json'))
    src = read_json(parent/'source5_inputs.json')
    for item in src.values(): bind(Path(item['path']),item['sha256'])
    baseline = read_json(parent/'baseline_identity.json')
    base = Path(baseline['reference_run'])
    dev = {str(base/'protocol'/baseline['development_partition']['artifacts']['aggregate']['path']):baseline['development_partition']['artifacts']['aggregate']['sha256']}
    for seed in SEEDS:
        key = f'seed_{seed}/{gate.PARTITION}'
        for group in ('reference/predictions','predictions/W50'):
            dev[str(base/group/key)] = baseline['artifacts'][group][key]
        for family in ('source5',):
            rel = f'predictions/{family}_{seed}.npz'
            dev[str(parent/rel)] = manifest[rel]
        for family in ('parent','gate'):
            rel = f'predictions/{family}_{seed}.npz'
            dev[str(old_gate/rel)] = gb['artifacts'][rel]
    truth = baseline['development_partition']['artifacts']['microwave']
    return dict(parent=str(parent),old_gate=str(old_gate),source_inputs=bound,source=src,
                development_unlabeled=dev,truth_original=dict(path=str(base/'protocol'/truth['path']),sha256=truth['sha256']),
                parent_manifest_sha256=sha256(parent/'final_delivery_manifest.json'),
                old_point_qualified=True,old_gate_qualified=True)


def check_paths(mapping):
    for path,digest in mapping.items(): require(sha256(Path(path))==digest,'Frozen input changed: '+path)


def validate_config(p):
    fixed = read_json(ROOT/'configs/microwave_source5_loss_20261025.json')
    require(p==fixed and p['authorized'] is False,'Default configuration must stay disabled and match preregistration')
    require(p['seeds']==SEEDS and p['maximum_new_fits']==3 and p['updates']==10584 and p['pos_weight']==1. and p['families']==FAMILIES,'Fixed matrix changed')


def prepare(run):
    run = Path(run)
    require(not run.exists(),'New preparation directory required')
    config = read_json(ROOT/'configs/microwave_source5_loss_20261025.json')
    validate_config(config)
    identity = parent_bindings(config)
    parent = Path(identity['parent'])
    with access_guard(run,'prepare',identity['source_inputs']) as access:
        check_paths(identity['source_inputs'])
        p = read_json(parent/'plan.json')
        require(all(p[k]==config[k] for k in ('seeds','checkpoints','steps_per_checkpoint','batch_size','positive_per_batch','learning_rate','cpu_threads','window_samples')),'Task10 fixed schedule differs')
        require(p['source_recipe']=='control_BCE50' and p['pos_weight']==50. and p['training_tail_gap_seconds']==1800,'Task10 recipe/date exception differs')
        require(p['house5_training_blocks']==read_json(temporal.f.PLAN)['house5_training_blocks'] and p['house5_validation_blocks']==read_json(temporal.f.PLAN)['house5_validation_blocks'],'Exact date exception changed')
        d = npz(parent/'train.npz',TRAIN_FIELDS)
        require([len(d['y']),int((d['y']>=200).sum())]==config['train_support'],'Training support differs')
        require(d['x'].shape==(336704,127) and np.array_equal(np.isfinite(d['x']),d['mask']),'Window/missing identity differs')
        require(set(np.unique(d['house']))=={0,1,2},'Training household identity differs')
        evidence = dict(pool_reuse='same frozen file; all rows/windows/masks/order unchanged',
            train_sha256=sha256(parent/'train.npz'),fields={k:gate.hash_array(d[k]) for k in TRAIN_FIELDS},
            support=config['train_support'],normalization_sha256=sha256(parent/'normalization.json'),seeds={})
        val = {f'block_{i}':npz(parent/f'validation/block_{i}.npz',VAL_FIELDS) for i in (4,5)}
        require([sum(len(v['y']) for v in val.values()),sum(int((v['y']>=200).sum()) for v in val.values())]==config['source_support'],'Source support differs')
        evidence['validation'] = {k:{field:gate.hash_array(v[field]) for field in VAL_FIELDS} for k,v in val.items()}
        for seed in SEEDS:
            ix = npz(parent/f'sampling_{seed}.npz',['indices'])['indices']
            require(ix.shape==(3528,256) and np.all((d['y'][ix]>=200).sum(1)==16),'Event quota differs')
            stream = BatchStream(d,'event',seed,p)
            for row in ix: np.testing.assert_array_equal(row,stream.next())
            expected = npz(parent/f'dry_exposure_{seed}.npz',['count'])['count']
            np.testing.assert_array_equal(np.bincount(ix.ravel(),minlength=len(d['y'])),expected)
            initial = torch.load(parent/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
            trained = torch.load(parent/f'models/source5/seed_{seed}/initial.pt',weights_only=True,map_location='cpu')
            require(initial.keys()==trained.keys() and all(torch.equal(initial[k],trained[k]) for k in initial),'Seed initialization differs')
            evidence['seeds'][str(seed)] = dict(indices_sha256=gate.hash_array(ix),sampling_file_sha256=sha256(parent/f'sampling_{seed}.npz'),initial_sha256=sha256(parent/f'initial_{seed}.pt'),event_queue_replayed_all_3528=True)
        run.mkdir(parents=True)
        shutil.copytree(ROOT/'src/virtual_submeter',run/'source/virtual_submeter',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(ROOT/'scripts/run_microwave_source5_loss.py',run/'run_microwave_source5_loss.py')
        shutil.copyfile(ROOT/'docs/MICROWAVE_SOURCE5_LOSS_PLAN_20261025.md',run/'protocol.md')
        shutil.copyfile(ROOT/'tests/test_microwave_source5_loss.py',run/'test_microwave_source5_loss.py')
        fresh(run/'workspace_identity.json',dict(root=str(ROOT)))
        fresh(run/'config.json',config)
        fresh(run/'identity.json',identity)
        fresh(run/'pool_identity.json',evidence)
        fresh(run/'parent_training_plan.json',p)
        fresh(run/'prepare_access.json',dict(reads=sorted(access['reads']),forbidden=access['forbidden'],house17_reads=0,new_fits=0,optimizer_updates=0,model_construction_blocked=True,optimizers_and_backward_blocked=True))
        fresh(run/'registration.json',dict(artifacts=inventory(run),authorized=False,maximum_new_fits=3,updates=10584))
    return dict(status='prepared_waiting_for_additional_human_3_fit_budget',new_fits=0,optimizer_updates=0)


def registered(run):
    reg = read_json(run/'registration.json')
    gate.check(run,reg['artifacts'])
    p,identity = read_json(run/'config.json'),read_json(run/'identity.json')
    validate_config(p)
    check_paths(identity['source_inputs'])
    return p,identity


def authorize(run,receipt):
    require(receipt is not None,'Additional 3-fit human budget is NOT authorized; no optimizer/model training started')
    a = read_json(receipt)
    require(a.get('authorized') is True and a.get('approval_origin')=='human' and a.get('additional_new_fits')==3
        and a.get('maximum_updates')==10584 and a.get('experiment')=='microwave_source5_loss_v1'
        and a.get('registration_sha256')==sha256(run/'registration.json')
        and a.get('human_authorization_quote','').strip() and a.get('seeds')==SEEDS,'Missing explicit additional human budget bound to this registration')
    return a


@phase('train')
def train(run,authorization=None):
    # This check precedes registration/data loading, model construction and optimizer creation.
    a = authorize(run,authorization)
    config,identity = registered(run)
    require(not (run/'training_started.json').exists() and not (run/'models').exists(),'No restart/retry permitted')
    fresh(run/'training_started.json',dict(authorization=a,maximum_new_fits=3,maximum_updates=10584))
    parent = Path(identity['parent'])
    d = npz(parent/'train.npz',TRAIN_FIELDS)
    blocks = {f'block_{i}':npz(parent/f'validation/block_{i}.npz',VAL_FIELDS) for i in (4,5)}
    p = read_json(run/'parent_training_plan.json')
    meta = read_json(parent/'normalization.json')
    rows = []
    for seed in SEEDS:
        indices = npz(parent/f'sampling_{seed}.npz',['indices'])['indices']
        initial = torch.load(parent/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
        rows.append(temporal.fit_one(run/f'models/unit_BCE/seed_{seed}',seed,p,d,blocks,meta,indices,initial,
            sha256(run/'registration.json'),pos_weight=1.,recipe='unit_BCE',arm='unit_BCE'))
    fresh(run/'new_point_rows.json',rows)
    fresh(run/'training_complete.json',dict(new_fits=3,updates=10584,control_retrains=0,artifacts=inventory(run/'models'),rows_sha256=sha256(run/'new_point_rows.json')))
    return dict(status='trained_three_unit_BCE_models',new_fits=3,updates=10584)


def training_guard(run):
    p,identity = registered(run)
    c = read_json(run/'training_complete.json')
    require(c['new_fits']==3 and c['updates']==10584 and c['control_retrains']==0 and c['rows_sha256']==sha256(run/'new_point_rows.json'),'Training budget/binding differs')
    gate.check(run/'models',c['artifacts'])
    return p,identity


def qualified(old,new):
    if [r['seed'] for r in old]!=SEEDS or [r['seed'] for r in new]!=SEEDS: return False
    return bool(all(not r['degenerate'] and r['source_gate_passed'] for r in new)
        and np.mean([r['f1'] for r in new])>=np.mean([r['f1'] for r in old])
        and all(b['recall']-a['recall']>=-.05 for a,b in zip(old,new))
        and sum(b['precision']>a['precision'] for a,b in zip(old,new))>=2)


def select_families(rows,eligibility):
    candidates = [a for a in FAMILIES if eligibility[a]]
    require(bool(candidates),'No source-qualified candidate')
    return max(candidates,key=lambda a:(gate.mean_rows(rows[a])['precision'],gate.mean_rows(rows[a])['f1'],-FAMILIES.index(a)))


def calibrate(blocks):
    """Exact Task11 L95 then H85 candidate sets and ranking, no new rule search."""
    low_blocks = {k:dict(values=d['p'][d['ix']],y=d['y']) for k,d in blocks.items()}
    candidates = np.unique(np.concatenate([d['p'][np.isfinite(d['p'])] for d in blocks.values()]))
    low,_ = gate.select_threshold(low_blocks,candidates,.95)
    if low is None: return None,None
    hc = np.unique(np.concatenate([d['h'][np.isfinite(d['h']) & (d['h']>=low)] for d in blocks.values()]))
    high_blocks = {k:dict(values=np.where(d['p'][d['ix']]>=low,d['h'][d['ix']],-1.),y=d['y']) for k,d in blocks.items()}
    high,_ = gate.select_threshold(high_blocks,hc,.85)
    return low,high


@phase('source_select')
def source_select(run):
    config,identity = training_guard(run)
    require(not (run/'source_selection.json').exists(),'Source already selected')
    parent,old_gate = Path(identity['parent']),Path(identity['old_gate'])
    old = read_json(old_gate/'source_selection.json')
    old_point = read_json(parent/'source_rows.json')['source5']
    old_gate_rows = [dict(**r,degenerate=False,source_gate_passed=True) for r in old['gate_rows']]
    rows = dict(old_point=old_point,old_gate=old_gate_rows,new_point=read_json(run/'new_point_rows.json'),new_gate=[])
    raw = gate.csv_rows(Path(identity['source']['aggregate']['path']),'refit_5')
    times = np.array([utc(v['timestamp']) for v in raw])
    recipes,block_rows = [],[]
    dest = run/'source_results'
    dest.mkdir(exist_ok=False)
    for seed in SEEDS:
        model = run/f'models/unit_BCE/seed_{seed}'
        meta = read_json(model/'model.json')
        scores = {}
        for i,lo,hi in temporal.f.VALIDATION:
            block = npz(parent/f'validation/block_{i}.npz',['t','y'])
            rr = [raw[j] for j in np.flatnonzero((times>=lo)&(times<hi))]
            t,p,_ = gate.infer(model,rr)
            ix = gate.align(t,block['t'])
            saved = npz(model/f'epoch_{meta["selected"]["epoch"]}_block_{i}.npz',['scores','t'])
            np.testing.assert_array_equal(saved['t'],block['t'])
            np.testing.assert_allclose(p[ix],saved['scores'],rtol=1e-5,atol=1e-7)
            np.testing.assert_array_equal(p[ix]>=meta['decision_threshold_probability'],saved['scores']>=meta['decision_threshold_probability'])
            scores[str(i)] = dict(t=t,p=p,h=gate.local_max(t,p),ix=ix,y=block['y']>=200)
        low,high = calibrate(scores)
        feasible = low is not None and high is not None
        metrics = []
        for key,d in scores.items():
            if feasible:
                m = gate.confusion(gate.gate_state(d['p'],d['h'],low,high)[d['ix']],d['y'])
                metrics.append(m)
                block_rows.append(dict(seed=seed,block=key,**m))
        if feasible: rows['new_gate'].append(dict(seed=seed,**gate.mean_rows(metrics),degenerate=any(m['degenerate'] for m in metrics),source_gate_passed=all(m['recall']>=.85 for m in metrics)))
        recipe = dict(seed=seed,seed_feasible=feasible,L=low,H=high,radius_samples=15,interval_seconds=8,parent=str(model),
            parent_hashes={k:sha256(model/k) for k in ('model.json','weights.pt')},algorithm_sha256=sha256(Path(gate.__file__)),
            reason=None if feasible else 'No feasible source L95/H85; no relaxation; point retained')
        recipes.append(recipe)
        fresh(dest/f'recipe_{seed}.json',recipe)
    eligibility = dict(old_point=identity['old_point_qualified'],old_gate=identity['old_gate_qualified'],
        new_point=qualified(rows['old_point'],rows['new_point']),new_gate=qualified(rows['old_gate'],rows['new_gate']))
    selected = select_families(rows,eligibility)
    fresh(run/'source_selection.json',dict(source_only=True,selected=selected,eligibility=eligibility,rows=rows,recipes=recipes,
        block_metrics=block_rows,artifacts=inventory(dest),training_complete_sha256=sha256(run/'training_complete.json')))
    return dict(selected=selected,eligibility=eligibility)


def source_guard(run):
    config,identity = training_guard(run)
    s = read_json(run/'source_selection.json')
    require(s['source_only'] and s['training_complete_sha256']==sha256(run/'training_complete.json') and s['selected']==select_families(s['rows'],s['eligibility']),'Source lock differs')
    gate.check(run/'source_results',s['artifacts'])
    return config,identity,s


def ordinary(run,family,seed,rows,diagnostic=False):
    config,identity,s = source_guard(run)
    require(family in FAMILIES and seed in SEEDS,'Unregistered inference request')
    require(diagnostic or s['eligibility'][family],'Unqualified family cannot be loaded for ordinary use')
    model = Path(identity['parent'])/f'models/source5/seed_{seed}' if family.startswith('old') else run/f'models/unit_BCE/seed_{seed}'
    t,p,meta = gate.infer(model,rows)
    if family.endswith('point'):
        state = np.where(np.isfinite(p),p>=meta['decision_threshold_probability'],-1).astype(np.int8)
    else:
        recipe = read_json(Path(identity['old_gate'])/f'source_results/recipe_{seed}.json') if family=='old_gate' else next(r for r in s['recipes'] if r['seed']==seed)
        require(recipe.get('seed_feasible',recipe.get('eligible',False)),'Infeasible gate; no fake thresholds')
        state = gate.gate_state(p,gate.local_max(t,p),recipe['L'],recipe['H'])
    return dict(t=t,scores=p,state=state)


def available(s):
    return [(a,seed) for a in FAMILIES for seed in SEEDS if a!='new_gate' or next(r for r in s['recipes'] if r['seed']==seed)['seed_feasible']]


@phase('freeze')
def freeze(run):
    config,identity,s = source_guard(run)
    check_paths(identity['development_unlabeled'])
    parent,old_gate = Path(identity['parent']),Path(identity['old_gate'])
    baseline = read_json(parent/'baseline_identity.json')
    base = Path(baseline['reference_run'])
    aggregate = base/'protocol'/baseline['development_partition']['artifacts']['aggregate']['path']
    rows = gate.csv_rows(aggregate,'refit_17')
    dest = run/'predictions'
    dest.mkdir(exist_ok=False)
    checks = []
    for seed in SEEDS:
        a = npz(base/'reference/predictions'/f'seed_{seed}'/gate.PARTITION,['stamps','raw'])
        w = npz(base/'predictions/W50'/f'seed_{seed}'/gate.PARTITION,['stamps','raw'])
        np.testing.assert_array_equal(a['stamps'],w['stamps'])
        np.testing.assert_array_equal(np.isfinite(a['raw']),np.isfinite(w['raw']))
        for arm,z in [('A',a),('W50',w)]: np.savez_compressed(dest/f'{arm}_{seed}.npz',t=z['stamps'],raw=z['raw'])
        for family,ss in available(s):
            if ss!=seed: continue
            z = ordinary(run,family,seed,rows,diagnostic=True)
            replay = ordinary(run,family,seed,rows,diagnostic=True)
            for key in ('t','scores','state'): np.testing.assert_array_equal(z[key],replay[key])
            np.testing.assert_array_equal(z['t'],a['stamps'])
            np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
            if family.startswith('old'):
                old = npz(old_gate/f'predictions/{"parent" if family=="old_point" else "gate"}_{seed}.npz',['t','scores','state','raw'])
                for key in ('t','scores','state'): np.testing.assert_array_equal(z[key],old[key])
                np.testing.assert_array_equal(old['raw'],a['raw'])
            np.savez_compressed(dest/f'{family}_{seed}.npz',**z,raw=a['raw'])
            checks.append(dict(family=family,seed=seed,ordinary_exact=True,raw_A_exact=True,original_support_exact=True))
    fresh(run/'inference_checks.json',checks)
    fresh(run/'freeze.json',dict(matrix=available(s),source_selection_sha256=sha256(run/'source_selection.json'),
        registration_sha256=sha256(run/'registration.json'),identity_sha256=sha256(run/'identity.json'),
        artifacts={'predictions/'+k:v for k,v in inventory(dest).items()}|{'inference_checks.json':sha256(run/'inference_checks.json')},truth_opened=False))
    return dict(status='all_available_predictions_and_A_W50_frozen',matrix=available(s))


def evaluation_guard(run):
    config,identity,s = source_guard(run)
    f = read_json(run/'freeze.json')
    require(f['matrix']==[list(v) for v in available(s)] and f['truth_opened'] is False,'Incomplete prediction freeze')
    for name in ('source_selection','registration','identity'):
        require(f[name+'_sha256']==sha256(run/(name+'.json')),'Freeze identity differs')
    expected = [f'predictions/{a}_{seed}.npz' for a,seed in available(s)]+[f'predictions/{a}_{seed}.npz' for a in ('A','W50') for seed in SEEDS]
    require(all(k in f['artifacts'] for k in expected),'Unbound candidate/control matrix')
    gate.check(run,f['artifacts'])
    checks = read_json(run/'inference_checks.json')
    require(len(checks)==len(available(s)) and all(c['ordinary_exact'] and c['raw_A_exact'] and c['original_support_exact'] for c in checks),'Ordinary inference verification incomplete')
    return config,identity,s


def score(run,truth):
    config,identity,s = evaluation_guard(run)
    rows,supports = [],[]
    for seed in SEEDS:
        a = npz(run/f'predictions/A_{seed}.npz',['t','raw'])
        w = npz(run/f'predictions/W50_{seed}.npz',['t','raw'])
        np.testing.assert_array_equal(a['t'],truth['t'])
        np.testing.assert_array_equal(a['t'],w['t'])
        mask = np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1])
        require(int(mask.sum())==137523,'Original W50 support differs')
        np.testing.assert_array_equal(mask,np.isfinite(truth['y'])&np.isfinite(a['raw'][:,1]))
        supports.append(dict(seed=seed,n=int(mask.sum()),mask_sha256=gate.hash_array(mask),t_sha256=gate.hash_array(a['t'][mask])))
        for arm,z in [('A',a),('W50',w)]: rows.append(dict(seed=seed,arm=arm,**gate.confusion((z['raw'][mask,1]>=200).astype(np.int8),truth['y'][mask]>=200)))
        for family,ss in available(s):
            if ss!=seed: continue
            z = npz(run/f'predictions/{family}_{seed}.npz',['t','raw','scores','state'])
            np.testing.assert_array_equal(z['t'],a['t'])
            np.testing.assert_array_equal(z['raw'],a['raw'])
            np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
            require((z['state'][mask]>=0).all(),'Lost W50 evaluation point')
            rows.append(dict(seed=seed,arm=family,**gate.confusion(z['state'][mask],truth['y'][mask]>=200)))
    return dict(rows=rows,supports=supports)


def summary(run,metrics):
    s = read_json(run/'source_selection.json')
    selected = s['selected']
    rows = metrics['rows']
    groups = {a:[next(r for r in rows if r['arm']==a and r['seed']==seed) for seed in SEEDS] for a in ('A','W50',selected)}
    base,new = groups['W50'],groups[selected]
    means = {a:gate.mean_rows(v)|dict(fp=float(np.mean([r['fp'] for r in v]))) for a,v in groups.items()}
    gates = dict(mean_P_at_least_008=means[selected]['precision']>=.08,
        each_R_at_least_085=all(r['recall']>=.85 for r in new),
        mean_F1_gain_at_least_002=means[selected]['f1']-means['W50']['f1']+1e-12>=.02,
        at_least_two_seed_P_F1_up=sum(b['precision']>a['precision'] and b['f1']>a['f1'] for a,b in zip(base,new))>=2,
        each_F1_drop_at_most_002=all(b['f1']-a['f1']>=-.02 for a,b in zip(base,new)),
        mean_FP_reduction_at_least_30_percent=means[selected]['fp']<=.7*means['W50']['fp'],
        no_new_all_on_or_off=all(not r['degenerate'] for r in new),support_137523_exact=all(v['n']==137523 for v in metrics['supports']))
    return dict(selected_before_development=selected,development_acceptance_passed=all(gates.values()),acceptance_gates=gates,
        formal_means=means,other_candidates_diagnostic_only=True,power_improvement_claim=False,default_changed=False,
        independent_new_house_validation=False,development_is_repeated_House17=True,new_fits=3,updates=10584)


@phase('evaluate')
def evaluate(run):
    config,identity,s = evaluation_guard(run)  # Must precede any truth hash/open.
    dest = run/'evaluation'
    dest.mkdir(exist_ok=False)
    fresh(dest/'started.json',dict(freeze_sha256=sha256(run/'freeze.json')))
    truth_info = identity['truth_original']
    require(sha256(Path(truth_info['path']))==truth_info['sha256'],'Original truth identity differs')
    with Path(truth_info['path']).open(encoding='utf-8',newline='') as stream:
        mapping = {utc(r['timestamp']):numeric(r['power_w']) for r in csv.DictReader(stream)}
    t = npz(run/'predictions/A_42.npz',['t'])['t']
    truth = dict(t=t,y=np.array([mapping.get(int(v),np.nan) for v in t],np.float64))
    np.savez_compressed(dest/'truth_once.npz',**truth)
    m = score(run,truth)
    fresh(dest/'metrics.json',m)
    fresh(dest/'summary.json',summary(run,m))
    gate.csv_save(dest/'confusions.csv',m['rows'])
    fresh(run/'evaluation_complete.json',dict(freeze_sha256=sha256(run/'freeze.json'),source_selection_sha256=sha256(run/'source_selection.json'),truth_csv_parses=1,artifacts=inventory(dest)))
    return summary(run,m)


@phase('verify')
def verify(run):
    # Preparation review never opens development or instantiates a model.
    if not (run/'training_started.json').exists():
        registered(run)
        return dict(status='prepared_only_verified',new_fits=0,updates=0,additional_budget_authorized=False)
    config,identity = training_guard(run)
    parent = Path(identity['parent'])
    blocks = {f'block_{i}':npz(parent/f'validation/block_{i}.npz',VAL_FIELDS) for i in (4,5)}
    replay_rows = []
    for seed in SEEDS:
        model = run/f'models/unit_BCE/seed_{seed}'
        meta,net = load_state_model(model)
        epochs = [{k:npz(model/f'epoch_{epoch}_{k}.npz',['scores'])['scores'] for k in blocks} for epoch in range(1,13)]
        chosen = temporal.choose(epochs,blocks,read_json(run/'parent_training_plan.json'))
        require(chosen==meta['selected'] and meta['recipe']=='unit_BCE' and sha256(model/f'epoch_{chosen["epoch"]}.pt')==meta['weights_sha256'],'Source point selection/weights differ')
        for k,b in blocks.items(): np.testing.assert_array_equal(predict_windows(meta,net,b['x']),epochs[chosen['epoch']-1][k])
        require(read_json(model/'progress.json')==dict(completed_updates=3528,attempted_updates=3528),'Update count differs')
        np.testing.assert_array_equal(npz(model/'exposure.npz',['count'])['count'],npz(parent/f'dry_exposure_{seed}.npz',['count'])['count'])
        ix = npz(parent/f'sampling_{seed}.npz',['indices'])['indices']
        with (model/'batches.csv').open(encoding='utf-8',newline='') as stream: batches = list(csv.DictReader(stream))
        require(len(batches)==3528 and all(r['indices_sha256']==temporal.array_hash(v) and int(r['positive'])==16 and int(r['negative'])==240 for r,v in zip(batches,ix)),'Actual batch sequence differs')
        replay_rows.append(temporal.parent.source_row(seed,chosen))
    require(replay_rows==read_json(run/'new_point_rows.json'),'Source rows differ')
    if (run/'source_selection.json').exists(): source_guard(run)
    if (run/'freeze.json').exists(): evaluation_guard(run)
    if (run/'evaluation_complete.json').exists():
        c = read_json(run/'evaluation_complete.json')
        require(c['freeze_sha256']==sha256(run/'freeze.json') and c['source_selection_sha256']==sha256(run/'source_selection.json') and c['truth_csv_parses']==1,'Evaluation binding differs')
        gate.check(run/'evaluation',c['artifacts'])
        truth = npz(run/'evaluation/truth_once.npz',['t','y'])
        m = score(run,truth)
        require(m==read_json(run/'evaluation/metrics.json') and summary(run,m)==read_json(run/'evaluation/summary.json'),'Evaluation replay differs')
        for r in m['rows']:
            z = npz(run/f'predictions/{r["arm"]}_{r["seed"]}.npz',['raw'] if r['arm'] in ('A','W50') else ['state'])
            w = npz(run/f'predictions/W50_{r["seed"]}.npz',['raw'])
            mask = np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1])
            pred = z['raw'][mask,1]>=200 if r['arm'] in ('A','W50') else z['state'][mask]
            counts = dict(tp=0,fp=0,fn=0,tn=0)
            for a,b in zip(pred,truth['y'][mask]>=200): counts['tp' if a and b else 'fp' if a else 'fn' if b else 'tn'] += 1
            require(all(r[k]==v for k,v in counts.items()),'Independent scalar confusion differs')
    return dict(status='verified',new_fits=3,updates=10584,control_retrains=0)


@phase('report')
def report(run):
    result = verify.__wrapped__(run)
    if not (run/'evaluation_complete.json').exists(): return result
    s = read_json(run/'evaluation/summary.json')
    path = run/'REPORT.zh.md'
    require(not path.exists(),'Report already exists')
    path.write_text('# House5 单变量损失对照\n\n'+
        f'来源预选：{s["selected_before_development"]}；正式开发验收：{s["development_acceptance_passed"]}。\n\n'+
        '3 新拟合 / 10584 更新；旧模型零重训。全部混淆矩阵见 evaluation/confusions.csv。\n\n'+
        'House17 为反复使用开发集；不声称独立泛化。功率和其他电器输出保持原 A，默认 UI 不切换。\n',encoding='utf-8')
    return s


def predict(run,family,seed,aggregate,output):
    run = Path(run)
    require(all(v is not None for v in (family,seed,aggregate,output)),'Family/seed/aggregate/output required')
    require(not Path(output).exists(),'Output exists')
    _,identity,_ = source_guard(run)
    with access_guard(run,'predict',[*identity['source_inputs'],Path(aggregate)]):
        with Path(aggregate).open(encoding='utf-8',newline='') as stream: rows = list(csv.DictReader(stream))
        z = ordinary(run,family,seed,rows)
        gate.csv_save(Path(output),[dict(timestamp=iso(int(t)),probability_on=None if not np.isfinite(p) else float(p),state=None if s<0 else bool(s)) for t,p,s in zip(z['t'],z['scores'],z['state'])])
    return dict(output=str(output),rows=len(z['t']),new_fits=0)
