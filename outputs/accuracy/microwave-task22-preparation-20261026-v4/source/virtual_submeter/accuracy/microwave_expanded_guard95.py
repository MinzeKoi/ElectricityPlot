"""Task18: fixed additional-source preparation; independent, disabled 3-fit gate.

Preparation never infers a candidate, calibrates, constructs an optimizer, or
opens development/final data. Execution requires a new registration-bound human
budget receipt before any data/model read. Old experiments remain immutable.
"""
import contextlib
import csv
import functools
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

from ..data.common import DataError, read_json, sha256, inventory, utc, iso
from ..models.center_mask_tcn import CenterMaskTCN, expand_context
from . import microwave_recall_guard95 as control
from . import microwave_source5_loss as loss
from . import microwave_event_gate as gate
from .microwave_sampling import BatchStream

ROOT = loss.ROOT
SEEDS = [42, 1019, 2026]
ARMS = ['A', 'W50', 'control_guard95', 'expanded_guard95']
CONFIG = 'configs/microwave_expanded_guard95_20261026.disabled.json'
SCRIPT = 'scripts/run_microwave_expanded_guard95.py'
TEST = 'tests/test_microwave_expanded_guard95.py'
PLAN = 'docs/MICROWAVE_EXPANDED_GUARD95_TASK18_20261026.md'
AUDIT = ROOT/'outputs/accuracy/microwave-source-event-coverage-20261026-v1'
TEMPORAL = ROOT/'outputs/accuracy/microwave-source5-temporal-20261025-v2'
CONTROL = ROOT/'outputs/accuracy/microwave-recall-guard95-preparation-20261025-v2'
EXPERIMENT = 'microwave_expanded_guard95_task18_v1'
require, fresh, npz = gate.require, gate.fresh_json, gate.npz


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def isolation(phase, allowed, *, training=False):
    """Exact numeric allowlist, deny protected JSON too; no local-directory bypass."""
    state = dict(active=True, reads=set(), forbidden=[])
    permit = {str(Path(p).resolve()).lower() for p in allowed}
    def audit(event, args):
        if event != 'open' or not state['active'] or not isinstance(args[0], str): return
        mode = args[1]
        if mode is not None and any(c in mode for c in 'wax+'): return
        if mode is None and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT): return
        name = str(Path(args[0]).resolve()).lower()
        numerical = name.endswith(('.npz','.csv','.pt','.npy','.parquet','.pkl','.h5','.hdf5','.arrow'))
        protected = numerical or any(v in name for v in ('truth_once','evaluation/','evaluation\\','frozen_test','refit_17','house_17'))
        # Registered evaluation/*.py are source, never numerical evidence.
        source = name.endswith('.py') and name in permit
        forbidden = protected and not source and name not in permit
        if re.search(r'(?:house|refit)[_-]?(?:8|9|20)(?:[_/\\.]|$)', name): forbidden = True
        if phase in ('prepare','verify_preparation','train','source') and protected and not source and any(v in name for v in ('truth_once','evaluation/','evaluation\\','frozen_test','refit_17','house_17')): forbidden = True
        if 'frozen_test' in name and name.endswith('.csv') and 'aggregate' not in name: forbidden = True
        if forbidden:
            state['forbidden'].append(name)
            raise DataError('Task18 phase forbids read: '+name)
        if protected: state['reads'].add(name)
    def blocked(*a, **k): raise DataError('Task18 preparation/inference: 0fit/0optimizer update')
    sys.addaudithook(audit)
    with contextlib.ExitStack() as stack:
        if not training:
            for obj, name in ((torch.Tensor,'backward'),(torch.optim.Optimizer,'add_param_group'),(torch.autograd,'backward'),(torch.autograd,'grad')):
                stack.enter_context(patch.object(obj, name, blocked))
        try: yield state
        finally: state['active'] = False


def authorize(run, receipt):
    # FIRST gate: only receipt and registration may be opened before this passes.
    require(receipt is not None, 'Task18 needs NEW explicit human 3fit/19404update authorization; preparation and old receipts are insufficient')
    a = read_json(Path(receipt))
    return validate_receipt(run,a)


def validate_receipt(run,a):
    require(a.get('authorized') is True and a.get('approval_origin') == 'human'
        and a.get('experiment') == EXPERIMENT and a.get('additional_new_fits') == 3
        and a.get('maximum_updates') == 19404 and a.get('seeds') == SEEDS
        and a.get('registration_sha256') == sha256(Path(run)/'registration.json')
        and a.get('human_authorization_quote','').strip()
        and a.get('human_authorization_reference','').strip()
        and a.get('approved_rule') == 'expanded_guard95_fixed_L95_H95_radius15',
        'Task18 receipt must bind NEW human budget 3fit/19404updates to this registration')
    return a


def metadata_bindings():
    m = read_json(AUDIT/'delivery_manifest.json')['files']
    require(len(m) == 19, 'Task17 delivery must have exactly 19 bound artifacts')
    b = {str(AUDIT/n): v for n,v in m.items()}
    b[str(AUDIT/'delivery_manifest.json')] = sha256(AUDIT/'delivery_manifest.json')
    c = read_json(CONTROL/'identity.json')  # Source identity + opaque dev hashes only.
    b.update(c['source_inputs'])
    cm = read_json(CONTROL/'final_delivery_manifest.json')['artifacts']
    for n in ('source_selection.json','source_complete.json','registration.json','identity.json'):
        b[str(CONTROL/n)] = cm[n]
    tm = read_json(TEMPORAL/'final_delivery_manifest.json')['artifacts']
    for n in ['train.npz','normalization.json','plan.json']+[f'initial_{s}.pt' for s in SEEDS]+[f'validation/block_{i}.npz' for i in (4,5)]:
        b[str(TEMPORAL/n)] = tm[n]
    baseline=read_json(TEMPORAL/'baseline_identity.json')
    for group in ('reference/models','models/W50'):
        for n,digest in baseline['artifacts'][group].items(): b[str(Path(c['baseline'])/group/n)]=digest
    for h in ('2','19'):
        p = ROOT/f'data/raw/refit_complete_20261022/households/CLEAN_House{h}.csv'
        r = p.with_suffix('.receipt.json')
        b[str(r)] = sha256(r)
        b[str(p)] = read_json(r)['sha256']
    return dict(source_inputs=b, control=str(CONTROL), temporal=str(TEMPORAL),
        parent=c['parent'], baseline=c['baseline'], source=c['source'],
        development_unlabeled=c['development_unlabeled'],truth_cache=c['truth_cache'],
        task17_stat={str(AUDIT/n):dict(size=(AUDIT/n).stat().st_size,mtime_ns=(AUDIT/n).stat().st_mtime_ns) for n in list(m)+['delivery_manifest.json']})


def build_added(t, a, y, house, low, high, event_offset, audit):
    """Exact audited core mask; C127 stops at native gaps, missing runs and role ends."""
    require(house in (0,1) and len(t)==len(a)==len(y), 'Invalid extra-house arrays')
    c, ids, events = audit.legal_centers(t,a,y,low,high)
    # Real fixed roles must start/end off, so no cropped active event can straddle.
    require(not (np.isfinite(y[0]) and y[0]>=200) and not (np.isfinite(y[-1]) and y[-1]>=200), 'Active role boundary needs whole-event exclusion audit; stop without changing dates')
    ext = expand_context(t,a,c,low,high)
    np.testing.assert_array_equal(ext['x'][:,48:79],np.lib.stride_tricks.sliding_window_view(a,31)[c-15])
    return dict(**ext,y=y[c].copy(),t=t[c].copy(),raw_index=c,
        event=np.where(ids[c]>=0,ids[c]+event_offset,-1),house=np.full(len(c),house,np.int64)), events


def original_roles(d):
    r = np.full(len(d['y']), 'original14d', dtype='U24')
    for block,lo,hi in loss.temporal.f.TRAIN:
        r[(d['house']==2)&(d['t']>=lo)&(d['t']<hi)] = f'house5_prefix_block_{block}'
    require(not ((d['house']==2)&(r=='original14d')).any(), 'Unmapped original House5 role')
    return r


def exposure_rows(d, count, expected):
    """Complete house/date/event ledger; centers retain expected and actual counts separately."""
    groups = {}
    for i in range(len(count)):
        key=(int(d['house'][i]),str(d['source_role'][i]),int(d['t'][i]//86400),int(d['event'][i]),bool(d['y'][i]>=200))
        v=groups.setdefault(key,[0,0.,0,0]);v[0]+=1;v[1]+=float(expected[i]);v[2]+=int(count[i]);v[3]+=int(count[i]>0)
    return [dict(house=['refit_2','refit_19','refit_5'][h],source_role=role,utc_date=iso(day*86400)[:10],event=e,positive=pos,support=n,expected_presentations=ex,actual_presentations=actual,unique_centers_seen=seen)
        for (h,role,day,e,pos),(n,ex,actual,seen) in sorted(groups.items())]


def expected_exposure(d, plan):
    p = d['y']>=200
    expected = np.full(len(p),plan['checkpoints']*plan['steps_per_checkpoint']*240/int((~p).sum()),np.float64)
    days={}
    for i in np.flatnonzero(p):
        day=(int(d['house'][i]),int(d['t'][i]//86400))
        days.setdefault(day,{}).setdefault(int(d['event'][i]),[]).append(int(i))
    for events in days.values():
        for centers in events.values():
            expected[centers]=plan['checkpoints']*plan['steps_per_checkpoint']*16/(len(days)*len(events)*len(centers))
    return expected


def prepare(run):
    run=Path(run)
    require(not run.exists(), 'Task18 requires a new directory; completed/started runs cannot be overwritten')
    cfg=read_json(ROOT/CONFIG)
    require(cfg['authorized'] is False and cfg['maximum_updates']==19404 and cfg['candidate_bounds']==read_json(AUDIT/'config.json')['candidate_bounds'], 'Fixed disabled Task18 recipe differs')
    identity=metadata_bindings()
    allowed=list(identity['source_inputs'])+[ROOT/CONFIG,ROOT/SCRIPT,ROOT/TEST,ROOT/PLAN]
    allowed += list((ROOT/'src/virtual_submeter').rglob('*.py'))
    allowed += [run/'source'/p.relative_to(ROOT/'src') for p in (ROOT/'src/virtual_submeter').rglob('*.py')]
    allowed += [run/n for n in ['train.npz','normalization.json','training_plan.json','support.json','identity.json','config.json','prepare_access.json','workspace_identity.json','authorization_template.disabled.json']
        +[f'extra/house_{h}.npz' for h in ('2','19')]+[f'validation/block_{i}.npz' for i in (4,5)]
        +[f'{name}_{s}.{ext}' for s in SEEDS for name,ext in [('sampling','npz'),('exposure','npz'),('exposure','csv'),('initial','pt')]]]
    with isolation('prepare',allowed) as access:
        loss.check_paths(identity['source_inputs'])
        audit=load_module(AUDIT/'audit.py','task18_frozen_source_audit')
        old=npz(TEMPORAL/'train.npz',loss.TRAIN_FIELDS)
        require([len(old['y']),int((old['y']>=200).sum())]==[336704,2275], 'Original support differs')
        all_d=[old]; roles=[original_roles(old)]; evidence={}; offset={}
        run.mkdir(parents=True)
        for hi,h in enumerate(('2','19')):
            lo,high=cfg['candidate_bounds'][h]
            arrays,raw_evidence=audit.raw_selected(h,lo,high,'Appliance5' if h=='2' else 'Appliance4',cfg)
            t,a,y,ac,yc=arrays
            expected_raw=next(r for r in json.loads((AUDIT/'raw_range_evidence.json').read_text(encoding='utf-8')) if r['house']==h)
            require(json.loads(json.dumps(raw_evidence))==expected_raw,'Task17 selected raw identity differs')
            off=int(old['event'][old['house']==hi].max())+1;offset[h]=off
            added,events=build_added(t,a,y,hi,lo,high,off,audit)
            support=[len(added['y']),int((added['y']>=200).sum()),len(set(added['event'][added['y']>=200]))]
            require(support==cfg['added_support'][h], 'Additional source counts differ: STOP, no relaxed mask/date change')
            all_d.append(added);roles.append(np.full(len(added['y']),'extended14d',dtype='U24'))
            (run/'extra').mkdir(exist_ok=True)
            np.savez_compressed(run/f'extra/house_{h}.npz',t=t,aggregate=a,y=y,aggregate_coverage=ac,label_coverage=yc)
            evidence[h]=dict(support=support,event_offset=off,raw=raw_evidence,events=events,
                raw_index_semantics='zero-based row in extra/house_'+h+'.npz native resampled role grid; never an original-file/global index')
        d={k:np.concatenate([v[k] for v in all_d]) for k in old}
        d['source_role']=np.concatenate(roles)
        for k,v in old.items():
            require(d[k].dtype==v.dtype and d[k][:len(v)].tobytes()==v.tobytes(),'Original array bytes/order/dtype changed: '+k)
        require([len(d['y']),int((d['y']>=200).sum()),int((d['y']<200).sum())]==cfg['train_support'],'Final support differs')
        loss.temporal.f.validate_identity(d)
        require(len(set(zip(d['house'].tolist(),d['source_role'].tolist(),d['t'].tolist())))==len(d['y']), 'Window identity collision')
        np.savez_compressed(run/'train.npz',**d)
        shutil.copyfile(TEMPORAL/'normalization.json',run/'normalization.json')
        plan=read_json(TEMPORAL/'plan.json')|{k:cfg[k] for k in ('checkpoints','steps_per_checkpoint','pos_weight','learning_rate','batch_size','positive_per_batch','uniform_negative_per_batch','cpu_threads')}
        fresh(run/'training_plan.json',plan)
        (run/'validation').mkdir()
        for i in (4,5): shutil.copyfile(TEMPORAL/f'validation/block_{i}.npz',run/f'validation/block_{i}.npz')
        norm=read_json(run/'normalization.json')
        require(norm['input_log_mean']==5.370561599731445 and norm['input_log_std']==.8337093591690063,'Original normalization differs')
        expected=expected_exposure(d,plan); sampling={}
        for seed in SEEDS:
            stream=BatchStream(d,'event',seed,plan)
            ix=np.stack([stream.next() for _ in range(6468)])
            require(ix.shape==(6468,256) and np.all((d['y'][ix]>=200).sum(1)==16),'Sampler quota differs')
            count=np.bincount(ix.ravel(),minlength=len(d['y']))
            np.savez_compressed(run/f'sampling_{seed}.npz',indices=ix)
            np.savez_compressed(run/f'exposure_{seed}.npz',count=count,expected=expected,house=d['house'],source_role=d['source_role'],t=d['t'],event=d['event'],positive=d['y']>=200)
            gate.csv_save(run/f'exposure_{seed}.csv',exposure_rows(d,count,expected))
            initial=torch.load(TEMPORAL/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
            torch.manual_seed(seed); model=CenterMaskTCN()
            require(sum(v.numel() for v in model.parameters())==5329,'Network parameter count differs')
            require(initial.keys()==model.state_dict().keys() and all(torch.equal(initial[k],model.state_dict()[k]) for k in initial),'Original seed initial tensors differ')
            shutil.copyfile(TEMPORAL/f'initial_{seed}.pt',run/f'initial_{seed}.pt')
            sampling[str(seed)]=dict(updates=6468,positive_presentations=int(count[d['y']>=200].sum()),negative_presentations=int(count[d['y']<200].sum()),unique_positive_centers=int((count[d['y']>=200]>0).sum()),unique_negative_centers=int((count[d['y']<200]>0).sum()),indices_sha256=gate.hash_array(ix),initial_tensor_exact=True)
        fresh(run/'support.json',dict(original=[336704,2275],added=evidence,combined=cfg['train_support'],sampling=sampling,
            negative_expected_presentations=6468*240/612928,original_negative_expected_presentations=3528*240/(336704-2275),
            limitations='Only coarse negative-center exposure matched. Positive/event/date exposures cannot all match. Data and update budget change together; no independent data-expansion causal claim.',
            original_fields={k:gate.hash_array(v) for k,v in old.items()},source_roles='original14d uses unchanged original house-local raw_index/event; House5 prefix retains its original block mapping; extended14d raw_index indexes its extra native role grid; event offset avoids old/new house-local event collisions'))
        fresh(run/'identity.json',identity)
        fresh(run/'config.json',cfg)
        shutil.copytree(ROOT/'src/virtual_submeter',run/'source/virtual_submeter',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for src,dst in ((SCRIPT,'run_microwave_expanded_guard95.py'),(TEST,'test_microwave_expanded_guard95.py'),(PLAN,'protocol.md'),('scripts/verify_microwave_expanded_guard95.py','verify_microwave_expanded_guard95.py')): shutil.copyfile(ROOT/src,run/dst)
        fresh(run/'workspace_identity.json',dict(root=str(ROOT)))
        fresh(run/'authorization_template.disabled.json',dict(authorized=False,approval_origin='human',experiment=EXPERIMENT,additional_new_fits=3,maximum_updates=19404,seeds=SEEDS,registration_sha256=None,approved_rule='expanded_guard95_fixed_L95_H95_radius15',human_authorization_quote=None,human_authorization_reference=None))
        fresh(run/'prepare_access.json',dict(reads=sorted(access['reads']),forbidden=access['forbidden'],new_fits=0,optimizer_updates=0,new_candidate_inference=0,new_candidate_calibration=0,development_reads=0,final_house_reads=0))
        fresh(run/'registration.json',dict(experiment=EXPERIMENT,authorized=False,maximum_new_fits=3,maximum_updates=19404,artifacts=inventory(run)))
    return dict(status='prepared_waiting_NEW_human_3fit_19404update_budget',registration_sha256=sha256(run/'registration.json'),new_fits=0,optimizer_updates=0)


def allowed_files(run, phase, *, extra=()):
    identity=read_json(run/'identity.json');reg=read_json(run/'registration.json')
    allowed=list(identity['source_inputs'])+[run/n for n in reg['artifacts']]
    allowed+=list(Path(__file__).resolve().parents[1].rglob('*.py'))
    if phase=='train':
        for seed in SEEDS:
            dest=run/f'models/expanded/seed_{seed}'
            allowed += [dest/n for n in ('started.json','progress.json','initial.pt','weights.pt','model.json','exposure.npz','history.json','batches.csv','complete.json')]
            allowed += [dest/f'epoch_{epoch}.pt' for epoch in range(1,13)]
            allowed += [dest/f'epoch_{epoch}_block_{block}.npz' for epoch in range(1,13) for block in (4,5)]
    if (run/'training_complete.json').exists(): allowed += [run/'models'/n for n in read_json(run/'training_complete.json')['artifacts']]
    if phase in ('freeze','predict','evaluate','verify','report'): allowed+=list(identity['development_unlabeled'])
    if phase in ('freeze','evaluate','verify','report'):
        allowed += [run/f'predictions/{a}_{s}.npz' for a in ARMS for s in SEEDS]
        allowed += [run/f'ordinary/{a}_{s}.csv' for a in ARMS for s in SEEDS]
    if phase in ('verify','report') and (run/'evaluation_complete.json').exists(): allowed += [run/'evaluation'/n for n in read_json(run/'evaluation_complete.json')['artifacts']]
    return allowed+list(extra)


def registered(run):
    reg=read_json(run/'registration.json');gate.check(run,reg['artifacts'])
    if (run/'final_delivery_manifest.json').exists():
        delivery=read_json(run/'final_delivery_manifest.json')
        require(delivery['registration_sha256']==sha256(run/'registration.json'),'Delivery registration changed')
        gate.check(run,delivery['artifacts'])
    cfg=read_json(run/'config.json');identity=read_json(run/'identity.json')
    require(cfg==read_json(ROOT/CONFIG) and cfg['authorized'] is False,'Fixed disabled config changed')
    for n,v in reg['artifacts'].items():
        if n.startswith('source/') and n.endswith('.py'):
            require(sha256(Path(__file__).resolve().parents[1]/n.removeprefix('source/virtual_submeter/'))==v,'Use registered source snapshot')
    loss.check_paths(identity['source_inputs'])
    for p,st in identity['task17_stat'].items():
        require(Path(p).stat().st_size==st['size'] and Path(p).stat().st_mtime_ns==st['mtime_ns'],'Task17 immutable artifact mtime/size changed')
    return cfg,identity


def execution(phase):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(run, authorization=None, **kw):
            run=Path(run)
            a=authorize(run,authorization)  # before identity/data/model reads
            extra=[kw['aggregate']] if phase=='predict' else []
            allowed=allowed_files(run,phase,extra=extra)
            if phase=='evaluate':
                with isolation(phase,allowed): identity,_=evaluation_guard(run)
                allowed.append(identity['truth_cache']['path'])
                allowed += [run/'evaluation'/n for n in ('started.json','metrics.json','summary.json','confusions.csv','paired.csv','paired_fp_segments.csv','truth_once.npz')]
            with isolation(phase,allowed,training=phase=='train') as access:
                result=fn(run,authorization=authorization,**kw)
            if phase not in ('predict',): fresh(run/(phase+'_access.json'),dict(reads=sorted(access['reads']),forbidden=access['forbidden'],authorization=a))
            return result
        return wrapped
    return decorate


@execution('train')
def train(run, authorization=None):
    cfg,identity=registered(run)
    require(not any((run/n).exists() for n in ('training_started.json','models')),'Task18 one-shot budget: restart/retry forbidden')
    fresh(run/'training_started.json',dict(authorization=authorize(run,authorization),maximum_new_fits=3,maximum_updates=19404))
    d=npz(run/'train.npz',loss.TRAIN_FIELDS+['source_role'])
    blocks=loss.temporal.validation(run);meta=read_json(run/'normalization.json');plan=read_json(run/'training_plan.json')
    rows=[]
    for seed in SEEDS:
        ix=npz(run/f'sampling_{seed}.npz',['indices'])['indices']
        initial=torch.load(run/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
        rows.append(loss.temporal.fit_one(run/f'models/expanded/seed_{seed}',seed,plan,d,blocks,meta,ix,initial,sha256(run/'registration.json'),pos_weight=1.,recipe='unit_BCE',arm='expanded'))
    fresh(run/'training_complete.json',dict(new_fits=3,updates=19404,control_retrains=0,rows=rows,artifacts=inventory(run/'models')))
    return dict(status='trained',new_fits=3,updates=19404)


def training_guard(run):
    cfg,identity=registered(run)
    c=read_json(run/'training_complete.json')
    require((c['new_fits'],c['updates'],c['control_retrains'])==(3,19404,0),'Task18 training budget differs')
    gate.check(run/'models',c['artifacts'])
    for seed in SEEDS:
        dest=run/f'models/expanded/seed_{seed}'
        require(read_json(dest/'progress.json')['completed_updates']==6468,'Training updates incomplete')
        initial=torch.load(dest/'initial.pt',map_location='cpu',weights_only=True)
        expected=torch.load(run/f'initial_{seed}.pt',map_location='cpu',weights_only=True)
        require(all(torch.equal(initial[k],expected[k]) for k in expected),'Actual initialization changed')
        actual=npz(dest/'exposure.npz',['count'])['count']
        np.testing.assert_array_equal(actual,npz(run/f'exposure_{seed}.npz',['count'])['count'])
        plan=read_json(run/'training_plan.json');blocks=loss.temporal.validation(run)
        scores=[]
        for epoch in range(1,13):
            score={}
            for key,b in blocks.items():
                cached=npz(dest/f'epoch_{epoch}_{key}.npz',['t','scores'])
                np.testing.assert_array_equal(cached['t'],b['t']);score[key]=cached['scores']
            scores.append(score)
        chosen=loss.temporal.choose(scores,blocks,plan)
        meta=read_json(dest/'model.json')
        require(meta['selected']==chosen and sha256(dest/'weights.pt')==sha256(dest/f'epoch_{chosen["epoch"]}.pt'),'Original checkpoint selection differs')
    return cfg,identity


def choose(control_rows, candidate_rows):
    eligible=loss.qualified(control_rows,candidate_rows)
    chosen='control_guard95'
    if eligible and tuple(gate.mean_rows(candidate_rows)[k] for k in ('precision','f1'))>tuple(gate.mean_rows(control_rows)[k] for k in ('precision','f1')): chosen='expanded_guard95'
    return dict(selected=chosen,expanded_eligible=eligible,source_rejection=None if eligible else 'expanded来源逐块95召回/非退化或相对冻结control保护未通过')


@execution('source')
def source(run, authorization=None):
    _,identity=training_guard(run)
    require(not (run/'source_started.json').exists(),'Source selection is one-shot')
    fresh(run/'source_started.json',dict(registration_sha256=sha256(run/'registration.json')))
    blocks=control.source_blocks(identity)
    raw=gate.csv_rows(Path(identity['source']['aggregate']['path']),'refit_5');times=np.array([utc(r['timestamp']) for r in raw])
    recipes=[];cr=[]
    for seed in SEEDS:
        model=run/f'models/expanded/seed_{seed}';meta=read_json(model/'model.json');scores={}
        for i,lo,hi in loss.temporal.f.VALIDATION:
            b=blocks[str(i)];rr=[raw[j] for j in np.flatnonzero((times>=lo)&(times<hi))]
            t,p,_=gate.infer(model,rr);ix=gate.align(t,b['t'])
            cached=npz(model/f'epoch_{meta["selected"]["epoch"]}_block_{i}.npz',['t','scores'])
            np.testing.assert_array_equal(cached['t'],b['t'])
            np.testing.assert_allclose(p[ix],cached['scores'],rtol=1e-5,atol=1e-7)
            scores[str(i)]=dict(t=t,p=p,h=gate.local_max(t,p),ix=ix,y=b['y']>=200)
        low,_=gate.select_threshold({k:dict(values=d['p'][d['ix']],y=d['y']) for k,d in scores.items()},np.unique(np.concatenate([d['p'][np.isfinite(d['p'])] for d in scores.values()])),.95)
        high=control.calibrate95(scores,low) if low is not None else None
        bm=[dict(block=k,**gate.confusion(gate.gate_state(d['p'],d['h'],low,high)[d['ix']],d['y'])) for k,d in scores.items()] if high is not None else []
        recipes.append(dict(seed=seed,L=low,H=high,feasible=high is not None,blocks=bm))
        if bm: cr.append(dict(seed=seed,**gate.mean_rows(bm),degenerate=any(m['degenerate'] for m in bm),source_gate_passed=all(m['recall']>=.95 for m in bm)))
    frozen=read_json(Path(identity['control'])/'source_selection.json');wr=frozen['rows']['guard95']
    require(frozen['guard95_eligible'] and frozen['selected']=='guard95','Frozen control no longer source qualified')
    choice=choose(wr,cr)
    fresh(run/'source_selection.json',dict(source_only=True,**choice,rows=dict(control_guard95=wr,expanded_guard95=cr),recipes=recipes,control_recipes=frozen['recipes'],
        registration_sha256=sha256(run/'registration.json'),training_sha256=sha256(run/'training_complete.json'),algorithm_sha256=sha256(Path(__file__))))
    fresh(run/'source_complete.json',dict(source_selection_sha256=sha256(run/'source_selection.json')))
    return choice


def source_guard(run, *, require_eligible=False):
    cfg,identity=training_guard(run);s=read_json(run/'source_selection.json')
    require(read_json(run/'source_complete.json')['source_selection_sha256']==sha256(run/'source_selection.json'),'Source seal differs')
    require(s['source_only'] and s['registration_sha256']==sha256(run/'registration.json') and s['training_sha256']==sha256(run/'training_complete.json') and s['algorithm_sha256']==sha256(Path(__file__)),'Source binding changed')
    require({k:s[k] for k in ('selected','expanded_eligible','source_rejection')}==choose(s['rows']['control_guard95'],s['rows']['expanded_guard95']),'Source qualification/selection changed')
    require([r['seed'] for r in s['recipes']]==SEEDS,'Incomplete source matrix')
    frozen=read_json(Path(identity['control'])/'source_selection.json')
    require(s['control_recipes']==frozen['recipes'] and s['rows']['control_guard95']==frozen['rows']['guard95'],'Frozen source control changed')
    derived=[]
    for recipe in s['recipes']:
        if not recipe['feasible']:
            require(not recipe['blocks'],'Infeasible candidate contains fabricated source metrics');continue
        require(0<=recipe['L']<=recipe['H']<=1 and [str(b['block']) for b in recipe['blocks']]==['4','5'],'Fixed L95/H95 block identity differs')
        for b in recipe['blocks']:
            tp,fp,fn,tn=(b[k] for k in ('tp','fp','fn','tn'))
            old=next(r for r in s['control_recipes'] if r['seed']==recipe['seed'])
            reference=next(v for v in old['blocks'] if str(v['block'])==str(b['block']))
            require(b['n']==tp+fp+fn+tn==reference['n'] and tp+fn==reference['tp']+reference['fn']
                and b['degenerate']==(tp+fp in (0,b['n'])) and b['precision']==tp/max(tp+fp,1) and b['recall']==tp/max(tp+fn,1) and b['f1']==2*tp/max(2*tp+fp+fn,1),'Source confusion arithmetic/support differs')
        bm=recipe['blocks']
        derived.append(dict(seed=recipe['seed'],**gate.mean_rows(bm),degenerate=any(m['degenerate'] for m in bm),source_gate_passed=all(m['recall']>=.95 for m in bm)))
    require(derived==s['rows']['expanded_guard95'],'Source row/block metrics differ')
    if require_eligible: require(s['expanded_eligible'] and all(r['feasible'] for r in s['recipes']),'Rejected candidate is source-only: development forbidden')
    return cfg,identity,s


def infer_arm(run, arm, seed, rows, identity, selection):
    if arm in ('A','W50'):
        from ..models.artifacts import load_model
        from ..models.prediction import predict_rows
        path=Path(identity['baseline'])/('reference/models' if arm=='A' else 'models/W50')/f'seed_{seed}'
        meta,model=load_model(path);pred=predict_rows(meta,model,rows)
        raw=np.array([[r['appliances'][a]['raw_power_w'] if r['appliances'][a]['raw_power_w'] is not None else np.nan for a in ('kettle','microwave','washing_machine')] for r in pred])
        corrected=np.array([[r['appliances'][a]['power_w'] if r['appliances'][a]['power_w'] is not None else np.nan for a in ('kettle','microwave','washing_machine')] for r in pred])
        return dict(t=np.array([utc(r['timestamp']) for r in rows]),raw=raw,corrected=corrected)
    require(arm in ARMS and seed in SEEDS,'Unknown fixed arm/seed')
    recipes=selection['recipes'] if arm=='expanded_guard95' else selection['control_recipes']
    recipe=next(r for r in recipes if r['seed']==seed)
    require(recipe['feasible'],'Infeasible recipe')
    parent=run/f'models/expanded/seed_{seed}' if arm=='expanded_guard95' else Path(identity['parent'])/f'models/unit_BCE/seed_{seed}'
    t,p,_=gate.infer(parent,rows)
    return dict(t=t,scores=p,state=gate.gate_state(p,gate.local_max(t,p),recipe['L'],recipe['H']))


def cli_rows(z):
    for i,t in enumerate(z['t']):
        row=dict(timestamp=iso(int(t)))
        if 'scores' in z:
            row.update(probability_on=None if not np.isfinite(z['scores'][i]) else float(z['scores'][i]),state=None if z['state'][i]<0 else int(z['state'][i]))
        else:
            for name in ('raw','corrected'):
                for j,a in enumerate(('kettle','microwave','washing_machine')): row[name+'_'+a]=None if not np.isfinite(z[name][i,j]) else float(z[name][i,j])
        yield row


@execution('predict')
def predict(run, authorization=None, *, arm, seed, aggregate, output):
    _,identity,s=source_guard(run,require_eligible=True)
    require(not Path(output).exists(),'Ordinary output cannot overwrite')
    rows=gate.csv_rows(Path(aggregate),'refit_17')
    z=infer_arm(run,arm,seed,rows,identity,s);gate.csv_save(Path(output),cli_rows(z))
    return dict(arm=arm,seed=seed,rows=len(z['t']),output=str(output))


def readback(path,z):
    with Path(path).open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    np.testing.assert_array_equal([utc(r['timestamp']) for r in rows],z['t'])
    if 'scores' in z:
        np.testing.assert_array_equal([float(r['probability_on']) if r['probability_on'] else np.nan for r in rows],z['scores'])
        np.testing.assert_array_equal([int(r['state']) if r['state'] else -1 for r in rows],z['state'])
    else:
        for k in ('raw','corrected'):
            for j,a in enumerate(('kettle','microwave','washing_machine')):
                np.testing.assert_array_equal([float(r[k+'_'+a]) if r[k+'_'+a] else np.nan for r in rows],z[k][:,j])


def freeze_matrix(run, identity, selection, rows, aggregate, authorization):
    """Common production/synthetic seal pipeline; all 12 public CLI outputs precede truth."""
    require(not (run/'predictions').exists(),'Freeze is one-shot')
    (run/'predictions').mkdir();(run/'ordinary').mkdir()
    checks=[];base=Path(identity['baseline'])
    for seed in SEEDS:
        a=npz(base/f'reference/predictions/seed_{seed}/{gate.PARTITION}',['stamps','raw','corrected'])
        w=npz(base/f'predictions/W50/seed_{seed}/{gate.PARTITION}',['stamps','raw'])
        astates=np.where(np.isfinite(a['corrected']),a['corrected']>=np.array([2000,200,20]),-1).astype(np.int8)
        for arm in ARMS:
            z=infer_arm(run,arm,seed,rows,identity,selection)
            np.testing.assert_array_equal(z['t'],a['stamps'])
            if arm in ('A','W50'):
                ref=a if arm=='A' else w
                np.testing.assert_array_equal(z['raw'],ref['raw'])
                if arm=='A': np.testing.assert_array_equal(z['corrected'],a['corrected'])
            else:
                np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
                other=astates.copy();other[:,1]=z['state']
                z.update(raw=a['raw'],corrected=a['corrected'],other_states=other)
            np.savez_compressed(run/f'predictions/{arm}_{seed}.npz',**z,**({'states':astates} if arm=='A' else {}))
            output=run/f'ordinary/{arm}_{seed}.csv'
            cli_command(run,arm,seed,aggregate,output,authorization)
            readback(output,z)
            checks.append(dict(arm=arm,seed=seed,ordinary_cli_exact=True,raw_A_and_other_states_exact=True,support_exact=True))
            print(f'freeze/readback {arm} seed={seed}: exact',flush=True)
    fresh(run/'inference_checks.json',dict(checks=checks))
    artifacts={'predictions/'+n:v for n,v in inventory(run/'predictions').items()}|{'ordinary/'+n:v for n,v in inventory(run/'ordinary').items()}|{'inference_checks.json':sha256(run/'inference_checks.json')}
    fresh(run/'freeze.json',dict(matrix=[[a,s] for a in ARMS for s in SEEDS],artifacts=artifacts,truth_opened=False,
        **{n+'_sha256':sha256(run/(n+'.json')) for n in ('registration','identity','source_selection','training_complete')}))
    return dict(status='frozen',matrix_size=12,public_CLI_readbacks=12)


def cli_command(run,arm,seed,aggregate,output,authorization):
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONUTF8']='1'
    result=subprocess.run([sys.executable,'-B',str(run/'run_microwave_expanded_guard95.py'),'predict','--run',str(run),'--authorization',str(authorization),'--arm',arm,'--seed',str(seed),'--aggregate',str(aggregate),'--output',str(output)],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
    require(result.returncode==0,'Public ordinary CLI failed: '+(result.stderr or result.stdout or '')[-3000:])


@execution('freeze')
def freeze(run,authorization=None):
    _,identity,s=source_guard(run,require_eligible=True)  # BEFORE any dev hash/shape/read
    loss.check_paths(identity['development_unlabeled'])
    aggregate=next(Path(p) for p in identity['development_unlabeled'] if p.endswith('aggregate.csv'))
    rows=gate.csv_rows(aggregate,'refit_17')
    return freeze_matrix(run,identity,s,rows,aggregate,authorization)


def evaluation_guard(run):
    _,identity,s=source_guard(run,require_eligible=True)
    frozen=read_json(run/'freeze.json')
    require(frozen['matrix']==[[a,s] for a in ARMS for s in SEEDS] and frozen['truth_opened'] is False,'Incomplete matrix')
    for n in ('registration','identity','source_selection','training_complete'): require(frozen[n+'_sha256']==sha256(run/(n+'.json')),'Freeze binding differs')
    expected={f'predictions/{a}_{s}.npz' for a in ARMS for s in SEEDS}|{f'ordinary/{a}_{s}.csv' for a in ARMS for s in SEEDS}|{'inference_checks.json'}
    require(set(frozen['artifacts'])==expected,'Incomplete/extra frozen matrix artifacts')
    gate.check(run,frozen['artifacts'])
    checks=read_json(run/'inference_checks.json')['checks']
    require([(c['arm'],c['seed']) for c in checks]==[(a,s) for s in SEEDS for a in ARMS] and all(all(c[k] for k in ('ordinary_cli_exact','raw_A_and_other_states_exact','support_exact')) for c in checks),'Incomplete ordinary CLI seal')
    loss.check_paths(identity['development_unlabeled'])
    for seed in SEEDS:
        a=npz(run/f'predictions/A_{seed}.npz',['t','raw','corrected','states'])
        original=npz(Path(identity['baseline'])/f'reference/predictions/seed_{seed}/{gate.PARTITION}',['stamps','raw','corrected'])
        w=npz(run/f'predictions/W50_{seed}.npz',['t','raw'])
        original_w=npz(Path(identity['baseline'])/f'predictions/W50/seed_{seed}/{gate.PARTITION}',['stamps','raw'])
        for key,oldkey in [('t','stamps'),('raw','raw'),('corrected','corrected')]: np.testing.assert_array_equal(a[key],original[oldkey])
        for key,oldkey in [('t','stamps'),('raw','raw')]: np.testing.assert_array_equal(w[key],original_w[oldkey])
        np.testing.assert_array_equal(a['states'],np.where(np.isfinite(a['corrected']),a['corrected']>=np.array([2000,200,20]),-1).astype(np.int8))
        for arm,key in [('control_guard95','control_recipes'),('expanded_guard95','recipes')]:
            z=npz(run/f'predictions/{arm}_{seed}.npz',['t','scores','state','raw','corrected','other_states'])
            r=next(r for r in s[key] if r['seed']==seed)
            np.testing.assert_array_equal(z['state'],gate.gate_state(z['scores'],gate.local_max(z['t'],z['scores']),r['L'],r['H']))
            for field in ('t','raw','corrected'): np.testing.assert_array_equal(z[field],a[field])
            np.testing.assert_array_equal(z['other_states'][:,[0,2]],a['states'][:,[0,2]])
            np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
    return identity,s


def score(run,truth):
    rows=[];supports=[];paired=[];segments=[]
    for seed in SEEDS:
        a=npz(run/f'predictions/A_{seed}.npz',['t','raw','corrected','states'])
        w=npz(run/f'predictions/W50_{seed}.npz',['t','raw'])
        np.testing.assert_array_equal(a['t'],truth['t']);np.testing.assert_array_equal(w['t'],truth['t'])
        mask=np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1])
        np.testing.assert_array_equal(mask,np.isfinite(truth['y'])&np.isfinite(a['raw'][:,1]))
        require(mask.sum()==137523,'Original development mask count differs')
        supports.append(dict(seed=seed,n=int(mask.sum()),mask_sha256=gate.hash_array(mask),t_sha256=gate.hash_array(a['t'][mask])))
        preds={arm:(a if arm=='A' else w)['raw'][mask,1]>=200 for arm in ('A','W50')}
        for arm in ARMS[2:]:
            z=npz(run/f'predictions/{arm}_{seed}.npz',['t','raw','corrected','scores','state','other_states'])
            for k in ('t','raw','corrected'): np.testing.assert_array_equal(z[k],a[k])
            np.testing.assert_array_equal(z['other_states'][:,[0,2]],a['states'][:,[0,2]])
            np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
            preds[arm]=z['state'][mask]
        y=truth['y'][mask]>=200;t=truth['t'][mask]
        for arm,p in preds.items(): rows.append(dict(seed=seed,arm=arm,**gate.confusion(p,y,t)))
        new=preds['expanded_guard95'].astype(bool)
        for arm in ('W50','control_guard95'):
            before=preds[arm].astype(bool)
            paired.append(dict(seed=seed,reference=arm,removed_fp=int((before&~new&~y).sum()),added_fp=int((~before&new&~y).sum()),restored_tp=int((~before&new&y).sum()),new_fn=int((before&~new&y).sum())))
            for kind,m in [('removed_fp',before&~new&~y),('added_fp',~before&new&~y)]:
                starts=np.flatnonzero(m&np.r_[True,~m[:-1]|(np.diff(t)!=8)])
                for start in starts:
                    end=start+1
                    while end<len(m) and m[end] and t[end]-t[end-1]==8: end+=1
                    segments.append(dict(seed=seed,reference=arm,kind=kind,start_utc=iso(int(t[start])),end_utc=iso(int(t[end-1]+8)),points=int(end-start)))
    return dict(rows=rows,supports=supports,paired=paired,paired_fp_segments=segments)


def summarize(selection, metrics):
    # Reuse the original eight formal gates, with the frozen source winner only.
    selected=selection['selected'];groups={a:[next(r for r in metrics['rows'] if r['arm']==a and r['seed']==s) for s in SEEDS] for a in ARMS}
    base,new=groups['W50'],groups[selected]
    means={a:gate.mean_rows(rs)|dict(fp=float(np.mean([r['fp'] for r in rs]))) for a,rs in groups.items()}
    gates=dict(mean_P_at_least_008=means[selected]['precision']>=.08,each_R_at_least_085=all(r['recall']>=.85 for r in new),
        mean_F1_gain_at_least_002=means[selected]['f1']-means['W50']['f1']+1e-12>=.02,
        at_least_two_seed_P_F1_up=sum(b['precision']>a['precision'] and b['f1']>a['f1'] for a,b in zip(base,new))>=2,
        each_F1_drop_at_most_002=all(b['f1']-a['f1']>=-.02 for a,b in zip(base,new)),
        mean_FP_reduction_at_least_30_percent=means[selected]['fp']<=.7*means['W50']['fp'],
        no_new_all_on_or_off=all(not r['degenerate'] for r in new),support_137523_exact=all(v['n']==137523 for v in metrics['supports']))
    return dict(selected_before_development=selected,acceptance_gates=gates,development_acceptance_passed=all(gates.values()),means=means,
        new_fits=3,optimizer_updates=19404,power_improvement_claim=False,default_changed=False,independent_new_house_validation=False,
        limitation='来源扩展和更新量同时改变；House17为反复使用的工程开发集；合成通过不能证明真实改善。')


@execution('evaluate')
def evaluate(run,authorization=None):
    identity,s=evaluation_guard(run)
    dest=run/'evaluation';dest.mkdir(exist_ok=False)  # once-only claim BEFORE truth access
    fresh(dest/'started.json',dict(freeze_sha256=sha256(run/'freeze.json')))
    info=identity['truth_cache']
    # Separate narrow truth scope exists only after complete source+CLI+freeze validation.
    with isolation('evaluate',[info['path']]+allowed_files(run,'evaluate')+[dest/n for n in ('started.json','metrics.json','summary.json','confusions.csv','paired.csv','paired_fp_segments.csv','truth_once.npz')]):
        require(sha256(Path(info['path']))==info['sha256'],'Old Task12 truth cache changed')
        truth=npz(Path(info['path']),['t','y'])
        np.savez_compressed(dest/'truth_once.npz',**truth)
        metrics=score(run,truth);summary=summarize(s,metrics)
        independently_count_predictions(run,truth,metrics)
        require(independent_scalar(metrics,s)==summary['acceptance_gates'],'Independent scalar gates disagree')
        fresh(dest/'metrics.json',metrics);fresh(dest/'summary.json',summary)
        gate.csv_save(dest/'confusions.csv',metrics['rows'])
        gate.csv_save(dest/'paired.csv',metrics['paired'])
        if metrics['paired_fp_segments']: gate.csv_save(dest/'paired_fp_segments.csv',metrics['paired_fp_segments'])
        fresh(run/'evaluation_complete.json',dict(freeze_sha256=sha256(run/'freeze.json'),truth_cache_sha256=info['sha256'],parent_truth_loads=1,original_truth_csv_reads=0,artifacts=inventory(dest)))
    return summary


def independently_count_predictions(run,truth,metrics):
    """Second path: scalar per-point counters, independent of vector score()."""
    for seed in SEEDS:
        w=npz(run/f'predictions/W50_{seed}.npz',['raw'])
        for arm in ARMS:
            fields=['raw'] if arm in ('A','W50') else ['state']
            z=npz(run/f'predictions/{arm}_{seed}.npz',fields)
            counts=dict(tp=0,fp=0,fn=0,tn=0)
            for i,target in enumerate(truth['y']):
                if not np.isfinite(target) or not np.isfinite(w['raw'][i,1]): continue
                pred=z['raw'][i,1]>=200 if arm in ('A','W50') else z['state'][i]
                require(pred in (0,1),'Scalar path lost support')
                counts['tp' if pred and target>=200 else 'fp' if pred else 'fn' if target>=200 else 'tn']+=1
            expected=next(r for r in metrics['rows'] if r['arm']==arm and r['seed']==seed)
            require(all(expected[k]==v for k,v in counts.items()),'Independent per-point count path disagrees')


def verify_preparation(run):
    run=Path(run)
    with isolation('verify_preparation',allowed_files(run,'verify_preparation')):
        cfg,identity=registered(run)
        d=npz(run/'train.npz',loss.TRAIN_FIELDS+['source_role']);old=npz(TEMPORAL/'train.npz',loss.TRAIN_FIELDS)
        require([len(d['y']),int((d['y']>=200).sum()),int((d['y']<200).sum())]==[615922,2994,612928],'Support differs')
        for k,v in old.items(): require(d[k].dtype==v.dtype and d[k][:len(v)].tobytes()==v.tobytes(),'Old rows changed')
        for i in (4,5): require(sha256(run/f'validation/block_{i}.npz')==sha256(TEMPORAL/f'validation/block_{i}.npz'),'Validation changed')
        require(sha256(run/'normalization.json')==sha256(TEMPORAL/'normalization.json'),'Normalization changed')
        plan=read_json(run/'training_plan.json')
        expected=expected_exposure(d,plan)
        for seed in SEEDS:
            ix=npz(run/f'sampling_{seed}.npz',['indices'])['indices'];stream=BatchStream(d,'event',seed,plan)
            require(ix.shape==(6468,256) and np.all((d['y'][ix]>=200).sum(1)==16),'Quotas changed')
            for row in ix: np.testing.assert_array_equal(row,stream.next())
            ex=npz(run/f'exposure_{seed}.npz',['count','expected'])
            np.testing.assert_array_equal(np.bincount(ix.ravel(),minlength=len(d['y'])),ex['count']);np.testing.assert_array_equal(expected,ex['expected'])
            require(sha256(run/f'initial_{seed}.pt')==sha256(TEMPORAL/f'initial_{seed}.pt'),'Initialization changed')
        return dict(status='preparation_verified',new_fits=0,optimizer_updates=0,development_reads=0,registration_sha256=sha256(run/'registration.json'))


def verify(run,authorization=None):
    run=Path(run)
    if not (run/'training_started.json').exists(): return verify_preparation(run)
    if authorization is None: validate_receipt(run,read_json(run/'training_started.json')['authorization'])
    else: authorize(run,authorization)
    # Verification uses the once-loaded local t/y copy, never the parent cache.
    with isolation('verify',allowed_files(run,'verify')):
        if not (run/'source_selection.json').exists(): training_guard(run);return dict(status='trained_source_pending')
        _,_,s=source_guard(run)
        if not s['expanded_eligible']:
            require(not (run/'freeze.json').exists() and not (run/'evaluation').exists(),'Source rejection leaked development')
            return dict(status='source_only_rejected',selected='control_guard95',development_reads=0)
        identity,s=evaluation_guard(run)
        if not (run/'evaluation_complete.json').exists(): return dict(status='frozen_before_truth',matrix_size=12)
        complete=read_json(run/'evaluation_complete.json')
        require(complete['freeze_sha256']==sha256(run/'freeze.json') and complete['truth_cache_sha256']==identity['truth_cache']['sha256'],'Evaluation binding changed')
        gate.check(run/'evaluation',complete['artifacts'])
        metrics=read_json(run/'evaluation/metrics.json')
        require(complete['parent_truth_loads']==1 and complete['original_truth_csv_reads']==0,'Once-only truth loading differs')
        truth=npz(run/'evaluation/truth_once.npz',['t','y'])
        require(score(run,truth)==metrics,'Repeated readonly point/confusion/mask/paired segment replay differs')
        independently_count_predictions(run,truth,metrics)
        scalar=independent_scalar(metrics,s)
        require(summarize(s,metrics)==read_json(run/'evaluation/summary.json') and scalar==read_json(run/'evaluation/summary.json')['acceptance_gates'],'Independent eight gates differ')
        # Full CLI CSV readback is repeatable and readonly.
        for arm in ARMS:
            for seed in SEEDS:
                with np.load(run/f'predictions/{arm}_{seed}.npz',allow_pickle=False) as f: z={k:f[k] for k in f.files}
                readback(run/f'ordinary/{arm}_{seed}.csv',z)
        return dict(status='execution_verified_readonly',matrix_size=12,parent_truth_reopens=0,local_truth_scalar_replay=True,independent_scalar_gates=scalar)


def independent_scalar(metrics,selection):
    """Independent count arithmetic; does not call production confusion/mean helpers."""
    groups={a:[] for a in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            r=next(r for r in metrics['rows'] if r['seed']==seed and r['arm']==arm)
            tp,fp,fn,tn=(int(r[k]) for k in ('tp','fp','fn','tn'));n=tp+fp+fn+tn
            p=tp/max(tp+fp,1);rec=tp/max(tp+fn,1);f1=2*tp/max(2*tp+fp+fn,1)
            require(all(r[k]==v for k,v in dict(n=n,precision=p,recall=rec,f1=f1,predicted_on=tp+fp,degenerate=(tp+fp in (0,n))).items()),'Independent scalar confusion mismatch')
            groups[arm].append(dict(p=p,r=rec,f=f1,fp=fp,n=n,on=tp+fp))
    w=groups['W50'];c=groups[selection['selected']]
    return dict(mean_P_at_least_008=sum(r['p'] for r in c)/3>=.08,each_R_at_least_085=min(r['r'] for r in c)>=.85,
        mean_F1_gain_at_least_002=sum(r['f'] for r in c)/3-sum(r['f'] for r in w)/3+1e-12>=.02,
        at_least_two_seed_P_F1_up=sum(c[i]['p']>w[i]['p'] and c[i]['f']>w[i]['f'] for i in range(3))>=2,
        each_F1_drop_at_most_002=min(c[i]['f']-w[i]['f'] for i in range(3))>=-.02,
        mean_FP_reduction_at_least_30_percent=sum(r['fp'] for r in c)<=.7*sum(r['fp'] for r in w),
        no_new_all_on_or_off=all(0<r['on']<r['n'] for r in c),support_137523_exact=all(v['n']==137523 for v in metrics['supports']))


def source_report(selection):
    s=selection
    text=f"来源唯一预选：{s['selected']}；expanded来源资格：{s['expanded_eligible']}；拒绝原因：{s['source_rejection']}。\n\n"
    text+='|来源块|模型|种子|L95|H95|TP|FP|FN|TN|P|R|F1|R≥95%且不退化|\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n'
    for arm,key in [('control_guard95','control_recipes'),('expanded_guard95','recipes')]:
        for recipe in s[key]:
            if not recipe['feasible']:
                text+=f"|无合格阈值|{arm}|{recipe['seed']}|{recipe['L']}|{recipe['H']}|—|—|—|—|—|—|—|失败|\n"
            for r in recipe['blocks']:
                text+=f"|{r['block']}|{arm}|{recipe['seed']}|{recipe['L']}|{recipe['H']}|{r['tp']}|{r['fp']}|{r['fn']}|{r['tn']}|{r['precision']:.6f}|{r['recall']:.6f}|{r['f1']:.6f}|{r['recall']>=.95 and not r['degenerate']}|\n"
    text+='\n|种子|Δ来源平均P（expanded−control）|ΔR|ΔF1|每块95%/非退化|R下降≤.05|P提升|\n|---|---|---|---|---|---|---|\n'
    for seed in SEEDS:
        before=next(r for r in s['rows']['control_guard95'] if r['seed']==seed)
        after=next((r for r in s['rows']['expanded_guard95'] if r['seed']==seed),None)
        if after is None: text+=f'|{seed}|无合格候选|—|—|失败|失败|失败|\n';continue
        text+=f"|{seed}|{after['precision']-before['precision']:+.6f}|{after['recall']-before['recall']:+.6f}|{after['f1']-before['f1']:+.6f}|{after['source_gate_passed'] and not after['degenerate']}|{after['recall']-before['recall']>=-.05}|{after['precision']>before['precision']}|\n"
    old=s['rows']['control_guard95'];new=s['rows']['expanded_guard95']
    mean_ok=len(new)==3 and sum(r['f1'] for r in new)>=sum(r['f1'] for r in old)
    two_ok=len(new)==3 and sum(b['precision']>a['precision'] for a,b in zip(old,new))>=2
    text+=f'\n整体来源平均F1不低于control：{mean_ok}；至少2种子P升：{two_ok}；排序为来源meanP/F1，完全平局保留control。\n'
    return text


def report(run,authorization=None,output=None):
    run=Path(run);v=verify(run,authorization)
    text='Task18 三模型工程对照\n\n'
    if not (run/'training_started.json').exists():
        support=read_json(run/'support.json')
        text+='准备完成，尚未训练。固定扩展House2（2014-10-14至10-28 UTC）与House19（2014-10-19至11-02 UTC），追加279218个中心/719个正中心，最终615922/2994/612928。原336704行每字段字节、House5前缀、34351/201验证支持和归一化完全保留。\n\n'
        text+='候选expanded_guard95，种子42/1019/2026；冻结control_guard95不重训。未来预算须另批3次拟合/19404更新（6468每种子，12×539），Adam .001、BCE权重1、256批次16正/240负、5329参数。'
        text+=f"负中心期望曝光{support['negative_expected_presentations']:.9f}（旧{support['original_negative_expected_presentations']:.9f}）；正中心/日期/事件曝光无法同时精确匹配。扩数据与增更新同时改变，不能独立证明数据扩展因果。\n\n"
        text+='|种子|批次/未来更新|正呈现|负呈现|不同正中心|不同负中心|\n|---|---|---|---|---|---|\n'
        for seed in SEEDS:
            row=support['sampling'][str(seed)]
            text+=f"|{seed}|{row['updates']}|{row['positive_presentations']}|{row['negative_presentations']}|{row['unique_positive_centers']} / 2994|{row['unique_negative_centers']} / 612928|\n"
        text+='\n每种子有42个正中心未抽到，均如实计入台账；不因此改动既定采样器或预算。新增719个正中心均被抽取，未抽到者来自原House5长片段。\n\n'
        independent=run/'independent_preparation_verification.json'
        if independent.exists():
            evidence=read_json(independent)
            text+=f"正样本户—日期组共{evidence['positive_day_groups']}，按House2/19/5为{evidence['positive_day_groups_by_house']}。House5正slot占比8/31→8/57；绝对期望正曝光{evidence['house5_old_positive_expected']:.4f}→{evidence['house5_new_positive_expected']:.4f}，约下降0.29%，基本接近原方案。三种子House5实际正曝光分别为"+str([r['house5_positive_actual'] for r in evidence['seeds']])+'。\n\n'
        text+='本轮0真实fit/0优化器更新/0新增候选推理校准/0开发与最终户读取。索引及每中心、户、日期、事件期望/实际曝光已封存；实际指确定性干运行采样，不是训练完成。\n\n'
        text+='执行顺序：新的人类专属预算回执→train→source→来源合格后freeze及完整12格公共CLI回读→一次旧Task12缓存评价→verify→report。来源拒绝只出来源报告，不进入开发。原八门槛不变，功率/电量无改善声称，UI不切换。合成验证不代表真实训练改善。\n'
    elif v['status']=='source_only_rejected':
        text+='来源资格失败，保留冻结control_guard95；0开发读取。\n\n'+source_report(read_json(run/'source_selection.json'))
    elif (run/'evaluation_complete.json').exists():
        m=read_json(run/'evaluation/metrics.json');s=read_json(run/'evaluation/summary.json');source=read_json(run/'source_selection.json')
        text+=source_report(source)+f"\n八门槛通过：{s['development_acceptance_passed']}。\n\n"
        text+='|分区/块|模型|种子|TP|FP|FN|TN|P|R|F1|\n|---|---|---|---|---|---|---|---|---|---|\n'
        for r in m['rows']: text+=f"|开发|{r['arm']}|{r['seed']}|{r['tp']}|{r['fp']}|{r['fn']}|{r['tn']}|{r['precision']:.6f}|{r['recall']:.6f}|{r['f1']:.6f}|\n"
        text+='\n|开发参照|种子|ΔP|ΔR|ΔF1|ΔFP|FP相对变化|\n|---|---|---|---|---|---|---|\n'
        for seed in SEEDS:
            c=next(r for r in m['rows'] if r['seed']==seed and r['arm']=='expanded_guard95')
            for arm in ('W50','control_guard95'):
                b=next(r for r in m['rows'] if r['seed']==seed and r['arm']==arm)
                ratio=(c['fp']/b['fp']-1) if b['fp'] else None
                text+=f"|{arm}|{seed}|{c['precision']-b['precision']:+.6f}|{c['recall']-b['recall']:+.6f}|{c['f1']-b['f1']:+.6f}|{c['fp']-b['fp']:+d}|{ratio}|\n"
        text+='\n门槛：\n'+''.join(f"- {k}：{val}\n" for k,val in s['acceptance_gates'].items())
        text+='\n逐种子配对FP/TP变化及FP片段见evaluation/paired.csv与paired_fp_segments.csv。原A全部功率列及其他电器状态不变，无功率、MAE、电量改善声称。House17反复用于工程开发，不能称独立泛化；扩数据与更新量同时改变。UI不切换。\n'
    else:
        text+='当前阶段：'+v['status']+'；尚无可报告的开发结论。\n'
        if (run/'source_selection.json').exists(): text+='\n'+source_report(read_json(run/'source_selection.json'))
    if output is not None:
        output=Path(output);require(not output.exists(),'Report output must be new')
        with output.open('x',encoding='utf-8') as f:f.write(text)
    return dict(verification=v,report_zh=text)
