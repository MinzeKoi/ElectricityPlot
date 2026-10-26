"""Task16: fixed same-seed top3 probability ensemble. Never trains.

Preparation only replays OLD single-checkpoint source85 selection. All new
ensemble inference, calibration and development access require a distinct receipt.
"""
import contextlib
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

from ..data.common import read_json, write_json, sha256, inventory, utc, iso
from ..models.center_mask_tcn import CenterMaskTCN, predict_windows, context_at_complete_centers
from ..models.arrays import aggregate_arrays
from . import microwave_recall_guard95 as guard
from . import microwave_source3_background_execution as shared
from . import microwave_source5_loss as loss
from . import microwave_event_gate as gate
from .microwave_sampling import rank

ROOT = loss.ROOT
LOSS = ROOT/'outputs/accuracy/microwave-source5-loss-preparation-20261025-v4'
TEMPORAL = ROOT/'outputs/accuracy/microwave-source5-temporal-20261025-v2'
CONTROL = ROOT/'outputs/accuracy/microwave-recall-guard95-preparation-20261025-v2'
CONFIG = 'configs/microwave_checkpoint_ensemble_20261026.disabled.json'
SCRIPT = 'scripts/run_microwave_checkpoint_ensemble.py'
TEST = 'tests/test_microwave_checkpoint_ensemble.py'
PLAN = 'docs/MICROWAVE_CHECKPOINT_ENSEMBLE_PLAN_20261026.md'
SEEDS = [42,1019,2026]
FAMILIES = ['control_guard95','ensemble_guard95']
require, fresh = gate.require, gate.fresh_json


def top3(points):
    require(len(points)==12 and [r['epoch'] for r in points]==list(range(1,13)), 'Complete unique twelve epochs required')
    eligible = [dict(r['selection'],epoch=r['epoch']) for r in points if r['selection'] is not None]
    require(len(eligible)>=3, 'Source feasibility rejected: fewer than three qualified checkpoints; no fallback')
    return sorted(eligible,key=lambda r:rank(r,r['epoch']))[:3]


def mean_probability(values):
    require(len(values)==3, 'Exactly three same-seed checkpoints required')
    require(all(v.dtype==np.float32 and v.shape==values[0].shape for v in values),'Probability dtype/shape differs')
    finite=np.isfinite(values[0])
    for v in values:
        require(np.array_equal(np.isfinite(v),finite), 'Finite support must be exact')
        require(not np.isinf(v).any() and ((v[finite]>=0)&(v[finite]<=1)).all(), 'Invalid probability')
    out=np.full(values[0].shape,np.nan,np.float32)
    out[finite]=(np.stack([v[finite] for v in values]).sum(axis=0,dtype=np.float64)/3).astype(np.float32)
    return out


def authorize(run,receipt):
    # Must precede metadata/numeric identity checks and any output creation.
    require(receipt is not None, 'Task16 new ensemble rule requires explicit new human authorization')
    a=read_json(Path(receipt))
    require(a.get('experiment')=='microwave_checkpoint_ensemble_task16_v1'
        and a.get('authorized') is True and a.get('approval_origin')=='human'
        and a.get('approved_rule')=='same_seed_source85_top3_equal_probability_L95_H95_radius15'
        and a.get('new_fits')==a.get('optimizer_updates')==0 and a.get('seeds')==SEEDS
        and a.get('registration_sha256')==sha256(Path(run)/'registration.json')
        and a.get('human_authorization_quote','').strip() and a.get('human_authorization_reference','').strip(),
        'Independent Task16 human receipt bound to this registration required; Task14/15 receipts invalid')
    return a


def source_files():
    files=[TEMPORAL/'plan.json',TEMPORAL/'normalization.json']
    files += [TEMPORAL/f'validation/block_{b}.npz' for b in (4,5)]
    for s in SEEDS:
        d=LOSS/f'models/unit_BCE/seed_{s}'
        files += [d/n for n in ('model.json','history.json','weights.pt','complete.json')]
        files += [d/f'epoch_{e}{suffix}' for e in range(1,13) for suffix in ('.pt','_block_4.npz','_block_5.npz')]
    files += [CONTROL/n for n in ('identity.json','source_selection.json','registration.json','freeze.json')]
    return files


@contextlib.contextmanager
def isolation(name,allowed):
    """Extend numerical guard to reject development metric JSON too."""
    state={'active':True}
    def audit(event,args):
        if event!='open' or not state['active'] or not isinstance(args[0],str): return
        mode=args[1]
        if mode is not None and any(c in mode for c in 'wax+'): return
        if mode is None and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT): return
        p=str(Path(args[0]).resolve()).lower().replace('\\','/')
        if re.search(r'(?:house|refit)[_-](?:8|9|20)(?:[_/.]|$)',p):
            raise gate.DataError('Final-house access forbidden')
        if name in ('prepare','verify_preparation','source_select','source_freeze_gate','evaluation_gate','predict_gate'):
            require(not any(v in p for v in ('/evaluation/','truth_once','/details_v2/','/report/development','refit_17','house_17')),'Development access forbidden in this phase')
    sys.addaudithook(audit)
    def no_model(*a,**k): raise gate.DataError('Preparation cannot construct a model')
    try:
        with guard.isolation('prepare' if name in ('prepare','verify_preparation') else name,allowed) as access:
            with contextlib.ExitStack() as stack:
                if name in ('prepare','verify_preparation'): stack.enter_context(patch.object(CenterMaskTCN,'__init__',no_model))
                yield access
    finally: state['active']=False


@contextlib.contextmanager
def phase(run,name,extra=()):
    identity=read_json(Path(run)/'identity.json')
    allowed=[Path(p) for p in identity['source_inputs']]+list(extra)
    # Exact local allowlist; preparation cannot bypass protection via local files.
    if name not in ('prepare','verify_preparation'):
        for folder in ('source_results','predictions','evaluation','ordinary'):
            root=Path(run)/folder
            if root.exists(): allowed += [p for p in root.iterdir() if p.suffix in ('.npz','.csv')]
    if name=='freeze':
        allowed += [Path(run)/f'predictions/{a}_{s}.npz' for a,s in matrix()]
        allowed += [Path(run)/f'ordinary/{a}_{s}.csv' for a in FAMILIES for s in SEEDS]
    if name=='source_select':
        allowed += [Path(run)/f'source_results/seed_{s}_block_{b}.npz' for s in SEEDS for b in (4,5)]
    if name=='evaluate':
        allowed += [Path(run)/'evaluation'/p for p in ('truth_once.npz','confusions.csv','fp_segments.csv')]
    with isolation(name,allowed) as state:
        try: yield
        finally:
            p=Path(run)/(name+'_access.json'); n=0
            while p.exists():
                n+=1; p=Path(run)/f'{name}_access_{n}.json'
            fresh(p,dict(reads=sorted(state['reads']),forbidden=state['forbidden'],new_fits=0,optimizer_updates=0))


def registered(run):
    run=Path(run); reg=read_json(run/'registration.json')
    gate.check(run,reg['artifacts'])
    p=read_json(run/'config.json')
    require(p['authorized'] is False and p['top_k']==3 and p['new_fits']==p['optimizer_updates']==0,'Disabled rule changed')
    identity=read_json(run/'identity.json')
    loss.check_paths(identity['source_inputs'])
    # Launching edited live code must not silently change the frozen program.
    package=Path(__file__).resolve().parents[1]
    frozen=run/'source/virtual_submeter'
    require({str(p.relative_to(package)):sha256(p) for p in package.rglob('*.py')}==
            {str(p.relative_to(frozen)):sha256(p) for p in frozen.rglob('*.py')},'Use registered executable source snapshot')
    return identity


def audit_checkpoints():
    p=read_json(TEMPORAL/'plan.json')
    blocks={f'block_{b}':gate.npz(TEMPORAL/f'validation/block_{b}.npz',['t','y','event']) for b in (4,5)}
    require([sum(len(z['y']) for z in blocks.values()),sum(int((z['y']>=200).sum()) for z in blocks.values())]==[34351,201],'Source support changed')
    result={}
    for seed in SEEDS:
        model=LOSS/f'models/unit_BCE/seed_{seed}'
        history=read_json(model/'history.json')['rows']; points=[]
        require(len(history)==12,'Incomplete old checkpoint history')
        for e in range(1,13):
            scores={}
            for b,d in blocks.items():
                z=gate.npz(model/f'epoch_{e}_{b}.npz',['t','scores'])
                np.testing.assert_array_equal(z['t'],d['t'])
                require(np.isfinite(z['scores']).all(),'Old source cache nonfinite')
                scores[b]=z['scores']
            chosen=loss.temporal.f.reselect_checkpoints([scores],blocks,p,expected_epochs=1)
            require(history[e-1]==dict(epoch=e,selection=chosen),'Old history/source85 replay differs')
            points.append(dict(epoch=e,selection=chosen))
        selected=top3(points)
        meta=read_json(model/'model.json')
        require(meta['recipe']=='unit_BCE' and meta['seed']==seed and meta['selected']['epoch']==selected[0]['epoch'],'Old single point identity differs')
        result[str(seed)]=dict(qualified=sum(r['selection'] is not None for r in points),selected=selected,rows=points)
    return result


def prepare(run):
    run=Path(run); require(not run.exists(),'Completed directories cannot be overwritten')
    run.mkdir(parents=True)
    files=source_files()
    # JSON identities only: no development numerical file is opened or hashed.
    c=read_json(CONTROL/'identity.json')
    files += [Path(v['path']) for v in c['source'].values()]
    # Bind every original old checkpoint to the original completion manifest.
    manifest=read_json(LOSS/'final_delivery_manifest.json')['artifacts']
    with isolation('prepare',files) as access:
        bindings={str(p.resolve()):sha256(p) for p in files}
        for v in c['source'].values(): require(bindings[str(Path(v['path']).resolve())]==v['sha256'],'Frozen source raw identity changed')
        tm=read_json(TEMPORAL/'final_delivery_manifest.json')['artifacts']
        for p in files:
            if p.is_relative_to(TEMPORAL): require(bindings[str(p.resolve())]==tm[p.relative_to(TEMPORAL).as_posix()],'Task10 source identity changed')
        seal=read_json(CONTROL/'freeze.json')
        require(seal['source_selection_sha256']==bindings[str((CONTROL/'source_selection.json').resolve())],'Task14 source selection seal changed')
        for p in files:
            if p.is_relative_to(LOSS):
                rel=p.relative_to(LOSS).as_posix()
                require(manifest[rel]==bindings[str(p.resolve())], 'Task12 completion identity differs')
        identity=dict(source_inputs=bindings,control_identity_sha256=sha256(CONTROL/'identity.json'),
                      new_fits=0,optimizer_updates=0,development_numeric_reads=0,final_reads=0)
        write_json(run/'identity.json',identity)
        try:
            audit=audit_checkpoints()
            result=dict(status='prepared_disabled',feasible=True,seeds=audit)
        except Exception as exc:
            fresh(run/'source_feasibility_rejection.json',dict(status='source_rejected',reason=str(exc),new_fits=0,development_reads=0))
            raise
        fresh(run/'checkpoint_audit.json',result)
        fresh(run/'prepare_access.json',dict(reads=sorted(access['reads']),forbidden=access['forbidden'],new_fits=0,optimizer_updates=0,development_reads=0,final_reads=0))
    shutil.copyfile(ROOT/CONFIG,run/'config.json')
    shutil.copyfile(ROOT/SCRIPT,run/Path(SCRIPT).name)
    shutil.copyfile(ROOT/TEST,run/Path(TEST).name)
    shutil.copyfile(ROOT/PLAN,run/'protocol.md')
    evidence=ROOT/'tmp/task16-checkpoint-ensemble-tests-20261026.log'
    require(evidence.exists(),'Run and retain synthetic test evidence before preparation')
    shutil.copyfile(evidence,run/'synthetic_tests.log')
    package=Path(__file__).resolve().parents[1]
    for p in package.rglob('*.py'):
        dest=run/'source/virtual_submeter'/p.relative_to(package)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
    write_json(run/'workspace_identity.json',dict(root=str(ROOT)))
    fresh(run/'registration.json',dict(experiment='microwave_checkpoint_ensemble_task16_v1',authorized=False,artifacts=inventory(run)))
    return dict(status='prepared_disabled',selected={s:[r['epoch'] for r in d['selected']] for s,d in audit.items()},new_fits=0,development_reads=0)


def checkpoint_infer(seed,epoch,rows):
    require(seed in SEEDS and epoch in range(1,13),'Unknown checkpoint identity')
    parent=LOSS/f'models/unit_BCE/seed_{seed}'
    meta=read_json(parent/'model.json')
    model=CenterMaskTCN()
    model.load_state_dict(torch.load(parent/f'epoch_{epoch}.pt',map_location='cpu',weights_only=True))
    t,a=aggregate_arrays(rows,8); x,centers=context_at_complete_centers(meta,t,a)
    p=np.full(len(t),np.nan,np.float32); p[centers]=predict_windows(meta,model,x)
    return t,p


def ensemble_infer(run,seed,rows,check_block=None):
    require(seed in SEEDS,'Cross-seed mixing forbidden')
    audit=read_json(Path(run)/'checkpoint_audit.json')['seeds'][str(seed)]
    points=top3(audit['rows']); require(points==audit['selected'],'Top3 selection changed')
    values=[]; times=None
    for point in points:
        t,p=checkpoint_infer(seed,point['epoch'],rows)
        if times is not None: np.testing.assert_array_equal(times,t)
        times=t; values.append(p)
        if check_block is not None:
            block=gate.npz(TEMPORAL/f'validation/block_{check_block}.npz',['t'])
            z=gate.npz(LOSS/f'models/unit_BCE/seed_{seed}/epoch_{point["epoch"]}_block_{check_block}.npz',['t','scores'])
            np.testing.assert_array_equal(z['t'],block['t']); ix=gate.align(t,z['t'])
            np.testing.assert_allclose(p[ix],z['scores'],rtol=1e-5,atol=1e-7)
            np.testing.assert_array_equal(p[ix]>=point['threshold'],z['scores']>=point['threshold'])
    return times,mean_probability(values)


def select_source(control,new,recipes):
    details=shared.qualification_details(control,new,recipes)
    ok=all(details.values()); selected='control_guard95'
    if ok:
        c,n=gate.mean_rows(control),gate.mean_rows(new)
        if (n['precision'],n['f1'])>(c['precision'],c['f1']): selected='ensemble_guard95'
    return dict(selected=selected,ensemble_eligible=ok,qualification=details,
                source_rejection=None if ok else 'Fixed source95/nondegenerate/control-relative qualification failed')


def source(run,authorization=None):
    run=Path(run); authorize(run,authorization)
    with phase(run,'source_select'):
        registered(run)
        require(not (run/'source_started.json').exists(),'Source stage is one-shot')
        fresh(run/'source_started.json',dict(registration_sha256=sha256(run/'registration.json')))
        require(audit_checkpoints()==read_json(run/'checkpoint_audit.json')['seeds'],'Top3 source85 replay changed')
        ci=read_json(CONTROL/'identity.json'); raw=gate.csv_rows(Path(ci['source']['aggregate']['path']),'refit_5')
        times=np.array([utc(r['timestamp']) for r in raw]); recipes=[]; new=[]
        dest=run/'source_results'; dest.mkdir()
        for seed in SEEDS:
            scores={}
            for b,lo,hi in loss.temporal.f.VALIDATION:
                rows=[raw[j] for j in np.flatnonzero((times>=lo)&(times<hi))]
                t,p=ensemble_infer(run,seed,rows,b)
                block=gate.npz(TEMPORAL/f'validation/block_{b}.npz',['t','y'])
                scores[str(b)]=dict(t=t,p=p,h=gate.local_max(t,p),ix=gate.align(t,block['t']),y=block['y']>=200)
                np.savez_compressed(dest/f'seed_{seed}_block_{b}.npz',**{k:v for k,v in scores[str(b)].items() if k!='h'})
            low,high=shared.calibrate_source95(scores)
            feasible=low is not None and high is not None
            metrics=[dict(block=k,**gate.confusion(gate.gate_state(d['p'],d['h'],low,high)[d['ix']],d['y'])) for k,d in scores.items()] if feasible else []
            if feasible: new.append(dict(seed=seed,**gate.mean_rows(metrics),degenerate=any(m['degenerate'] for m in metrics),source_gate_passed=all(m['recall']>=.95 for m in metrics)))
            recipes.append(dict(seed=seed,L=low,H=high,feasible=feasible,blocks=metrics,radius_samples=15))
        control=read_json(CONTROL/'source_selection.json')['rows']['guard95']
        decision=select_source(control,new,recipes)
        fresh(run/'source_selection.json',dict(source_only=True,**decision,rows=dict(control_guard95=control,ensemble_guard95=new),recipes=recipes))
        fresh(run/'source_complete.json',dict(source_selection_sha256=sha256(run/'source_selection.json'),artifacts=inventory(dest)))
        write_source_report(run,read_json(run/'source_selection.json'))
        return decision


def source_guard(run):
    registered(run); run=Path(run)
    done=read_json(run/'source_complete.json'); gate.check(run/'source_results',done['artifacts'])
    require(sha256(run/'source_selection.json')==done['source_selection_sha256'],'Source seal changed')
    s=read_json(run/'source_selection.json'); new=[]; recipes=[]
    control=read_json(CONTROL/'source_selection.json')['rows']['guard95']
    require(s['rows']['control_guard95']==control,'Frozen control source differs')
    for recipe in s['recipes']:
        scores={}; seed=recipe['seed']
        for b in (4,5):
            z=gate.npz(run/f'source_results/seed_{seed}_block_{b}.npz',['t','p','ix','y'])
            original=gate.npz(TEMPORAL/f'validation/block_{b}.npz',['t','y'])
            np.testing.assert_array_equal(z['t'][z['ix']],original['t']); np.testing.assert_array_equal(z['y'],original['y']>=200)
            scores[str(b)]={**z,'h':gate.local_max(z['t'],z['p'])}
        low,high=shared.calibrate_source95(scores)
        require((low,high)==(recipe['L'],recipe['H']),'Independent L95/H95 replay differs')
        feasible=low is not None and high is not None; metrics=[]
        if feasible:
            metrics=[dict(block=k,**shared.scalar_confusion(((d['p'][d['ix']]>=low)&(d['h'][d['ix']]>=high)).astype(np.int8),d['y'])) for k,d in scores.items()]
            for a,b in zip(recipe['blocks'],metrics): require(a==b,'Independent source counts differ')
            new.append(dict(seed=seed,**gate.mean_rows(metrics),degenerate=any(m['degenerate'] for m in metrics),source_gate_passed=all(m['recall']>=.95 for m in metrics)))
        recipes.append(dict(seed=seed,L=low,H=high,feasible=feasible,blocks=metrics,radius_samples=15))
    require(s['recipes']==recipes and s['rows']['ensemble_guard95']==new,'Source recipe replay differs')
    decision=select_source(control,new,recipes)
    require(all(s[k]==v for k,v in decision.items()),'Independent source qualification differs')
    return s


def ordinary(run,family,seed,rows):
    require(seed in SEEDS and family in FAMILIES,'Unknown family or seed')
    if family=='control_guard95':
        recipe=next(r for r in read_json(CONTROL/'source_selection.json')['recipes'] if r['seed']==seed)
        t,p,_=gate.infer(LOSS/f'models/unit_BCE/seed_{seed}',rows)
    else:
        recipe=next(r for r in read_json(Path(run)/'source_selection.json')['recipes'] if r['seed']==seed)
        t,p=ensemble_infer(run,seed,rows)
    require(recipe['feasible'],'Source recipe infeasible')
    return dict(t=t,scores=p,state=gate.gate_state(p,gate.local_max(t,p),recipe['L'],recipe['H']))


def matrix():
    return [[a,s] for a in ('A','W50',*FAMILIES) for s in SEEDS]


def freeze(run,authorization=None):
    run=Path(run); authorize(run,authorization)
    with phase(run,'source_freeze_gate'):
        s=source_guard(run); require(s['ensemble_eligible'],'Source rejected; development prohibited')
    ref=read_json(CONTROL/'identity.json')
    with phase(run,'freeze',list(ref['development_unlabeled'])):
        require(not (run/'predictions').exists(),'Freeze is one-shot')
        loss.check_paths(ref['development_unlabeled'])
        base=Path(ref['baseline']); agg=next(Path(p) for p in ref['development_unlabeled'] if p.endswith('aggregate.csv'))
        rows=gate.csv_rows(agg,'refit_17'); dest=run/'predictions'; dest.mkdir(); checks=[]
        (run/'ordinary').mkdir()
        for seed in SEEDS:
            a=gate.npz(base/f'reference/predictions/seed_{seed}/{gate.PARTITION}',['stamps','raw','corrected'])
            w=gate.npz(base/f'predictions/W50/seed_{seed}/{gate.PARTITION}',['stamps','raw'])
            np.testing.assert_array_equal(a['stamps'],w['stamps']); np.testing.assert_array_equal(np.isfinite(a['raw']),np.isfinite(w['raw']))
            states=np.where(np.isfinite(a['corrected']),a['corrected']>=np.array([2000,200,20]),-1).astype(np.int8)
            np.savez_compressed(dest/f'A_{seed}.npz',t=a['stamps'],raw=a['raw'],corrected=a['corrected'],states=states)
            np.savez_compressed(dest/f'W50_{seed}.npz',t=w['stamps'],raw=w['raw'])
            for family in FAMILIES:
                z=ordinary(run,family,seed,rows); replay=ordinary(run,family,seed,rows)
                for k in z: np.testing.assert_array_equal(z[k],replay[k])
                np.testing.assert_array_equal(z['t'],a['stamps']); np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
                other=states.copy(); other[:,1]=z['state']
                np.savez_compressed(dest/f'{family}_{seed}.npz',**z,raw=a['raw'],corrected=a['corrected'],other_states=other)
                saved=gate.npz(dest/f'{family}_{seed}.npz',list(z)+['raw','corrected','other_states'])
                for k in z: np.testing.assert_array_equal(saved[k],replay[k])
                np.testing.assert_array_equal(saved['raw'],a['raw']); np.testing.assert_array_equal(saved['corrected'],a['corrected'])
                np.testing.assert_array_equal(saved['other_states'][:,[0,2]],states[:,[0,2]])
                # Execute the registered public CLI and read its text output back.
                out=run/f'ordinary/{family}_{seed}.csv'
                proc=subprocess.run([sys.executable,'-B',str(run/Path(SCRIPT).name),'predict','--run',str(run),
                    '--authorization',str(Path(authorization).resolve()),'--family',family,'--seed',str(seed),
                    '--aggregate',str(agg),'--output',str(out)],capture_output=True,text=True)
                require(proc.returncode==0,'Registered ordinary CLI failed: '+proc.stderr)
                with out.open(encoding='utf-8',newline='') as stream: ordinary_rows=list(csv.DictReader(stream))
                rt=np.array([utc(r['timestamp']) for r in ordinary_rows],np.int64)
                rp=np.array([float(r['probability_on']) if r['probability_on'] else np.nan for r in ordinary_rows],np.float32)
                rs=np.array([int(r['state']) if r['state'] else -1 for r in ordinary_rows],np.int8)
                for left,right in ((rt,z['t']),(rp,z['scores']),(rs,z['state'])): np.testing.assert_array_equal(left,right)
                checks.append(dict(seed=seed,family=family,ordinary_exact=True,cli_exact=True))
        fresh(run/'inference_checks.json',dict(checks=checks))
        fresh(run/'freeze.json',dict(matrix=matrix(),source_selection_sha256=sha256(run/'source_selection.json'),registration_sha256=sha256(run/'registration.json'),
            artifacts={'predictions/'+k:v for k,v in inventory(dest).items()}|{'ordinary/'+k:v for k,v in inventory(run/'ordinary').items()}|{'inference_checks.json':sha256(run/'inference_checks.json')}))
        return dict(status='frozen',selected=s['selected'],matrix=matrix())


def evaluation_guard(run):
    run=Path(run); s=source_guard(run); f=read_json(run/'freeze.json')
    require(s['ensemble_eligible'] and f['matrix']==matrix(),'Incomplete twelve-cell freeze')
    require(set(f['artifacts'])=={f'predictions/{a}_{s}.npz' for a,s in matrix()}|{f'ordinary/{a}_{s}.csv' for a in FAMILIES for s in SEEDS}|{'inference_checks.json'},'Missing or extra frozen cells')
    for name in ('source_selection','registration'): require(f[name+'_sha256']==sha256(run/(name+'.json')),'Freeze binding changed')
    gate.check(run,f['artifacts'])
    checks=read_json(run/'inference_checks.json')['checks']
    require([(r['family'],r['seed']) for r in checks]==[(a,s) for s in SEEDS for a in FAMILIES] and all(r['ordinary_exact'] and r['cli_exact'] for r in checks),'Missing ordinary exact replay')
    return s


def development_metrics(run,truth,scalar=False):
    rows=[]; supports=[]; pairs=[]; segments=[]
    count=shared.scalar_confusion if scalar else gate.confusion
    for seed in SEEDS:
        a=gate.npz(run/f'predictions/A_{seed}.npz',['t','raw','corrected','states'])
        w=gate.npz(run/f'predictions/W50_{seed}.npz',['t','raw'])
        for z in (a,w): np.testing.assert_array_equal(z['t'],truth['t'])
        valid=np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1]); y=truth['y']>=200
        np.testing.assert_array_equal(valid,np.isfinite(truth['y'])&np.isfinite(a['raw'][:,1]))
        require(valid.sum()==137523,'Effective support changed')
        supports.append(dict(seed=seed,n=int(valid.sum()),mask_sha256=gate.hash_array(valid),t_sha256=gate.hash_array(a['t'][valid])))
        states={name:z['raw'][:,1]>=200 for name,z in (('A',a),('W50',w))}
        for family in FAMILIES:
            z=gate.npz(run/f'predictions/{family}_{seed}.npz',['t','raw','corrected','other_states','scores','state'])
            for k in ('t','raw','corrected'): np.testing.assert_array_equal(z[k],a[k])
            np.testing.assert_array_equal(z['other_states'][:,[0,2]],a['states'][:,[0,2]])
            np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
            np.testing.assert_array_equal(z['state']>=0,np.isfinite(z['scores']))
            states[family]=z['state']==1
        for family,pred in states.items(): rows.append(dict(seed=seed,arm=family,**count(pred[valid].astype(np.int8),y[valid])))
        after=states['ensemble_guard95']
        for reference in ('W50','control_guard95'):
            before=states[reference]; removed=valid&~y&before&~after; added=valid&~y&~before&after
            pairs.append(dict(seed=seed,reference=reference,removed_fp=int(removed.sum()),added_fp=int(added.sum()),restored_tp=int((valid&y&~before&after).sum()),new_fn=int((valid&y&before&~after).sum())))
            for kind,mask in (('reference_fp',valid&~y&before),('ensemble_fp',valid&~y&after),('removed_fp',removed),('added_fp',added)):
                segments.extend(dict(seed=seed,reference=reference,kind=kind,**r) for r in shared.contiguous_rows(truth['t'],mask))
    return dict(rows=rows,supports=supports),pairs,segments


def evaluate(run,authorization=None):
    run=Path(run); authorize(run,authorization)
    with phase(run,'evaluation_gate'):
        s=evaluation_guard(run); require(not (run/'evaluation').exists(),'Evaluation is one-shot; no repeat')
    info=read_json(CONTROL/'identity.json')['truth_cache']
    with phase(run,'evaluate',[info['path']]):
        dest=run/'evaluation'; dest.mkdir(); fresh(dest/'started.json',dict(freeze_sha256=sha256(run/'freeze.json')))
        require(sha256(Path(info['path']))==info['sha256'],'Task12 cached truth changed')
        truth=gate.npz(Path(info['path']),['t','y'])
        metrics,pairs,segments=development_metrics(run,truth)
        # Production summary and independent verification deliberately use
        # different implementations of the original eight acceptance gates.
        result=loss.summary(run,metrics)
        result.update(new_fits=0,updates=0)
        np.savez_compressed(dest/'truth_once.npz',**truth)
        fresh(dest/'metrics.json',metrics); fresh(dest/'summary.json',result)
        gate.csv_save(dest/'confusions.csv',metrics['rows']); fresh(dest/'paired_analysis.json',dict(paired=pairs,diagnostic_only=True))
        gate.csv_save(dest/'fp_segments.csv',segments)
        fresh(run/'evaluation_complete.json',dict(truth_cache_loads=1,original_truth_csv_reads=0,freeze_sha256=sha256(run/'freeze.json'),artifacts=inventory(dest)))
        return result


def verify(run,authorization=None):
    run=Path(run)
    if not (run/'source_started.json').exists():
        with phase(run,'verify_preparation'):
            registered(run)
        return dict(status='preparation_verified_disabled',new_fits=0,optimizer_updates=0,development_reads=0)
    authorize(run,authorization)
    with phase(run,'verify'):
        s=source_guard(run)
        if (run/'freeze.json').exists(): evaluation_guard(run)
        if (run/'evaluation_complete.json').exists():
            done=read_json(run/'evaluation_complete.json'); gate.check(run/'evaluation',done['artifacts'])
            require(done['truth_cache_loads']==1 and done['original_truth_csv_reads']==0 and done['freeze_sha256']==sha256(run/'freeze.json'),'Evaluation seal differs')
            truth=gate.npz(run/'evaluation/truth_once.npz',['t','y'])
            metrics,pairs,segments=development_metrics(run,truth,scalar=True)
            require(metrics==read_json(run/'evaluation/metrics.json'),'Independent scalar counts differ')
            require(pairs==read_json(run/'evaluation/paired_analysis.json')['paired'],'Independent paired changes differ')
            with (run/'evaluation/fp_segments.csv').open(encoding='utf-8',newline='') as stream: saved=list(csv.DictReader(stream))
            require(saved==[{k:str(v) for k,v in r.items()} for r in segments],'Independent FP segments differ')
            independent=shared.independent_formal_summary(metrics['rows'],metrics['supports'],s['selected'])
            summary=read_json(run/'evaluation/summary.json')
            require(all(summary[k]==independent[k] for k in independent),'Independent eight gates differ')
        return dict(status='execution_verified',new_fits=0,optimizer_updates=0)


def immutable_report(path,lines):
    content='\n\n'.join(lines)+'\n'
    if path.exists(): require(path.read_text(encoding='utf-8')==content,'Existing report differs; no overwrite')
    else: path.write_text(content,encoding='utf-8')


def metric_table(rows,include_arm=True):
    header='|seed|候选|P|R|F1|TP|FP|FN|TN|有效点数|' if include_arm else '|seed|块|P|R|F1|TP|FP|FN|TN|有效点数|'
    lines=[header,'|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        name=r.get('arm',r.get('block',''))
        lines.append('|'+ '|'.join([str(r['seed']),str(name),*[f"{r[k]:.9f}" for k in ('precision','recall','f1')],*[str(r[k]) for k in ('tp','fp','fn','tn','n')]])+'|')
    return '\n'.join(lines)


def write_source_report(run,s):
    lines=['# Task16 来源阶段报告',
           '结论：来源通过。' if s['ensemble_eligible'] else '结论：来源拒绝。禁止生成开发预测、读取开发数值或开发真值。',
           f"唯一来源预选：`{s['selected']}`。完全平局保留 control_guard95。",
           '|来源资格|结果|\n|---|---|\n'+'\n'.join(f"|{k}|{'通过' if v else '拒绝'}|" for k,v in s['qualification'].items()),
           '|seed|L95|H95|可行|\n|---|---|---|---|\n'+'\n'.join(f"|{r['seed']}|{r['L']}|{r['H']}|{r['feasible']}|" for r in s['recipes'])]
    blocks=[dict(seed=r['seed'],**b) for r in s['recipes'] for b in r['blocks']]
    if blocks: lines += ['融合候选逐来源块：',metric_table(blocks,False)]
    # Frozen control block metrics come only from its source recipes.
    cs=read_json(CONTROL/'source_selection.json')
    control_blocks=[dict(seed=r['seed'],**b) for r in cs['recipes'] for b in r.get('blocks',[])]
    if control_blocks: lines += ['冻结 control_guard95 逐来源块：',metric_table(control_blocks,False)]
    c=s['rows']['control_guard95']; new=s['rows']['ensemble_guard95']
    lines += ['|seed|候选|平均P|平均R|平均F1|\n|---|---|---|---|---|\n'+'\n'.join(
        f"|{r['seed']}|{arm}|{r['precision']:.9f}|{r['recall']:.9f}|{r['f1']:.9f}|" for arm,rows in s['rows'].items() for r in rows)]
    lines += ['|seed|ΔP 对control|ΔR 对control|ΔF1 对control|\n|---|---|---|---|\n'+'\n'.join(
        f"|{b['seed']}|{b['precision']-a['precision']:+.9f}|{b['recall']-a['recall']:+.9f}|{b['f1']-a['f1']:+.9f}|" for a,b in zip(c,new))]
    lines += ['新训练0次、优化器更新0次。来源资格与唯一预选在开发之前冻结；开发结果不能改选。',
              '检查点共享训练与来源验证，错误相关且有来源选择过拟合风险；不能把三个检查点视作三个独立复现。']
    immutable_report(Path(run)/'source_report.md',lines)


def report(run,authorization=None):
    run=Path(run); result=verify(run,authorization)
    prepared=not (run/'source_started.json').exists()
    lines=['# Task16 同种子 top3 等权概率融合'+('准备报告' if prepared else '执行报告'),
           '新训练 **0** 次；优化器更新 **0** 次；Task15 增强模型不纳入。最终8/9/20保持隔离，UI默认未切换。']
    if prepared: lines += ['状态：**准备完成，配置禁用**。真实融合推理、融合源校准与开发评价均未执行。House17 数值、数值hash与shape均未读取。',
                           '后续执行必须由根另行取得绑定本快照 registration.json 的明确人类 Task16 新规则授权；Task14 历史授权无效。']
    lines += ['|seed|合格旧检查点数|固定top3 epoch|\n|---|---|---|\n'+'\n'.join(
        f"|{seed}|{d['qualified']}|{', '.join(str(r['epoch']) for r in d['selected'])}|" for seed,d in read_json(run/'checkpoint_audit.json')['seeds'].items()),
        '36个旧单点检查点均从来源缓存重放 source85 资格和排序，并与 history exact 对照。融合为逐seed sigmoid概率 float64求和再转float32，NaN不填充，支持必须exact一致。',
        '根另行独立重放36个旧检查点，确认相同top3；该证据为外部交叉核对，本快照仍以自身 audit/access/identity 为依据。',
        '检查点共享训练、初始化与来源验证块，存在统计依赖和来源验证过拟合风险。House17是重复工程开发，不是独立泛化证据。旧单点错误高度重叠，不能保证融合大幅提升。',
        '[完整固定协议](protocol.md) · [旧来源审计](checkpoint_audit.json) · [来源身份](identity.json) · [准备读取审计](prepare_access.json) · [冻结注册](registration.json)']
    if (run/'source_selection.json').exists():
        s=read_json(run/'source_selection.json'); write_source_report(run,s)
        lines += [f"来源{'通过' if s['ensemble_eligible'] else '拒绝'}；唯一预选 `{s['selected']}`。",'[来源完整报告](source_report.md)']
    if (run/'evaluation_complete.json').exists():
        metrics=read_json(run/'evaluation/metrics.json'); summary=read_json(run/'evaluation/summary.json')
        lines += [f"开发原八门槛：{'全部通过' if summary['development_acceptance_passed'] else '未全部通过'}；评价保持来源预选不变。",
            '|原验收门槛|结果|\n|---|---|\n'+'\n'.join(f"|{k}|{'通过' if v else '拒绝'}|" for k,v in summary['acceptance_gates'].items()),metric_table(metrics['rows']),
            '[逐seed混淆](evaluation/confusions.csv) · [成对FP变化](evaluation/paired_analysis.json) · [连续FP片段](evaluation/fp_segments.csv) · [12格封存](freeze.json)',
            '生产八门槛由原 Task12 summary 计算；独立复核使用单独标量混淆和独立八门槛实现，核对全部计数与判定。']
    path=run/('preparation_report.md' if prepared else 'execution_report.md')
    immutable_report(path,lines)
    return dict(report=str(path),**result)


def predict(run,authorization=None,*,family=None,seed=None,aggregate=None,output=None):
    run=Path(run); authorize(run,authorization)
    with phase(run,'predict_gate'):
        s=source_guard(run); require(s['ensemble_eligible'],'Source rejected')
    family=family or s['selected']
    require(seed in SEEDS and family in FAMILIES and aggregate and output and not Path(output).exists(),'Fresh output, aggregate, frozen family and seed required')
    with phase(run,'predict',[aggregate]):
        with Path(aggregate).open(encoding='utf-8',newline='') as stream: rows=list(csv.DictReader(stream))
        z=ordinary(run,family,seed,rows)
        gate.csv_save(Path(output),[dict(timestamp=iso(int(t)),probability_on=float(p) if np.isfinite(p) else '',state=int(v) if v>=0 else '') for t,p,v in zip(z['t'],z['scores'],z['state'])])
        return dict(output=str(output),family=family,seed=seed)
