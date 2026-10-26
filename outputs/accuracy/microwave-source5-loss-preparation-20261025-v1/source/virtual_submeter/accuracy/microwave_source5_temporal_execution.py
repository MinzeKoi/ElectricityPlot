"""Authorized fixed control_BCE50 temporal experiment; at most three starts."""
import contextlib
import csv
import functools
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from ..data.common import DataError, inventory, read_json, write_json, sha256, utc, iso
from ..models.arrays import aggregate_arrays, numeric
from ..models.center_mask_tcn import (CenterMaskTCN, tensor_input, expand_context,
    load_state_model, predict_windows, predict_state_rows, context_at_complete_centers, KIND)
from ..models.state_selection import pr_curve
from . import microwave_source5_temporal as f
from .microwave_diagnosis import load_npz, metrics, save_csv, segments
from .microwave_sampling import BatchStream, check_artifacts, rank
from .microwave_background import array_hash
from . import microwave_tcn_execution as parent

ROOT = f.ROOT
WORKSPACE = Path(read_json(ROOT/'workspace_identity.json')['root']) if (ROOT/'workspace_identity.json').exists() else ROOT
UP = WORKSPACE/'outputs/accuracy/microwave-tcn-20261025-v3'
LOSS = WORKSPACE/'outputs/accuracy/microwave-loss-correction-20261025-v2'
EXEC_PLAN = WORKSPACE/'configs/microwave_source5_temporal_execution_20261025.json'


@contextlib.contextmanager
def read_guard(run, allowed=(), truth=False):
    """Restrict numerical inputs; record data/weight reads, including hash reads."""
    state = dict(active=True, reads=set(), forbidden=[])
    permitted = {str(Path(v).resolve()).lower() for v in allowed}
    local = str(Path(run).resolve()).lower()+os.sep
    def audit(event,args):
        if event != 'open' or not state['active'] or not isinstance(args[0],str): return
        name = str(Path(args[0]).resolve()).lower()
        mode = args[1]
        if mode is not None and any(c in mode for c in 'wax+'): return
        if mode is None and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT): return
        protected = name.endswith(('.npz','.csv','.pt'))
        forbidden = (not truth and ('truth_once' in name or 'evaluation_labels' in name))
        if protected and name not in permitted and not name.startswith(local): forbidden = True
        if forbidden:
            state['forbidden'].append(name)
            raise DataError('Unregistered numerical/truth input: '+name)
        if protected: state['reads'].add(name)
    sys.addaudithook(audit)
    try: yield state
    finally: state['active']=False


def execution_plan():
    p = read_json(f.PLAN)
    e = read_json(EXEC_PLAN)
    if (e['source_recipe'] != 'control_BCE50' or e['maximum_new_fits'] != 3
            or e['seeds'] != f.SEEDS or e['execution_condition_confirmed'] is not True):
        raise DataError('Only the authorized three-seed control_BCE50 matrix is supported')
    return {**p, **e, 'pos_weight':50.}


def verify_condition(receipt):
    if f.condition_receipt(receipt) != 'control_BCE50':
        raise DataError('Actual experiment is locked to control_BCE50')
    selection = read_json(LOSS/'source_selection.json')
    freeze = read_json(LOSS/'freeze.json')
    if (selection['selected_arm'] != 'TCN127' or not selection['source_only']
            or sha256(LOSS/'source_selection.json') != receipt['task8_source_selection_sha256']
            or sha256(LOSS/'freeze.json') != receipt['task8_freeze_sha256']
            or freeze['source_selection_sha256'] != receipt['task8_source_selection_sha256']
            or any(r['eligible'] for r in selection['arms'] if r['arm'] != 'TCN127')):
        raise DataError('Task8 frozen source selection differs from authorized control recipe')
    identity = read_json(LOSS/'parent_identity.json')
    if (Path(identity['run']).resolve() != UP.resolve() or sha256(UP/'freeze.json') != identity['freeze_sha256']
            or sha256(UP/'registration.json') != identity['registration_sha256']):
        raise DataError('Task8 control is not bound to Task7v3')
    return dict(task8_freeze_sha256=sha256(LOSS/'freeze.json'),
                task8_source_selection_sha256=sha256(LOSS/'source_selection.json'),
                control_freeze_sha256=sha256(UP/'freeze.json'),
                control_registration_sha256=sha256(UP/'registration.json'),
                source_recipe='control_BCE50',source_only=True,development_scores_read=False)


def source_inputs():
    audit = read_json(ROOT/'docs/MICROWAVE_SOURCE_VALIDATION_AUDIT_20261025.json')
    candidates = [v for v in audit['access_log'] if v['kind']=='source_data'
                  and 'source_validation_refit_5_0' in v['path']]
    if len({v['path'] for v in candidates}) != 2:
        raise DataError('Expected exactly audited House5 aggregate/microwave inputs')
    return {Path(v['path']).stem:dict(path=str((ROOT/v['path']).resolve()),sha256=v['sha256']) for v in candidates}


def bounded_phase(phase):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(run,*args,**kwargs):
            run=Path(run)
            control=read_json(run/'control_identity.json')
            allowed=[Path(control['run'])/'models'/k for k in control['models']]
            if phase in ('freeze','evaluate','verify','report'):
                baseline=read_json(run/'baseline_identity.json')
                base=Path(baseline['reference_run'])
                for seed in f.SEEDS:
                    for relative in (f'reference/predictions/seed_{seed}',f'predictions/W50/seed_{seed}'):
                        allowed.append(base/relative/f'{parent.PARTITION}.npz')
                allowed.append(base/'protocol'/baseline['development_partition']['artifacts']['aggregate']['path'])
                if phase=='evaluate':
                    allowed.append(base/'protocol'/baseline['development_partition']['artifacts']['microwave']['path'])
            truth=phase=='evaluate' or (phase in ('verify','report') and (run/'evaluation_complete.json').exists())
            with read_guard(run,allowed,truth=truth): return function(run,*args,**kwargs)
        return wrapped
    return decorate


def prepare(run, authorization):
    run=Path(run)
    if run.exists(): raise DataError('New run required; no overwrite')
    receipt=read_json(authorization)
    condition=verify_condition(receipt)
    p=execution_plan()
    inputs=source_inputs()
    allowed=[UP/'TCN127.npz', *[UP/f'validation/TCN127/block_{i}.npz' for i in (4,5)],
             *[Path(v['path']) for v in inputs.values()]]
    control_hashes=read_json(UP/'training_complete.json')['artifacts']
    bound={k:v for k,v in control_hashes.items() if k.startswith('TCN127/')}
    allowed += [UP/'models'/k for k in bound]
    with read_guard(run,allowed) as access:
        reg=read_json(UP/'registration.json')
        for key in ('TCN127.npz','normalization.json','baseline_identity.json','baseline_settings.json','source_identity.json'):
            check_artifacts(UP,{key:reg['artifacts'][key]})
        check_artifacts(UP/'models',bound)
        run.mkdir(parents=True)
        # Full executable snapshot, no mutable shared imports after registration.
        shutil.copytree(ROOT/'src/virtual_submeter',run/'source/virtual_submeter',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(ROOT/'scripts/run_microwave_source5_temporal.py',run/'run_microwave_source5_temporal.py')
        shutil.copyfile(ROOT/'scripts/predict_microwave_tcn.py',run/'predict_microwave_tcn.py')
        shutil.copyfile(ROOT/'docs/MICROWAVE_SOURCE5_TEMPORAL_PLAN_20261025.md',run/'protocol.md')
        for name in ('test_microwave_source5_temporal.py','test_microwave_source5_temporal_execution.py'):
            shutil.copyfile(ROOT/'tests'/name,run/name)
        shutil.copyfile(EXEC_PLAN,run/'execution_plan.json')
        shutil.copyfile(authorization,run/'condition_authorization.json')
        for key in ('normalization.json','baseline_identity.json','baseline_settings.json','source_identity.json'):
            shutil.copyfile(UP/key,run/key)
        write_json(run/'plan.json',p)
        write_json(run/'workspace_identity.json',dict(root=str(WORKSPACE.resolve())))
        write_json(run/'condition_identity.json',condition)
        write_json(run/'control_identity.json',dict(run=str(UP.resolve()),models=bound,
            training_complete_sha256=sha256(UP/'training_complete.json')))
        write_json(run/'source5_inputs.json',inputs)
        raw={}
        for channel,item in inputs.items():
            if sha256(Path(item['path'])) != item['sha256']: raise DataError('Audited House5 input changed')
            with Path(item['path']).open(encoding='utf-8',newline='') as stream:
                raw[channel]=list(csv.DictReader(stream))
            if {r['household_id'] for r in raw[channel]} != {'refit_5'}:
                raise DataError('Wrong source household')
        t,a=aggregate_arrays(raw['aggregate'],8)
        lt=np.array([utc(r['timestamp']) for r in raw['microwave']],np.int64)
        if not np.array_equal(t,lt): raise DataError('Source label/aggregate timestamps differ')
        y=np.array([numeric(r['power_w']) for r in raw['microwave']],np.float32)
        original=load_npz(UP/'TCN127.npz')
        if (len(original['y']),int((original['y']>=200).sum())) != (259806,849):
            raise DataError('Original partial pool differs')
        added=[f.build_block(t,a,y,i,'train') for i in range(4)]
        added={k:np.concatenate([d[k] for d in added]) for k in added[0]}
        if (len(added['y']),int((added['y']>=200).sum())) != (76898,1426):
            raise DataError('Added House5 support does not match frozen audit')
        normalization=read_json(run/'normalization.json')
        combined=f.append_training(original,added,normalization,read_json(UP/'normalization.json'))
        np.savez_compressed(run/'train.npz',**combined)
        np.savez_compressed(run/'added_identity.npz',**{k:added[k] for k in ('t','y','house','house_name','block','event','raw_index','mask_identity')})
        val={}
        (run/'validation').mkdir()
        for i,low,high in f.VALIDATION:
            key=f'block_{i}'
            check_artifacts(UP,{f'validation/TCN127/{key}.npz':reg['artifacts'][f'validation/TCN127/{key}.npz']})
            old=load_npz(UP/f'validation/TCN127/{key}.npz')
            val[key]=f.build_block(t,a,y,i,'validation',old)
            # Exact input symmetry with the frozen source control evaluation.
            np.testing.assert_array_equal(val[key]['x'],old['x'])
            np.savez_compressed(run/f'validation/{key}.npz',**val[key])
        if (sum(len(d['y']) for d in val.values()),sum(int((d['y']>=200).sum()) for d in val.values())) != (34351,201):
            raise DataError('Retained strict31 validation support differs')
        train_t=set(added['t'].tolist())
        if any(train_t.intersection(d['t'].tolist()) for d in val.values()): raise DataError('Center role overlap')
        train_events=set(added['event'][added['y']>=200].tolist())
        if any(train_events.intersection(d['event'][d['y']>=200].tolist()) for d in val.values()): raise DataError('Event role overlap')
        support=dict(original=[259806,849],added=[76898,1426],combined=[336704,2275],validation=[34351,201],
            normalization_exact=sha256(run/'normalization.json')==sha256(UP/'normalization.json'),
            original_rows_exact=True,house5_numeric_index=2,context_and_events_within_block=True,
            validation_inputs_equal_original_frozen=True,train_validation_center_and_event_disjoint=True)
        write_json(run/'support.json',support)
        # Bind pool identities and replay all indices without fitting.
        inits={}
        for seed in f.SEEDS:
            stream=BatchStream(combined,'event',seed,p)
            indices=np.stack([stream.next() for _ in range(3528)])
            if not np.all((combined['y'][indices]>=200).sum(axis=1)==16): raise DataError('Quota differs')
            np.savez_compressed(run/f'sampling_{seed}.npz',indices=indices)
            counts=np.bincount(indices.ravel(),minlength=len(combined['y']))
            np.savez_compressed(run/f'dry_exposure_{seed}.npz',count=counts)
            initial,_=parent.configure(seed,p)
            saved=torch.load(UP/f'models/TCN127/seed_{seed}/initial.pt',weights_only=True,map_location='cpu')
            for k,v in initial.state_dict().items():
                if not torch.equal(v,saved[k]): raise DataError('Same-seed initial parameters differ')
            shutil.copyfile(UP/f'models/TCN127/seed_{seed}/initial.pt',run/f'initial_{seed}.pt')
            inits[str(seed)]=sha256(run/f'initial_{seed}.pt')
        write_json(run/'initializations.json',inits)
        write_json(run/'matrix.json',dict(arms=f.ARMS,seeds=f.SEEDS,new_fits=3,control_fits=0,recipe='control_BCE50',updates=3528))
        write_json(run/'runtime.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__))
        write_json(run/'preparation_access.json',dict(reads=sorted(access['reads']),forbidden=access['forbidden'],house17_truth_reads=0,real_fits=0))
        write_json(run/'registration.json',dict(schema='source5_temporal_execution_v1',artifacts=inventory(run),
            actual_fits_at_registration=0,maximum_new_starts=3,source_recipe='control_BCE50',
            compatible_with_original_cross_household_protocol=False,authorization_quote=p['human_authorization_quote']))
    return registered(run)[1]


def registered(run):
    run=Path(run)
    reg=read_json(run/'registration.json')
    check_artifacts(run,reg['artifacts'])
    p,m=read_json(run/'plan.json'),read_json(run/'matrix.json')
    if (m != dict(arms=f.ARMS,seeds=f.SEEDS,new_fits=3,control_fits=0,recipe='control_BCE50',updates=3528)
            or p['source_recipe']!='control_BCE50' or p['checkpoints']!=12 or p['steps_per_checkpoint']!=294
            or p['batch_size']!=256 or p['positive_per_batch']!=16 or p['pos_weight']!=50.
            or p['learning_rate']!=.001 or p['cpu_threads']!=4): raise DataError('Fixed experiment changed')
    control=read_json(run/'control_identity.json')
    if sha256(UP/'training_complete.json')!=control['training_complete_sha256']: raise DataError('Control training identity changed')
    check_artifacts(UP/'models',control['models'])
    return p,m


def validation(run):
    return {f'block_{i}':load_npz(Path(run)/f'validation/block_{i}.npz') for i in (4,5)}


def choose(checkpoints,blocks,p):
    selected=f.reselect_checkpoints(checkpoints,blocks,p,expected_epochs=p['checkpoints'])
    if selected is not None: return {**selected,'source_gate_passed':True}
    return dict(epoch=p['checkpoints'],threshold=.5,source_gate_passed=False,
                block_metrics={k:metrics(checkpoints[-1][k],d['y'],.5) for k,d in blocks.items()})


def write_selected(dest,seed,arm,chosen,meta,epoch_path,registration,updates,recipe='control_BCE50'):
    shutil.copyfile(epoch_path,dest/'weights.pt')
    info=dict(**{**meta,'window_samples':127},kind=KIND,inference_support='strict_complete_31',truth_threshold_w=200,
              decision_threshold_probability=chosen['threshold'],selected=chosen,
              arm=arm,seed=seed,updates=updates,parameter_count=5329,recipe=recipe,
              weights_sha256=sha256(dest/'weights.pt'),registration_sha256=registration,
              same_household_temporal_engineering_experiment=True,power_improvement_claim=False)
    write_json(dest/'model.json',info)
    return parent.source_row(seed,chosen)


def fit_one(dest,seed,p,d,blocks,meta,indices,initial,registration,*,pos_weight=50.,recipe='control_BCE50',arm='source5'):
    """Used identically by real execution and the small synthetic integration test."""
    meta={**meta,'window_samples':127}
    dest.mkdir(parents=True,exist_ok=False)
    if (pos_weight,recipe) not in ((50.,'control_BCE50'),(1.,'unit_BCE')):
        raise DataError('Unsupported fixed loss recipe')
    write_json(dest/'started.json',dict(seed=seed,arm=arm,maximum_updates=p['checkpoints']*p['steps_per_checkpoint']))
    write_json(dest/'progress.json',dict(completed_updates=0,attempted_updates=0))
    model,opt=parent.configure(seed,p)
    for k,v in model.state_dict().items():
        if not torch.equal(v,initial[k]): raise DataError('Initialization changed')
    torch.save(initial,dest/'initial.pt')
    x=tensor_input(meta,d['x'],d['mask'])
    y=torch.from_numpy((d['y']>=200).astype(np.float32))
    exposure=np.zeros(len(y),np.int64)
    history,batches,all_scores=[],[],[]
    started=time.perf_counter()
    for epoch in range(1,p['checkpoints']+1):
        model.train()
        for step in range(p['steps_per_checkpoint']):
            update=(epoch-1)*p['steps_per_checkpoint']+step
            idx=indices[update]
            write_json(dest/'progress.json',dict(completed_updates=update,attempted_updates=update+1))
            positive=d['y'][idx]>=200
            if positive.sum()!=16 or len(idx)!=256: raise DataError('Training quota changed')
            opt.zero_grad(set_to_none=True)
            point=torch.nn.functional.binary_cross_entropy_with_logits(model(x[idx]),y[idx],pos_weight=torch.tensor(float(pos_weight)),reduction='none')
            loss=point.mean()
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise DataError('Nonfinite loss/gradient; no automatic retry')
            opt.step()
            write_json(dest/'progress.json',dict(completed_updates=update+1,attempted_updates=update+1))
            np.add.at(exposure,idx,1)
            batches.append(dict(update=update+1,positive=16,negative=240,loss=float(loss.detach()),
                positive_loss_sum=float(point[positive].sum().detach()),negative_loss_sum=float(point[~positive].sum().detach()),
                indices_sha256=array_hash(idx)))
        scores={k:predict_windows(meta,model,v['x']) for k,v in blocks.items()}
        all_scores.append(scores)
        point=f.reselect_checkpoints([scores],blocks,p,expected_epochs=1)
        history.append(dict(epoch=epoch,selection=point))
        torch.save(model.state_dict(),dest/f'epoch_{epoch}.pt')
        for k,v in scores.items(): np.savez_compressed(dest/f'epoch_{epoch}_{k}.npz',scores=v,t=blocks[k]['t'])
        write_json(dest/'history.json',dict(rows=history))
        save_csv(dest/'batches.csv',batches)
        print(f'source5 seed={seed} checkpoint={epoch}/{p["checkpoints"]} eligible={point is not None} seconds={time.perf_counter()-started:.1f}',flush=True)
    chosen=choose(all_scores,blocks,p)
    row=write_selected(dest,seed,arm,chosen,meta,dest/f'epoch_{chosen["epoch"]}.pt',registration,len(indices),recipe)
    np.savez_compressed(dest/'exposure.npz',count=exposure,t=d['t'],house=d['house'],event=d['event'])
    write_json(dest/'complete.json',dict(updates=len(indices),seconds=time.perf_counter()-started,artifacts=inventory(dest)))
    return row


def select_control(run,p,blocks,meta):
    rows=[]
    # Read frozen validation predictions only after proving same inputs at prepare.
    for seed in f.SEEDS:
        dest=run/f'models/control/seed_{seed}'
        dest.mkdir(parents=True,exist_ok=False)
        old=UP/f'models/TCN127/seed_{seed}'
        scores=[]
        for epoch in range(1,13):
            value={}
            for key,block in blocks.items():
                arr=load_npz(old/f'epoch_{epoch}_{key}.npz')
                np.testing.assert_array_equal(arr['t'],block['t'])
                value[key]=arr['scores']
                shutil.copyfile(old/f'epoch_{epoch}_{key}.npz',dest/f'epoch_{epoch}_{key}.npz')
            scores.append(value)
        chosen=choose(scores,blocks,p)
        row=write_selected(dest,seed,'control',chosen,meta,old/f'epoch_{chosen["epoch"]}.pt',sha256(run/'registration.json'),0)
        # Prove saved old epoch probabilities reproduce on identical input arrays.
        info,model=load_state_model(dest)
        for key,block in blocks.items():
            np.testing.assert_array_equal(predict_windows(info,model,block['x']),scores[chosen['epoch']-1][key])
        write_json(dest/'complete.json',dict(retrained=False,artifacts=inventory(dest)))
        rows.append(row)
    return rows


@bounded_phase('train')
def train(run):
    run=Path(run)
    p,m=registered(run)
    if (run/'training_started.json').exists() or (run/'models').exists(): raise DataError('Training already attempted; no restart')
    write_json(run/'training_started.json',dict(registration_sha256=sha256(run/'registration.json'),maximum_new_starts=3))
    blocks=validation(run)
    meta={**read_json(run/'normalization.json'),'window_samples':127}
    rows={'control':select_control(run,p,blocks,meta),'source5':[]}
    d=load_npz(run/'train.npz')
    for seed in f.SEEDS:
        indices=load_npz(run/f'sampling_{seed}.npz')['indices']
        initial=torch.load(run/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
        rows['source5'].append(fit_one(run/f'models/source5/seed_{seed}',seed,p,d,blocks,meta,indices,initial,sha256(run/'registration.json')))
    write_json(run/'source_rows.json',rows)
    source_select(run)
    write_json(run/'training_complete.json',dict(new_fits=3,control_retrains=0,total_updates=10584,
        source_rows_sha256=sha256(run/'source_rows.json'),source_selection_sha256=sha256(run/'source_selection.json'),
        artifacts=inventory(run/'models')))
    return dict(status='trained_three_fixed_models',source_selection=read_json(run/'source_selection.json'))


def source_select(run):
    run=Path(run)
    choice=f.select_source(read_json(run/'source_rows.json'))
    path=run/'source_selection.json'
    if path.exists():
        if read_json(path)!=choice: raise DataError('Frozen source selection changed')
    else: write_json(path,choice)
    return choice


def verify_training(run,ordinary=True):
    run=Path(run)
    p,_=registered(run)
    complete=read_json(run/'training_complete.json')
    if complete['new_fits']!=3 or complete['total_updates']!=10584: raise DataError('Fit/update budget differs')
    check_artifacts(run/'models',complete['artifacts'])
    for name in ('source_rows','source_selection'):
        if complete[name+'_sha256']!=sha256(run/(name+'.json')): raise DataError('Source result changed')
    blocks=validation(run)
    replay={a:[] for a in f.ARMS}
    for arm in f.ARMS:
        for seed in f.SEEDS:
            dest=run/f'models/{arm}/seed_{seed}'
            meta,model=load_state_model(dest)
            scores=[]
            for epoch in range(1,13):
                value={}
                for key,block in blocks.items():
                    z=load_npz(dest/f'epoch_{epoch}_{key}.npz')
                    np.testing.assert_array_equal(z['t'],block['t'])
                    value[key]=z['scores']
                scores.append(value)
            chosen=choose(scores,blocks,p)
            if chosen!=meta['selected']: raise DataError('Checkpoint/threshold selection does not reproduce')
            epoch_path=(UP/f'models/TCN127/seed_{seed}' if arm=='control' else dest)/f'epoch_{chosen["epoch"]}.pt'
            if sha256(epoch_path)!=meta['weights_sha256']: raise DataError('Selected epoch weights differ')
            if ordinary:
                for key,block in blocks.items():
                    np.testing.assert_array_equal(predict_windows(meta,model,block['x']),scores[chosen['epoch']-1][key])
            replay[arm].append(parent.source_row(seed,chosen))
            if arm=='source5':
                progress=read_json(dest/'progress.json')
                if progress!=dict(completed_updates=3528,attempted_updates=3528): raise DataError('Update record differs')
                exposure=load_npz(dest/'exposure.npz')['count']
                np.testing.assert_array_equal(exposure,load_npz(run/f'dry_exposure_{seed}.npz')['count'])
                with (dest/'batches.csv').open(encoding='utf-8',newline='') as stream: batches=list(csv.DictReader(stream))
                indices=load_npz(run/f'sampling_{seed}.npz')['indices']
                if len(batches)!=3528: raise DataError('Batch count differs')
                for row,ix in zip(batches,indices):
                    if row['indices_sha256']!=array_hash(ix) or int(row['positive'])!=16 or int(row['negative'])!=240:
                        raise DataError('Sampling replay differs')
                saved=torch.load(dest/'initial.pt',weights_only=True,map_location='cpu')
                original=torch.load(run/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
                for key in saved:
                    if not torch.equal(saved[key],original[key]): raise DataError('Initialization changed')
    if replay!=read_json(run/'source_rows.json') or f.select_source(replay)!=read_json(run/'source_selection.json'):
        raise DataError('Source-only result does not reproduce')
    return True


def verify_references(run):
    b=read_json(run/'baseline_identity.json')
    base=Path(b['reference_run'])
    for seed in f.SEEDS:
        key=f'seed_{seed}/{parent.PARTITION}.npz'
        for group,path in (('reference/predictions',base/'reference/predictions'),('predictions/W50',base/'predictions/W50')):
            check_artifacts(path,{key:b['artifacts'][group][key]})


@bounded_phase('freeze')
def freeze(run):
    run=Path(run)
    verify_training(run)
    verify_references(run)
    if (run/'predictions').exists() or (run/'freeze.json').exists(): raise DataError('Freeze already attempted')
    (run/'predictions').mkdir()
    rows=parent.target_input(run)  # Aggregate only; no truth cache is touched.
    t,a=aggregate_arrays(rows,8)
    thresholds=read_json(run/'baseline_settings.json')['on_thresholds_w']
    checks=[]
    for arm in f.ARMS:
        for seed in f.SEEDS:
            meta,model=load_state_model(run/f'models/{arm}/seed_{seed}')
            x,c=context_at_complete_centers(meta,t,a)
            scores=np.full(len(t),np.nan,np.float32)
            scores[c]=predict_windows(meta,model,x)
            ordinary=predict_state_rows(meta,model,rows)
            ordinary_scores=np.array([np.nan if v['probability_on'] is None else v['probability_on'] for v in ordinary],np.float32)
            state=np.where(np.isfinite(scores),scores>=meta['decision_threshold_probability'],-1).astype(np.int8)
            np.testing.assert_array_equal(scores,ordinary_scores)
            np.testing.assert_array_equal(state,np.array([-1 if v['state'] is None else int(v['state']) for v in ordinary],np.int8))
            base,w50=parent.reference_arrays(run,seed),parent.reference_arrays(run,seed,'W50')
            for ref in (base,w50):
                np.testing.assert_array_equal(t,ref['stamps'])
                np.testing.assert_array_equal(np.isfinite(scores),np.isfinite(ref['raw'][:,1]))
            np.testing.assert_array_equal(a,base['aggregate'])
            candidate={'stamps':t,'raw':base['raw'].copy()}
            from .microwave_fullsource import aligned_unchanged
            aligned_unchanged(parent.unchanged_view(base,thresholds),parent.unchanged_view(candidate,thresholds))
            np.savez_compressed(run/f'predictions/{arm}_{seed}.npz',t=t,scores=scores,state=state,raw=base['raw'],aggregate=a)
            checks.append(dict(arm=arm,seed=seed,ordinary_scores_exact=True,ordinary_state_exact=True,
                               raw_power_other_devices_exact=True,input_centers=len(c),weights_sha256=meta['weights_sha256']))
    write_json(run/'inference_checks.json',dict(checks=checks,house17_labels_opened=False))
    write_json(run/'freeze.json',dict(predictions=[f'{arm}/{s}' for arm in f.ARMS for s in f.SEEDS],
        source_selection_sha256=sha256(run/'source_selection.json'),training_complete_sha256=sha256(run/'training_complete.json'),
        registration_sha256=sha256(run/'registration.json'),house17_labels_opened=False,
        artifacts={**{'predictions/'+k:v for k,v in inventory(run/'predictions').items()},
                   'inference_checks.json':sha256(run/'inference_checks.json')}))
    evaluation_guard(run)
    return dict(status='six_predictions_frozen_before_truth',source_selection=read_json(run/'source_selection.json'))


def evaluation_guard(run):
    registered(run)
    z=read_json(run/'freeze.json')
    if (z['predictions']!=[f'{a}/{s}' for a in f.ARMS for s in f.SEEDS]
            or z['source_selection_sha256']!=sha256(run/'source_selection.json')
            or z['training_complete_sha256']!=sha256(run/'training_complete.json')
            or z['registration_sha256']!=sha256(run/'registration.json') or z['house17_labels_opened']):
        raise DataError('Incomplete or mutated pre-truth freeze')
    check_artifacts(run,z['artifacts'])
    checks=read_json(run/'inference_checks.json')['checks']
    if len(checks)!=6 or not all(v['ordinary_scores_exact'] and v['ordinary_state_exact'] and v['raw_power_other_devices_exact'] for v in checks):
        raise DataError('Ordinary inference/common-support gates incomplete')


def score(run,truth):
    rows,supports=[],{}
    for seed in f.SEEDS:
        base,w50=parent.reference_arrays(run,seed),parent.reference_arrays(run,seed,'W50')
        mask=np.isfinite(truth)&np.isfinite(w50['raw'][:,1])
        if int(mask.sum())!=137523: raise DataError('Expected 137523 original W50 evaluation points')
        np.testing.assert_array_equal(mask,np.isfinite(truth)&np.isfinite(base['raw'][:,1]))
        supports[str(seed)]=dict(n=int(mask.sum()),mask_sha256=array_hash(mask),timestamps_sha256=array_hash(base['stamps'][mask]))
        for arm,data in (('A',base),('W50',w50)):
            rows.append(dict(arm=arm,seed=seed,**parent.confusion(data['raw'][mask,1]>=200,truth[mask]>=200)))
        for arm in f.ARMS:
            data=load_npz(run/f'predictions/{arm}_{seed}.npz')
            np.testing.assert_array_equal(data['t'],base['stamps'])
            np.testing.assert_array_equal(data['raw'],base['raw'])
            if not np.isfinite(data['scores'][mask]).all() or (data['state'][mask]<0).any(): raise DataError('Lost W50 support')
            rows.append(dict(arm=arm,seed=seed,**parent.confusion(data['state'][mask].astype(bool),truth[mask]>=200)))
    return rows,supports


def summarize(run,rows,supports):
    p=read_json(run/'plan.json')
    groups={arm:[next(r for r in rows if r['arm']==arm and r['seed']==s) for s in f.SEEDS] for arm in ['A','W50']+f.ARMS}
    means={arm:{k:float(np.mean([r[k] for r in values])) for k in ('precision','recall','f1','fp')} for arm,values in groups.items()}
    selected=read_json(run/'source_selection.json')['selected_arm']
    passed=selected is not None and f.development_gate(groups['W50'],groups[selected],p)
    source=read_json(run/'source_selection.json')
    source5_eligible=next(r['eligible'] for r in source['arms'] if r['arm']=='source5')
    return dict(selected_before_development=selected,development_acceptance_passed=bool(passed),means=means,common_support=supports,
        source5_beats_symmetric_control_at_source=bool(selected=='source5' and source5_eligible),
        source5_development_deltas_vs_control={k:means['source5'][k]-means['control'][k] for k in ('precision','recall','f1','fp')},
        declared_house5_training_improvement=bool(passed and selected=='source5'),
        only_source_preselected_candidate_accepted=True,development_diagnostics_do_not_change_selection=True,
        power_improvement_claim=False,default_model_changed=False,final_houses_accessed=False,
        validation_is_fresh_independent=False,development_only_previously_inspected_house17=True,additional_training_permitted=False)


@bounded_phase('evaluate')
def evaluate(run):
    run=Path(run)
    evaluation_guard(run)  # Must precede opening/hashing any truth or truth cache.
    verify_references(run)
    verify_training(run,ordinary=False)
    dest=run/'evaluation'
    dest.mkdir(exist_ok=False)
    write_json(dest/'started.json',dict(freeze_sha256=sha256(run/'freeze.json'),source_selection_sha256=sha256(run/'source_selection.json')))
    b=read_json(run/'baseline_identity.json')
    artifact=b['development_partition']['artifacts']['microwave']
    path=Path(b['reference_run'])/'protocol'/artifact['path']
    if sha256(path)!=artifact['sha256']: raise DataError('Original W50 development truth changed')
    with path.open(encoding='utf-8',newline='') as stream:
        mapping={utc(v['timestamp']):numeric(v['power_w']) for v in csv.DictReader(stream)}
    t=parent.reference_arrays(run,42)['stamps']
    truth=np.array([mapping.get(int(v),np.nan) for v in t],np.float64)
    np.savez_compressed(dest/'truth_once.npz',t=t,y=truth)
    rows,supports=score(run,truth)
    write_json(dest/'metrics.json',dict(rows=rows,supports=supports))
    summary=summarize(run,rows,supports)
    write_json(dest/'summary.json',summary)
    save_csv(dest/'comparison.csv',rows)
    write_json(run/'evaluation_complete.json',dict(freeze_sha256=sha256(run/'freeze.json'),
        source_selection_sha256=sha256(run/'source_selection.json'),truth_original_sha256=artifact['sha256'],
        truth_csv_parses=1,artifacts=inventory(dest)))
    return summary


@bounded_phase('verify')
def verify(run):
    run=Path(run)
    verify_training(run)
    evaluation_guard(run)
    verify_references(run)
    if (run/'evaluation_complete.json').exists():
        complete=read_json(run/'evaluation_complete.json')
        if complete['freeze_sha256']!=sha256(run/'freeze.json') or complete['source_selection_sha256']!=sha256(run/'source_selection.json'):
            raise DataError('Evaluation not bound to freeze')
        check_artifacts(run/'evaluation',complete['artifacts'])
        truth=load_npz(run/'evaluation/truth_once.npz')
        rows,supports=score(run,truth['y'])
        if dict(rows=rows,supports=supports)!=read_json(run/'evaluation/metrics.json') or summarize(run,rows,supports)!=read_json(run/'evaluation/summary.json'):
            raise DataError('Cached once-only evaluation does not reproduce')
    return dict(status='verified',real_new_starts=3,real_new_updates=10584,control_retrains=0)


@bounded_phase('report')
def report(run):
    run=Path(run)
    verify(run)
    if (run/'report').exists(): raise DataError('Report exists; cannot overwrite')
    out=run/'report'
    out.mkdir()
    summary=read_json(run/'evaluation/summary.json')
    rows=read_json(run/'evaluation/metrics.json')['rows']
    source=read_json(run/'source_rows.json')
    save_csv(out/'source_comparison.csv',[dict(arm=a,**r) for a in f.ARMS for r in source[a]])
    save_csv(out/'development_comparison.csv',rows)
    truth=load_npz(run/'evaluation/truth_once.npz')
    blocks=validation(run)
    for arm in f.ARMS:
        for seed in f.SEEDS:
            model=run/f'models/{arm}/seed_{seed}'
            meta=read_json(model/'model.json')
            for key,block in blocks.items():
                scores=load_npz(model/f'epoch_{meta["selected"]["epoch"]}_{key}.npz')['scores']
                curve=pr_curve(scores,block['y']>=200)
                save_csv(out/f'source_PR_{arm}_{seed}_{key}.csv',[{k:v[i].item() for k,v in curve.items()} for i in range(len(curve['threshold']))])
            data=load_npz(run/f'predictions/{arm}_{seed}.npz')
            mask=np.isfinite(truth['y'])&np.isfinite(data['scores'])
            curve=pr_curve(data['scores'][mask],truth['y'][mask]>=200)
            save_csv(out/f'development_PR_{arm}_{seed}.csv',[{k:v[i].item() for k,v in curve.items()} for i in range(len(curve['threshold']))])
            fp=mask&(truth['y']<200)&(data['state']==1)
            w50=parent.reference_arrays(run,seed,'W50')
            compare=mask&(truth['y']<200)&(w50['raw'][:,1]>=200)
            save_csv(out/f'false_positive_points_{arm}_{seed}.csv',[
                dict(timestamp=iso(int(data['t'][i])),score=float(data['scores'][i]),
                     candidate_fp=bool(fp[i]),W50_fp=bool(compare[i]),aggregate_w=float(data['aggregate'][i]),truth_w=float(truth['y'][i]))
                for i in np.flatnonzero(fp|compare)])
    lines=['# House5同户时间工程试验交付（2026-10-25）','',
        f'来源唯一预选：{summary["selected_before_development"]}。相对原W50整体开发验收：{"通过" if summary["development_acceptance_passed"] else "未通过"}。',
        f'新增House5训练在来源对称比较胜过旧控制：{summary["source5_beats_symmetric_control_at_source"]}；可归因于新增训练的达标改善声明：{summary["declared_house5_training_improvement"]}。','',
        '本任务3次新启动，各3528更新，共10584更新；旧控制0重训。只执行control_BCE50；无新搜索、追加拟合或开发改选。',
        '训练新增House5前四冻结块，末端UTC 2014-11-22T23:30:00Z，保留1800秒带；最后两块C31验证34351/201。原2/19 partial 259806/849逐行保留，新增76898/1426，合并336704/2275。原归一化字节相同。',
        '控制与新增均在相同block4/5输入重选checkpoint/公共阈值；原六块阈值不用于对照。',
        '人类授权：“允许这个独立时间对照试验”。本试验明确改变House5仅验证与来源训练/验证整户隔离；不符合原跨户协议。默认守卫保留。',
        '最后4日已历史用于选模，只称复用来源时间验证；House17已查看，仅开发验收，非新独立泛化证据。最终8/9/20不访问。',
        '全部6份开发预测、模型与唯一source预选先封存，普通推理完全一致及原W50同支持通过后才一次解析真值；每seed有效137523点。',
        '原A电器功率、其他电器状态不变；微波炉仅状态分类，未改善功率，不切默认。','',
        '|组|平均P|平均R|平均F1|平均FP|','|---|---:|---:|---:|---:|']
    for a,m in summary['means'].items(): lines.append(f'|{a}|{m["precision"]:.6f}|{m["recall"]:.6f}|{m["f1"]:.6f}|{m["fp"]:.2f}|')
    lines+=['','逐seed混淆矩阵见development_comparison.csv；来源与开发PR、逐点误报对照保存在本目录。',
            '开发两组数据只作冻结后诊断，唯一可验收者仍是事前source预选组。新增数据索引流不同，保持抽样规则/配额/seed，不声称逐批相同。','',
            '普通预测命令（模型目录可替换为所选组和seed）：','```powershell',
            f'.venv-m2/Scripts/python.exe -B "{run}/predict_microwave_tcn.py" --model "{run}/models/{summary["selected_before_development"] or "control"}/seed_42" --aggregate <总表CSV> --output <新输出CSV>',
            '```']
    (out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'summary.json',summary)
    write_json(run/'delivery_manifest.json',dict(artifacts=inventory(run),source_recipe='control_BCE50',new_fits=3,control_retrains=0))
    return dict(report=str(out/'REPORT.zh.md'),**summary)
