"""Task 8: fixed sampling loss correction, with immutable local lifecycle."""
import csv
import hashlib
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from ..data.common import DataError, read_json, write_json, sha256, inventory, utc
from ..models.arrays import aggregate_arrays, numeric
from ..models.center_mask_tcn import (CenterMaskTCN, tensor_input, predict_windows,
    load_state_model, context_at_complete_centers, predict_state_rows)
from .microwave_sampling import BatchStream, check_artifacts, rank
from .microwave_diagnosis import load_npz, metrics, save_csv
from .microwave_fullsource import select_checkpoint, development_gate, aligned_unchanged
from . import microwave_tcn_execution as parent
from .microwave_background import array_hash

ROOT = Path(__file__).resolve().parents[3]
ARMS = ['importance_BCE50', 'unit_BCE']
ORDER = ['TCN127'] + ARMS
SEEDS = [42, 1019, 2026]


def probabilities(d, p):
    for key in ['y','house','t','event']:
        if np.asarray(d[key]).ndim != 1 or len(d[key]) != len(d['y']) or not np.isfinite(d[key]).all():
            raise DataError('Invalid/missing training identity or label')
    if np.any(d['y'] < 0) or np.any(d['house'] < 0) or np.any(d['house'] != np.floor(d['house'])) or np.any(d['t'] != np.floor(d['t'])):
        raise DataError('Invalid training center/household/label')
    sampler = BatchStream(d, 'event', 42, p)
    if len(set(zip(d['house'].tolist(), d['t'].tolist()))) != len(d['y']):
        raise DataError('Duplicate household/center identity')
    q = p['positive_per_batch']/p['batch_size']
    prob = np.full(len(d['y']), (1-q)/len(sampler.negative), np.float64)
    supports = []
    for (house, day), events in sorted(sampler.groups.items()):
        for event, centers in sorted(events.items()):
            prob[centers] = q/(len(sampler.groups)*len(events)*len(centers))
            supports.append(dict(house=house, utc_day=day, event=event, centers=len(centers),
                                 groups=len(sampler.groups), events_in_group=len(events)))
    importance = 1/(len(prob)*prob)
    if not np.isfinite(importance).all() or np.any(importance <= 0):
        raise DataError('Nonfinite/nonpositive importance')
    np.testing.assert_allclose(prob.sum(), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(prob[d['y'] >= 200].sum(), q, rtol=0, atol=1e-12)
    return prob, importance, supports


def losses(logits, y, arm, importance):
    if arm not in ARMS or not torch.isfinite(y).all() or not torch.isfinite(importance).all() or (importance <= 0).any():
        raise DataError('Invalid loss inputs')
    point = torch.nn.functional.binary_cross_entropy_with_logits(logits, y,
        pos_weight=torch.tensor(50. if arm == ARMS[0] else 1.), reduction='none')
    return point*importance if arm == ARMS[0] else point


def select_source(rows, p):
    if set(rows) != set(ORDER) or any([r['seed'] for r in rows[a]] != SEEDS for a in ORDER):
        raise DataError('Complete three-arm paired source matrix required')
    old, summaries = rows['TCN127'], []
    for arm in ORDER:
        rr = rows[arm]
        ok = all(r['source_gate_passed'] and not r['degenerate'] and r['recall'] >= .85 for r in rr)
        if arm != 'TCN127':
            ok &= (np.mean([r['f1'] for r in rr]) >= np.mean([r['f1'] for r in old])
                   and all(b['recall']-a['recall'] >= -.05 for a,b in zip(old,rr))
                   and sum(b['precision'] > a['precision'] for a,b in zip(old,rr)) >= 2)
        summaries.append(dict(arm=arm, eligible=bool(ok), mean_precision=float(np.mean([r['precision'] for r in rr])),
                              mean_f1=float(np.mean([r['f1'] for r in rr]))))
    eligible = [r for r in summaries if r['eligible']]
    best = max(eligible,key=lambda r:(r['mean_precision'],r['mean_f1'],-ORDER.index(r['arm']))) if eligible else None
    return dict(selected_arm=best['arm'] if best else None, arms=summaries, source_only=True,
                improvement_candidate=best is not None and best['arm'] in ARMS)


def parent_check(path):
    delivery = read_json(ROOT/'docs/MICROWAVE_TCN_DELIVERY_20261025.json')
    if not delivery['independent_preparation_and_delivery_checks_passed']:
        raise DataError('Parent correctness review not terminal/pass')
    for name, digest in delivery['artifacts'].items():
        if sha256(ROOT/name) != digest:
            raise DataError('Parent source/config/document identity changed: '+name)
    if sha256(path/'freeze.json') != delivery['freeze_sha256']:
        raise DataError('Parent freeze changed')
    check_artifacts(path, read_json(path/'registration.json')['artifacts'])
    check_artifacts(path/'models', read_json(path/'training_complete.json')['artifacts'])
    check_artifacts(path, read_json(path/'freeze.json')['artifacts'])
    review = read_json(path/'independent_delivery_verification.json')
    if (review['status'] != 'independently_verified_completed_delivery' or
            sha256(path/'independent_delivery_verification.json') != delivery['independent_delivery_verification_sha256']):
        raise DataError('Parent independent review changed')
    return dict(run=str(path.resolve()), freeze_sha256=sha256(path/'freeze.json'),
                registration_sha256=sha256(path/'registration.json'), delivery_sha256=sha256(ROOT/'docs/MICROWAVE_TCN_DELIVERY_20261025.json'),
                prior_failed_starts=1, prior_successful_fits=6, prior_updates=21462)


def prepare(run, plan_path):
    if run.exists():
        raise DataError('Existing run cannot be overwritten')
    p = read_json(plan_path)
    up = ROOT/p['parent_run']
    identity = parent_check(up)
    # Only these named training/validation NPZs are permitted during preparation.
    allowed = {str((up/'TCN127.npz').resolve()).lower()}
    allowed.update(str((up/f'sampling_{seed}.npz').resolve()).lower() for seed in SEEDS)
    allowed.update(str(v.resolve()).lower() for v in (up/'validation/TCN127').glob('*.npz'))
    state = dict(active=True, reads=set())
    def guard(event, args):
        if event != 'open' or not state['active'] or not isinstance(args[0],str):
            return
        name = str(Path(args[0]).resolve()).lower()
        mode = args[1]
        if mode is not None and any(c in mode for c in 'wax+'):
            return
        if mode is None:
            import os
            if args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT):
                return
        if 'truth_once' in name or 'evaluation_labels' in name or (name.endswith('.npz') and name not in allowed and not Path(name).is_relative_to(run.resolve())):
            raise DataError('Forbidden prepare read: '+name)
        if name.endswith('.npz'):
            state['reads'].add(name)
    sys.addaudithook(guard)
    try:
        run.mkdir(parents=True)
        shutil.copytree(up/'source',run/'source')
        shutil.copyfile(Path(__file__),run/'source/virtual_submeter/accuracy/microwave_loss_correction.py')
        shutil.copyfile(ROOT/'scripts/run_microwave_loss_correction.py',run/'run_microwave_loss_correction.py')
        shutil.copyfile(ROOT/'tests/test_microwave_loss_correction.py',run/'test_microwave_loss_correction.py')
        shutil.copyfile(ROOT/'docs/MICROWAVE_LOSS_CORRECTION_PLAN_20261025.md',run/'protocol.md')
        shutil.copyfile(plan_path,run/'plan.json')
        for name in ['TCN127.npz','normalization.json','baseline_identity.json','baseline_settings.json','source_identity.json','validation_identity.json']:
            shutil.copyfile(up/name,run/name)
        shutil.copytree(up/'validation/TCN127',run/'validation')
        write_json(run/'parent_identity.json',identity)
        write_json(run/'reference_source_rows.json',{'TCN127':read_json(up/'source_rows.json')['TCN127']})
        d = load_npz(run/'TCN127.npz')
        if len(d['y']) != 259806 or int((d['y'] >= 200).sum()) != 849:
            raise DataError('Training support differs')
        prob, weight, supports = probabilities(d,p)
        np.savez_compressed(run/'weights.npz',probability=prob, importance=weight, house=d['house'],t=d['t'],event=d['event'],positive=d['y']>=200)
        save_csv(run/'weight_support.csv',supports)
        save_csv(run/'weight_manifest.csv',[dict(index=i,house=int(d['house'][i]),t=int(d['t'][i]),event=int(d['event'][i]),positive=bool(d['y'][i]>=200),probability=prob[i],importance=weight[i]) for i in range(len(prob))])
        write_json(run/'weight_math.json',dict(q=.0625,p=849/259806,event_BCE50_mass=50/65,
            uniform_BCE50_mass=50*849/(50*849+259806-849),positive_formula='q/(G*E_g*C_ge)',negative_formula='(1-q)/N_negative',
            importance_formula='1/(N*draw_probability_i)',clipping=False,normalization=False,
            sampling='Each hierarchy queue is shuffled without replacement; long-run marginal, not conditional independent unbiasedness given the full past. Finite cycle counts differ by at most one at each queue.'))
        inits = {}
        for seed in SEEDS:
            shutil.copyfile(up/f'sampling_{seed}.npz',run/f'sampling_{seed}.npz')
            idx = load_npz(run/f'sampling_{seed}.npz')['indices']
            stream = BatchStream(d,'event',seed,p)
            for batch in idx:
                np.testing.assert_array_equal(batch,stream.next())
            torch.manual_seed(seed)
            model = CenterMaskTCN()
            saved = torch.load(up/f'models/TCN127/seed_{seed}/initial.pt',weights_only=True,map_location='cpu')
            for key,value in model.state_dict().items():
                if not torch.equal(value,saved[key]):
                    raise DataError('Parent initialization differs')
            shutil.copyfile(up/f'models/TCN127/seed_{seed}/initial.pt',run/f'initial_{seed}.pt')
            inits[str(seed)] = sha256(run/f'initial_{seed}.pt')
        write_json(run/'initializations.json',inits)
        write_json(run/'preparation_access.json',dict(npz_reads=sorted(state['reads']),forbidden_reads=[],actual_fits=0))
        write_json(run/'registration.json',dict(artifacts=inventory(run),actual_fits=0,house17_truth_reads=0,maximum_new_fits=6))
    finally:
        state['active'] = False
    return verify_preparation(run)


def verify_preparation(run):
    check_artifacts(run,read_json(run/'registration.json')['artifacts'])
    p = read_json(run/'plan.json')
    if p['arms'] != ARMS or p['seeds'] != SEEDS or p['maximum_fits'] != 6:
        raise DataError('Invalid experiment matrix')
    d, w = load_npz(run/'TCN127.npz'),load_npz(run/'weights.npz')
    prob,importance,_ = probabilities(d,p)
    for key in ['house','t','event']:
        np.testing.assert_array_equal(w[key],d[key])
    np.testing.assert_array_equal(prob,w['probability'])
    np.testing.assert_array_equal(importance,w['importance'])
    return dict(status='prepared_verified',new_fits=0)


def authorize(run, path, phase):
    a = read_json(path)
    if (a.get('registration_sha256') != sha256(run/'registration.json') or a.get('maximum_fits') != 6
            or not a.get(phase) or not a.get('coordinator_execution_message')):
        raise DataError('Authorization not bound to registered six-fit experiment')
    if phase == 'evaluate' and a.get('freeze_sha256') != sha256(run/'freeze.json'):
        raise DataError('Evaluation not bound to freeze')
    return a


def fit_one(run, arm, seed, p, d, blocks, meta):
    dest = run/f'models/{arm}/seed_{seed}'
    dest.mkdir(parents=True,exist_ok=False)
    # Persist the start before the first update; any engineering failure is terminal.
    write_json(dest/'started.json',dict(arm=arm,seed=seed,updates=0))
    model,optimizer = parent.configure(seed,p)
    initial = torch.load(run/f'initial_{seed}.pt',map_location='cpu',weights_only=True)
    for key,value in model.state_dict().items():
        if not torch.equal(value,initial[key]):
            raise DataError('Initial parameters differ')
    torch.save(model.state_dict(),dest/'initial.pt')
    x,y = tensor_input(meta,d['x'],d['mask']),torch.from_numpy((d['y']>=200).astype(np.float32))
    importance = torch.from_numpy(load_npz(run/'weights.npz')['importance'].astype(np.float32))
    indices = load_npz(run/f'sampling_{seed}.npz')['indices']
    exposure = np.zeros(len(y),np.int64)
    history,batches,chosen,best = [],[],None,None
    started = time.perf_counter()
    for epoch in range(1,p['checkpoints']+1):
        model.train()
        for step in range(p['steps_per_checkpoint']):
            update = (epoch-1)*p['steps_per_checkpoint']+step
            idx = indices[update]
            optimizer.zero_grad(set_to_none=True)
            point = losses(model(x[idx]),y[idx],arm,importance[idx])
            loss = point.mean()
            if not torch.isfinite(loss):
                raise DataError('Nonfinite loss')
            loss.backward()
            if any(not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise DataError('Nonfinite gradient')
            optimizer.step()
            write_json(dest/'progress.json',dict(updates=update+1))
            np.add.at(exposure,idx,1)
            pos = y[idx].bool()
            pl,nl = float(point[pos].sum().detach()),float(point[~pos].sum().detach())
            batches.append(dict(update=update+1,positive=int(pos.sum()),negative=int((~pos).sum()),loss=float(loss.detach()),positive_loss_sum=pl,negative_loss_sum=nl,
                positive_effective_loss_fraction=pl/(pl+nl),batch_indices_sha256=array_hash(idx),
                nominal_positive_weight_mass=float((importance[idx][pos]*50).sum()/((importance[idx][pos]*50).sum()+importance[idx][~pos].sum())) if arm==ARMS[0] else .0625))
        scores = {k:predict_windows(meta,model,b['x']) for k,b in blocks.items()}
        point = select_checkpoint(scores,blocks,p)
        torch.save(model.state_dict(),dest/f'epoch_{epoch}.pt')
        for k,scores_k in scores.items():
            np.savez_compressed(dest/f'epoch_{epoch}_{k}.npz',scores=scores_k,t=blocks[k]['t'])
        history.append(dict(epoch=epoch,selection=point))
        if point is not None and (best is None or rank(point,epoch)<best):
            best,chosen = rank(point,epoch),dict(epoch=epoch,**point,source_gate_passed=True)
        write_json(dest/'history.json',dict(rows=history))
        save_csv(dest/'batches.csv',batches)
        write_json(dest/'progress.json',dict(updates=update+1))
        print(f'{arm} seed={seed} checkpoint={epoch} eligible={point is not None}',flush=True)
    if chosen is None:
        chosen = dict(epoch=p['checkpoints'],threshold=.5,source_gate_passed=False,
            block_metrics={k:metrics(scores[k],b['y'],.5) for k,b in blocks.items()})
    shutil.copyfile(dest/f"epoch_{chosen['epoch']}.pt",dest/'weights.pt')
    info = dict(**meta,kind='microwave_center_mask_tcn_v1',inference_support='strict_complete_31',truth_threshold_w=200,
        decision_threshold_probability=chosen['threshold'],selected=chosen,source_gate_passed=chosen['source_gate_passed'],
        arm=arm,seed=seed,weights_sha256=sha256(dest/'weights.pt'),updates=len(indices),seconds=time.perf_counter()-started,
        initial_sha256=sha256(run/f'initial_{seed}.pt'),sampling_sha256=sha256(run/f'sampling_{seed}.npz'),
        registration_sha256=sha256(run/'registration.json'),positive_seen=int(exposure[d['y']>=200].sum()),negative_seen=int(exposure[d['y']<200].sum()))
    write_json(dest/'model.json',info)
    np.savez_compressed(dest/'exposure.npz',count=exposure,t=d['t'],house=d['house'],event=d['event'])
    grouped = {}
    for i,n in enumerate(exposure):
        key = int(d['house'][i]),int(d['t'][i]//86400),int(d['event'][i])
        row = grouped.setdefault(key,dict(house=key[0],utc_day=key[1],event=key[2],available=0,unique_seen=0,exposure=0))
        row['available']+=1; row['unique_seen']+=int(n>0); row['exposure']+=int(n)
    save_csv(dest/'date_event_exposure.csv',list(grouped.values()))
    write_json(dest/'complete.json',dict(artifacts=inventory(dest)))
    return parent.source_row(seed,chosen)


def train(run, authorization):
    verify_preparation(run)
    authorize(run,authorization,'train')
    if (run/'training_started.json').exists() or (run/'models').exists():
        raise DataError('No restart/extra fits permitted')
    write_json(run/'training_started.json',dict(authorization=read_json(authorization),maximum_fits=6))
    p,d = read_json(run/'plan.json'),load_npz(run/'TCN127.npz')
    blocks = {f'block_{i}':load_npz(run/f'validation/block_{i}.npz') for i in range(6)}
    meta = {**read_json(run/'normalization.json'),'window_samples':127}
    rows = read_json(run/'reference_source_rows.json')
    for arm in ARMS:
        rows[arm] = [fit_one(run,arm,seed,p,d,blocks,meta) for seed in SEEDS]
    write_json(run/'source_rows.json',rows)
    write_json(run/'source_selection.json',select_source(rows,p))
    write_json(run/'training_complete.json',dict(fits=6,completed_models=[f'{a}/{s}' for a in ARMS for s in SEEDS],
        artifacts=inventory(run/'models'),source_rows_sha256=sha256(run/'source_rows.json'),source_selection_sha256=sha256(run/'source_selection.json')))
    return dict(status='six_new_fits_complete',selection=read_json(run/'source_selection.json'))


def verify_training(run, predictions=False):
    verify_preparation(run)
    p = read_json(run/'plan.json')
    complete = read_json(run/'training_complete.json')
    if complete['fits'] != 6 or complete['completed_models'] != [f'{a}/{s}' for a in ARMS for s in SEEDS]:
        raise DataError('Incomplete six-fit matrix')
    check_artifacts(run/'models',complete['artifacts'])
    for key in ['source_rows','source_selection']:
        if sha256(run/(key+'.json')) != complete[key+'_sha256']:
            raise DataError('Source selection binding changed')
    d = load_npz(run/'TCN127.npz')
    blocks = {f'block_{i}':load_npz(run/f'validation/block_{i}.npz') for i in range(6)}
    rows = read_json(run/'reference_source_rows.json')
    for arm in ARMS:
        rows[arm] = []
        for seed in SEEDS:
            path = run/f'models/{arm}/seed_{seed}'
            indices = load_npz(run/f'sampling_{seed}.npz')['indices']
            exposure = load_npz(path/'exposure.npz')
            np.testing.assert_array_equal(exposure['count'],np.bincount(indices.ravel(),minlength=len(d['y'])))
            for key in ['house','t','event']:
                np.testing.assert_array_equal(exposure[key],d[key])
            with (path/'batches.csv').open(encoding='utf-8',newline='') as stream:
                batches = list(csv.DictReader(stream))
            if len(batches) != 3528:
                raise DataError('Update budget differs')
            for i,(batch,idx) in enumerate(zip(batches,indices)):
                if batch['batch_indices_sha256'] != array_hash(idx) or int(batch['update']) != i+1 or int(batch['positive']) != 16 or int(batch['negative']) != 240:
                    raise DataError('Batch replay differs')
            initial = torch.load(run/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
            actual = torch.load(path/'initial.pt',weights_only=True,map_location='cpu')
            if any(not torch.equal(v,actual[k]) for k,v in initial.items()):
                raise DataError('Paired initial parameters differ')
            meta,model = load_state_model(path)
            chosen,best = None,None
            for epoch in range(1,13):
                scores = {}
                for k,b in blocks.items():
                    cache = load_npz(path/f'epoch_{epoch}_{k}.npz')
                    np.testing.assert_array_equal(cache['t'],b['t'])
                    scores[k] = cache['scores']
                point = select_checkpoint(scores,blocks,p)
                if point is not None and (best is None or rank(point,epoch)<best):
                    best,chosen = rank(point,epoch),dict(epoch=epoch,**point,source_gate_passed=True)
            if chosen is None:
                chosen = dict(epoch=12,threshold=.5,source_gate_passed=False,block_metrics={k:metrics(scores[k],b['y'],.5) for k,b in blocks.items()})
            if chosen != meta['selected'] or sha256(path/'weights.pt') != sha256(path/f"epoch_{chosen['epoch']}.pt"):
                raise DataError('Checkpoint/threshold does not reproduce')
            for k,b in blocks.items():
                np.testing.assert_array_equal(predict_windows(meta,model,b['x']),load_npz(path/f"epoch_{chosen['epoch']}_{k}.npz")['scores'])
            rows[arm].append(parent.source_row(seed,chosen))
    if rows != read_json(run/'source_rows.json') or select_source(rows,p) != read_json(run/'source_selection.json'):
        raise DataError('Source preselection does not reproduce')
    if predictions:
        frozen = read_json(run/'freeze.json')
        if set(frozen['predictions']) != {f'{a}/{s}' for a in ORDER for s in SEEDS}:
            raise DataError('Incomplete nine-prediction matrix')
        check_artifacts(run,frozen['artifacts'])
        for key,name in [('registration_sha256','registration.json'),('training_sha256','training_complete.json'),('source_selection_sha256','source_selection.json')]:
            if frozen[key] != sha256(run/name):
                raise DataError('Freeze dependency differs')
        if not frozen['ordinary_inference_matches'] or not frozen['common_support_and_other_devices_match']:
            raise DataError('Ordinary/common support check missing')
    return True


def freeze(run):
    verify_training(run)
    if (run/'predictions').exists():
        raise DataError('Prediction attempt exists')
    (run/'predictions').mkdir()
    up = Path(read_json(run/'parent_identity.json')['run'])
    pf = read_json(up/'freeze.json')
    rows = parent.target_input(run)
    t,aggregate = aggregate_arrays(rows,8)
    checks = []
    for arm in ORDER:
        for seed in SEEDS:
            path = up/f'models/TCN127/seed_{seed}' if arm == 'TCN127' else run/f'models/{arm}/seed_{seed}'
            meta,model = load_state_model(path)
            x,centers = context_at_complete_centers(meta,t,aggregate)
            scores = np.full(len(t),np.nan,np.float32)
            scores[centers] = predict_windows(meta,model,x)
            ordinary = predict_state_rows(meta,model,rows)
            np.testing.assert_array_equal(scores,np.array([r['probability_on'] if r['probability_on'] is not None else np.nan for r in ordinary],np.float32))
            state = np.where(np.isfinite(scores),scores>=meta['decision_threshold_probability'],-1).astype(np.int8)
            np.testing.assert_array_equal(state,np.array([-1 if r['state'] is None else int(r['state']) for r in ordinary],np.int8))
            base,w50 = parent.reference_arrays(run,seed),parent.reference_arrays(run,seed,'W50')
            np.testing.assert_array_equal(t,base['stamps'])
            np.testing.assert_array_equal(aggregate,base['aggregate'])
            for a in [base,w50]:
                np.testing.assert_array_equal(np.isfinite(scores),np.isfinite(a['raw'][:,1]))
            if arm == 'TCN127':
                name = f'predictions/TCN127_{seed}.npz'
                if sha256(up/name) != pf['artifacts'][name]:
                    raise DataError('Parent control prediction changed')
                # Named non-label fields only. No parent development truth is opened.
                with np.load(up/name,allow_pickle=False) as z:
                    np.testing.assert_array_equal(scores,z['scores'])
                    np.testing.assert_array_equal(state,z['state'])
                    np.testing.assert_array_equal(t,z['t'])
                    np.testing.assert_array_equal(base['raw'],z['raw'])
            np.savez_compressed(run/f'predictions/{arm}_{seed}.npz',t=t,scores=scores,state=state,raw=base['raw'],aggregate=aggregate)
            checks.append(dict(arm=arm,seed=seed,ordinary_scores_exact=True,ordinary_state_exact=True,raw_A_all_devices_exact=True,
                input_support_sha256=array_hash(t[centers]),valid_input_centers=len(centers),threshold=meta['decision_threshold_probability'],weights_sha256=meta['weights_sha256']))
    write_json(run/'inference_checks.json',dict(checks=checks,house17_truth_reads=0))
    write_json(run/'freeze.json',dict(predictions=[f'{a}/{s}' for a in ORDER for s in SEEDS],registration_sha256=sha256(run/'registration.json'),
        training_sha256=sha256(run/'training_complete.json'),source_selection_sha256=sha256(run/'source_selection.json'),
        artifacts={**{'predictions/'+k:v for k,v in inventory(run/'predictions').items()},'inference_checks.json':sha256(run/'inference_checks.json')},
        ordinary_inference_matches=True,common_support_and_other_devices_match=True,house17_truth_reads=0))
    return dict(status='nine_predictions_frozen_before_truth')


def score(run, truth):
    rows,supports = [],{}
    for seed in SEEDS:
        base,w50 = parent.reference_arrays(run,seed),parent.reference_arrays(run,seed,'W50')
        mask = np.isfinite(truth)&np.isfinite(w50['raw'][:,1])
        if int(mask.sum()) != 137523:
            raise DataError('Common support must be 137523')
        supports[str(seed)] = dict(n=int(mask.sum()),mask_sha256=array_hash(mask),timestamps_sha256=array_hash(base['stamps'][mask]))
        for a,d in [('A',base),('W50',w50)]:
            rows.append(dict(arm=a,seed=seed,**parent.confusion(d['raw'][mask,1]>=200,truth[mask]>=200)))
        for arm in ORDER:
            a = load_npz(run/f'predictions/{arm}_{seed}.npz')
            np.testing.assert_array_equal(a['t'],base['stamps'])
            np.testing.assert_array_equal(a['raw'],base['raw'])
            np.testing.assert_array_equal(mask,np.isfinite(truth)&np.isfinite(a['scores']))
            if np.any(a['state'][mask]<0):
                raise DataError('Missing candidate decision')
            rows.append(dict(arm=arm,seed=seed,**parent.confusion(a['state'][mask].astype(bool),truth[mask]>=200)))
    return rows,supports


def evaluate(run, authorization):
    authorize(run,authorization,'evaluate')
    verify_training(run,predictions=True)
    if (run/'evaluation').exists():
        raise DataError('Evaluation attempt exists')
    (run/'evaluation').mkdir()
    write_json(run/'evaluation/started.json',dict(freeze_sha256=sha256(run/'freeze.json'),authorization=read_json(authorization)))
    b = read_json(run/'baseline_identity.json')
    artifact = b['development_partition']['artifacts']['microwave']
    path = Path(b['reference_run'])/'protocol'/artifact['path']
    if sha256(path) != artifact['sha256']:
        raise DataError('Truth identity changed')
    with path.open(encoding='utf-8',newline='') as stream:
        mapping = {utc(r['timestamp']):numeric(r['power_w']) for r in csv.DictReader(stream)}
    t = parent.reference_arrays(run,42)['stamps']
    truth = np.array([mapping.get(int(s),np.nan) for s in t],np.float64)
    np.savez_compressed(run/'evaluation/truth_once.npz',t=t,y=truth)
    rows,supports = score(run,truth)
    groups = {a:[r for r in rows if r['arm']==a] for a in ['A','W50']+ORDER}
    selected = read_json(run/'source_selection.json')['selected_arm']
    means = {a:{k:float(np.mean([r[k] for r in rr])) for k in ['precision','recall','f1','fp']} for a,rr in groups.items()}
    passed = selected is not None and development_gate(groups['W50'],groups[selected],read_json(run/'plan.json'))
    summary = dict(selected_before_development=selected,development_acceptance_passed=bool(passed),
        declared_new_improvement=bool(passed and selected in ARMS),means=means,supports=supports,
        default_model_changed=False,final_houses_accessed=False,development_only_previously_inspected_house17=True,
        raw_power_all_devices_unchanged=True,additional_fits_permitted=False)
    write_json(run/'evaluation/metrics.json',dict(rows=rows,supports=supports))
    write_json(run/'evaluation/summary.json',summary)
    save_csv(run/'evaluation/comparison.csv',rows)
    write_json(run/'evaluation_complete.json',dict(artifacts=inventory(run/'evaluation'),truth_csv_parses=1,freeze_sha256=sha256(run/'freeze.json')))
    return summary


def verify(run):
    if not (run/'training_complete.json').exists():
        return verify_preparation(run)
    verify_training(run,predictions=(run/'freeze.json').exists())
    if (run/'evaluation_complete.json').exists():
        check_artifacts(run/'evaluation',read_json(run/'evaluation_complete.json')['artifacts'])
        rows,supports = score(run,load_npz(run/'evaluation/truth_once.npz')['y'])
        if dict(rows=rows,supports=supports) != read_json(run/'evaluation/metrics.json'):
            raise DataError('Confusion/support replay differs')
    result = dict(status='verified',new_fits=6,reused_control_fits=3,new_updates=21168,
                  default_model_changed=False,final_houses_accessed=False)
    write_json(run/'verification.json',result)
    return result


def report(run):
    verify(run)
    summary = read_json(run/'evaluation/summary.json')
    lines = ['# 微波炉事件采样损失权重校正对照（任务8）','',
        '结论：'+('通过开发验收。' if summary['development_acceptance_passed'] else '未通过开发验收，保留负结果。'),
        f"来源唯一预选：{summary['selected_before_development']}；查看开发成绩后未改选。",'',
        '| 组 | 均P | 均R | 均F1 | 均FP |','|---|---:|---:|---:|---:|']
    for a,m in summary['means'].items():
        lines.append(f"| {a} | {m['precision']:.8f} | {m['recall']:.8f} | {m['f1']:.8f} | {m['fp']:.2f} |")
    lines += ['', '新预算：两损失×三seed，共6启动、6有效拟合、21,168更新；控制TCN127三个模型直接复用。父任务另有1次294更新失败和6有效拟合（21,462更新），不计入本任务。',
        '训练House2/19原14天，259806中心/849正类；六个House5双日块；127上下文、真实归一化、mask和gap截断均沿用父封存。每批16正/240负、256、Adam .001、3528更新、CPU4。',
        'importance_BCE50：逐点BCE50乘1/(Nπ_i)，π正=q/(G E_g C_ge)、π负=(1-q)/N_negative，不截断、不归一化。固定logits的长期边际加权期望等于自然均匀BCE50；队列逐层打乱无放回循环，给定完整历史后的每步不独立，也不承诺每个有限前缀严格无偏。',
        'unit_BCE：相同事件采样批次，BCE pos_weight=1且单位逐点权，目标仍为事件采样分布下的单位BCE，不等于自然均匀BCE。',
        '原事件BCE50正类名义质量0.76923077；自然均匀BCE50约0.14084224。该目标差异不证明它是原误报唯一原因。实际逐批正负损失贡献见models各目录batches.csv。',
        '全部9组开发预测、阈值和普通推理一致性在首次读真值前封存；共同评价支持137523点。House17已在旧任务查看，只称开发验收，最终House8/9/20未访问。微波炉功率、其他电器功率及状态沿原A逐点不变；默认未切换。',
        '来源门槛：六块R≥.85非退化；最差P/均P/F1/较早轮次。新臂均F1≥控制、每seed召回下降≤.05、至少2seed P提高。按均P/均F1与控制/importance/unit顺序唯一预选，无额外绝对来源P门槛。',
        '开发门槛对W50：均P≥.08、各seed R≥.85、均F1提升≥.02（1e-12容差）、至少2seed P/F1同升、任F1下降≤.02、FP下降≥30%、非退化。',
        '', '逐seed混淆矩阵与开发PR：evaluation/comparison.csv；源PR与门槛：source_rows.json；来源checkpoint PR：models/*/seed_*/history.json；权重清单：weight_manifest.csv；事件支持：weight_support.csv；批次精确索引：sampling_*.npz；曝光：models/*/seed_*/exposure.npz与date_event_exposure.csv。',
        '普通加载：virtual_submeter.models.center_mask_tcn.load_state_model(模型目录)，kind保持microwave_center_mask_tcn_v1、window_samples=127。独立封存入口run_microwave_loss_correction.py支持prepare/train/freeze/evaluate/verify/report。',
        '', '| arm | seed | TP | FP | FN | TN | P | R | F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in read_json(run/'evaluation/metrics.json')['rows']:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {r['precision']:.8f} | {r['recall']:.8f} | {r['f1']:.8f} |")
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(run/'report.json',summary)
    write_json(run/'delivery.json',dict(status='completed',summary=summary,verification=read_json(run/'verification.json'),artifacts=inventory(run)))
    return dict(status='report_complete',report=str((run/'report.md').resolve()))
