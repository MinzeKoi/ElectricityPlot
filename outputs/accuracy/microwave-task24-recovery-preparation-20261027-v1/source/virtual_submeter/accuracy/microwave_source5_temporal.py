"""Explicit House5 temporal exception. Preparation is metadata only, zero data reads.

The real-data execution adapter is deliberately not activated by this preparation
version. Pure array helpers below are exercised only on synthetic fixtures here.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from ..data.common import DataError, read_json, write_json, sha256, utc
from ..models.center_mask_tcn import (CenterMaskTCN, expand_context, tensor_input,
                                      load_state_model, predict_windows, KIND)
from .microwave_diagnosis import segments, split_mask
from .microwave_fullsource import select_checkpoint, development_gate
from .microwave_sampling import BatchStream, rank, check_artifacts

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT/'configs/microwave_source5_temporal_20261025.json'
SEEDS = [42, 1019, 2026]
ARMS = ['control', 'source5']
RECIPES = ('control_BCE50', 'importance_BCE50', 'unit_BCE')
TRAIN = [(i, utc(a), utc(b)) for i, a, b in (
    (0, '2014-11-15T00:00:00Z', '2014-11-17T00:00:00Z'),
    (1, '2014-11-17T00:00:00Z', '2014-11-19T00:00:00Z'),
    (2, '2014-11-19T00:00:00Z', '2014-11-21T00:00:00Z'),
    (3, '2014-11-21T00:00:00Z', '2014-11-22T23:30:00Z'))]
VALIDATION = [(4, utc('2014-11-23T00:00:00Z'), utc('2014-11-25T00:00:00Z')),
              (5, utc('2014-11-25T00:00:00Z'), utc('2014-11-27T00:00:00Z'))]
P_REF = 849/259806


def permit(house, role, channel, block, low, high):
    """Only exact approved House5 intervals, never a general protocol bypass."""
    allowed = TRAIN if role == 'train' else VALIDATION if role == 'validation' else []
    if (house != 'refit_5' or channel not in ('aggregate', 'microwave')
            or (block, low, high) not in allowed):
        raise DataError('Only exact approved House5 block/role/channel is permitted')


def _grid(t, a, y):
    if (any(np.asarray(v).ndim != 1 for v in (t, a, y)) or len(t) < 31
            or len(t) != len(a) or len(t) != len(y) or not np.isfinite(t).all()
            or np.any(t != np.floor(t)) or np.any(t % 8) or np.any(np.diff(t) <= 0)
            or np.isinf(a).any() or np.isinf(y).any()
            or np.any(a[np.isfinite(a)] < 0) or np.any(y[np.isfinite(y)] < 0)):
        raise DataError('Invalid timestamp, aggregate or label grid; missing must remain NaN')


def build_block(t, a, y, block, role, frozen=None):
    """Array-only builder; caller must separately authorize every real-data read.

    Events are segmented on the complete input grid before role/block selection,
    so an event straddling a boundary is excluded in its entirety.
    """
    t, a, y = np.asarray(t), np.asarray(a), np.asarray(y)
    _grid(t, a, y)
    bounds = TRAIN if role == 'train' else VALIDATION if role == 'validation' else []
    matched = [r for r in bounds if r[0] == block]
    if len(matched) != 1:
        raise DataError('Unknown frozen block or role')
    _, low, high = matched[0]
    permit('refit_5', role, 'microwave', block, low, high)
    ids, events = segments(t, y)
    c = np.arange(15, len(t)-15)
    core = np.lib.stride_tricks.sliding_window_view(a, 31)
    times = np.lib.stride_tricks.sliding_window_view(t, 31)
    observed = np.isfinite(core)
    keep = (split_mask(t, c, ids, events, low, high)
            & (np.diff(times, axis=1) == 8).all(axis=1)
            & np.isfinite(a[c]) & np.isfinite(y[c]))
    if role == 'train':
        keep &= (observed.sum(axis=1) >= 24)
        keep &= ~((~observed[:, :-1]) & (~observed[:, 1:])).any(axis=1)
    else:
        keep &= observed.all(axis=1)
        if frozen is None:
            raise DataError('Validation requires original frozen strict31 centers/labels')
        wanted = np.asarray(frozen['t'])
        keep &= np.isin(t[c], wanted)
        if not np.array_equal(t[c[keep]], wanted):
            raise DataError('Frozen validation centers changed or violate approved boundaries')
        np.testing.assert_array_equal(y[c[keep]], frozen['y'])
    cc = c[keep]
    ext = expand_context(t, a, cc, low, high)
    np.testing.assert_array_equal(ext['x'][:, 48:79], core[keep])
    if frozen is not None and 'x' in frozen:
        fx = frozen['x']
        np.testing.assert_array_equal(ext['x'][:, 48:79], fx if fx.shape[1] == 31 else fx[:,48:79])
    return dict(**ext, y=y[cc].copy(), t=t[cc].copy(), raw_index=cc,
                house=np.full(len(cc), 2, np.int64), event=ids[cc],
                house_name=np.full(len(cc), 'refit_5', dtype='U8'),
                block=np.full(len(cc), block, np.int64))


def validate_identity(d):
    n = len(d['y'])
    for k in ('y', 'house', 't', 'event'):
        v = np.asarray(d[k])
        if v.shape != (n,) or not np.isfinite(v).all():
            raise DataError('Missing/nonfinite center identity or label')
    if (not n or np.any(d['y'] < 0) or not np.isin(d['house'], [0,1,2]).all()
            or np.any(d['t'] != np.floor(d['t'])) or np.any(d['t'] % 8)
            or np.any(d['event'] != np.floor(d['event']))
            or np.any(d['event'][d['y'] >= 200] < 0)
            or len(set(zip(d['house'].tolist(), d['t'].tolist()))) != n):
        raise DataError('Invalid/duplicate house+UTC center or positive event identity')
    if 'house_name' in d:
        names = np.array(['refit_2','refit_19','refit_5'])
        if not np.array_equal(d['house_name'], names[d['house'].astype(int)]):
            raise DataError('House string and numeric index disagree')


def append_training(original, added, normalization, original_normalization):
    """Preserve every original row/column and the original normalization values."""
    validate_identity(original)
    validate_identity(added)
    if (not np.isin(original['house'], [0,1]).all() or not (added['house'] == 2).all()
            or normalization != original_normalization or set(original) - set(added)):
        raise DataError('Original rows/normalization or added House5 identity changed')
    result = {k: np.concatenate([v, added[k]]) for k,v in original.items()}
    validate_identity(result)
    for k,v in original.items():
        np.testing.assert_array_equal(result[k][:len(v)], v)
    return result


def fixed_prior_weights(d, p):
    """Long-run event sampler marginal; fixed old prior, no clip/renormalize."""
    validate_identity(d)
    if p['batch_size'] != 256 or p['positive_per_batch'] != 16:
        raise DataError('Fixed 16/240 sampling quota required')
    stream = BatchStream(d, 'event', 42, p)
    pos = d['y'] >= 200
    if not pos.any() or pos.all():
        raise DataError('Both classes required')
    q = 16/256
    actual = np.full(len(pos), (1-q)/int((~pos).sum()), np.float64)
    for events in stream.groups.values():
        for centers in events.values():
            actual[centers] = q/(len(stream.groups)*len(events)*len(centers))
    target = np.where(pos, P_REF/int(pos.sum()), (1-P_REF)/int((~pos).sum()))
    weights = target/actual
    np.testing.assert_allclose([actual.sum(), target.sum(), target[pos].sum()], [1,1,P_REF], atol=1e-12, rtol=0)
    return actual, target, weights


def point_loss(logits, y, recipe, weights):
    if (recipe not in RECIPES or not torch.isfinite(logits).all()
            or not torch.isfinite(y).all() or not ((y == 0) | (y == 1)).all()
            or weights.shape != y.shape or not torch.isfinite(weights).all() or (weights <= 0).any()):
        raise DataError('Invalid fixed recipe or BCE inputs')
    value = torch.nn.functional.binary_cross_entropy_with_logits(logits, y,
        pos_weight=logits.new_tensor(1. if recipe == 'unit_BCE' else 50.), reduction='none')
    return value*weights if recipe == 'importance_BCE50' else value


def reselect_checkpoints(checkpoints, blocks, p, expected_epochs=12):
    """Both old and new models call this with identical block4/5 input arrays."""
    if set(blocks) != {'block_4','block_5'} or len(checkpoints) != expected_epochs:
        raise DataError('Exactly retained blocks4/5 and complete checkpoint matrix required')
    for i, low, high in VALIDATION:
        d = blocks[f'block_{i}']
        if (np.any(d['t']-120 < low) or np.any(d['t']+128 > high)
                or not np.isfinite(d['y']).all()):
            raise DataError('Validation center/core outside frozen block')
    choices = []
    for epoch, scores in enumerate(checkpoints, 1):
        if set(scores) != set(blocks):
            raise DataError('Checkpoint must contain exactly the same two validation blocks')
        if any(np.any((scores[k] < 0) | (scores[k] > 1)) for k in scores):
            raise DataError('Invalid probability')
        point = select_checkpoint(scores, blocks, p)
        if point is not None and point['supported_blocks'] == ['block_4','block_5']:
            choices.append({**point, 'epoch': epoch})
    return min(choices, key=lambda r: rank(r, r['epoch'])) if choices else None


def select_source(rows):
    if set(rows) != set(ARMS) or any([r['seed'] for r in rows[a]] != SEEDS for a in ARMS):
        raise DataError('Complete control/source5 x three seed source matrix required')
    old, summaries = rows['control'], []
    for arm in ARMS:
        rr = rows[arm]
        if any(not np.isfinite([r[k] for k in ('precision','recall','f1')]).all()
               or any(not 0 <= r[k] <= 1 for k in ('precision','recall','f1')) for r in rr):
            raise DataError('Invalid source metrics')
        ok = all(r['source_gate_passed'] and not r['degenerate'] and r['recall'] >= .85 for r in rr)
        if arm == 'source5':
            ok &= (np.mean([r['f1'] for r in rr]) >= np.mean([r['f1'] for r in old])
                   and all(b['recall']-a['recall'] >= -.05 for a,b in zip(old,rr))
                   and sum(b['precision'] > a['precision'] for a,b in zip(old,rr)) >= 2)
        summaries.append(dict(arm=arm, eligible=bool(ok), mean_precision=float(np.mean([r['precision'] for r in rr])),
                              mean_f1=float(np.mean([r['f1'] for r in rr]))))
    eligible = [r for r in summaries if r['eligible']]
    best = max(eligible,key=lambda r:(r['mean_precision'],r['mean_f1'],-ARMS.index(r['arm']))) if eligible else None
    return dict(selected_arm=best['arm'] if best else None, source_only=True, arms=summaries)


def condition_receipt(receipt):
    """Validate a future coordinator receipt, without inferring it from scores."""
    if (receipt.get('task8_development_passed') is not False
            or receipt.get('source_recipe') not in RECIPES
            or receipt.get('maximum_new_fits') != 3
            or not receipt.get('coordinator_message')
            or receipt.get('source_thread_id') != '01a16d7b-478e-7780-a999-88abe71e8c94'
            or any(len(receipt.get(k,'')) != 64 for k in ('task8_source_selection_sha256','task8_freeze_sha256'))):
        raise DataError('Await coordinator confirmation of Task8 failure and unique source recipe')
    return receipt['source_recipe']


def prepare(run, plan_path=PLAN):
    """Only copy code/config/docs; never open source arrays or any household data."""
    run = Path(run)
    p = read_json(plan_path)
    if p != read_json(PLAN) or p['execution_condition_confirmed'] or p['source_recipe'] is not None:
        raise DataError('Preparation plan must retain the pending execution condition')
    run.mkdir(parents=True, exist_ok=False)
    files = [PLAN, Path(__file__), ROOT/'scripts/run_microwave_source5_temporal.py',
             ROOT/'tests/test_microwave_source5_temporal.py', ROOT/'docs/MICROWAVE_SOURCE5_TEMPORAL_PLAN_20261025.md']
    artifacts = {}
    for path in files:
        dst = run/path.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
        artifacts[str(path.relative_to(ROOT)).replace('\\','/')] = sha256(dst)
    write_json(run/'registration.json', dict(schema='source5_preparation_v1', artifacts=artifacts,
        actual_real_fits=0, real_training_arrays_built=False, household_data_reads=0,
        execution_condition_confirmed=False, source_recipe=None, maximum_new_fits=3))
    return verify(run)


def verify(run):
    r = read_json(Path(run)/'registration.json')
    check_artifacts(Path(run), r['artifacts'])
    if r['actual_real_fits'] != 0 or r['real_training_arrays_built'] or r['execution_condition_confirmed']:
        raise DataError('Preparation-only registration changed')
    return dict(status='prepared_code_only_waiting_for_condition', actual_real_fits=0,
                real_training_arrays_built=False, execution_adapter_enabled=False)


def pending_stage(run, stage, authorization=None):
    verify(run)
    if authorization is not None:
        condition_receipt(read_json(authorization))
    raise DataError(f'{stage}: real execution adapter remains disabled in preparation v1; await coordinator follow-up')


def report(run):
    result = verify(run)
    dest = Path(run)/'PREPARATION_REPORT.md'
    if dest.exists():
        raise DataError('Report exists; cannot overwrite')
    dest.write_text('独立House5同户时间工程试验：仅代码和合成测试准备。\n\n'
        '真实拟合0；未制作真实House5训练池；未读取House17或最终8/9/20数据。\n'
        '等待协调者确认任务8未达标并给出唯一source预选配方。\n'
        '两项获批例外为House5仅验证和来源训练/验证家庭隔离；默认协议不变。\n'
        '最后4日有历史选模接触，仅为复用来源时间验证；不宣称独立验证或功率改善。\n', encoding='utf-8')
    return result


def synthetic_roundtrip(run, recipe):
    """Tiny generated fixture -> one update -> validation -> ordinary model load.

    This is NOT a real trial, a counted real fit, or the 3528-update protocol.
    """
    if recipe not in RECIPES:
        raise DataError('Unknown recipe')
    run = Path(run)
    run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    x = rng.uniform(0,2000,(64,127)).astype(np.float32)
    y = np.r_[np.full(16,300), np.zeros(48)].astype(np.float32)
    d = dict(y=y,house=np.r_[np.zeros(32,int),np.ones(16,int),np.full(16,2)],
             t=np.arange(64)*8,event=np.where(y>=200,np.arange(64)//4,-1))
    p = read_json(PLAN)
    _, _, w = fixed_prior_weights(d,p)
    model = CenterMaskTCN()
    meta = dict(kind=KIND,truth_threshold_w=200,window_samples=127,inference_support='strict_complete_31',
                input_log_mean=5.,input_log_std=1.,decision_threshold_probability=.5,cpu_threads=4,batch_size=256)
    ix = BatchStream(d,'event',42,p).next()
    opt = torch.optim.Adam(model.parameters(),lr=.001)
    opt.zero_grad()
    point_loss(model(tensor_input(meta,x[ix])),torch.tensor(y[ix]>=200,dtype=torch.float32),
               recipe,torch.tensor(w[ix],dtype=torch.float32)).mean().backward()
    opt.step()
    before = predict_windows(meta,model,x)
    torch.save(model.state_dict(),run/'weights.pt')
    meta['weights_sha256'] = sha256(run/'weights.pt')
    write_json(run/'model.json',meta)
    loaded, restored = load_state_model(run)
    after = predict_windows(loaded,restored,x)
    np.testing.assert_array_equal(before,after)
    return dict(synthetic_only=True,real_fits=0,updates=1,recipe=recipe,
                ordinary_load_prediction_equal=True,parameter_count=sum(v.numel() for v in model.parameters()))
