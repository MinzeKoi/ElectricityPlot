"""Zero-fit, source-only decision calibration of twelve frozen regression models."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil

import numpy as np

from ..data.common import DataError, read_json, write_json, sha256, utc, safe_child
from ..data.protocol import read_partition
from ..models.arrays import aggregate_arrays, labels_at_centers
from ..models.artifacts import load_model
from ..models.prediction import predict_rows
from .data_expansion import inventory, compact_metrics
from .precision import (SOURCE_VALIDATION as SV, DEVELOPMENT as DEV, SEEDS,
                        validate_frozen as validate_parent, validate_protocol,
                        development_gate)
from .state_experiment import state_metrics

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'pyproject.toml').is_file() and (p/'src/virtual_submeter').is_dir())
FAMILIES = ['W50', 'W10', 'W1', 'W50_HN']
PREFIX = 'MICROWAVE_REGRESSION_CALIBRATION'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    if path.exists():
        if read_json(path) != value:
            raise DataError('Refusing to overwrite ' + str(path))
    else:
        write_json(path, value)


def check_files(base, files):
    for name, digest in files.items():
        if sha256(safe_child(base, name)) != digest:
            raise DataError('Sealed artifact changed: ' + name)


def context(run):
    reg = read_json(run/'registration.json')
    check_files(ROOT, reg['inputs'])
    check_files(run, reg['execution'])
    return reg, ROOT/reg['plan']['parent_run']


def prepare(run, config):
    if run.exists():
        raise DataError('New run directory required')
    plan = read_json(config)
    if (plan['families'] != FAMILIES or plan['seeds'] != SEEDS or plan['maximum_fits'] != 0
            or plan['truth_threshold_w'] != 200 or plan['minimum_block_recall'] != .85
            or plan['expected_development_support'] != 137523):
        raise DataError('Finite registered recipe changed')
    parent = ROOT/plan['parent_run']
    frozen = validate_parent(parent)
    block_parent = (ROOT/plan['block_support']).parent
    block_registration = read_json(block_parent/'registration.json')
    check_files(block_parent, {k: v for k, v in block_registration['artifacts'].items()
                              if k.startswith('validation/') or k in ('support.json', 'upstream_protocol.json')})
    validate_protocol(read_json(parent/'protocol/protocol.json'))
    if frozen['groups'] != FAMILIES:
        raise DataError('Wrong frozen family identity')
    inputs = {}
    paths = [config, ROOT/'docs'/f'{PREFIX}_PLAN_20261025.md', ROOT/plan['block_support'],
             parent/'freeze.json', parent/'registration.json', parent/'base_settings.json']
    paths += [p for p in (ROOT/plan['block_support']).parent.joinpath('validation').glob('block_*.npz')]
    paths += [block_parent/'registration.json', block_parent/'upstream_protocol.json']
    paths += [p for p in (parent/'protocol').rglob('*') if p.is_file()]
    for g in FAMILIES:
        for s in SEEDS:
            directory = parent/f'models/{g}/seed_{s}'
            meta, _ = load_model(directory)
            if (meta['checkpoint_metric'] != 'normalized_mae' or meta['target_labels_read']
                    or meta['protocol_sha256'] != sha256(parent/'protocol/protocol.json')
                    or meta['training_households'] != ['refit_19', 'refit_2']
                    or meta['validation_households'] != ['refit_5']):
                raise DataError('Model is not the original source-MAE checkpoint')
            paths += [directory/'model.json', directory/'weights.pt']
    paths += [parent/c['path'] for c in frozen['cases']]
    paths += [parent/f'reference/predictions/seed_{s}/{p}.npz' for s in SEEDS for p in (SV, DEV)]
    # Only bytes are hashed here: no development arrays, labels, or metrics are parsed.
    for p in paths:
        inputs[p.relative_to(ROOT).as_posix()] = sha256(p)
    run.mkdir(parents=True)
    shutil.copytree(ROOT/'src/virtual_submeter', run/'source/virtual_submeter',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(ROOT/'scripts/run_microwave_regression_calibration.py', run/'run_microwave_regression_calibration.py')
    shutil.copyfile(ROOT/'tests/test_microwave_regression_calibration.py', run/'test_microwave_regression_calibration.py')
    shutil.copyfile(config, run/'plan.json')
    execution = {'source/'+k: v for k, v in inventory(run/'source').items()}
    execution.update({n: sha256(run/n) for n in ('run_microwave_regression_calibration.py',
                      'test_microwave_regression_calibration.py', 'plan.json')})
    save(run/'registration.json', {'registered_at': now(), 'plan': plan, 'inputs': inputs,
         'execution': execution, 'maximum_fits': 0, 'development_labels_parsed': False,
         'final_households_accessed': False, 'original_protocol_sha256': sha256(parent/'protocol/protocol.json')})
    return {'status': 'prepared', 'models': 12, 'fits': 0}


def aligned_truth(rows, stamps, expected_house):
    if any(r['household_id'] != expected_house for r in rows):
        raise DataError('Label household mismatch')
    if len(set(int(t) for t in stamps)) != len(stamps) or np.any(np.diff(stamps) <= 0):
        raise DataError('Prediction time identity is ambiguous')
    return labels_at_centers({'microwave': rows}, ['microwave'], stamps, np.arange(len(stamps)))[:, 0]


def decisions(score, threshold):
    result = np.full(len(score), -1, np.int8)  # -1 is unavailable, never off.
    if threshold is not None:
        valid = np.isfinite(score)
        result[valid] = (score[valid] >= threshold).astype(np.int8)
    return result


def exact_curve(score, truth, thresholds=None):
    valid = np.isfinite(score) & np.isfinite(truth)
    p, y = score[valid], truth[valid] >= 200
    thresholds = np.unique(p) if thresholds is None else thresholds
    order = np.argsort(p, kind='stable')
    p, y = p[order], y[order]
    suffix = np.r_[np.cumsum(y[::-1], dtype=np.int64)[::-1], 0]
    indices = np.searchsorted(p, thresholds, side='left')
    tp, on = suffix[indices], len(p)-indices
    fp, fn = on-tp, int(y.sum())-tp
    precision = np.divide(tp, on, out=np.zeros(len(on), float), where=on > 0)
    recall = tp/int(y.sum()) if y.any() else np.zeros(len(on), float)
    denom = 2*tp+fp+fn
    f1 = np.divide(2*tp, denom, out=np.zeros(len(on), float), where=denom > 0)
    return {'threshold': thresholds, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'mixed': (on > 0) & (on < len(p))}


def calibrate(score, truth, block_masks):
    thresholds = np.unique(score[np.isfinite(score) & np.isfinite(truth) & np.any(block_masks, axis=0)])
    curves = [exact_curve(score[m], truth[m], thresholds) for m in block_masks]
    allowed = np.ones(len(thresholds), bool)
    for c in curves:
        allowed &= (c['recall'] >= .85) & c['mixed']
    pr = {'threshold': thresholds}
    for i, c in enumerate(curves):
        pr.update({f'block_{i}_{k}': v for k, v in c.items() if k != 'threshold'})
    if not allowed.any():
        return {'threshold': None, 'passed': False, 'reason': 'no_single_nondegenerate_R85_threshold'}, pr
    worst_p = np.min([c['precision'] for c in curves], axis=0)
    mean_p = np.mean([c['precision'] for c in curves], axis=0)
    mean_f1 = np.mean([c['f1'] for c in curves], axis=0)
    idx = np.flatnonzero(allowed)
    best = idx[np.lexsort((thresholds[idx], mean_f1[idx], mean_p[idx], worst_p[idx]))[-1]]
    threshold = float(thresholds[best])
    return {'threshold': threshold, 'passed': True,
            'worst_block_precision': float(worst_p[best]), 'mean_block_precision': float(mean_p[best]),
            'mean_block_f1': float(mean_f1[best]),
            'blocks': [state_metrics(score[m], truth[m], threshold) for m in block_masks]}, pr


def choose(rows):
    keyed = {(r['candidate'], r['seed']): r for r in rows}
    if len(rows) != 12 or len(keyed) != 12 or set(keyed) != {(g, s) for g in FAMILIES for s in SEEDS}:
        raise DataError('Complete twelve-model source matrix required')
    base = [keyed['W50', s] for s in SEEDS]
    base_ok = all(r['passed'] for r in base)
    families = []
    for g in FAMILIES:
        cur = [keyed[g, s] for s in SEEDS]
        ok = base_ok and all(r['passed'] for r in cur)
        mean_p = float(np.mean([r['metrics']['precision'] for r in cur])) if ok else None
        mean_f1 = float(np.mean([r['metrics']['f1'] for r in cur])) if ok else None
        if ok and g != 'W50':
            ok = (mean_f1 >= np.mean([r['metrics']['f1'] for r in base])
                  and all(r['metrics']['recall'] >= b['metrics']['recall']-.05 for r, b in zip(cur, base))
                  and sum(r['metrics']['precision'] > b['metrics']['precision'] for r, b in zip(cur, base)) >= 2)
        families.append({'candidate': g, 'eligible': bool(ok), 'mean_precision': mean_p, 'mean_f1': mean_f1})
    eligible = [r for r in families if r['eligible']]
    selected = max(eligible, key=lambda r: (r['mean_precision'], r['mean_f1'], -FAMILIES.index(r['candidate']))) if eligible else None
    return {'selected': selected['candidate'] if selected else None, 'families': families,
            'selection_data': 'House5_six_frozen_UTC_blocks_only'}


def old_arrays(parent, g, s, partition):
    with np.load(parent/f'predictions/{g}/seed_{s}/{partition}.npz', allow_pickle=False) as a:
        return {k: a[k] for k in ('stamps', 'aggregate', 'raw')}


def replay(run, parent, g, s, partition, threshold, rows):
    target = run/f'replays/{g}_{s}_{partition}.npz'
    receipt = target.with_suffix('.json')
    if receipt.exists():
        r = read_json(receipt)
        if sha256(target) != r['sha256'] or r['threshold'] != threshold:
            raise DataError('Interrupted replay receipt changed')
        return r
    expected = old_arrays(parent, g, s, partition)
    stamps, aggregate = aggregate_arrays(rows, 8)
    if not np.array_equal(stamps, expected['stamps']) or not np.array_equal(aggregate, expected['aggregate'], equal_nan=True):
        raise DataError('Inference time/aggregate identity mismatch')
    meta, model = load_model(parent/f'models/{g}/seed_{s}')
    ordinary = predict_rows(meta, model, rows)
    actual = np.array([r['appliances']['microwave']['raw_power_w'] if
                       r['appliances']['microwave']['raw_power_w'] is not None else np.nan for r in ordinary])
    original = expected['raw'][:, 1]
    if not np.array_equal(actual, original, equal_nan=True):
        raise DataError('Full ordinary raw-score replay differs from frozen model output')
    state = decisions(original, threshold)
    if not np.array_equal(state, decisions(actual, threshold)):
        raise DataError('Ordinary decision replay differs')
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, stamps=stamps, score=original, state=state)
    r = {'candidate': g, 'seed': s, 'partition': partition, 'threshold': threshold,
         'path': target.relative_to(run).as_posix(), 'sha256': sha256(target),
         'rows': len(stamps), 'maximum_raw_difference_w': 0., 'state_equal': True,
         'ordinary_model_path': str(parent/f'models/{g}/seed_{s}'), 'labels_read': False}
    save(receipt, r)
    print(f'ordinary replay {g} seed{s} {partition}: {len(stamps)} rows', flush=True)
    return r


def stage_check(run, name):
    stage = read_json(run/(name+'.json'))
    check_files(run, stage['artifacts'])
    return stage


def source_select(run):
    reg, parent = context(run)
    if (run/'source_complete.json').exists():
        return stage_check(run, 'source_complete')
    data = read_partition(parent/'protocol/protocol.json', SV, 'calibration', include_labels=True)
    stamps, _ = aggregate_arrays(data['aggregate'], 8)
    truth = aligned_truth(data['microwave'], stamps, 'refit_5')
    support = read_json(ROOT/reg['plan']['block_support'])['refit_5']
    for channel in ('aggregate', 'microwave'):
        artifact = next(p for p in read_json(parent/'protocol/protocol.json')['partitions'] if p['partition_id'] == SV)['artifacts'][channel]
        if artifact['sha256'] != support['inputs'][channel]['sha256']:
            raise DataError('Frozen block data differs from original regression protocol')
    blocks = [b for b in support['blocks'].values() if b['supported']]
    if len(blocks) != 6:
        raise DataError('Exactly six frozen supported blocks required')
    masks = []
    for i, b in enumerate(blocks):
        with np.load((ROOT/reg['plan']['block_support']).parent/f'validation/block_{i}.npz', allow_pickle=False) as a:
            times = a['t']
            if (len(np.unique(times)) != len(times) or np.any(times < utc(b['start']))
                    or np.any(times >= utc(b['end'])) or not np.isin(times, stamps).all()):
                raise DataError('Frozen block timestamp identity invalid')
            if not np.array_equal(truth[np.searchsorted(stamps, times)], a['y'], equal_nan=True):
                raise DataError('Frozen block labels differ at matching timestamps')
            masks.append(np.isin(stamps, times))
    masks = np.array(masks)
    rows, replays = [], []
    (run/'source_pr').mkdir(exist_ok=True)
    for g in FAMILIES:
        for s in SEEDS:
            score = old_arrays(parent, g, s, SV)['raw'][:, 1]
            for b, mask in zip(blocks, masks):
                valid = mask & np.isfinite(score) & np.isfinite(truth)
                if int(valid.sum()) != b['windows'] or int((valid & (truth >= 200)).sum()) != b['positive_windows']:
                    raise DataError('Block support differs after timestamp alignment')
            result, pr = calibrate(score, truth, masks)
            np.savez_compressed(run/f'source_pr/{g}_{s}.npz', **pr)
            if result['passed']:
                result['metrics'] = state_metrics(np.where(masks.any(axis=0), score, np.nan), truth, result['threshold'])
            result.update(candidate=g, seed=s)
            rows.append(result)
            replays.append(replay(run, parent, g, s, SV, result['threshold'], data['aggregate']))
    selection = choose(rows)
    save(run/'source_rows.json', rows)
    save(run/'source_selection.json', selection)
    save(run/'source_support.json', {'household': 'refit_5', 'blocks': blocks,
         'alignment': 'household_id + UTC timestamp, no row-index transfer',
         'truth_sha256': hashlib.sha256(truth.tobytes()).hexdigest(), 'development_labels_parsed': False})
    return seal(run, 'source_complete', ['source_pr', 'source_rows.json', 'source_selection.json', 'source_support.json', 'replays'],
                {'selected': selection['selected'], 'completed_at': now(), 'development_labels_parsed': False})


def seal(run, name, names, extra):
    files = {}
    for n in names:
        p = run/n
        files.update({n+'/'+k: v for k, v in inventory(p).items()} if p.is_dir() else {n: sha256(p)})
    result = {**extra, 'artifacts': files}
    save(run/(name+'.json'), result)
    return result


def freeze(run):
    reg, parent = context(run)
    source = stage_check(run, 'source_complete')
    if (run/'freeze.json').exists():
        return validate_freeze(run)
    rows = read_partition(parent/'protocol/protocol.json', DEV, 'inference')['aggregate']
    receipts = []
    recipes = []
    for r in read_json(run/'source_rows.json'):
        g, s = r['candidate'], r['seed']
        receipts.append(replay(run, parent, g, s, DEV, r['threshold'], rows))
        directory = parent/f'models/{g}/seed_{s}'
        recipes.append({'candidate': g, 'seed': s, 'model_path': str(directory),
                        'model_sha256': sha256(directory/'model.json'), 'weights_sha256': sha256(directory/'weights.pt'),
                        'threshold_w': r['threshold'], 'source_gate_passed': r['passed']})
    save(run/'recipe.json', {'schema_version': 1, 'score': 'ordinary microwave raw_power_w before consistency correction',
         'decision': 'raw_power_w >= threshold_w; missing -> unavailable', 'selected': source['selected'],
         'models': recipes, 'source_selection_sha256': sha256(run/'source_selection.json'),
         'source_protocol_sha256': reg['original_protocol_sha256'], 'power_output': 'original_A_unchanged',
         'default_ui_changed': False})
    return seal(run, 'freeze', ['replays', 'recipe.json', 'source_complete.json', 'source_rows.json', 'source_selection.json'],
                {'frozen_at': now(), 'cases': receipts, 'source_complete_sha256': sha256(run/'source_complete.json'),
                 'development_labels_parsed': False, 'matrix_count': 12})


def validate_freeze(run):
    context(run)
    stage_check(run, 'source_complete')
    f = stage_check(run, 'freeze')
    expected = {(g, s, DEV) for g in FAMILIES for s in SEEDS}
    if (len(f['cases']) != 12 or {(c['candidate'], c['seed'], c['partition']) for c in f['cases']} != expected
            or f['development_labels_parsed'] is not False
            or f['source_complete_sha256'] != sha256(run/'source_complete.json')):
        raise DataError('Complete prelabel freeze and source selection required')
    return f


def gate(rows, selected):
    if selected is None:
        return {'precision_exit_passed': False, 'checks': {'source_preselection_exists': False}}
    plan = {'minimum_precision': .08, 'minimum_recall': .85, 'minimum_f1_gain': .02,
            'maximum_seed_f1_drop': .02, 'minimum_positive_seeds': 2, 'minimum_fp_reduction': .30}
    # W50 here is exclusively the original fixed-200W baseline; calibrated W50 is W50_CAL.
    result = development_gate(rows, selected+'_CAL', plan)
    result['checks']['exact_original_support'] = all(r['valid_samples'] == 137523 for r in rows)
    result['precision_exit_passed'] = all(result['checks'].values())
    return result


def score_development(run, parent, truth, stamps, write_pr=False):
    rows, diagnostics, fp_rows = [], [], []
    for s in SEEDS:
        base = old_arrays(parent, 'W50', s, DEV)
        if not np.array_equal(stamps, base['stamps']):
            raise DataError('Cached truth timestamps differ from original A')
        base_mask = np.isfinite(base['raw'][:, 1]) & np.isfinite(truth)
        for g in ['W50_FIXED'] + FAMILIES:
            original = old_arrays(parent, 'W50' if g == 'W50_FIXED' else g, s, DEV)
            if not np.array_equal(stamps, original['stamps']):
                raise DataError('Candidate timestamps differ')
            if not np.array_equal(original['raw'][:, [0, 2]], base['raw'][:, [0, 2]], equal_nan=True):
                raise DataError('Other appliance raw power/state changed')
            if g == 'W50_FIXED':
                score, threshold, candidate = base['raw'][:, 1], 200., 'W50'
            else:
                with np.load(run/f'replays/{g}_{s}_{DEV}.npz', allow_pickle=False) as a:
                    score = a['score'].copy()
                threshold = next(r['threshold'] for r in read_json(run/'source_rows.json') if r['candidate'] == g and r['seed'] == s)
                candidate = g+'_CAL'
            if not np.array_equal(np.isfinite(score) & np.isfinite(truth), base_mask) or int(base_mask.sum()) != 137523:
                raise DataError('Original baseline 137523-point support not retained')
            if threshold is None:
                metrics = {'calibration_failed': True, **state_metrics(np.full(len(score), np.nan), truth, 200)}
            else:
                metrics = state_metrics(score, truth, threshold)
            row = {'candidate': candidate, 'seed': s, 'partition': DEV, 'appliance': 'microwave',
                   'view': 'raw', 'decision_threshold_w': threshold, **metrics}
            rows.append(row)
            if g != 'W50_FIXED':
                diagnostics.append({'candidate': g, 'seed': s, **compact_metrics(original['raw'][:, 1], truth, 200)})
            if threshold is not None:
                fp = base_mask & (score >= threshold) & (truth < 200)
                base_fp = base_mask & (base['raw'][:, 1] >= 200) & (truth < 200)
                fp_rows.append({'candidate': candidate, 'seed': s, 'fp': int(fp.sum()),
                    'baseline_fp': int(base_fp.sum()), 'removed_fp': int((base_fp & ~fp).sum()),
                    'new_fp': int((~base_fp & fp).sum()), 'fp_reduction_fraction': 1-float(fp.sum()/base_fp.sum())})
            if write_pr:
                (run/'development_pr').mkdir(exist_ok=True)
                np.savez_compressed(run/f'development_pr/{candidate}_{s}.npz', **exact_curve(score, truth))
    selected = read_json(run/'source_selection.json')['selected']
    # Failed calibration rows remain diagnostic, excluded from the existing gate's finite metrics.
    gate_rows = [r for r in rows if r['candidate'] in ('W50', str(selected)+'_CAL')]
    return {'rows': rows, 'original_regression_power_diagnostics': diagnostics, 'false_positive_comparison': fp_rows,
            'summary': gate(gate_rows, selected), 'other_devices_raw_power_and_state_unchanged': True,
            'combination_microwave_power': 'original_A', 'power_improvement_claimed': False}


def evaluate(run):
    frozen = validate_freeze(run)  # Mandatory barrier before any development label cache access.
    reg, parent = context(run)
    if (run/'evaluation_complete.json').exists():
        return stage_check(run, 'evaluation_complete')
    cache_parent = (ROOT/reg['plan']['block_support']).parent
    receipt = read_json(cache_parent/'evaluation_complete.json')
    protocol = read_json(parent/'protocol/protocol.json')
    artifact = next(p for p in protocol['partitions'] if p['partition_id'] == DEV)['artifacts']['microwave']
    cache = cache_parent/'evaluation/truth_once.npz'
    if receipt['truth_original_sha256'] != artifact['sha256'] or sha256(cache) != receipt['artifacts']['truth_once.npz']:
        raise DataError('Historical truth cache does not match original development protocol')
    save(run/'evaluation_started.json', {'freeze_sha256': sha256(run/'freeze.json'),
         'source_selection_sha256': sha256(run/'source_selection.json'),
         'truth_cache_path': str(cache), 'truth_cache_sha256': sha256(cache),
         'truth_original_sha256': artifact['sha256'], 'cache_receipt_sha256': sha256(cache_parent/'evaluation_complete.json')})
    with np.load(cache, allow_pickle=False) as a:
        stamps, truth = a['t'], a['y']
    np.savez_compressed(run/'development_truth.npz', stamps=stamps, truth=truth)
    result = score_development(run, parent, truth, stamps, write_pr=True)
    save(run/'results.json', result)
    for name, data in [('comparison.csv', result['rows']), ('false_positives.csv', result['false_positive_comparison']),
                       ('power_diagnostics.csv', result['original_regression_power_diagnostics'])]:
        with (run/name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            writer.writeheader()
            writer.writerows(data)
    return seal(run, 'evaluation_complete', ['results.json', 'comparison.csv', 'false_positives.csv',
                'power_diagnostics.csv', 'development_pr', 'development_truth.npz', 'evaluation_started.json'],
                {'completed_at': now(), 'freeze_sha256': sha256(run/'freeze.json'), 'fits': 0,
                 'truth_csv_reopened': False, 'summary': result['summary']})


def verify(run):
    frozen = validate_freeze(run)
    reg, parent = context(run)
    rows = read_json(run/'source_rows.json')
    if choose(rows) != read_json(run/'source_selection.json'):
        raise DataError('Source preselection cannot be reproduced')
    # Rebuild threshold choices from sealed exact PR arrays, without any development data.
    for row in rows:
        with np.load(run/f"source_pr/{row['candidate']}_{row['seed']}.npz", allow_pickle=False) as a:
            valid = np.all([a[f'block_{i}_mixed'] & (a[f'block_{i}_recall'] >= .85) for i in range(6)], axis=0)
            idx = np.flatnonzero(valid)
            best = None
            if len(idx):
                wp = np.min([a[f'block_{i}_precision'] for i in range(6)], axis=0)
                mp = np.mean([a[f'block_{i}_precision'] for i in range(6)], axis=0)
                mf = np.mean([a[f'block_{i}_f1'] for i in range(6)], axis=0)
                best = float(a['threshold'][idx[np.lexsort((a['threshold'][idx], mf[idx], mp[idx], wp[idx]))[-1]]])
            if best != row['threshold']:
                raise DataError('Source threshold does not reproduce')
    for c in frozen['cases']:
        with np.load(run/c['path'], allow_pickle=False) as a:
            if not np.array_equal(a['state'], decisions(a['score'], c['threshold'])):
                raise DataError('Frozen state does not follow source-only threshold')
    result = {'status': 'verified', 'models': 12, 'fits': 0, 'full_ordinary_replays': 24,
              'development_label_csv_reopened': False, 'final_households_accessed': False,
              'verification_kind': 'local automated readback, not independent member or machine'}
    if (run/'evaluation_complete.json').exists():
        stage = stage_check(run, 'evaluation_complete')
        if stage['freeze_sha256'] != sha256(run/'freeze.json'):
            raise DataError('Evaluation freeze binding changed')
        with np.load(run/'development_truth.npz', allow_pickle=False) as a:
            rebuilt = score_development(run, parent, a['truth'], a['stamps'])
        if rebuilt != read_json(run/'results.json'):
            raise DataError('Development metrics cannot be rebuilt from frozen cache')
        result['summary'] = rebuilt['summary']
    save(run/'verification.json', result)
    return result


def predict_recipe(recipe_path, candidate, seed, input_path, output):
    if output.exists():
        raise DataError('New output required')
    recipe = read_json(recipe_path)
    r = next(r for r in recipe['models'] if r['candidate'] == candidate and r['seed'] == seed)
    if r['threshold_w'] is None:
        raise DataError('This model has no admissible source threshold')
    directory = Path(r['model_path'])
    if sha256(directory/'model.json') != r['model_sha256'] or sha256(directory/'weights.pt') != r['weights_sha256']:
        raise DataError('Recipe model hash mismatch')
    with input_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    meta, model = load_model(directory)
    ordinary = predict_rows(meta, model, rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['household_id', 'timestamp', 'raw_score_w', 'threshold_w', 'state'])
        writer.writeheader()
        for row in ordinary:
            score = row['appliances']['microwave']['raw_power_w']
            writer.writerow({'household_id': row['household_id'], 'timestamp': row['timestamp'], 'raw_score_w': score,
                             'threshold_w': r['threshold_w'], 'state': None if score is None else int(score >= r['threshold_w'])})
    return {'output': str(output), 'rows': len(rows), 'fits': 0}


def report(run):
    verification = verify(run)
    result = read_json(run/'results.json')
    selected = read_json(run/'source_selection.json')['selected']
    lines = ['# 冻结回归分数状态阈值实验', '', f'来源预选：{selected}；完整出口通过：{result["summary"]["precision_exit_passed"]}。', '',
             '0次新拟合；12个旧模型，24份来源/开发完整普通推理重放raw分数及阈值状态逐点一致。', '',
             '|候选|seed|阈值W|TP|FP|FN|P|R|F1|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['rows']:
        lines.append('|'+ '|'.join(str(r[k]) for k in ('candidate', 'seed', 'decision_threshold_w', 'true_positive',
                                                      'false_positive', 'false_negative', 'precision', 'recall', 'f1'))+'|')
    lines += ['', '正式基准为原W50固定200W。来源控制为单独校准W50；两者不能互换。', '',
              '候选阈值由House5六个UTC双日块精确PR决定，全部块召回保护85%。House17只在来源选择和12份开发预测封存后统一评价，未据开发结果改选。', '',
              '单户开发已多次查看，不能外推跨户收益。最终8/9/20未访问。默认UI未切换。组合中三设备原始功率仍用A，水壶/洗衣机状态逐点不变。旧回归功率误差另见power_diagnostics.csv，本任务不声称改善功率。', '',
              '精确来源PR：source_pr/；开发PR（仅评价后诊断）：development_pr/；逐seed混淆：comparison.csv；误报对照：false_positives.csv。', '',
              '权重及数据溯源：registration.json / recipe.json / source_support.json / evaluation_started.json。冻结、评价及核验分别见freeze.json / evaluation_complete.json / verification.json。', '',
              '这是独立状态头新协议，不改变真实200W标签，也不代表旧固定200W实验通过。失败不触发追加阈值算法、家族或权重搜索。']
    text = '\n'.join(lines)+'\n'
    path = run/'report.md'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise DataError('Report differs')
    path.write_text(text, encoding='utf-8')
    return {'report': str(path), **verification}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'source_select', 'freeze', 'evaluate', 'verify', 'report', 'predict'])
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/accuracy/microwave-regression-calibration-20261025-v1')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/microwave_regression_calibration_20261025.json')
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--candidate', choices=FAMILIES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.run.resolve(), args.config.resolve())
    elif args.stage == 'predict':
        result = predict_recipe(args.recipe, args.candidate, args.seed, args.input, args.output)
    else:
        result = globals()[args.stage](args.run.resolve())
    print(__import__('json').dumps(result, ensure_ascii=False, indent=2))
