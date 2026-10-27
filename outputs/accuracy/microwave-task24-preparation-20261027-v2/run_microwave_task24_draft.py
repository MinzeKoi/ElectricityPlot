"""Draft Task24 training adapter. No executable authorization exists yet."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def require_budget(run, receipt):
    if receipt is None:
        raise PermissionError('Task24 needs a new human budget bound to the final reviewed registration; no fit started')
    run = Path(run).resolve()
    registration_path = run / 'registration.json'
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    approval = json.loads(Path(receipt).read_text(encoding='utf-8'))
    expected = {'authorized': True, 'approval_origin': 'human', 'experiment': 'microwave_task24_multihouse_v1',
                'registration_sha256': sha(registration_path), 'additional_new_fits': 3,
                'maximum_updates': 10584, 'seeds': [42, 1019, 2026]}
    if any(approval.get(k) != v for k, v in expected.items()) or not approval.get('human_authorization_quote', '').strip() or not approval.get('human_authorization_reference', '').strip():
        raise PermissionError('A new, exact Task24 budget and human authorization reference are required')
    if registration.get('status') != 'sealed_pending_review' or registration.get('authorized') is not False:
        raise PermissionError('Preparation must be sealed before independent review')
    for rel, digest in registration['artifacts'].items():
        if sha(run / rel) != digest:
            raise PermissionError('Registered artifact changed: ' + rel)
    for path, digest in registration['inputs'].items():
        if sha(Path(path)) != digest:
            raise PermissionError('Registered input changed: ' + path)
    code = {p.relative_to(run).as_posix() for p in (run / 'source').rglob('*.py')}
    if not code or not code.issubset(registration['artifacts']) or 'run_microwave_task24_draft.py' not in registration['artifacts']:
        raise PermissionError('All executable source and this adapter must be registered')
    for name in ('independent_review_1.json', 'independent_review_2.json'):
        if approval.get('review_sha256', {}).get(name) != sha(run / name):
            raise PermissionError('Human budget must bind both review files')
        review = json.loads((run / name).read_text(encoding='utf-8'))
        if review.get('passed') is not True or review.get('reviewed_registration_sha256') != approval['registration_sha256']:
            raise PermissionError('Both independent read-only reviews must pass for this registration')
    if Path(__file__).resolve() != (run / 'run_microwave_task24_draft.py').resolve():
        raise PermissionError('Execute the immutable registered adapter')
    return approval


def source_eligible(control, rows, checkpoints, qualified):
    return (len(rows) == 3 and [v['seed'] for v in checkpoints] == [42, 1019, 2026]
            and all(v['source_gate_passed'] and not v['degenerate'] for v in rows)
            and all(v['source_gate_passed'] for v in checkpoints) and qualified(control, rows))


def train(run, receipt):
    run = Path(run).resolve()
    approval = require_budget(run, receipt)
    if (run / 'training_started.json').exists() or (run / 'models').exists():
        raise PermissionError('Training was already attempted; no retry is authorized')
    sys.path.insert(0, str(run / 'source'))
    import numpy as np
    import torch
    from virtual_submeter.accuracy.microwave_task21 import isolation
    from virtual_submeter.accuracy.microwave_source5_temporal_execution import fit_one
    from virtual_submeter.data.common import inventory
    registration = json.loads((run / 'registration.json').read_text(encoding='utf-8'))
    sources = registration['training_sources']
    allowed = [Path(x) for x in registration['inputs']] + [run / x for x in registration['artifacts']]
    required = [sources[k] for k in ('old_pool', 'index', 'plan', 'normalization')]
    required += sources['window_shards'] + list(sources['validation'].values()) + list(sources['batches'].values()) + list(sources['initial'].values())
    if not {Path(p).resolve() for p in required}.issubset({Path(p).resolve() for p in allowed}):
        raise PermissionError('Every training source must be registered')
    with isolation(run, allowed, training=True):
        with np.load(sources['old_pool'], allow_pickle=False) as old:
            original = {k: old[k] for k in ('x', 'mask', 'y', 't', 'house', 'event', 'raw_index')}
        with np.load(sources['index'], allow_pickle=False) as z:
            index = {k: z[k] for k in ('y', 't', 'house', 'event', 'raw_index')}
        for key in ('y', 't', 'house', 'event', 'raw_index'):
            np.testing.assert_array_equal(original[key], index[key][:336704])
        parts = []
        offset = 336704
        for path in sources['window_shards']:
            with np.load(path, allow_pickle=False) as z:
                end = offset + len(z['y'])
                for key in ('t', 'house', 'raw_index'):
                    np.testing.assert_array_equal(z[key], index[key][offset:end])
                np.testing.assert_array_equal(z['y'] >= 200, index['y'][offset:end] >= 200)
                if not np.isfinite(z['y']).all() or not np.array_equal(np.isfinite(z['x']), z['mask']):
                    raise ValueError('Invalid source windows or labels')
                parts.append({k: z[k] for k in ('x', 'mask', 'y')})
                offset = end
        if offset != 1393867:
            raise ValueError('Combined source identity incomplete')
        d = {k: np.concatenate([original[k]] + [p[k] for p in parts]) for k in ('x', 'mask', 'y')}
        d.update({k: index[k] for k in ('t', 'house', 'event')})
        if len(d['y']) != 1393867 or int((d['y'] >= 200).sum()) != 4389:
            raise ValueError('Combined source support differs; no training performed')
        plan = json.loads(Path(sources['plan']).read_text(encoding='utf-8'))
        meta = json.loads(Path(sources['normalization']).read_text(encoding='utf-8'))
        if [plan[k] for k in ('checkpoints', 'steps_per_checkpoint', 'batch_size', 'positive_per_batch', 'uniform_negative_per_batch', 'learning_rate')] != [12, 294, 256, 16, 240, .001]:
            raise ValueError('Original training plan differs')
        blocks = {}
        for block, path in sources['validation'].items():
            with np.load(path, allow_pickle=False) as z:
                blocks[block] = {k: z[k] for k in z.files}
        batches = {}
        for seed in (42, 1019, 2026):
            with np.load(sources['batches'][str(seed)], allow_pickle=False) as z:
                indices = z['indices']
            if indices.shape != (3528, 256) or indices.min() < 0 or indices.max() >= len(d['y']) or not ((d['y'][indices] >= 200).sum(axis=1) == 16).all():
                raise ValueError('Registered full batch matrix differs')
            batches[seed] = indices
        with (run / 'training_started.json').open('x', encoding='utf-8') as f:
            json.dump({'approval': approval, 'maximum_fits': 3, 'maximum_updates': 10584}, f)
        result = []
        for seed in (42, 1019, 2026):
            initial = torch.load(sources['initial'][str(seed)], map_location='cpu', weights_only=True)
            result.append(fit_one(run / f'models/unit_BCE/seed_{seed}', seed, plan, d, blocks, meta, batches[seed],
                                  initial, sha(run / 'registration.json'), pos_weight=1., recipe='unit_BCE',
                                  arm='task24', channels=16))
        with (run / 'training_complete.json').open('x', encoding='utf-8') as f:
            json.dump({'new_fits': 3, 'updates': 10584, 'control_retrains': 0, 'rows': result,
                       'artifacts': inventory(run / 'models')}, f)


def source_select(run, receipt):
    """Apply the previously registered Task14 source guard; no development input."""
    run = Path(run).resolve()
    require_budget(run, receipt)
    sys.path.insert(0, str(run / 'source'))
    import numpy as np
    from virtual_submeter.accuracy import microwave_source5_loss as loss
    from virtual_submeter.accuracy import microwave_event_gate as gate
    from virtual_submeter.accuracy import microwave_recall_guard95 as guard95
    from virtual_submeter.accuracy.microwave_task21 import isolation
    from virtual_submeter.data.common import inventory
    reg = json.loads((run / 'registration.json').read_text(encoding='utf-8'))
    sources = reg['training_sources']
    complete = json.loads((run / 'training_complete.json').read_text(encoding='utf-8'))
    if complete['new_fits'] != 3 or complete['updates'] != 10584:
        raise ValueError('Complete candidate matrix required')
    if (run / 'source_results').exists() or (run / 'source_selection.json').exists():
        raise PermissionError('Source selection is one-shot')
    allowed = [Path(x) for x in reg['inputs']] + [run / x for x in reg['artifacts']]
    required = [sources['source_control'], sources['source5_aggregate']] + list(sources['validation'].values())
    if not {Path(p).resolve() for p in required}.issubset({Path(p).resolve() for p in allowed}):
        raise PermissionError('Source control, aggregate and validation must be registered')
    allowed += [run / 'training_complete.json'] + [run / 'models' / x for x in complete['artifacts']]
    with isolation(run, allowed):
        gate.check(run / 'models', complete['artifacts'])
        control = json.loads(Path(sources['source_control']).read_text(encoding='utf-8'))
        raw = gate.csv_rows(Path(sources['source5_aggregate']), 'refit_5')
        times = np.array([loss.utc(row['timestamp']) for row in raw])
        recipes, rows = [], []
        (run / 'source_results').mkdir()
        for seed in (42, 1019, 2026):
            model = run / f'models/unit_BCE/seed_{seed}'
            meta = json.loads((model / 'model.json').read_text(encoding='utf-8'))
            scores = {}
            for block, low, high in loss.temporal.f.VALIDATION:
                with np.load(sources['validation'][f'block_{block}'], allow_pickle=False) as z:
                    tval, yval = z['t'], z['y']
                t, probability, _ = gate.infer(model, [raw[j] for j in np.flatnonzero((times >= low) & (times < high))])
                ix = gate.align(t, tval)
                with np.load(model / f'epoch_{meta["selected"]["epoch"]}_block_{block}.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(tval, z['t'])
                    np.testing.assert_allclose(probability[ix], z['scores'], rtol=1e-5, atol=1e-7)
                scores[str(block)] = {'t': t, 'p': probability, 'h': gate.local_max(t, probability), 'ix': ix, 'y': yval >= 200}
            low, _ = gate.select_threshold({k: {'values': v['p'][v['ix']], 'y': v['y']} for k, v in scores.items()},
                np.unique(np.concatenate([v['p'][np.isfinite(v['p'])] for v in scores.values()])), .95)
            high = guard95.calibrate95(scores, low) if low is not None else None
            metrics = [dict(block=k, **gate.confusion(gate.gate_state(v['p'], v['h'], low, high)[v['ix']], v['y'])) for k, v in scores.items()] if high is not None else []
            recipe = {'seed': seed, 'L': low, 'H': high, 'feasible': high is not None, 'blocks': metrics,
                'schema': 'event_gate_recipe_v1', 'eligible': high is not None, 'radius_samples': 15,
                'interval_seconds': 8, 'parent': str(model), 'parent_hashes': {n: sha(model / n) for n in ('model.json', 'weights.pt')},
                'algorithm_sha256': sha(Path(gate.__file__)), 'score_semantics': 'uncalibrated sigmoid state score'}
            with (run / f'source_results/recipe_{seed}.json').open('x', encoding='utf-8') as f:
                json.dump(recipe, f, allow_nan=False)
            for block, values in scores.items():
                np.savez_compressed(run / f'source_results/scores_{seed}_{block}.npz', **values)
            recipes.append(recipe)
            if metrics:
                rows.append(dict(seed=seed, **gate.mean_rows(metrics), degenerate=any(v['degenerate'] for v in metrics),
                                 source_gate_passed=all(v['recall'] >= .95 for v in metrics)))
        eligible = source_eligible(control['rows']['guard95'], rows, complete['rows'], loss.qualified)
        selection = {'source_only': True, 'selected': 'unit_BCE' if eligible else 'control_guard95',
                     'eligible': eligible, 'recipes': recipes, 'rows': rows,
                     'training_sha256': sha(run / 'training_complete.json')}
        with (run / 'source_selection.json').open('x', encoding='utf-8') as f:
            json.dump(selection, f, allow_nan=False)
        with (run / 'source_complete.json').open('x', encoding='utf-8') as f:
            json.dump({'source_sha256': sha(run / 'source_selection.json'), 'registration_sha256': sha(run / 'registration.json'),
                       'artifacts': inventory(run / 'source_results')}, f)
    return {'source_eligible': eligible, 'development_reads': 0}


def development_stage(run, receipt, command):
    """Open the legacy development identity only after the source guard passes."""
    run = Path(run).resolve()
    require_budget(run, receipt)
    sys.path.insert(0, str(run / 'source'))
    import types
    from virtual_submeter.accuracy import microwave_source5_loss as loss
    from virtual_submeter.accuracy import microwave_event_gate as gate
    from virtual_submeter.accuracy import microwave_task24_acceptance as acceptance
    from virtual_submeter.accuracy.microwave_task21 import isolation, load, fresh, require
    from virtual_submeter.data.common import inventory, read_json
    task = types.SimpleNamespace(SEEDS=[42, 1019, 2026], loss=loss, gate=gate, isolation=isolation,
        load=load, fresh=fresh, require=require, inventory=inventory, read_json=read_json, sha256=sha)
    task.authorize = lambda path, authorization: require_budget(path, authorization)
    task.registered = lambda path: require_budget(path, receipt)
    def allowed(path):
        reg = read_json(path / 'registration.json')
        paths = [Path(x) for x in reg['inputs']] + [path / x for x in reg['artifacts']]
        paths += [Path(receipt), path / 'registration.json', path / 'independent_review_1.json', path / 'independent_review_2.json']
        paths += [path / x for x in ('training_complete.json', 'source_selection.json', 'source_complete.json',
                                     'future_identity.json', 'freeze.json', 'inference_checks.json')]
        for folder in ('models', 'source_results', 'predictions'):
            if (path / folder).exists():
                paths.extend(path / folder / item for item in inventory(path / folder))
        return paths
    task.execution_allowed = allowed
    task.source_guard = lambda path: acceptance.source_guard(path, task=task)
    task.evaluation_guard = lambda path: acceptance.evaluation_guard(path, task=task)
    with isolation(run, allowed(run)):
        task.source_guard(run)
    reg = read_json(run / 'registration.json')
    legacy = reg['future_identity_source']
    with isolation(run, allowed(run) + [Path(legacy['path'])]):
        require(sha(Path(legacy['path'])) == legacy['sha256'], 'Legacy identity differs')
        target = run / 'future_identity.json'
        if not target.exists():
            with target.open('x', encoding='utf-8', newline='') as out:
                out.write(Path(legacy['path']).read_text(encoding='utf-8'))
        require(sha(target) == legacy['sha256'], 'Future identity differs')
    return getattr(acceptance, command)(run, receipt, task=task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'source_select', 'freeze', 'evaluate'])
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    result = development_stage(args.run, args.authorization, args.command) if args.command in ('freeze', 'evaluate') else globals()[args.command](args.run, args.authorization)
    print(json.dumps(result))
