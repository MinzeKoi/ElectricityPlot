"""Task24 adaptation of the fixed Task22 source/freeze/evaluation pipeline.

The caller supplies the registered experiment context; no stage runs on import.
"""
from pathlib import Path
import numpy as np

def source_guard(run, *, task):
    task.registered(run)
    s = task.read_json(run / 'source_selection.json')
    seal = task.read_json(run / 'source_complete.json')
    task.require(seal['source_sha256'] == task.sha256(run / 'source_selection.json') and seal['registration_sha256'] == task.sha256(run / 'registration.json'), 'Source seal differs')
    task.gate.check(run / 'source_results', seal['artifacts'])
    task.require(s['training_sha256'] == task.sha256(run / 'training_complete.json') and s['source_only'] and s['eligible'] and (s['selected'] == 'unit_BCE') and ([r['seed'] for r in s['recipes']] == task.SEEDS) and all((r['feasible'] for r in s['recipes'])) and all(r['feasible'] and len(r['blocks'])==2 and all(b['recall']>=.95 and not b['degenerate'] for b in r['blocks']) for r in s['recipes']) and task.loss.qualified(task.read_json(Path(task.read_json(run / 'registration.json')['training_sources']['source_control']))['rows']['guard95'], s['rows']), 'Source rejected/incomplete: 0 development reads')
    control = task.read_json(Path(task.read_json(run / 'registration.json')['training_sources']['source_control']))['rows']['guard95']
    task.require((np.mean([r['precision'] for r in s['rows']]), np.mean([r['f1'] for r in s['rows']]))
                 > (np.mean([r['precision'] for r in control]), np.mean([r['f1'] for r in control])),
                 'Control wins source ranking or tie: 0 development reads')
    task.require(all(task.read_json(run / f"source_results/recipe_{r['seed']}.json") == r for r in s['recipes']), 'Source recipe differs from sealed selection')
    complete = task.read_json(run / 'training_complete.json')
    task.require(complete['new_fits'] == 3 and complete['updates'] == 10584 and [v['seed'] for v in complete['rows']] == task.SEEDS and all(v['source_gate_passed'] for v in complete['rows']), 'Original85 checkpoint guard failed')
    task.gate.check(run / 'models', task.read_json(run / 'training_complete.json')['artifacts'])
    return s

def freeze(run, authorization=None, *, task):
    run = Path(run).resolve()
    task.authorize(run, authorization)
    with task.isolation(run, task.execution_allowed(run)):
        s = task.source_guard(run)
    identity = task.read_json(run / 'future_identity.json')
    dev = identity['development_unlabeled']
    base = Path(identity['baseline'])
    baseline_paths = [base / group / f'seed_{seed}' / task.gate.PARTITION
                      for seed in task.SEEDS for group in ('reference/predictions', 'predictions/W50')]
    allowed_dev = {Path(path).resolve() for path in dev}
    task.require(all(path.resolve() in allowed_dev for path in baseline_paths),
                 'Missing sealed baseline identity: 0 development reads')
    with task.isolation(run, task.execution_allowed(run) + list(dev)):
        task.require(not (run / 'predictions').exists(), 'One-shot freeze; no retry')
        task.loss.check_paths(dev)
        raw = task.gate.csv_rows(next((Path(p) for p in dev if p.endswith('aggregate.csv'))), 'refit_17')
        (run / 'predictions').mkdir()
        checks = []
        for seed in task.SEEDS:
            a = task.load(base / f'reference/predictions/seed_{seed}/{task.gate.PARTITION}', ['stamps', 'raw', 'corrected'])
            w = task.load(base / f'predictions/W50/seed_{seed}/{task.gate.PARTITION}', ['stamps', 'raw'])
            np.testing.assert_array_equal(a['stamps'], w['stamps'])
            states = np.where(np.isfinite(a['corrected']), a['corrected'] >= np.array([2000, 200, 20]), -1).astype(np.int8)
            np.savez_compressed(run / f'predictions/A_{seed}.npz', t=a['stamps'], raw=a['raw'], corrected=a['corrected'], states=states)
            np.savez_compressed(run / f'predictions/W50_{seed}.npz', t=w['stamps'], raw=w['raw'])
            recipe = next((r for r in s['recipes'] if r['seed'] == seed))
            t, p, _ = task.gate.infer(run / f'models/unit_BCE/seed_{seed}', raw)
            ordinary = task.gate.ordinary(run / f'source_results/recipe_{seed}.json', raw)
            np.testing.assert_array_equal(t, ordinary['t'])
            np.testing.assert_array_equal(p, ordinary['scores'])
            np.testing.assert_array_equal(t, a['stamps'])
            np.testing.assert_array_equal(np.isfinite(p), np.isfinite(a['raw'][:, 1]))
            st = task.gate.gate_state(p, task.gate.local_max(t, p), recipe['L'], recipe['H'])
            other = states.copy()
            other[:, 1] = st
            np.testing.assert_array_equal(st, ordinary['state'])
            np.savez_compressed(run / f'predictions/unit_BCE_{seed}.npz', t=t, scores=p, state=st, raw=a['raw'], corrected=a['corrected'], other_states=other)
            checks.append(dict(seed=seed, ordinary_exact=True, public_api='microwave_event_gate.ordinary', recipe_sha256=task.sha256(run / f'source_results/recipe_{seed}.json'), raw_A_exact=True, other_states_A_exact=True, support_exact=True))
        task.fresh(run / 'inference_checks.json', dict(checks=checks))
        task.fresh(run / 'freeze.json', dict(matrix=[[a, s] for a in ('A', 'W50', 'unit_BCE') for s in task.SEEDS], registration_sha256=task.sha256(run / 'registration.json'), source_sha256=task.sha256(run / 'source_selection.json'), artifacts={'predictions/' + k: v for k, v in task.inventory(run / 'predictions').items()} | {'inference_checks.json': task.sha256(run / 'inference_checks.json')}, truth_opened=False))
    return dict(frozen=True, truth_reads=0)

def evaluation_guard(run, *, task):
    task.source_guard(run)
    f = task.read_json(run / 'freeze.json')
    expected = {f'predictions/{a}_{s}.npz' for a in ('A', 'W50', 'unit_BCE') for s in task.SEEDS} | {'inference_checks.json'}
    task.require(set(f['artifacts']) == expected and f['matrix'] == [[a, s] for a in ('A', 'W50', 'unit_BCE') for s in task.SEEDS] and (f['truth_opened'] is False) and (f['registration_sha256'] == task.sha256(run / 'registration.json')) and (f['source_sha256'] == task.sha256(run / 'source_selection.json')), 'Incomplete/unbound frozen development matrix')
    task.gate.check(run, f['artifacts'])
    checks = task.read_json(run / 'inference_checks.json')['checks']
    task.require([v['seed'] for v in checks] == task.SEEDS and all((v['ordinary_exact'] and v['raw_A_exact'] and v['other_states_A_exact'] and v['support_exact'] and (v['public_api'] == 'microwave_event_gate.ordinary') and (v['recipe_sha256'] == task.sha256(run / f"source_results/recipe_{v['seed']}.json")) for v in checks)), 'Public inference verification matrix incomplete')

def evaluate(run, authorization=None, *, task):
    run = Path(run).resolve()
    task.authorize(run, authorization)
    with task.isolation(run, task.execution_allowed(run)):
        task.evaluation_guard(run)
    identity = task.read_json(run / 'future_identity.json')
    truth = identity['truth_cache']
    with task.isolation(run, task.execution_allowed(run) + [truth['path']]):
        task.require(not (run / 'evaluation').exists(), 'One development acceptance only')
        (run / 'evaluation').mkdir()
        task.fresh(run / 'evaluation/started.json', dict(freeze_sha256=task.sha256(run / 'freeze.json')))
        task.require(task.sha256(Path(truth['path'])) == truth['sha256'], 'Truth cache binding changed')
        z = task.load(truth['path'], ['t', 'y'])
        rows = []
        supports = []
        np.savez_compressed(run / 'evaluation/truth_once.npz', **z)
        for seed in task.SEEDS:
            a = task.load(run / f'predictions/A_{seed}.npz')
            w = task.load(run / f'predictions/W50_{seed}.npz')
            c = task.load(run / f'predictions/unit_BCE_{seed}.npz')
            for v in (a, w, c):
                np.testing.assert_array_equal(v['t'], z['t'])
            mask = np.isfinite(z['y']) & np.isfinite(w['raw'][:, 1])
            np.testing.assert_array_equal(mask, np.isfinite(z['y']) & np.isfinite(a['raw'][:, 1]))
            np.testing.assert_array_equal(np.isfinite(c['scores']), np.isfinite(a['raw'][:, 1]))
            task.require(mask.sum() == 137523, 'Original effective support differs')
            np.testing.assert_array_equal(c['raw'], a['raw'])
            np.testing.assert_array_equal(c['corrected'], a['corrected'])
            np.testing.assert_array_equal(c['other_states'][:, [0, 2]], a['states'][:, [0, 2]])
            for arm, v in (('A', a), ('W50', w), ('unit_BCE', c)):
                state = v['state'][mask] if arm == 'unit_BCE' else (v['raw'][mask, 1] >= 200).astype(np.int8)
                rows.append(dict(arm=arm, seed=seed, **task.gate.confusion(state, z['y'][mask] >= 200)))
            supports.append(dict(seed=seed, n=int(mask.sum()), mask_sha256=task.gate.hash_array(mask)))
        metrics = dict(rows=rows, supports=supports)
        result = task.loss.summary(run, metrics)
        task.fresh(run / 'evaluation/metrics.json', metrics)
        task.fresh(run / 'evaluation/summary.json', result)
        task.fresh(run / 'evaluation_complete.json', dict(freeze_sha256=task.sha256(run / 'freeze.json'), artifacts=task.inventory(run / 'evaluation')))
    return result
