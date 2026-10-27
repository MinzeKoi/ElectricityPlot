import unittest
import tempfile
import hashlib
import numpy as np
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from virtual_submeter.accuracy import microwave_task24_acceptance as adapter


class GuardFixture:
    SEEDS = [42, 1019, 2026]

    def __init__(self):
        self.run = Path('synthetic-run').resolve()
        self.reads = []
        self.recipes = [dict(seed=s, feasible=True, blocks=[dict(recall=.96, degenerate=False)] * 2) for s in self.SEEDS]
        self.selection = dict(training_sha256='training', source_only=True, eligible=True, selected='unit_BCE', recipes=self.recipes, rows=[dict(seed=s) for s in self.SEEDS])
        self.values = {
            'registration.json': {'training_sources': {'source_control': 'synthetic-control.json'}},
            'synthetic-control.json': {'rows': {'guard95': []}},
            'source_selection.json': self.selection,
            'source_complete.json': {'source_sha256': 'source', 'registration_sha256': 'registration', 'artifacts': {'sealed': 'x'}},
            'training_complete.json': {'new_fits': 3, 'updates': 10584, 'rows': [dict(seed=s, source_gate_passed=True) for s in self.SEEDS], 'artifacts': {'sealed': 'x'}},
        }
        for r in self.recipes:
            self.values[f"recipe_{r['seed']}.json"] = r.copy()
        self.gate = SimpleNamespace(check=Mock())
        self.loss = SimpleNamespace(qualified=Mock(return_value=True))

    def read_json(self, path):
        name = Path(path).name
        self.reads.append(name)
        if name not in self.values:
            raise AssertionError(f'unexpected read: {name}')
        return self.values[name]

    def sha256(self, path):
        name = Path(path).name
        if name not in {'source_selection.json', 'registration.json', 'training_complete.json'}:
            raise AssertionError(f'unexpected hash: {name}')
        return {'source_selection.json': 'source', 'registration.json': 'registration', 'training_complete.json': 'training'}[name]

    def require(self, condition, message):
        if not condition:
            raise PermissionError(message)

    def registered(self, run):
        pass

    def authorize(self, run, authorization):
        if authorization != 'synthetic-approval':
            raise PermissionError('missing authorization')

    def isolation(self, run, allowed):
        return nullcontext()

    def execution_allowed(self, run):
        return []

    def source_guard(self, run):
        return adapter.source_guard(run, task=self)

    def evaluation_guard(self, run):
        return adapter.evaluation_guard(run, task=self)


class Task24AcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.task = GuardFixture()

    def test_complete_source_passes(self):
        result = adapter.source_guard(self.task.run, task=self.task)
        self.assertEqual(result['selected'], 'unit_BCE')
        self.assertEqual(self.task.gate.check.call_count, 2)

    def test_rejections_precede_any_development_access(self):
        mutations = [
            lambda f: f.selection.update(eligible=False, selected='control_guard95'),
            lambda f: f.values['source_complete.json'].update(source_sha256='altered'),
            lambda f: f.selection['recipes'][0]['blocks'][0].update(recall=.94),
            lambda f: f.values['training_complete.json'].update(updates=10583),
            lambda f: f.values['training_complete.json']['rows'][0].update(source_gate_passed=False),
            lambda f: f.values['recipe_42.json'].update(feasible=False),
            lambda f: setattr(f.loss.qualified, 'return_value', False),
        ]
        for mutate in mutations:
            for operation in (adapter.freeze, adapter.evaluate):
                fixture = GuardFixture()
                mutate(fixture)
                with self.subTest(operation=operation.__name__, mutation=mutate), self.assertRaises(PermissionError):
                    operation(fixture.run, 'synthetic-approval', task=fixture)
                self.assertNotIn('future_identity.json', fixture.reads)
                self.assertNotIn('freeze.json', fixture.reads)

    def test_synthetic_freeze_and_evaluation_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            f = GuardFixture()
            f.run = Path(directory)
            n = 137523
            times = np.arange(n, dtype=np.int64)
            raw = np.zeros((n, 3), dtype=np.float32)
            scores = np.full(n, .1, dtype=np.float32)
            state = np.zeros(n, dtype=np.int8)
            base = Path('synthetic-baseline')
            dev = {'synthetic-aggregate.csv': 'digest'}
            dev.update({str(base / group / f'seed_{seed}' / 'synthetic-partition'): 'digest'
                        for seed in f.SEEDS for group in ('reference/predictions', 'predictions/W50')})
            f.values['future_identity.json'] = {'development_unlabeled': dev,
                'baseline': str(base), 'truth_cache': {'path': 'synthetic-truth', 'sha256': 'truth'}}
            f.values['source_selection.json']['recipes'] = [dict(r, L=.2, H=.3) for r in f.recipes]
            f.recipes = f.values['source_selection.json']['recipes']
            for recipe in f.recipes:
                f.values[f"recipe_{recipe['seed']}.json"] = recipe.copy()
            original_sha = f.sha256
            f.sha256 = lambda p: ('truth' if Path(p).name == 'synthetic-truth' else
                'recipe' if Path(p).name.startswith('recipe_') else
                'freeze' if Path(p).name == 'freeze.json' else 'checks' if Path(p).name == 'inference_checks.json' else original_sha(p))
            f.loss.check_paths = Mock()
            f.gate.PARTITION = 'synthetic-partition'
            f.gate.csv_rows = Mock(return_value=[])
            f.gate.infer = Mock(return_value=(times, scores, None))
            f.gate.ordinary = Mock(return_value={'t': times, 'scores': scores, 'state': state})
            f.gate.local_max = Mock(return_value=scores)
            f.gate.gate_state = Mock(return_value=state)
            f.gate.confusion = Mock(return_value={'precision': 1., 'recall': 1., 'f1': 1., 'fp': 0, 'degenerate': False})
            f.gate.hash_array = Mock(return_value='synthetic-mask')
            def load(path, keys=None):
                path = Path(path)
                if path.name == 'synthetic-truth':
                    return {'t': times, 'y': np.zeros(n)}
                if path.name == 'synthetic-partition':
                    return {'stamps': times, 'raw': raw, 'corrected': raw}
                with np.load(path, allow_pickle=False) as z:
                    return {k: z[k] for k in z.files}
            f.load = load
            f.inventory = lambda path: {p.name: 'artifact' for p in Path(path).iterdir() if p.is_file()}
            def fresh(path, value):
                if Path(path).exists():
                    raise FileExistsError(path)
                Path(path).write_text('synthetic', encoding='utf-8')
                f.values[Path(path).name] = value
            f.fresh = fresh
            def summary(run, metrics):
                self.assertEqual(len(metrics['rows']), 9)
                self.assertEqual([v['arm'] for v in metrics['rows']], ['A', 'W50', 'unit_BCE'] * 3)
                self.assertTrue(all(v['n'] == n for v in metrics['supports']))
                from virtual_submeter.accuracy import microwave_source5_loss
                with patch.object(microwave_source5_loss, 'read_json', return_value={'selected': 'unit_BCE'}):
                    result = microwave_source5_loss.summary(run, metrics)
                self.assertEqual(len(result['acceptance_gates']), 8)
                return result
            f.loss.summary = summary
            self.assertEqual(adapter.freeze(f.run, 'synthetic-approval', task=f), {'frozen': True, 'truth_reads': 0})
            result = adapter.evaluate(f.run, 'synthetic-approval', task=f)
            self.assertFalse(result['development_acceptance_passed'])
            self.assertEqual(len(result['acceptance_gates']), 8)
            self.assertTrue((f.run / 'evaluation_complete.json').exists())

    def test_baselines_require_exact_allowlist_under_real_isolation(self):
        from virtual_submeter.accuracy.microwave_task21 import isolation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'synthetic-baseline'
            aggregate = root / 'aggregate.csv'
            aggregate.write_text('synthetic', encoding='utf-8')
            times = np.arange(3, dtype=np.int64)
            raw = np.zeros((3, 3), dtype=np.float32)
            scores = np.full(3, .1, dtype=np.float32)
            state = np.zeros(3, dtype=np.int8)
            dev = {str(aggregate): hashlib.sha256(aggregate.read_bytes()).hexdigest()}
            for seed in (42, 1019, 2026):
                for group in ('reference/predictions', 'predictions/W50'):
                    path = base / group / f'seed_{seed}' / 'synthetic.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, stamps=times, raw=raw, corrected=raw)
                    dev[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            baseline_keys = [key for key in dev if key != str(aggregate)]
            for missing_key in baseline_keys + [None]:
                f = GuardFixture()
                f.run = root / ('missing-' + str(baseline_keys.index(missing_key)) if missing_key else 'complete')
                f.run.mkdir()
                f.isolation = isolation
                f.gate.PARTITION = 'synthetic.npz'
                f.values['future_identity.json'] = {'baseline': str(base), 'development_unlabeled': dict(dev)}
                if missing_key:
                    f.values['future_identity.json']['development_unlabeled'].pop(missing_key)
                f.values['source_selection.json']['recipes'] = [dict(r, L=.2, H=.3) for r in f.recipes]
                for recipe in f.values['source_selection.json']['recipes']:
                    f.values[f"recipe_{recipe['seed']}.json"] = recipe.copy()
                f.gate.csv_rows = Mock(side_effect=lambda path, _: Path(path).read_text(encoding='utf-8'))
                f.gate.infer = Mock(return_value=(times, scores, None))
                f.gate.ordinary = Mock(return_value={'t': times, 'scores': scores, 'state': state})
                f.gate.local_max = Mock(return_value=scores)
                f.gate.gate_state = Mock(return_value=state)
                def check_paths(mapping):
                    for path, expected in mapping.items():
                        f.require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, 'identity mismatch')
                f.loss.check_paths = check_paths
                def load(path, keys):
                    with np.load(path, allow_pickle=False) as z:
                        return {key: z[key] for key in keys}
                f.load = load
                f.inventory = lambda path: {p.name: 'synthetic' for p in Path(path).iterdir() if p.is_file()}
                original_sha = f.sha256
                f.sha256 = lambda p: ('synthetic' if Path(p).name.startswith('recipe_') or
                    Path(p).name == 'inference_checks.json' else original_sha(p))
                def fresh(path, value):
                    Path(path).write_text('synthetic', encoding='utf-8')
                f.fresh = fresh
                if missing_key:
                    with self.assertRaisesRegex(PermissionError, 'Missing sealed baseline'):
                        adapter.freeze(f.run, 'synthetic-approval', task=f)
                    f.gate.csv_rows.assert_not_called()
                else:
                    self.assertEqual(adapter.freeze(f.run, 'synthetic-approval', task=f),
                                     {'frozen': True, 'truth_reads': 0})
                    self.assertEqual(len(list((f.run / 'predictions').glob('*.npz'))), 9)
                    f.gate.csv_rows.assert_called_once()

    def test_missing_budget_fails_before_source_or_development(self):
        for operation in (adapter.freeze, adapter.evaluate):
            with self.assertRaises(PermissionError):
                operation(self.task.run, None, task=self.task)
        self.assertEqual(self.task.reads, [])

    def test_incomplete_frozen_matrix_rejected_before_truth(self):
        self.task.values['freeze.json'] = {'artifacts': {}, 'matrix': [], 'truth_opened': False,
            'registration_sha256': 'registration', 'source_sha256': 'source'}
        with self.assertRaises(PermissionError):
            adapter.evaluate(self.task.run, 'synthetic-approval', task=self.task)
        self.assertNotIn('future_identity.json', self.task.reads)


if __name__ == '__main__':
    unittest.main()

