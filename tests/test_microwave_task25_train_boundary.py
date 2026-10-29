"""Exercise the real training isolation up to a substituted first fit call."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('task24_train_boundary', ROOT / 'scripts/run_microwave_task25.py')
task24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task24)
from virtual_submeter.accuracy import microwave_source5_temporal_execution as temporal


class ReachedFirstFit(Exception):
    pass


class FakeArchive:
    def __init__(self, values):
        self.values = values
        self.files = list(values)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __getitem__(self, key): return self.values[key]


class TrainingBoundary(unittest.TestCase):
    def test_real_workspace_isolation_reaches_first_substituted_fit(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as td:
            run = Path(td).resolve()
            count, old_n = 1393867, 336704
            all_y = np.zeros(count, np.float32)
            all_y[:2275] = 300
            all_y[old_n:old_n + 2114] = 300
            t = np.arange(count, dtype=np.int64)
            house = np.zeros(count, np.int64)
            house[old_n:] = 3
            event = np.zeros(count, np.int64)
            raw = np.arange(count, dtype=np.int64)
            values = {}
            def file(name, value):
                path = run / name
                path.write_bytes(b'fixture')
                values[str(path)] = value
                return str(path)
            def window(n):
                return np.broadcast_to(np.zeros(127, np.float32), (n, 127))
            def mask(n):
                return np.broadcast_to(np.ones(127, bool), (n, 127))
            old = file('old.npz', {'x': window(old_n), 'mask': mask(old_n), 'y': all_y[:old_n],
                't': t[:old_n], 'house': house[:old_n], 'event': event[:old_n], 'raw_index': raw[:old_n]})
            index = file('index.npz', {'y': all_y, 't': t, 'house': house, 'event': event, 'raw_index': raw})
            shards = []
            start = old_n
            for i, size in enumerate(np.array_split(np.arange(count-old_n), 8)):
                end = start + len(size)
                shards.append(file(f'part_{i}.npz', {'x': window(end-start), 'mask': mask(end-start),
                    'y': all_y[start:end], 't': t[start:end], 'house': house[start:end], 'raw_index': raw[start:end]}))
                start = end
            old_positive = np.flatnonzero((all_y >= 200) & (house <= 2))[:11]
            new_positive = np.flatnonzero((all_y >= 200) & (house >= 3))[:6]
            negative = np.flatnonzero(all_y < 200)[:240]
            cycle = np.stack([np.r_[old_positive[:n], new_positive[:16-n], negative] for n in (11,11,10)])
            batch = np.tile(cycle, (1176, 1))
            batches = {str(s): file(f'batch_{s}.npz', {'indices': batch}) for s in (42, 1019, 2026)}
            initial = {str(s): file(f'initial_{s}.pt', {}) for s in (42, 1019, 2026)}
            validation = {f'block_{b}': file(f'block_{b}.npz', {'y': np.zeros(1), 't': np.zeros(1)}) for b in (4, 5)}
            plan = run / 'plan.json'
            plan.write_text(json.dumps({'checkpoints': 12, 'steps_per_checkpoint': 294, 'batch_size': 256,
                'positive_per_batch': 16, 'uniform_negative_per_batch': 240, 'learning_rate': .001}), encoding='utf-8')
            norm = run / 'normalization.json'
            norm.write_text('{}', encoding='utf-8')
            sources = {'old_pool': old, 'index': index, 'window_shards': shards, 'plan': str(plan),
                'normalization': str(norm), 'validation': validation, 'batches': batches, 'initial': initial}
            files = [old, index, *shards, str(plan), str(norm), *validation.values(), *batches.values(), *initial.values()]
            (run / 'registration.json').write_text(json.dumps({'inputs': {p: 'synthetic' for p in files},
                'artifacts': {}, 'training_sources': sources}), encoding='utf-8')
            real_concat = np.concatenate
            def fake_load(path, *args, **kwargs):
                with Path(path).open('rb'):
                    pass
                return FakeArchive(values[str(path)])
            def concat(arrays, *args, **kwargs):
                if arrays[0].ndim == 2 and arrays[0].shape[1] == 127:
                    return np.broadcast_to(arrays[0][0], (sum(len(a) for a in arrays), 127))
                return real_concat(arrays, *args, **kwargs)
            with patch.object(task24, 'require_budget', return_value={}), patch.object(np, 'load', side_effect=fake_load), \
                 patch.object(np, 'concatenate', side_effect=concat), patch.object(torch, 'load', return_value={}), \
                 patch.object(temporal, 'fit_one', side_effect=ReachedFirstFit) as fit:
                with self.assertRaises(ReachedFirstFit):
                    task24.train(run, 'synthetic-only')
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(fit.call_args.kwargs['channels'], 16)
            self.assertEqual(fit.call_args.kwargs['pos_weight'], 1.)
            self.assertTrue((run / 'training_started.json').exists())
            self.assertFalse((run / 'models').exists())


if __name__ == '__main__':
    unittest.main()
