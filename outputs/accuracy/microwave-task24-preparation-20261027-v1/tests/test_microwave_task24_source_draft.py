"""Synthetic end-to-end source selection; no model inference or training."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('task24_source_draft', ROOT / 'scripts/run_microwave_task24_draft.py')
task24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task24)
from virtual_submeter.accuracy import microwave_event_gate as gate
from virtual_submeter.accuracy import microwave_source5_loss as loss
from virtual_submeter.data.common import inventory


class SourceSelection(unittest.TestCase):
    def run_case(self, probabilities):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            validation = {}
            raw = []
            for block, lo, _ in loss.temporal.f.VALIDATION:
                t = lo + np.arange(4, dtype=np.int64) * 240
                path = run / f'block_{block}.npz'
                np.savez_compressed(path, t=t, y=np.array([0, 0, 300, 300], np.float32))
                validation[f'block_{block}'] = str(path)
                raw.extend({'timestamp': datetime.fromtimestamp(int(x), timezone.utc).isoformat()} for x in t)
            control = run / 'control.json'
            control.write_text(json.dumps({'rows': {'guard95': [dict(seed=s, f1=.6, precision=.5, recall=1.,
                degenerate=False, source_gate_passed=True) for s in (42, 1019, 2026)]}}), encoding='utf-8')
            aggregate = run / 'aggregate.csv'
            aggregate.write_text('fixture', encoding='utf-8')
            for seed in (42, 1019, 2026):
                model = run / f'models/unit_BCE/seed_{seed}'
                model.mkdir(parents=True)
                (model / 'model.json').write_text(json.dumps({'selected': {'epoch': 1}}), encoding='utf-8')
                (model / 'weights.pt').write_bytes(b'fixture')
                for block in (4, 5):
                    with np.load(validation[f'block_{block}']) as z:
                        np.savez_compressed(model / f'epoch_1_block_{block}.npz', t=z['t'], scores=probabilities)
            registration = {'inputs': {str(p): 'fixture' for p in [control, aggregate, *map(Path, validation.values()), Path(gate.__file__)]},
                'artifacts': {}, 'training_sources': {'source_control': str(control), 'source5_aggregate': str(aggregate), 'validation': validation}}
            (run / 'registration.json').write_text(json.dumps(registration), encoding='utf-8')
            (run / 'training_complete.json').write_text(json.dumps({'new_fits': 3, 'updates': 10584,
                'rows': [{'seed': s, 'source_gate_passed': True} for s in (42, 1019, 2026)],
                'artifacts': inventory(run / 'models')}), encoding='utf-8')
            def infer(_, rows):
                return np.array([loss.utc(r['timestamp']) for r in rows]), probabilities, {}
            with patch.object(task24, 'require_budget', return_value={}), \
                 patch.object(gate, 'csv_rows', return_value=raw), patch.object(gate, 'infer', side_effect=infer):
                result = task24.source_select(run, 'synthetic-only')
            selected = json.loads((run / 'source_selection.json').read_text(encoding='utf-8'))
            self.assertFalse(any(p.name.startswith('prediction') for p in run.iterdir()))
            return result, selected

    def test_eligible_source_selects_candidate(self):
        result, selected = self.run_case(np.array([.1, .2, .8, .9]))
        self.assertTrue(result['source_eligible'])
        self.assertEqual(selected['selected'], 'unit_BCE')

    def test_infeasible_source_retains_control(self):
        result, selected = self.run_case(np.array([.5, .5, .5, .5]))
        self.assertFalse(result['source_eligible'])
        self.assertEqual(selected['selected'], 'control_guard95')


if __name__ == '__main__':
    unittest.main()
