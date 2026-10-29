import unittest
import numpy as np
import torch

from task26_centered import post_source as p, source_gate as s
from task26_centered.test_source_gate import row
from virtual_submeter.accuracy.microwave_source5_temporal import VALIDATION


class ThirdChannel(torch.nn.Module):
    def forward(self, x):
        return x[:, 2, 63]


class PostSourceTest(unittest.TestCase):
    def test_ordinary_uses_centered_channel_and_preserves_support(self):
        t = np.arange(200, dtype=np.int64) * 8
        a = np.linspace(100, 2090, 200)
        meta = dict(kind='task26_centered_tcn_v1', window_samples=127,
                    inference_support='strict_complete_31', input_log_mean=0.,
                    input_log_std=1., batch_size=32)
        out = p.ordinary(meta, ThirdChannel(), t, a, .4, .4)
        np.testing.assert_array_equal(out['t'], t)
        self.assertTrue(np.isnan(out['scores'][:15]).all())
        self.assertTrue(np.isnan(out['scores'][-15:]).all())
        np.testing.assert_allclose(out['scores'][15:-15], .5, atol=1e-6)
        self.assertTrue(np.all(out['state'][:15] == -1))
        with self.assertRaises(ValueError):
            p.ordinary({**meta, 'kind': 'old_kind'}, ThirdChannel(), t, a, .4, .4)

    def test_authorization_and_replay(self):
        blocks, full = {}, {}
        for i, low, _ in VALIDATION:
            key = f'block_{i}'
            t = low + 160 + np.arange(20) * 8
            y = np.array([250] * 6 + [0] * 14)
            values = np.array([.9] * 6 + [.1] * 14)
            blocks[key] = dict(t=t, y=y, event=np.array([1] * 3 + [2] * 3 + [0] * 14))
            full[key] = dict(t=t, p=values)
        checkpoints = {seed: [{k: d['p'] for k,d in full.items()}] * 12 for seed in s.SEEDS}
        grids = {seed: full for seed in s.SEEDS}
        control = [row(seed, .5, .6) for seed in s.SEEDS]
        decision = s.select_source(checkpoints, blocks, grids, control, 'reg')
        opts = dict(receipt_verified=True, registration_verified=True, source_artifacts_verified=True, development_in_scope=True)
        self.assertTrue(p.source_precondition(checkpoints, blocks, grids, control, decision, 'reg', **opts))
        for key in opts:
            self.assertFalse(p.source_precondition(checkpoints, blocks, grids, control, decision, 'reg', **(opts | {key: False})))
        self.assertFalse(p.source_precondition(checkpoints, blocks, grids, control, decision | {'selected': 'control_guard95'}, 'reg', **opts))
        self.assertFalse(p.source_precondition(checkpoints, blocks, grids, control, decision, 'changed', **opts))

    def test_eight_development_gates(self):
        rows = []
        for seed in s.SEEDS:
            for arm in ('A','W50','task26'):
                candidate = arm == 'task26'
                rows.append(dict(arm=arm, seed=seed, precision=.4 if candidate else .2,
                    recall=.95, f1=.5 if candidate else .3, fp=50 if candidate else 100, degenerate=False))
        self.assertTrue(p.development_summary(rows, [{'n':137523}]*3)['development_acceptance_passed'])
        self.assertFalse(p.development_summary(rows, [{'n':137523}]*2)['development_acceptance_passed'])


if __name__ == '__main__':
    unittest.main()
