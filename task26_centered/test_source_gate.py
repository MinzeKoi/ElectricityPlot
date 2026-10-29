import unittest
import numpy as np

from task26_centered import source_gate as s
from virtual_submeter.accuracy.microwave_source5_temporal import VALIDATION


def row(seed, p, f, recall=1):
    return dict(seed=seed, precision=p, recall=recall, f1=f,
                degenerate=False, source_gate_passed=True)


class GateTest(unittest.TestCase):
    def test_relative_and_ranking(self):
        seeds = s.SEEDS
        checkpoints = dict.fromkeys(seeds, {'epoch': 1})
        old = [row(x, .5, .6) for x in seeds]
        new = [row(x, .6, .7) for x in seeds]
        self.assertEqual(s.decide(old, new, checkpoints)['selected'], 'task26')
        self.assertFalse(s.decide(old, [new[0]], checkpoints)['source_selected_candidate'])
        self.assertFalse(s.decide(old, [new[0], old[1], old[2]], checkpoints)['eligible'])
        self.assertFalse(s.decide(old, [row(x, .6, .5) for x in seeds], checkpoints)['eligible'])
        tie = [row(seeds[0], .6, .6), row(seeds[1], .6, .6), row(seeds[2], .3, .6)]
        self.assertTrue(s.decide(old, tie, checkpoints)['eligible'])
        self.assertEqual(s.decide(old, tie, checkpoints)['selected'], 'control_guard95')
        worse = [row(seeds[0], .51, .6), row(seeds[1], .51, .6), row(seeds[2], .1, .6)]
        self.assertTrue(s.decide(old, worse, checkpoints)['eligible'])
        self.assertEqual(s.decide(old, worse, checkpoints)['selected'], 'control_guard95')

    def test_full_pipeline_and_checkpoint_replay(self):
        blocks, full = {}, {}
        for i, low, _ in VALIDATION:
            key = f'block_{i}'
            t = low + 160 + np.arange(20) * 8
            y = np.array([250] * 6 + [0] * 14)
            p = np.array([.9] * 6 + [.1] * 14)
            blocks[key] = dict(t=t, y=y, event=np.array([1] * 3 + [2] * 3 + [0] * 14))
            full[key] = dict(t=t, p=p)
        scores = {k: d['p'] for k, d in full.items()}
        checkpoints = {seed: [scores] * 12 for seed in s.SEEDS}
        old = [row(seed, .5, .6) for seed in s.SEEDS]
        result = s.select_source(checkpoints, blocks, {seed: full for seed in s.SEEDS}, old, 'synthetic-registration')
        self.assertEqual(result['selected'], 'task26')
        self.assertTrue(all(r['checkpoint']['epoch'] == 1 and r['feasible'] for r in result['recipes']))
        self.assertTrue(all(len(r['blocks']) == 2 for r in result['recipes']))
        outside = {seed: dict(full) for seed in s.SEEDS}
        outside[42]['block_4'] = dict(t=full['block_4']['t'] - 1000, p=full['block_4']['p'])
        with self.assertRaises(ValueError):
            s.select_source(checkpoints, blocks, outside, old, 'synthetic-registration')
        broken = {seed: dict(full) for seed in s.SEEDS}
        broken[42]['block_4'] = dict(t=full['block_4']['t'], p=np.full(20, .2))
        with self.assertRaises(AssertionError):
            s.select_source(checkpoints, blocks, broken, old, 'synthetic-registration')


if __name__ == '__main__':
    unittest.main()


