"""Guard actual time/context isolation and capability-probe decisions."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from virtual_submeter.accuracy import microwave_diagnosis as d
from virtual_submeter.models.arrays import windows, labels_at_centers
from virtual_submeter.data.common import iso, read_json


class MicrowaveDiagnosisTest(unittest.TestCase):
    def test_missing_labels_break_events_not_off_and_partial_events_not_small(self):
        t = np.arange(10, dtype=np.int64) * 8
        y = np.array([0, 300, 300, 0, np.nan, 300, 0, 300, 300, 300])
        ids, events = d.segments(t, y)
        np.testing.assert_array_equal(ids, [-1, 0, 0, -1, -1, 1, -1, 2, 2, 2])
        self.assertEqual([e["complete"] for e in events], [True, False, False])

    def test_whole_event_crossing_boundary_excluded_and_context_disjoint(self):
        t = np.arange(300, dtype=np.int64) * 8
        y = np.zeros(len(t))
        y[98:103] = 300
        ids, events = d.segments(t, y)
        x, c = windows(t, np.ones(len(t), np.float32), 31, 8)
        left = d.split_mask(t, c, ids, events, 0, 100*8)
        right = d.split_mask(t, c, ids, events, 100*8, 300*8)
        self.assertFalse(np.isin(ids[c[left]], [0]).any())
        self.assertFalse(np.isin(ids[c[right]], [0]).any())
        self.assertLess(t[c[left][-1]] + 120, t[c[right][0]] - 120)

    def test_label_timestamp_alignment_and_missing_context_real_api(self):
        t = np.arange(70, dtype=np.int64) * 8
        a = np.ones(len(t), np.float32)
        a[35] = np.nan
        x, c = windows(t, a, 31, 8)
        self.assertFalse(((c >= 20) & (c <= 50)).any())
        rows = [{"timestamp": iso(int(t[i])), "power_w": str(300 if i == 17 else 0)} for i in range(70) if i != 16]
        y = labels_at_centers({"microwave": rows}, ["microwave"], t, c)[:, 0]
        self.assertTrue(np.isnan(y[c == 16]).all())
        self.assertEqual(y[c == 17][0], 300)

    def test_weighted_bce_gradient_really_changes_original_network(self):
        plan = read_json(Path(__file__).resolve().parents[1] / "configs/microwave_diagnosis_20261025.json")
        model, optimizer = d.configure(42, plan)
        x = torch.ones(12, 31)
        y = torch.tensor([0., 1.] * 6)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=torch.tensor(50.))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertFalse(d.small_success({"precision": 1., "recall": 1., "ap": 1., "bce": .11}, plan))
        self.assertTrue(d.small_success({"precision": 1., "recall": 1., "ap": 1., "bce": .01}, plan))

    def test_exact_pr_ap_ties_and_confusion(self):
        m = d.metrics(np.array([.9, .9, .1, .1]), np.array([300, 0, 300, 0]), .5)
        self.assertEqual([m[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(m["ap"], .5)
        upper = d.diagnostic_upper(np.array([.9, .9, .1, .1]), np.array([300, 0, 300, 0]))
        self.assertTrue(upper["only_diagnostic_not_deployable"])


if __name__ == "__main__":
    unittest.main()
