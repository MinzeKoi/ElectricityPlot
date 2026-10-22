"""Check audit booleans against the production resampler, including boundaries."""

from collections import Counter
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.data.prepare import Sample, resample

spec = importlib.util.spec_from_file_location("r0_audit", ROOT / "scripts/audit_refit_r0.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class R0AuditTest(unittest.TestCase):
    def test_vectorized_support_matches_production_without_exposing_values(self):
        rng = np.random.default_rng(42)
        for trial in range(30):
            stamps = np.r_[trial % 8, trial % 8 + np.cumsum(rng.integers(1, 90, 150))]
            validity = {name: rng.random(len(stamps)) > 0.08 for name in ("aggregate", "kettle")}
            issues = rng.random(len(stamps)) < 0.2
            samples = [Sample(int(t), {name: 1.0 if validity[name][i] and (name == "aggregate" or not issues[i]) else None
                                       for name in validity}, {name: set() for name in validity}) for i, t in enumerate(stamps)]
            expected = list(resample(samples, 8, {"raw_intervals_seconds": Counter()}))
            origin, timing, before, actual = audit.strict_masks(stamps, validity, issues)
            self.assertEqual(origin, expected[0][0])
            for name in validity:
                np.testing.assert_array_equal(actual[name], [item[1][name]["power_w"] is not None for item in expected])
                decomposition = ((~timing).sum() + (timing & ~before[name]).sum() + (before[name] & ~actual[name]).sum())
                self.assertEqual(decomposition, (~actual[name]).sum())

    def test_search_finds_earliest_later_window_and_rejects_future_gap(self):
        plan = {"window_days": 28, "split_gap_seconds": 3600, "minimum_coverage": 0.9, "minimum_valid_days_per_partition": 7}
        mask = np.ones(29 * 10800, dtype=bool)
        mask[:20000] = False
        result = audit.search_windows({"aggregate": mask, "kettle": mask}, 0, plan)
        self.assertIsNotNone(result["first_pass"])
        self.assertNotEqual(result["first_pass"]["start"], audit.iso(0))
        self.assertTrue(result["first_pass"]["all_present_channels_pass"])
        bad = mask.copy()
        bad[14 * 10800:] = False
        self.assertIsNone(audit.search_windows({"aggregate": mask, "kettle": bad}, 0, plan)["first_pass"])

    def test_label_gaps_do_not_change_query_input_support(self):
        aggregate = np.ones(1800, dtype=bool)
        baseline = audit.query_support(aggregate)
        labels = np.zeros(1800, dtype=bool)
        self.assertEqual(baseline, audit.query_support(aggregate))
        self.assertEqual(baseline["complete_aggregate_slots"], 12)
        self.assertEqual(baseline["complete_slots_with_31_bin_context"], 10)
        self.assertEqual(audit.query_support(labels)["complete_aggregate_slots"], 0)


if __name__ == "__main__":
    unittest.main()
