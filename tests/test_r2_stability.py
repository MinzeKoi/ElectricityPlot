"""A single independently versioned L1 repair and its inference boundary."""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.common import DataError, read_json, write_json

try:
    import numpy as np
    import torch
    NUMERIC = True
except ImportError:
    NUMERIC = False

from test_r2 import R2ExperimentTest as _R2ExperimentTest


@unittest.skipUnless(NUMERIC, "R2 stability needs CPU dependencies")
class StabilityLearnerTest(unittest.TestCase):
    def setUp(self):
        self.settings = read_json(ROOT / "configs/r1_learners_20261022.json")

    def test_mixed_answers_clamp_bias_and_retain_historical_l1(self):
        from virtual_submeter.accuracy.learners import fit_parameters
        x = np.array([[1., -4.], [1., .4]])
        bags = [(x, 1.), (x, 0.), (x, 0.)]
        prior = np.array([0., 1.])
        old, _ = fit_parameters(prior, bags, None, self.settings, "L1")
        new, history = fit_parameters(prior, bags, None, self.settings, "L1_stable")
        self.assertLess(old[0], -.4)
        self.assertAlmostEqual(new[0], -.25)
        self.assertEqual(new[1], 1.)
        self.assertEqual((x @ old >= 0).tolist(), [False, False])
        self.assertEqual((x @ new >= 0).tolist(), [False, True])
        self.assertFalse(history["source_replay"])
        np.testing.assert_array_equal(prior, [0., 1.])
        positive, _ = fit_parameters(prior, [(x, 1.), (x, 1.), (x, 1.), (x, 0.)], None, self.settings, "L1_stable")
        self.assertAlmostEqual(positive[0], .25)

    def test_no_answers_and_single_class_keep_old_behavior(self):
        from virtual_submeter.accuracy.learners import fit_parameters
        prior = np.array([0., 1.])
        for bags in ([], [(np.array([[1., 2.]]), 0.)], [(np.array([[1., -2.]]), 1.)]):
            old, _ = fit_parameters(prior, bags, None, self.settings, "L1")
            new, _ = fit_parameters(prior, bags, None, self.settings, "L1_stable")
            np.testing.assert_array_equal(new, old)

    def test_new_identity_cannot_be_substituted_into_old_schema(self):
        from virtual_submeter.accuracy.dynamic_session import load_plan
        plan = load_plan(ROOT / "configs/r2_stability_20261023.json")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            for key, value in (("schema_version", 2), ("learner", "L1"), ("experiment_id", "r2-l1-limited-v1"),
                               ("minimum_random_f1_gain", .01), ("uncertainty_weight", .6)):
                write_json(path, {**plan, key: value})
                with self.subTest(key=key), self.assertRaises(DataError):
                    load_plan(path)


@unittest.skipUnless(NUMERIC, "R2 stability needs CPU dependencies")
class StabilityExperimentTest(_R2ExperimentTest):
    # Inherit the full interruption/replay, freeze barrier, ordinary inference,
    # unchanged power and corruption test with the independent repair plan.
    def setUp(self):
        super().setUp()
        self.query_plan_path = ROOT / "configs/r2_stability_20261023.json"


# unittest also discovers TestCase class aliases imported into a module.
del _R2ExperimentTest

if __name__ == "__main__":
    unittest.main()
