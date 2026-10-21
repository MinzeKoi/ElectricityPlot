"""Fixed-prefix fairness, weak semantics, source isolation and A2 round trips."""

import copy
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.common import DataError, iso, read_json, sha256, write_json

try:
    import numpy as np
    import torch
    NUMERIC = True
except ImportError:
    NUMERIC = False


@unittest.skipUnless(NUMERIC, "A2 requires the M2 CPU environment")
class LearnerTest(unittest.TestCase):
    def setUp(self):
        from virtual_submeter.accuracy.fixed_queries import load_plan
        self.plan = load_plan(ROOT / "configs/a2_learners.json")
        self.plan["target_epochs"] = 30

    def test_pool_gradient_and_positive_bag_do_not_label_every_point(self):
        from virtual_submeter.accuracy.learners import pool, fit_parameters
        for method in ("max", "smooth"):
            logits = np.array([-4.0, -3.0, 2.0])
            value, derivative = pool(logits, method, 0.25)
            for index in range(len(logits)):
                perturbed = logits.copy()
                perturbed[index] += 1e-6
                self.assertAlmostEqual((pool(perturbed, method, 0.25)[0] - value) / 1e-6, derivative[index], places=5)
            self.assertAlmostEqual(derivative.sum(), 1)
        x = np.array([[1.0, -4], [1.0, -3], [1.0, 2]])
        weight, _ = fit_parameters(np.array([0.0, 1.0]), [(x, 1.0)], None, self.plan, "L1")
        self.assertEqual((x @ weight >= 0).tolist(), [False, False, True])

    def test_no_usable_answers_is_exact_identity_for_every_new_variant(self):
        from virtual_submeter.accuracy.learners import fit_parameters, VARIANTS
        prior = np.array([0.1, 1.2, -0.2])
        for variant in VARIANTS[1:]:
            weight, history = fit_parameters(prior, [], None, self.plan, variant)
            np.testing.assert_array_equal(weight, prior)
            self.assertEqual(history["update_rule"], "exact_source_identity")

    def test_single_class_bias_is_bounded_and_slopes_are_preserved(self):
        from virtual_submeter.accuracy.learners import fit_parameters
        prior = np.array([0.0, 1.0, 0.0])
        x = np.array([[1.0, -1, 0.1], [1.0, 2, 0.2]])
        replay = (x, np.array([0.0, 1.0]))
        for variant in ("L1", "L2_max", "L2_smooth"):
            first, _ = fit_parameters(prior, [(x, 0.0)], replay, self.plan, variant)
            second, _ = fit_parameters(prior, [(x, 0.0)], replay, self.plan, variant)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first[1:], prior[1:])
            self.assertLessEqual(abs(first[0] - prior[0]), self.plan["single_class_bias_bound"] + 1e-12)
        mixed, _ = fit_parameters(prior, [(x, 0.0), (x, 1.0)], replay, self.plan, "L2_max")
        self.assertLessEqual(np.linalg.norm(mixed - prior), self.plan["parameter_radius"] + 1e-12)

    def test_skip_unsure_and_unqueried_device_preserve_source(self):
        from virtual_submeter.accuracy.learners import fit_adapter, score_adapter
        rows = [{"timestamp": iso(i * 8), "aggregate_power_w": 500.0, "appliances": {
            a: {"raw_power_w": float(i * 100), "raw_state": i >= 2} for a in ("kettle", "microwave")}} for i in range(4)]
        meta = {"appliances": ["kettle", "microwave"], "settings": {"on_thresholds_w": {"kettle": 200, "microwave": 200}, "hidden_units": 2}}
        records = [{"appliance_type": "kettle", "response": "unsure", "window_start": iso(0), "window_end": iso(32)}]
        records += [{**records[0], "response": "skipped"}]
        adapter, _ = fit_adapter(rows, records, meta, self.plan, "L1")
        self.assertEqual(adapter["active_appliances"], [])
        values = score_adapter(rows, meta, adapter)
        np.testing.assert_array_equal(values[:, 0], [0, 0, 1, 1])

    def test_score_support_cannot_be_changed_or_missing_normalizer_loaded(self):
        from virtual_submeter.accuracy.learners import apply_scores, validate_adapter
        rows = [{"appliances": {"kettle": {"raw_power_w": None}}}]
        with self.assertRaises(DataError):
            apply_scores(rows, ["kettle"], np.array([[0.5]]))
        meta = {"appliances": ["kettle"], "settings": {"hidden_units": 2}}
        head = {"schema_version": 1, "variant": "L2_max", "coefficients": {"kettle": [0, 1, 0, 0]}, "active_appliances": ["kettle"], "normalizer": {"mean": [0, 0], "std": [1, 0]}}
        with self.assertRaises(DataError):
            validate_adapter(head, meta)

    def test_configuration_and_missing_metrics_cannot_pass_exit(self):
        from virtual_submeter.accuracy.fixed_queries import load_plan, summarize
        summary = summarize([], self.plan)
        self.assertFalse(summary["development_exit_passed"])
        self.assertTrue(all(group["f1"] is None for group in summary["groups"]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            invalid = copy.deepcopy(self.plan)
            invalid["budgets"] = [5, 10]
            write_json(path, invalid)
            with self.assertRaises(DataError):
                load_plan(path)


@unittest.skipUnless(NUMERIC, "A2 requires the M2 CPU environment")
class ExperimentTest(unittest.TestCase):
    def setUp(self):
        import test_m2_baseline as baseline
        baseline.BaselineTest.setUp(self)
        with contextlib.redirect_stderr(io.StringIO()):
            baseline.BaselineTest.train(self)
        protocol = read_json(self.manifest)
        protocol.update(accuracy_usage="development_only")
        write_json(self.manifest, protocol)
        metadata = read_json(self.model / "model.json")
        metadata.update(protocol_sha256=sha256(self.manifest), state_threshold_view="raw_power_w")
        write_json(self.model / "model.json", metadata)
        self.a1 = self.root / "a1"
        self.a1.mkdir()
        import shutil
        shutil.copytree(self.manifest.parent, self.a1 / "protocol")
        shutil.copytree(self.model, self.a1 / "selected/seed_42")
        write_json(self.a1 / "baseline_freeze.json", {"status": "source_baseline_weights_frozen", "primary_exit_passed": True,
                   "protocol_sha256": sha256(self.manifest), "models": {"42": {"path": "selected/seed_42",
                   "model_manifest_sha256": sha256(self.model / "model.json"), "weights_sha256": metadata["weights"]["sha256"]}}})
        self.plan = read_json(ROOT / "configs/a2_learners.json")
        self.plan.update(source_seeds=[42], query_seeds=[42], budgets=[0, 5], main_budget=5, source_epochs=5, target_epochs=5)
        self.plan_path = self.root / "a2_plan.json"
        write_json(self.plan_path, self.plan)
        self.output = self.root / "a2"

    def test_freeze_barrier_roundtrip_fair_prefix_and_corruption(self):
        from virtual_submeter.accuracy.fixed_queries import freeze, evaluate, verify
        from virtual_submeter.data.protocol import read_partition
        from virtual_submeter.models.artifacts import load_model
        from virtual_submeter.models.prediction import predict_rows, predict
        original = Path.open

        def deny_future(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            # Hash/copy bytes may touch hidden labels; parsed CSV must not.
            if "evaluation_labels" in path.parts and "r" in mode and "b" not in mode:
                raise AssertionError("Future labels opened before A2 freeze")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", deny_future), contextlib.redirect_stderr(io.StringIO()):
            result = freeze(self.a1, self.plan_path, ROOT / "configs/m3_adaptation.json", self.output)
        self.assertEqual(result["cases"], 14)
        self.assertFalse(result["development_future_labels_opened"])
        frozen = self.output / "frozen"
        manifest = frozen / "protocol/protocol.json"
        base_dir = frozen / "bases/seed_42"
        base_meta, base_model = load_model(base_dir)
        aggregates = read_partition(manifest, "frozen_test_house_3", "inference")["aggregate"]
        base_rows = predict_rows(base_meta, base_model, aggregates)
        for variant in self.plan["variants"]:
            zero_dir = frozen / "cases/seed_42/house_3/query_42" / variant / "budget_0/model"
            metadata, model = load_model(zero_dir)
            rows = predict_rows(metadata, model, aggregates)
            if not variant.startswith("L2"):
                self.assertEqual(rows, base_rows)
            five_dir = zero_dir.parent.parent / "budget_5/model"
            metadata, model = load_model(five_dir)
            rows = predict_rows(metadata, model, aggregates)
            with np.load(five_dir.parent / "states.npz", allow_pickle=False) as saved:
                for index, row in enumerate(rows):
                    for column, appliance in enumerate(metadata["appliances"]):
                        value = saved["scores"][index, column]
                        self.assertEqual(row["appliances"][appliance]["state"], bool(value >= 0.5) if np.isfinite(value) else None)
                        self.assertEqual(row["appliances"][appliance]["power_w"], base_rows[index]["appliances"][appliance]["power_w"])
            with self.assertRaises(DataError):
                predict_rows(metadata, model, [{**r, "household_id": "other"} for r in aggregates])
        # Ordinary prediction/export must accept the new score columns.
        predict(five_dir, self.root / "normal_prediction", manifest, "frozen_test_house_3")
        result = evaluate(self.output)
        self.assertEqual(result["metric_rows"], 42)
        with patch.object(Path, "open", deny_future):
            self.assertEqual(verify(self.output)["status"], "a2_verified")
        with self.assertRaises(DataError):
            evaluate(self.output)
        (self.output / "summary.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(DataError):
            verify(self.output)

    def test_model_side_source_and_inference_never_parse_target_labels(self):
        from virtual_submeter.accuracy.fixed_queries import train_source_head
        original = Path.open
        def guard(path, *args, **kwargs):
            if any(p in path.parts for p in ("oracle_labels", "evaluation_labels")):
                raise AssertionError("State-head source training read target labels")
            return original(path, *args, **kwargs)
        directory = self.root / "source_head"
        directory.mkdir()
        with patch.object(Path, "open", guard):
            head, replay = train_source_head(self.model, self.manifest, self.plan, directory)
        self.assertFalse(head["target_labels_read"])
        self.assertTrue(all(len(y) > 0 for _, y in replay.values()))


if __name__ == "__main__":
    unittest.main()
