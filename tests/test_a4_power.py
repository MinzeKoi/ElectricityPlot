"""A4 weak-label boundary, coupled inference, paired freeze and calibration identity."""

import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.data.common import DataError, read_json, write_json

try:
    import numpy as np
    import torch
    NUMERIC = True
except ImportError:
    NUMERIC = False


@unittest.skipUnless(NUMERIC, "A4 requires the M2 CPU environment")
class PowerTest(unittest.TestCase):
    def test_missing_metrics_never_pass_and_numeric_config_is_strict(self):
        import tempfile
        from virtual_submeter.accuracy.power import load_plan
        from virtual_submeter.accuracy.power_experiment import summarize
        plan = load_plan(ROOT / "configs/a4_power.json")
        summary = summarize([], plan, read_json(ROOT / "configs/a2_learners.json"))
        self.assertFalse(summary["h3_development_target_met"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            for key, value in (("ridge", True), ("upper_quantile", 1), ("minimum_active_samples", 2.5)):
                write_json(path, {**plan, key: value})
                with self.assertRaises(DataError):
                    load_plan(path)

    def test_source_branch_and_weak_scores_do_not_fill_positive_window(self):
        from virtual_submeter.accuracy.power import fit_branch, conditional_power
        normalizer = {"mean": [0.0], "std": [1.0]}
        metadata = {"appliances": ["kettle"], "settings": {"hidden_units": 1, "on_thresholds_w": {"kettle": 100}},
                    "accuracy_state_head": {"variant": "L2_max", "normalizer": normalizer}}
        x = {"kettle": np.column_stack((np.ones(12), np.zeros(12), np.linspace(0, 1, 12)))}
        head = fit_branch(x, np.full((12, 1), 1000.0), metadata, normalizer, read_json(ROOT / "configs/a4_power.json"))
        amplitude = conditional_power(head, ["kettle"], np.array([[0.0], [0.0], [0.0]]))
        raw = amplitude[:, 0] * np.array([0.001, 0.002, 0.9])
        np.testing.assert_allclose(raw, [1, 2, 900], atol=1e-8)
        self.assertFalse(head["target_power_labels_read"])


@unittest.skipUnless(NUMERIC, "A4 requires the M2 CPU environment")
class ExperimentTest(unittest.TestCase):
    def setUp(self):
        from test_a2_learners import ExperimentTest as A2Test
        A2Test.setUp(self)
        self.plan.update(variants=["L2_max"])
        write_json(self.plan_path, self.plan)

    def test_freeze_without_target_labels_pairing_calibration_roundtrip_and_tamper(self):
        from virtual_submeter.accuracy import fixed_queries as fixed, dynamic_queries as dynamic
        from virtual_submeter.accuracy import power_experiment as power
        from virtual_submeter.models.artifacts import load_model
        from virtual_submeter.models.prediction import predict, predict_rows
        from virtual_submeter.data.protocol import read_partition
        from virtual_submeter.trust.calibration import load_calibration, power_fingerprint
        original_summary = fixed.summarize
        def passing(*args):
            result = original_summary(*args)
            result.update(development_exit_passed=True, selected_development_learner="L2_max")
            return result
        a3 = self.root / "a3"
        query_plan = read_json(ROOT / "configs/a3_queries.json")
        query_plan["strategies"] = ["random", "uncertainty", "proposed"]
        query_path = self.root / "query.json"
        write_json(query_path, query_plan)
        with patch.object(fixed, "summarize", passing), contextlib.redirect_stderr(io.StringIO()):
            fixed.freeze(self.a1, self.plan_path, ROOT / "configs/m3_adaptation.json", self.output)
            fixed.evaluate(self.output)
            dynamic.freeze(self.output, query_path, a3)
            dynamic.evaluate(a3)
        original_open = Path.open
        def deny_target_labels(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if any(p in path.parts for p in ("oracle_labels", "evaluation_labels")) and "r" in mode and "b" not in mode:
                raise AssertionError("A4 parsed target labels before freeze")
            return original_open(path, *args, **kwargs)
        a4 = self.root / "a4"
        plan = read_json(ROOT / "configs/a4_power.json")
        plan.update(minimum_active_samples=2, minimum_calibration_samples=2)
        path = self.root / "power.json"
        write_json(path, plan)
        with patch.object(Path, "open", deny_target_labels), contextlib.redirect_stderr(io.StringIO()):
            result = power.freeze(a3, path, a4)
        self.assertEqual(result["cases"], 12)
        frozen = a4 / "frozen"
        cases = read_json(a4 / "run.json")["cases"]
        coupled = next(c for c in cases if c["mode"] == "coupled" and c["budget"] == 5)
        control = next(c for c in cases if c["mode"] == "state_only" and c["budget"] == 5 and c["variant"] == coupled["variant"])
        metadata, model = load_model(frozen / coupled["path"] / "model")
        control_meta, _ = load_model(frozen / control["path"] / "model")
        self.assertEqual(metadata["weights"], control_meta["weights"])
        self.assertNotEqual(power_fingerprint(metadata), power_fingerprint(control_meta))
        with self.assertRaises(DataError):
            load_calibration(frozen / coupled["path"] / "calibration", control_meta)
        changed = copy.deepcopy(metadata)
        changed["accuracy_state_head"]["coefficients"]["kettle"][0] += 0.01
        with self.assertRaises(DataError):
            load_calibration(frozen / coupled["path"] / "calibration", changed)
        aggregates = read_partition(frozen / "protocol/protocol.json", coupled["future_partition"], "inference")["aggregate"]
        rows = predict_rows(metadata, model, aggregates)
        payload = read_json(frozen / coupled["base_prediction"])
        materialized = power.materialize(frozen, coupled, payload)
        for a in metadata["appliances"]:
            np.testing.assert_allclose([r["appliances"][a]["raw_power_w"] or np.nan for r in rows],
                                       [r["appliances"][a]["raw_power_w"] or np.nan for r in materialized], rtol=1e-6, equal_nan=True)
        predict(frozen / coupled["path"] / "model", self.root / "ordinary", frozen / "protocol/protocol.json", coupled["future_partition"],
                calibration_dir=frozen / coupled["path"] / "calibration")
        self.assertIn("kettle_state_score", (self.root / "ordinary/predictions.csv").read_text(encoding="utf-8").splitlines()[0])
        html = (self.root / "ordinary/report.html").read_text(encoding="utf-8")
        self.assertIn("A4 功率联动", html)
        self.assertNotIn("本模型未经目标家庭适配", html)
        self.assertNotIn("M2 未发起用户确认", html)
        with patch.object(power, "source_fingerprint", return_value="changed"):
            with self.assertRaises(DataError):
                power.evaluate(a4)
        self.assertFalse(read_json(a4 / "run.json")["development_future_labels_opened"])
        result = power.evaluate(a4)
        self.assertEqual(result["metric_rows"], 36)
        with patch.object(Path, "open", deny_target_labels):
            self.assertEqual(power.verify(a4)["status"], "a4_verified")
        with self.assertRaises(DataError):
            power.evaluate(a4)
        (a4 / "comparison.csv").write_text("bad", encoding="utf-8")
        with self.assertRaises(DataError):
            power.verify(a4)


if __name__ == "__main__":
    unittest.main()
