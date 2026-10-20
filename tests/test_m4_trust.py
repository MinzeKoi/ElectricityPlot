"""M4 leakage barriers, interval semantics, paired stress and failed-run recovery."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.common import DataError, iso, read_json, sha256, write_json
from virtual_submeter.data.protocol import read_partition
from virtual_submeter.trust.output import reconcile, validate_output
from virtual_submeter.trust.calibration import apply_intervals
from virtual_submeter.evaluation.metrics import attribution_metrics, interval_metrics
import test_m2_baseline as baseline

try:
    import numpy
    NUMERIC = True
except ImportError:
    NUMERIC = False


class TrustMathTest(unittest.TestCase):
    def test_consistency_never_scales_up_or_explains_away_residual(self):
        self.assertEqual(reconcile(1000, [100, 200]), ([100, 200], 1))
        self.assertEqual(reconcile(100, [100, 100]), ([50, 50], 0.5))
        self.assertEqual(reconcile(0, [10, 0]), ([0, 0], 0))
        for total, values in ((-1, [0]), (10, [-1]), (10, [float("nan")])):
            with self.assertRaises(DataError):
                reconcile(total, values)

    def test_interval_coverage_and_attribution_use_common_valid_support(self):
        rows = [{"household_id": "3", "timestamp": iso(i * 8), "aggregate_power_w": 100,
                 "raw_excess_w": 0 if i < 2 else None, "unexplained_power_w": 100 if i < 2 else None,
                 "quality_flags": [], "appliances": {"kettle": {"raw_power_w": 0 if i < 2 else None, "power_w": 0 if i < 2 else None}}} for i in range(3)]
        cal = {"source_aggregate_range_w": [0, 100], "quantiles": {"kettle": {"raw": {"radius_w": 10}, "corrected": {"radius_w": 10}}}}
        apply_intervals(rows, cal)
        labels = [{"household_id": "3", "timestamp": iso(i * 8), "power_w": str(v)} for i, v in enumerate((5, 20, 999))]
        metrics = interval_metrics(rows, labels, "kettle")
        self.assertEqual(metrics["raw"]["empirical_coverage"], 0.5)
        self.assertEqual(metrics["corrected"]["mean_width_w"], 10)
        self.assertEqual(metrics["raw"]["valid_samples"], 2)
        attribution = attribution_metrics(rows, {"kettle": labels}, ["kettle"], 8)
        self.assertEqual(attribution["corrected"]["attributed_energy_fraction"], 0)
        self.assertEqual(attribution["all_unexplained_control"]["mean_appliance_mae_w"], 12.5)
        self.assertGreater(attribution["all_unexplained_control"]["missed_known_energy_kwh"], 0)
        self.assertIsNone(rows[2]["appliances"]["kettle"]["lower_w"])


@unittest.skipUnless(NUMERIC, "M4 prediction requires NumPy")
class TrustPipelineTest(unittest.TestCase):
    def setUp(self):
        baseline.BaselineTest.setUp(self)
        baseline.BaselineTest.train(self, "constant_mean")
        self.calibration = self.root / "calibration"
        self.trust_settings = ROOT / "configs/m4_trust.json"

    def calibrate(self):
        from virtual_submeter.trust.calibration import calibrate
        return calibrate(self.model, self.manifest, self.trust_settings, self.calibration)

    def predict(self):
        from virtual_submeter.models.prediction import predict
        return predict(self.model, self.root / "prediction", self.manifest, "frozen_test_house_3", calibration_dir=self.calibration)

    def test_calibration_reads_only_source_validation_and_predict_reads_no_labels(self):
        from virtual_submeter.trust.calibration import load_calibration
        from virtual_submeter.models.artifacts import load_model
        original = Path.open
        def guarded(path, *args, **kwargs):
            if any(p in ("oracle_labels", "evaluation_labels", "source_train_house_2") for p in path.parts):
                raise AssertionError("Calibration/inference opened forbidden labels")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded), patch("virtual_submeter.trust.calibration.read_partition", wraps=read_partition) as reader:
            self.calibrate()
            self.predict()
        self.assertEqual({call.args[1] for call in reader.call_args_list}, {"source_validation_house_5"})
        cal = load_calibration(self.calibration, load_model(self.model)[0])
        self.assertFalse(cal["target_labels_read"])
        self.assertFalse(cal["coverage_guarantee"])
        payload = read_json(self.root / "prediction/predictions.json")
        validate_output(payload)
        valid = [r for r in payload["rows"] if r["raw_excess_w"] is not None]
        self.assertTrue(valid)
        self.assertTrue(all(v["confidence"] is None for r in valid for v in r["appliances"].values()))
        self.assertTrue(all(v["lower_w"] <= v["power_w"] <= v["upper_w"] for r in valid for v in r["appliances"].values()))
        self.assertIn("kettle_raw_lower_w", (self.root / "prediction/predictions.csv").read_text(encoding="utf-8"))

    def test_calibration_consumer_cannot_read_target_or_training_labels(self):
        for partition in ("source_train_house_2", "target_adaptation_house_3", "frozen_test_house_3"):
            with self.assertRaises(DataError):
                read_partition(self.manifest, partition, "calibration", True)

    def test_calibration_rejects_changed_power_model_and_tampered_provenance(self):
        from virtual_submeter.trust.calibration import load_calibration
        from virtual_submeter.models.artifacts import load_model
        self.calibrate()
        metadata, _ = load_model(self.model)
        changed = copy.deepcopy(metadata)
        changed["constant_power_w"][0] += 1
        with self.assertRaises(DataError):
            load_calibration(self.calibration, changed)
        changed = copy.deepcopy(metadata)
        changed["adaptation"] = "weak_window_state_only"
        changed["state_adapter"] = {"arbitrary": "ignored by power fingerprint"}
        self.assertEqual(load_calibration(self.calibration, changed)["status"], "calibrated")
        (self.calibration / "settings.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(DataError):
            load_calibration(self.calibration, metadata)
        with self.assertRaises(DataError):
            self.predict()
        self.assertFalse((self.root / "prediction").exists())

    def test_insufficient_calibration_and_invalid_settings_publish_nothing(self):
        from virtual_submeter.trust.calibration import calibrate, load_settings
        settings = read_json(self.trust_settings)
        path = self.root / "m4.json"
        settings["minimum_samples"] = 100000
        write_json(path, settings)
        with self.assertRaises(DataError):
            calibrate(self.model, self.manifest, path, self.calibration)
        self.assertFalse(self.calibration.exists())
        for level in (0, 1, True):
            settings["interval_level"] = level
            write_json(path, settings)
            with self.assertRaises(DataError):
                load_settings(path)

    def test_recovery_rejects_corrupt_prediction_and_changed_calibration(self):
        from virtual_submeter.app.results import load_result
        self.calibrate()
        self.predict()
        directory = self.root / "prediction"
        valid = load_result(directory)
        path = directory / "predictions.json"
        corrupted = copy.deepcopy(valid)
        row = next(r for r in corrupted["rows"] if r["raw_excess_w"] is not None)
        row["appliances"]["kettle"]["power_w"] += 10
        write_json(path, corrupted)
        with self.assertRaises(DataError):
            load_result(directory)
        write_json(path, valid)
        (directory / "calibration.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(DataError):
            load_result(directory)

    def test_recovery_binds_downloads_and_metadata_to_prediction(self):
        from virtual_submeter.app.results import load_result
        self.calibrate()
        self.predict()
        directory = self.root / "prediction"
        for name in ("predictions.csv", "report.html", "run.json"):
            path = directory / name
            original = path.read_bytes()
            if name == "run.json":
                record = read_json(path)
                record["model_id"] = "stale-model"
                write_json(path, record)
            else:
                path.write_text("corrupt export", encoding="utf-8")
            with self.assertRaises(DataError):
                load_result(directory)
            path.write_bytes(original)
        self.assertEqual(load_result(directory)["status"], "complete")

    def test_calibration_rejects_inconsistent_level_and_sampling_interval(self):
        from virtual_submeter.trust.calibration import load_calibration
        from virtual_submeter.models.artifacts import load_model
        self.calibrate()
        metadata, _ = load_model(self.model)
        path = self.calibration / "calibration.json"
        original = read_json(path)
        for field, value in (("interval_level", 0.8), ("sample_interval_seconds", 16)):
            write_json(path, {**original, field: value})
            with self.assertRaises(DataError):
                load_calibration(self.calibration, metadata)

    def test_unavailable_states_and_uncalibrated_intervals_cannot_be_fabricated(self):
        self.calibrate()
        self.predict()
        original = read_json(self.root / "prediction/predictions.json")
        for field, value in (("sample_interval_seconds", 0), ("calibration", None)):
            payload = copy.deepcopy(original)
            payload[field] = value
            with self.assertRaises(DataError):
                validate_output(payload)
        payload = copy.deepcopy(original)
        missing = next(r for r in payload["rows"] if r["raw_excess_w"] is None)
        missing["appliances"]["kettle"]["state_score"] = 0
        with self.assertRaises(DataError):
            validate_output(payload)

    def test_evaluation_reports_intervals_attribution_and_all_unexplained_control(self):
        from virtual_submeter.evaluation.runner import evaluate
        self.calibrate()
        self.predict()
        evaluate(self.root / "prediction", self.manifest, self.root / "evaluation")
        evaluation = read_json(self.root / "evaluation/metrics.json")
        self.assertGreater(evaluation["intervals"]["kettle"]["corrected"]["valid_samples"], 0)
        self.assertGreater(evaluation["attribution"]["all_unexplained_control"]["missed_known_energy_kwh"], 0)
        self.assertEqual(evaluation["consistency"]["corrected_exceedance_rate"], 0)
        html = (self.root / "evaluation/report.html").read_text(encoding="utf-8")
        for text in ("归因比例与误差", "全部留作未解释对照", "平均区间宽度", "数据质量标记"):
            self.assertIn(text, html)

    def test_synthetic_provenance_reaches_predictions_evaluation_and_report(self):
        from virtual_submeter.evaluation.runner import evaluate
        from virtual_submeter.models.training import train
        protocol = read_json(self.manifest)
        protocol["data_kind"] = "synthetic_contract_fixture"
        write_json(self.manifest, protocol)
        self.model = self.root / "synthetic_model"
        train(self.manifest, self.settings_path, self.model, "constant_mean")
        self.calibrate()
        self.predict()
        payload = read_json(self.root / "prediction/predictions.json")
        self.assertEqual(payload["data_kind"], "synthetic_contract_fixture")
        self.assertEqual(payload["model_data_kind"], "synthetic_contract_fixture")
        evaluate(self.root / "prediction", self.manifest, self.root / "evaluation")
        evaluation = read_json(self.root / "evaluation/metrics.json")
        self.assertIn("Synthetic contract fixture", evaluation["notice"])
        html = (self.root / "evaluation/report.html").read_text(encoding="utf-8")
        self.assertIn("不代表 REFIT 实测效果", html)
        self.assertNotIn("当前真实文件前缀", html)

    def test_background_predictions_are_frozen_before_evaluation_labels_open(self):
        from virtual_submeter.evaluation.stress import stress_test
        from virtual_submeter.evaluation.runner import evaluate
        self.calibrate()
        def guarded(manifest, partition, consumer, include_labels=False):
            if include_labels:
                markers = list(self.root.glob(".stress-*/run/frozen_predictions.json"))
                self.assertEqual(len(markers), 1)
                record = read_json(markers[0])
                self.assertTrue(record["predictions_frozen_before_test_labels"])
                self.assertEqual(record["injected_sha256"], sha256(markers[0].parent / "injected/predictions.json"))
            return read_partition(manifest, partition, consumer, include_labels)
        with patch("virtual_submeter.evaluation.stress.read_partition", side_effect=guarded):
            stress_test(self.model, self.manifest, "frozen_test_house_3", self.trust_settings, self.root / "stress", self.calibration)
        stress = read_json(self.root / "stress/stress.json")
        self.assertEqual(set(stress["paired"]["groups"]), {"background_present", "background_absent"})
        self.assertGreater(stress["paired"]["injected_energy_kwh_on_common_support"], 0)
        self.assertGreater(stress["paired"]["incremental_known_assignment_fraction"], 0)
        self.assertTrue((self.root / "stress/report.html").exists())
        with self.assertRaises(DataError):
            evaluate(self.root / "stress/injected", self.manifest, self.root / "invalid_evaluation")
        with self.assertRaises(DataError):
            stress_test(self.model, self.manifest, "frozen_test_house_3", self.trust_settings, self.root / "stress", self.calibration)

    def test_stress_failure_publishes_no_complete_run(self):
        from virtual_submeter.evaluation.stress import stress_test
        self.calibrate()
        with patch("virtual_submeter.evaluation.stress.predict", side_effect=DataError("inference failed")):
            with self.assertRaises(DataError):
                stress_test(self.model, self.manifest, "frozen_test_house_3", self.trust_settings, self.root / "failed_stress", self.calibration)
        self.assertFalse((self.root / "failed_stress").exists())

    def test_verification_reopens_no_labels_and_detects_changed_csv(self):
        from virtual_submeter.evaluation.stress import stress_test
        from virtual_submeter.trust.verification import verify
        self.calibrate()
        # Exact config bytes, including nonstandard formatting, are preserved.
        settings = self.root / "compact.json"
        settings.write_text(json.dumps(read_json(self.trust_settings)), encoding="utf-8")
        output = self.root / "stress"
        stress_test(self.model, self.manifest, "frozen_test_house_3", settings, output, self.calibration)
        original = Path.open
        def guarded(path, *args, **kwargs):
            if any(p in ("oracle_labels", "evaluation_labels", "source_labels") for p in path.parts):
                raise AssertionError("Verifier opened labels")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            result = verify(output)
        self.assertEqual(result["status"], "m4_artifacts_verified")
        self.assertFalse(result["test_labels_reopened"])
        (output / "injected/predictions.csv").write_text("corrupt", encoding="utf-8")
        with self.assertRaises(DataError):
            verify(output)

    def test_cli_exposes_m4_and_reports_invalid_calibration_without_success(self):
        from virtual_submeter.cli import main
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["calibrate-output", "--model", str(self.model), "--manifest", str(self.manifest), "--trust-config", str(self.trust_settings), "--output", str(self.calibration), "--json"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "calibration_complete")

    def test_dashboard_displays_calibrated_results_and_clears_corrupt_exports(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install Streamlit to exercise M4 result recovery")
        self.calibrate()
        script = "from pathlib import Path\nfrom virtual_submeter.app.dashboard import main\nmain(Path(" + repr(str(self.root)) + "))"
        script_path = self.root / "app_test.py"
        script_path.write_text(script, encoding="utf-8")
        app = AppTest.from_file(script_path, default_timeout=30).run()
        app.text_input[0].set_value(str(self.model)).run()
        app.text_input[1].set_value(str(self.calibration)).run()
        next(button for button in app.button if button.label == "开始分析").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 1)
        self.assertTrue(any("没有跨家庭覆盖保证" in caption.value for caption in app.caption))
        import pandas as pd
        from virtual_submeter.app.charts import history_chart
        chart = history_chart(pd.DataFrame({"时间 UTC": pd.to_datetime([iso(0), iso(8)], utc=True), "估计 W": [10, None]})).to_dict()
        self.assertEqual(chart["encoding"]["x"]["scale"]["type"], "utc")
        self.assertEqual(chart["mark"]["invalid"], "break-paths-show-domains")
        directory = Path(app.session_state["prediction_directory"])
        (directory / "predictions.csv").write_text("corrupt", encoding="utf-8")
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertNotIn("prediction_directory", app.session_state)
        self.assertTrue(any("结果不可用" in error.value for error in app.error))
        next(button for button in app.button if button.label == "开始分析").click().run()
        self.assertEqual(len(app.metric), 1)
        (self.calibration / "settings.json").write_text("{}", encoding="utf-8")
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertNotIn("prediction_directory", app.session_state)


if __name__ == "__main__":
    unittest.main()
