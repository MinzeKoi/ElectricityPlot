"""M2 numeric correctness, source-only fitting, checkpoint and end-to-end tests."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.config import load_config
from virtual_submeter.data.common import DataError, iso, read_json, write_json
from virtual_submeter.data.prepare import prepare
from virtual_submeter.data.protocol import freeze, read_partition
from virtual_submeter.evaluation.metrics import appliance_metrics

try:
    import numpy as np
    import torch
    NUMERIC = True
except ImportError:
    NUMERIC = False


def tearDownModule():
    # Streamlit 1.50 creates a module-global script directory on import. Explicit
    # cleanup avoids its implicit-cleanup warning under ResourceWarning=error.
    module = sys.modules.get("streamlit.testing.v1.app_test")
    if module is not None:
        directory = module.TMP_DIR
        if Path(directory.name).resolve().parent != Path(tempfile.gettempdir()).resolve():
            raise RuntimeError("Unexpected Streamlit test temporary directory")
        directory.cleanup()


def prediction_rows(values):
    return [{"household_id": "3", "timestamp": iso(i * 8), "appliances": {"kettle": {"power_w": value}}} for i, value in enumerate(values)]


class MetricTest(unittest.TestCase):
    def test_power_state_and_energy_use_common_valid_support(self):
        rows = prediction_rows([10, 30, None, 20])
        labels = [{"household_id": "3", "timestamp": iso(i * 8), "power_w": str(value)} for i, value in enumerate([0, 20, 999, 0])]
        result = appliance_metrics(rows, labels, "kettle", 15, 8)
        self.assertEqual(result["valid_samples"], 3)
        self.assertAlmostEqual(result["mae_w"], 40 / 3)
        self.assertEqual(result["active_mae_w"], 10)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1)
        self.assertAlmostEqual(result["energy"]["absolute_error_kwh"], 40 * 8 / 3600000)
        self.assertEqual(result["evaluation_coverage"], 0.75)
        self.assertNotIn("percentage_error", result["energy"])

    def test_no_valid_pairs_and_undefined_state_metrics_are_null(self):
        labels = [{"household_id": "3", "timestamp": iso(0), "power_w": "0"}]
        result = appliance_metrics(prediction_rows([0]), labels, "kettle", 15, 8)
        self.assertIsNone(result["f1"])
        self.assertIsNone(result["active_mae_w"])
        result = appliance_metrics(prediction_rows([None]), labels, "kettle", 15, 8)
        self.assertIsNone(result["mae_w"])
        self.assertIsNone(result["energy"]["truth_kwh"])
        self.assertEqual(result["valid_samples"], 0)

    def test_duplicates_or_nonfinite_truth_cannot_produce_metrics(self):
        labels = [{"household_id": "3", "timestamp": iso(0), "power_w": "0"}]
        with self.assertRaises(DataError):
            appliance_metrics(prediction_rows([1]), labels * 2, "kettle", 15, 8)
        with self.assertRaises(DataError):
            appliance_metrics(prediction_rows([1]), [{**labels[0], "power_w": "nan"}], "kettle", 15, 8)


@unittest.skipUnless(NUMERIC, "Install M2 CPU dependencies to run training tests")
class BaselineTest(unittest.TestCase):
    def setUp(self):
        from virtual_submeter.models.settings import load_settings
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        self.protocol_dir = self.root / "data/processed/refit_m1_protocol"
        self.manifest = self.protocol_dir / "protocol.json"
        self.settings = load_settings(ROOT / "configs/m2_baseline.json")
        self.settings.update({"window_samples": 7, "channels": 2, "hidden_units": 4, "epochs": 2, "batch_size": 32, "cpu_threads": 1})
        self.settings_path = self.root / "settings.json"
        write_json(self.settings_path, self.settings)
        mapping = read_json(ROOT / "configs/refit_channels.json")["households"]
        inputs = {}
        for house in ("2", "3", "5"):
            source = self.root / f"house_{house}.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["Unix", "Aggregate", *[f"Appliance{i}" for i in range(1, 10)], "Issues"])
                writer.writeheader()
                for index in range(160):
                    on = index % 20 >= 10
                    row = {f"Appliance{i}": 0 for i in range(1, 10)}
                    for appliance, value in (("kettle", 2500 if on else 0), ("microwave", 400 if on else 0), ("washing_machine", 40 if on else 0)):
                        row[mapping[house][appliance][0]] = value
                    row.update({"Unix": index * 8, "Aggregate": 3440 if on else 500, "Issues": 0})
                    writer.writerow(row)
            inputs[house] = source
        prepare(inputs, self.prepared, ROOT / "configs/refit_channels.json")
        plan = read_json(ROOT / "configs/m1_split.json")
        plan["max_model_context_seconds"] = 56
        plan_path = self.root / "plan.json"
        write_json(plan_path, plan)
        config = replace(load_config(ROOT / "configs/default.toml"), split_gap_seconds=56, query_window_seconds=16)
        freeze(self.prepared, plan_path, self.protocol_dir, config)
        self.model = self.root / "model"

    def train(self, kind="small_seq2point", output=None):
        from virtual_submeter.models.training import train
        return train(self.manifest, self.settings_path, output or self.model, kind)

    def predict(self, partition="frozen_test_house_3", output=None):
        from virtual_submeter.models.prediction import predict
        return predict(self.model, output or self.root / "prediction", self.manifest, partition)

    def test_train_reload_predict_evaluate_and_reports(self):
        from virtual_submeter.evaluation.runner import evaluate
        self.train()
        self.predict()
        result = evaluate(self.root / "prediction", self.manifest, self.root / "evaluation")
        self.assertEqual(result["status"], "evaluation_complete")
        model = read_json(self.model / "model.json")
        self.assertEqual(model["training_households"], ["2"])
        self.assertEqual(model["validation_households"], ["5"])
        self.assertFalse(model["target_labels_read"])
        self.assertIsNone(model["confidence"])
        metrics = read_json(self.root / "evaluation/metrics.json")
        self.assertEqual(set(metrics["metrics"]), {"kettle", "microwave", "washing_machine"})
        self.assertGreater(metrics["metrics"]["kettle"]["corrected"]["valid_samples"], 0)
        self.assertTrue((self.root / "prediction/predictions.csv").exists())
        html = (self.root / "evaluation/report.html").read_text(encoding="utf-8")
        self.assertIn("运行片段 MAE", html)
        self.assertNotIn("https://cdn", html)
        self.predict(output=self.root / "reloaded_prediction")
        self.assertEqual(read_json(self.root / "prediction/predictions.json")["rows"], read_json(self.root / "reloaded_prediction/predictions.json")["rows"])

    def test_training_reads_only_source_labels_and_predict_reads_no_labels(self):
        with patch("virtual_submeter.models.training.read_partition", wraps=read_partition) as loader:
            self.train("constant_mean")
        self.assertEqual({call.args[1] for call in loader.call_args_list}, {"source_train_house_2", "source_validation_house_5"})
        with patch("virtual_submeter.models.prediction.read_partition", wraps=read_partition) as loader:
            self.predict()
        self.assertTrue(all(not call.kwargs.get("include_labels", False) for call in loader.call_args_list))

    def test_context_beyond_frozen_gap_is_rejected(self):
        self.settings["window_samples"] = 9
        write_json(self.settings_path, self.settings)
        with self.assertRaisesRegex(DataError, "context"):
            self.train()
        self.assertFalse(self.model.exists())

    def test_window_generation_never_crosses_missing_observations_or_time_gaps(self):
        from virtual_submeter.models.arrays import windows
        aggregate = np.arange(20, dtype=np.float32)
        aggregate[10] = np.nan
        stamps = np.arange(20) * 8
        stamps[16:] += 8
        x, centers = windows(stamps, aggregate, 7, 8)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(all(not 7 <= center <= 13 for center in centers))
        self.assertTrue(all(not 13 <= center <= 18 for center in centers))

    def test_excess_is_scaled_down_and_small_predictions_leave_residual(self):
        from virtual_submeter.models.artifacts import load_model
        from virtual_submeter.models.prediction import predict_rows
        self.train("constant_mean")
        metadata, constants = load_model(self.model)
        data = read_partition(self.manifest, "frozen_test_house_3", "inference")
        rows = predict_rows(metadata, np.array([1000, 1000, 1000]), data["aggregate"])
        valid = [row for row in rows if row["raw_excess_w"] is not None]
        off = next(row for row in valid if row["aggregate_power_w"] == 500)
        self.assertAlmostEqual(sum(item["power_w"] for item in off["appliances"].values()), 500)
        self.assertEqual(off["unexplained_power_w"], 0)
        self.assertEqual(off["raw_excess_w"], 2500)
        on = next(row for row in valid if row["aggregate_power_w"] == 3440)
        self.assertEqual(on["unexplained_power_w"], 440)
        self.assertTrue(all(a["confidence"] is None for a in off["appliances"].values()))
        self.assertIsNone(rows[0]["appliances"]["kettle"]["state"])

    def test_corrupt_checkpoint_or_untrained_model_is_refused(self):
        from virtual_submeter.models.artifacts import load_model
        self.train()
        weights = self.model / "weights.pt"
        weights.write_bytes(weights.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(DataError, "hash"):
            load_model(self.model)
        model = read_json(self.model / "model.json")
        model["status"] = "failed"
        write_json(self.model / "model.json", model)
        with self.assertRaises(DataError):
            load_model(self.model)

    def test_evaluation_rejects_removed_rows_and_adaptation_answers(self):
        from virtual_submeter.evaluation.runner import evaluate
        self.train("zero")
        self.predict()
        path = self.root / "prediction/predictions.json"
        payload = read_json(path)
        payload["rows"] = payload["rows"][1:]
        write_json(path, payload)
        with self.assertRaises(DataError):
            evaluate(path.parent, self.manifest, self.root / "evaluation")
        self.predict(partition="target_adaptation_house_3", output=self.root / "adaptation")
        with self.assertRaises(DataError):
            evaluate(self.root / "adaptation", self.manifest, self.root / "evaluation")

    def test_existing_runs_are_immutable_and_single_device_can_be_trained(self):
        from virtual_submeter.models.training import train
        train(self.manifest, self.settings_path, self.model, "constant_mean", ["kettle"])
        self.assertEqual(read_json(self.model / "model.json")["appliances"], ["kettle"])
        with self.assertRaises(DataError):
            self.train()
        self.predict()
        with self.assertRaises(DataError):
            self.predict()

    def test_invalid_or_too_short_user_input_does_not_publish_success(self):
        from virtual_submeter.models.prediction import predict
        self.train("zero")
        input_path = self.root / "aggregate.csv"
        input_path.write_text("household_id,timestamp,aggregate_power_w\n3,1970-01-01T00:00:00Z,100\n", encoding="utf-8")
        with self.assertRaises(DataError):
            predict(self.model, self.root / "prediction", input_path=input_path)
        self.assertFalse((self.root / "prediction").exists())

    def test_same_seed_reproduces_learned_predictions(self):
        from virtual_submeter.models.artifacts import load_model
        from virtual_submeter.models.prediction import predict_rows
        self.train()
        second = self.root / "second_model"
        self.train(output=second)
        data = read_partition(self.manifest, "frozen_test_house_3", "inference")["aggregate"]
        a, model_a = load_model(self.model)
        b, model_b = load_model(second)
        self.assertEqual(predict_rows(a, model_a, data), predict_rows(b, model_b, data))

    def test_imported_user_aggregate_can_be_predicted_without_any_labels(self):
        from virtual_submeter.models.prediction import predict
        self.train("constant_mean")
        source = self.root / "user.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["household_id", "timestamp", "aggregate_power_w"])
            for i in range(30):
                writer.writerow(["new_home", iso(i * 8), 1000])
        imported = self.root / "imported"
        prepare({"new_home": source}, imported, ROOT / "configs/refit_channels.json", kind="aggregate")
        predict(self.model, self.root / "user_prediction", input_path=imported / "aggregate/house_new_home.csv")
        payload = read_json(self.root / "user_prediction/predictions.json")
        self.assertIsNone(payload["protocol_sha256"])
        self.assertEqual(payload["purpose"], "user_import_no_ground_truth")
        self.assertTrue(all(row["household_id"] == "new_home" for row in payload["rows"]))

    def test_dashboard_predicts_and_clears_results_when_model_becomes_invalid(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install Streamlit to exercise the dashboard")
        self.model = self.root / "outputs/m2-20261020/seq2point"
        self.train("constant_mean")
        script = "from pathlib import Path\nfrom virtual_submeter.app.dashboard import main\nmain(Path(" + repr(str(self.root)) + "))"
        app = AppTest.from_string(script, default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        next(button for button in app.button if button.label == "开始分析").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 1)
        next(button for button in app.button if button.label == "生成分表评价报告").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.dataframe), 0)
        app.text_input[0].set_value("missing_model").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.dataframe), 0)

    def test_dashboard_has_no_fake_results_when_model_is_absent(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install Streamlit to exercise the dashboard")
        script = "from pathlib import Path\nfrom virtual_submeter.app.dashboard import main\nmain(Path(" + repr(str(self.root)) + "))"
        app = AppTest.from_string(script, default_timeout=30).run()
        next(button for button in app.button if button.label == "开始分析").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertGreater(len(app.error), 0)

    def test_dashboard_m3_confirm_budget_adapt_compare_and_invalidate(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install Streamlit to exercise M3 confirmations")
        self.model = self.root / "outputs/m2-20261020/seq2point"
        self.train("constant_mean")
        (self.root / "configs").mkdir()
        (self.root / "configs/m3_adaptation.json").write_bytes((ROOT / "configs/m3_adaptation.json").read_bytes())
        script = "from pathlib import Path\nfrom virtual_submeter.app.dashboard import main\nmain(Path(" + repr(str(self.root)) + "))"
        app = AppTest.from_string(script, default_timeout=30).run()
        next(box for box in app.selectbox if box.label == "分析区间").select("target_adaptation_house_3").run()
        next(button for button in app.button if button.label == "新建确认会话").click().run()
        self.assertEqual(len(app.exception), 0)
        for _ in range(5):
            next(radio for radio in app.radio if radio.label == "你的回答").set_value("skipped")
            next(button for button in app.button if button.label == "提交回答").click().run()
            self.assertEqual(len(app.exception), 0)
        self.assertEqual(next(metric for metric in app.metric if metric.label == "确认次数").value, "5 / 5")
        next(button for button in app.button if button.label == "生成适配模型").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.success), 0)
        next(button for button in app.button if button.label == "比较冻结测试区间的适配前后结果").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("m3_comparison", app.session_state)
        next(box for box in app.selectbox if box.label == "确认预算").select(10).run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("m3_comparison", app.session_state)
        self.assertNotIn("m3_model_directory", app.session_state)


if __name__ == "__main__":
    unittest.main()
