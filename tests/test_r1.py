"""R1 checkpoint objective, independent-network packing and role isolation."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from virtual_submeter.data.common import DataError, read_json, sha256, write_json

try:
    import numpy as np
    import torch
    NUMERIC = True
except ImportError:
    NUMERIC = False


@unittest.skipUnless(NUMERIC, "R1 requires M2 CPU dependencies")
class R1Test(unittest.TestCase):
    def fixture(self):
        from test_m2_baseline import BaselineTest
        fixture = BaselineTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_packed_network_retains_each_device_power_and_temporal_features(self):
        from virtual_submeter.accuracy.r1 import merge_single_models
        from virtual_submeter.models.training import train
        from virtual_submeter.models.artifacts import load_model
        from virtual_submeter.models.prediction import predict_rows
        from virtual_submeter.data.protocol import read_partition
        from virtual_submeter.accuracy.learners import temporal_features
        fixture = self.fixture()
        directories = []
        for appliance in ("kettle", "microwave", "washing_machine"):
            directory = fixture.root / appliance
            train(fixture.manifest, fixture.settings_path, directory, appliances=[appliance])
            directories.append(directory)
        packed = fixture.root / "packed"
        merge_single_models(directories, packed)
        meta, model = load_model(packed)
        inputs = read_partition(fixture.manifest, "frozen_test_house_3", "inference")["aggregate"]
        combined = predict_rows(meta, model, inputs)
        hidden = temporal_features(meta, model, inputs)
        for index, directory in enumerate(directories):
            single_meta, single_model = load_model(directory)
            single = predict_rows(single_meta, single_model, inputs)
            device = single_meta["appliances"][0]
            expected = [r["appliances"][device]["raw_power_w"] for r in single]
            actual = [r["appliances"][device]["raw_power_w"] for r in combined]
            np.testing.assert_allclose(np.array(actual, dtype=float), np.array(expected, dtype=float), rtol=1e-5, atol=1e-5, equal_nan=True)
            width = single_meta["settings"]["hidden_units"]
            np.testing.assert_allclose(hidden[:, index * width:(index + 1) * width], temporal_features(single_meta, single_model, inputs),
                                       rtol=1e-5, atol=1e-6, equal_nan=True)
        changed = read_json(directories[0] / "model.json")
        changed["input_log_mean"] += 1
        write_json(directories[0] / "model.json", changed)
        with self.assertRaises(DataError):
            merge_single_models(directories, fixture.root / "invalid")

    def test_primary_checkpoint_uses_source_f1_then_mae(self):
        from virtual_submeter.models.training import train
        fixture = self.fixture()
        target = fixture.root / "f1"
        train(fixture.manifest, fixture.settings_path, target, checkpoint_metric="primary_f1")
        epochs = read_json(target / "history.json")["epochs"]
        chosen = min(epochs, key=lambda e: (-e["source_primary_f1"], e["validation_normalized_mae"]))
        meta = read_json(target / "model.json")
        self.assertEqual(meta["selected_epoch"], chosen["epoch"])
        self.assertEqual(meta["validation_normalized_mae"], chosen["validation_normalized_mae"])
        self.assertEqual(meta["training_households"], ["2"])
        self.assertFalse(meta["target_labels_read"])
        with self.assertRaises(DataError):
            train(fixture.manifest, fixture.settings_path, fixture.root / "bad", checkpoint_metric="target_f1")

    def test_f1_checkpoint_can_choose_higher_mae_to_retain_events(self):
        from virtual_submeter.models.training import train
        fixture = self.fixture()
        x = np.ones((10, 7), dtype=np.float32)
        y = np.zeros((10, 1), dtype=np.float32)
        y[0] = 2500

        class ControlledNetwork(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.1))
                self.validation_calls = 0

            def forward(self, values):
                if self.training:
                    return self.weight.expand(len(values), 1)
                self.validation_calls += 1
                output = torch.zeros(len(values), 1)
                if self.validation_calls == 1:
                    output[0] = 1999 / 2500
                else:
                    output[:] = 1000 / 2500
                    output[0] = 3000 / 2500
                return output

        for metric, expected_epoch in (("normalized_mae", 1), ("primary_f1", 2)):
            with patch("virtual_submeter.models.training.dataset", return_value=(x, y, {"source": 10})), \
                    patch("virtual_submeter.models.network.SmallSeq2Point", ControlledNetwork):
                target = fixture.root / metric
                train(fixture.manifest, fixture.settings_path, target, appliances=["kettle"], checkpoint_metric=metric)
            self.assertEqual(read_json(target / "model.json")["selected_epoch"], expected_epoch)

    def test_r1_rejects_final_house_roles_before_data_access(self):
        from virtual_submeter.accuracy.r1 import prepare
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = read_json(ROOT / "configs/r1_baseline_20261022.json")
            plan["development_households"] = ["8"]
            write_json(root / "plan.json", plan)
            with self.assertRaisesRegex(DataError, "frozen R0 roles"):
                prepare(root / "missing", root / "missing2", root / "plan.json", ROOT / "configs/m2_baseline.json", root / "run")
            self.assertFalse((root / "run").exists())

    def test_candidate_cache_preserves_hash_checks_and_hides_label_metadata(self):
        from virtual_submeter.adaptation.session import session_candidates, _candidate_cache
        _candidate_cache.cache_clear()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "prediction/predictions.json"
            write_json(path, {"test": "immutable_visible_prediction"})
            partition = {"role": "target_adaptation", "household_id": "17", "partition_id": "target_adaptation_house_17",
                         "start": "2014-01-01T00:00:00Z", "end": "2014-01-02T00:00:00Z", "artifacts": {"label": "hidden_hash_a"}}
            protocol = {"partitions": [partition], "sample_interval_seconds": 8, "query_window_seconds": 1200}
            session = {"prediction_sha256": sha256(path), "partition_id": partition["partition_id"], "settings": {}}
            with patch("virtual_submeter.adaptation.session.candidates", return_value=[{"query_id": "original"}]) as build:
                first = session_candidates(root, session, protocol)
                first[0]["query_id"] = "caller_mutated"
                partition["artifacts"]["label"] = "hidden_hash_b"
                self.assertEqual(session_candidates(root, session, protocol)[0]["query_id"], "original")
                self.assertEqual(build.call_count, 1)
                self.assertNotIn("artifacts", build.call_args.args[1])
                write_json(path, {"test": "corrupted"})
                with self.assertRaises(DataError):
                    session_candidates(root, session, protocol)
                session["prediction_sha256"] = sha256(path)
                session_candidates(root, session, protocol)
                self.assertEqual(build.call_count, 2)
        _candidate_cache.cache_clear()

    def test_fixed_report_uses_actual_development_household(self):
        from virtual_submeter.accuracy.fixed_queries import render_report
        summary = {"groups": [{"variant": "L1", "budget": 10, "f1": 0.2, "f1_delta": 0.01,
                               "adaptation_degenerate_cases": 0, "households": [{"household_id": "17"}]}],
                   "main_budget": 10, "development_exit_passed": True, "internal_gain_target_met": False,
                   "selected_development_learner": "L1"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            render_report(root, summary, [])
            page = (root / "report.html").read_text(encoding="utf-8")
            self.assertIn("House 17", page)
            self.assertNotIn("House 3", page)


if __name__ == "__main__":
    unittest.main()
