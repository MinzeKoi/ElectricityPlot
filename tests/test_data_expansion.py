"""Data expansion boundaries, matched optimization and native power integration."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import torch

from virtual_submeter.accuracy.data_expansion import admission, compact_metrics, load_plan, partition
from virtual_submeter.data.common import DataError, iso, read_json, write_json
from virtual_submeter.data.ukdale import native_to_grid, timestamp_us, find_channels


class ExpansionTest(unittest.TestCase):
    def fixture(self):
        from test_m2_baseline import BaselineTest
        fixture = BaselineTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_plan_cannot_reassign_reserved_or_validation_homes(self):
        plan = load_plan(ROOT / "configs/data_expansion_20261024.json")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            for key, value in (("source_train", ["2", "8"]), ("additional_sources", ["5"]),
                               ("development", ["20"]), ("ukdale_train", ["1", "5"])):
                write_json(path, {**plan, key: value})
                with self.assertRaises(DataError):
                    load_plan(path)

    def test_reserved_partition_rejected_before_any_file_access(self):
        with patch("virtual_submeter.accuracy.data_expansion.read_json", side_effect=AssertionError("data opened")):
            with self.assertRaises(DataError):
                partition(Path("absent"), "8", "source_train", 0, 8, "bad", Path("absent"), {}, {})

    def test_missing_label_is_not_zero_and_training_roles_remain_enforced(self):
        from virtual_submeter.models.training import dataset
        fixture = self.fixture()
        protocol = read_json(fixture.manifest)
        part = next(p for p in protocol["partitions"] if p["role"] == "source_train")
        import copy
        second = copy.deepcopy(part)
        second["partition_id"] += "_second"
        protocol["partitions"].append(second)
        del part["artifacts"]["microwave"]
        protocol["partial_source_labels"] = True
        write_json(fixture.manifest, protocol)
        _, y, _ = dataset(fixture.manifest, [part, second], ["kettle", "microwave"], fixture.settings, 8)
        self.assertTrue(np.isnan(y[:len(y) // 2, 1]).all())
        self.assertTrue(np.isfinite(y[len(y) // 2:, 1]).all())
        self.assertTrue(np.isfinite(y[:, 0]).all())
        forbidden = next(p for p in protocol["partitions"] if p["role"] == "frozen_test")
        with self.assertRaises(DataError):
            dataset(fixture.manifest, [forbidden], ["kettle"], fixture.settings, 8)

    def test_matched_steps_reproduce_legacy_and_do_not_grow_with_dataset(self):
        from virtual_submeter.models.training import dataset, train
        from virtual_submeter.models.artifacts import load_model
        fixture = self.fixture()
        protocol = read_json(fixture.manifest)
        train_parts = [p for p in protocol["partitions"] if p["role"] == "source_train"]
        x, _, _ = dataset(fixture.manifest, train_parts, ["kettle"], fixture.settings, 8)
        legacy = fixture.root / "legacy"
        train(fixture.manifest, fixture.settings_path, legacy, appliances=["kettle"])
        steps = (len(x) + fixture.settings["batch_size"] - 1) // fixture.settings["batch_size"]
        protocol["expansion_training"] = {"input_log_mean": float(np.log1p(x).mean()), "input_log_std": float(np.log1p(x).std()), "steps_per_checkpoint": steps}
        write_json(fixture.manifest, protocol)
        matched = fixture.root / "matched"
        train(fixture.manifest, fixture.settings_path, matched, appliances=["kettle"])
        meta, model = load_model(matched)
        _, old = load_model(legacy)
        self.assertEqual(meta["optimizer_steps"], steps * fixture.settings["epochs"])
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        with patch("virtual_submeter.models.training.dataset", return_value=(np.tile(x, (3, 1)), np.ones((len(x) * 3, 1), np.float32), {"source": len(x) * 3})):
            doubled = fixture.root / "more_data"
            train(fixture.manifest, fixture.settings_path, doubled, appliances=["kettle"])
        self.assertEqual(read_json(doubled / "model.json")["optimizer_steps"], meta["optimizer_steps"])

    def test_external_support_audit_does_not_compute_activity(self):
        rows = [{"timestamp": iso(i * 8), "power_w": str(i % 2 * 3000)} for i in range(8)]
        agg = [{"aggregate_power_w": "4000"} for _ in rows]
        plan = {"minimum_coverage": .9, "minimum_observed_days": 0, "minimum_complete_events": 2}
        with patch("virtual_submeter.accuracy.data_expansion.event_support", side_effect=AssertionError("activity inspected")):
            stats = admission(rows, agg, 2000, plan, audit_events=False)
        self.assertIsNone(stats["complete_threshold_events"])
        self.assertTrue(stats["included"])

    def test_compact_metrics_agree_with_production_metrics(self):
        from virtual_submeter.evaluation.metrics import appliance_metrics
        p = np.array([0., 2500., 2600., np.nan, 0., 2300.])
        y = np.array([0., 2400., 0., 2400., np.nan, 2500.])
        rows = [{"household_id": "x", "timestamp": iso(i * 8), "appliances": {"kettle": {
            "raw_power_w": float(v) if np.isfinite(v) else None, "raw_state": bool(v >= 2000) if np.isfinite(v) else None}}} for i, v in enumerate(p)]
        labels = [{"household_id": "x", "timestamp": iso(i * 8), "power_w": str(v) if np.isfinite(v) else ""} for i, v in enumerate(y)]
        actual = compact_metrics(p, y, 2000)
        expected = appliance_metrics(rows, labels, "kettle", 2000, 8, "raw_power_w")
        for key in ("valid_samples", "precision", "recall", "f1", "mae_w", "active_mae_w"):
            self.assertAlmostEqual(actual[key], expected[key])
        self.assertAlmostEqual(actual["energy_absolute_error_kwh"], expected["energy"]["absolute_error_kwh"])

    def test_native_support_preserves_fractional_time_energy_and_gaps(self):
        stamps = timestamp_us(["0.0", "6.0", "12.0", "18.0", "24.0"])
        values, coverage = native_to_grid(stamps, [0, 800, 1600, 0, 0], 6, 0, 24)
        np.testing.assert_allclose(values, [200, 1200, 400])
        np.testing.assert_array_equal(coverage, [1, 1, 1])
        values, coverage = native_to_grid(timestamp_us(["0.5", "6.5", "20.0", "24.0"]), [100, 100, 100, 100], 6, 0, 24)
        self.assertTrue(np.isnan(values).all())
        self.assertTrue((coverage < 1).all())
        np.testing.assert_array_equal(timestamp_us(["1382000000.123456"]), [1382000000123456])
        with self.assertRaises(DataError):
            native_to_grid([0, 0], [1, 1], 1, 0, 8)

    def test_mixed_and_washer_dryer_labels_are_not_independent_devices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "labels.dat").write_text("1 aggregate\n2 kettle_radio\n3 washer_dryer\n4 washing_machine_microwave_breadmaker\n5 microwave\n", encoding="utf-8")
            self.assertEqual(set(find_channels(root)), {"aggregate", "microwave"})

    def test_partial_components_pack_and_compact_predictions_match_ordinary_inference(self):
        import copy
        from virtual_submeter.accuracy.data_expansion import predict_compact
        from virtual_submeter.accuracy.r1 import merge_single_models
        from virtual_submeter.data.protocol import read_partition
        from virtual_submeter.models.training import train, dataset
        from virtual_submeter.models.artifacts import load_model
        from virtual_submeter.models.prediction import predict_rows
        fixture = self.fixture()
        protocol = read_json(fixture.manifest)
        source = next(p for p in protocol["partitions"] if p["role"] == "source_train")
        extra = copy.deepcopy(source)
        extra["partition_id"] += "_partial"
        del extra["artifacts"]["kettle"]
        protocol["partitions"].append(extra)
        protocol["partial_source_labels"] = True
        write_json(fixture.manifest, protocol)
        x, _, _ = dataset(fixture.manifest, [source, extra], ["kettle", "microwave", "washing_machine"], fixture.settings, 8)
        protocol["expansion_training"] = {"input_log_mean": float(np.log1p(x).mean()), "input_log_std": float(np.log1p(x).std()), "steps_per_checkpoint": 2}
        write_json(fixture.manifest, protocol)
        components = []
        for device in ("kettle", "microwave", "washing_machine"):
            path = fixture.root / device
            train(fixture.manifest, fixture.settings_path, path, appliances=[device])
            components.append(path)
        packed = fixture.root / "packed"
        merge_single_models(components, packed)
        meta, model = load_model(packed)
        self.assertEqual(meta["source_train_windows"]["kettle"][extra["partition_id"]], 0)
        self.assertGreater(meta["source_train_windows"]["microwave"][extra["partition_id"]], 0)
        pid = next(p["partition_id"] for p in protocol["partitions"] if p["role"] == "frozen_test")
        data = read_partition(fixture.manifest, pid, "inference")
        ordinary = predict_rows(meta, model, data["aggregate"])
        output = fixture.root / "compact.npz"
        predict_compact(packed, fixture.manifest, pid, output)
        with np.load(output) as arrays:
            for i, device in enumerate(meta["appliances"]):
                for compact, key in (("raw", "raw_power_w"), ("corrected", "power_w")):
                    expected = np.array([r["appliances"][device][key] for r in ordinary], dtype=float)
                    np.testing.assert_allclose(arrays[compact][:, i], expected, rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_gain_gate_rejects_a_sacrificed_primary_device_and_support_changes(self):
        from virtual_submeter.accuracy.data_expansion import paired_gains
        plan = load_plan(ROOT / "configs/data_expansion_20261024.json")
        metrics = []
        for group in ("A", "B"):
            for seed in plan["seeds"]:
                for device in ("kettle", "microwave", "washing_machine"):
                    delta = 0 if group == "A" else -.03 if device == "kettle" else .09
                    metrics.append({"group": group, "seed": seed, "appliance": device, "view": "raw",
                                    "partition": "frozen_test_refit_17_0", "support_sha256": "same",
                                    "state_pattern": "mixed", "f1": .4 + delta})
        self.assertFalse(paired_gains(metrics, "A", "B", plan)["internal_target_met"])
        for metric in metrics:
            if metric["group"] == "B":
                metric["f1"] = .43
        self.assertTrue(paired_gains(metrics, "A", "B", plan)["internal_target_met"])
        metrics[-1]["support_sha256"] = "different"
        with self.assertRaises(DataError):
            paired_gains(metrics, "A", "B", plan)

    def test_missing_seed_blocks_evaluation_before_any_label_access(self):
        from virtual_submeter.accuracy.data_expansion import evaluate
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(run / "plan.json", load_plan(ROOT / "configs/data_expansion_20261024.json"))
            write_json(run / "settings.json", read_json(ROOT / "configs/m2_baseline.json"))
            cases = []
            for group in ("A", "B", "C"):
                write_json(run / f"protocols/{group}/protocol.json", {"partitions": [
                    {"partition_id": "future", "household_id": "refit_17", "role": "frozen_test"}]})
                cases.append({"group": group, "seed": 42, "partition_id": "future"})
            write_json(run / "freeze_refit.json", {"predictions": cases})
            with patch("virtual_submeter.accuracy.data_expansion.check_registered"), \
                    patch("virtual_submeter.accuracy.data_expansion.read_partition", side_effect=AssertionError("label access")):
                with self.assertRaisesRegex(DataError, "every seed"):
                    evaluate(run)


if __name__ == "__main__":
    unittest.main()
