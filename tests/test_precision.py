"""Precision experiments: source-only weighting, paired guards and freeze barrier."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from virtual_submeter.accuracy import precision
from virtual_submeter.data.common import DataError, iso, read_json, sha256, write_json
from virtual_submeter.models.training import dataset, negative_sample_factors, train


class PrecisionTest(unittest.TestCase):
    def plan(self):
        return precision.load_plan(ROOT / "configs/precision_20261024.json")

    def fixture(self):
        from test_m2_baseline import BaselineTest
        result = BaselineTest()
        result.setUp()
        self.addCleanup(result.doCleanups)
        return result

    def test_registered_plan_cannot_change_recipe_or_targets(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            for key, value in (("source_group", "C"), ("seeds", [42]), ("minimum_recall", .5),
                               ("loss_weights", [50, 10, 1, .1]), ("maximum_models", 15)):
                write_json(path, {**plan, key: value})
                with self.assertRaises(DataError):
                    precision.load_plan(path)

    def test_negative_weight_alignment_is_identity_based_and_rejects_wrong_labels(self):
        keys = [("2", 16), ("19", 0), ("2", 8), ("2", 24)]
        y = np.array([[0.], [300.], [np.nan], [0.]])
        plan = {"schema_version": 1, "appliance": "microwave", "factor": 4,
                "windows": [{"household_id": "2", "timestamp": iso(24)}, {"household_id": "2", "timestamp": iso(16)}]}
        np.testing.assert_array_equal(negative_sample_factors(keys, y, ["microwave"], [200], plan)[:, 0], [4, 1, 1, 4])
        for bad in ([{"household_id": "19", "timestamp": iso(0)}],
                    [{"household_id": "2", "timestamp": iso(8)}],
                    [{"household_id": "17", "timestamp": iso(16)}], plan["windows"] * 2):
            with self.assertRaises(DataError):
                negative_sample_factors(keys, y, ["microwave"], [200], {**plan, "windows": bad})
        empty = negative_sample_factors(keys, y, ["microwave"], [200], {**plan, "windows": []})
        np.testing.assert_array_equal(empty, np.ones_like(y))

    def test_empty_weighting_reproduces_actual_training_and_non_source_labels_are_refused(self):
        fixture = self.fixture()
        protocol = read_json(fixture.manifest)
        parts = [p for p in protocol["partitions"] if p["role"] == "source_train"]
        x, _, _, keys = dataset(fixture.manifest, parts, ["microwave"], fixture.settings, 8, return_keys=True)
        self.assertEqual(len(x), len(keys))
        self.assertEqual({k[0] for k in keys}, {"2"})
        protocol["expansion_training"] = {"input_log_mean": float(np.log1p(x).mean()),
                                           "input_log_std": float(np.log1p(x).std()), "steps_per_checkpoint": 2}
        write_json(fixture.manifest, protocol)
        negative = fixture.root / "negatives.json"
        weight_plan = {"schema_version": 1, "appliance": "microwave", "factor": 4, "windows": [],
                       "protocol_sha256": sha256(fixture.manifest), "source_model_sha256": "0" * 64}
        write_json(negative, weight_plan)
        baseline, empty = fixture.root / "ordinary", fixture.root / "empty"
        train(fixture.manifest, fixture.settings_path, baseline, appliances=["microwave"])
        train(fixture.manifest, fixture.settings_path, empty, appliances=["microwave"], negative_weights_path=negative)
        from virtual_submeter.models.artifacts import load_model
        _, original = load_model(baseline)
        metadata, reproduced = load_model(empty)
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, reproduced.state_dict()[name]))
        self.assertEqual(metadata["optimizer_steps"], 4)
        _, y, _, keys = dataset(fixture.manifest, parts, ["microwave"], fixture.settings, 8, return_keys=True)
        off = [i for i in range(len(keys)) if y[i, 0] < 200][:2]
        weighted_windows = [{"household_id": keys[i][0], "timestamp": iso(keys[i][1])} for i in off]
        trained = []
        for index, order in enumerate((weighted_windows, list(reversed(weighted_windows)))):
            write_json(negative, {**weight_plan, "windows": order})
            destination = fixture.root / f"weighted_{index}"
            train(fixture.manifest, fixture.settings_path, destination, appliances=["microwave"], negative_weights_path=negative)
            trained.append(load_model(destination)[1])
        self.assertTrue(any(not torch.equal(v, trained[0].state_dict()[k]) for k, v in original.state_dict().items()))
        for key, value in trained[0].state_dict().items():
            self.assertTrue(torch.equal(value, trained[1].state_dict()[key]))
        write_json(negative, {**weight_plan, "windows": [{"household_id": "3", "timestamp": iso(0)}]})
        with self.assertRaises(DataError):
            train(fixture.manifest, fixture.settings_path, fixture.root / "forbidden", appliances=["microwave"], negative_weights_path=negative)
        future = next(p for p in protocol["partitions"] if p["role"] == "frozen_test")
        with self.assertRaises(DataError):
            dataset(fixture.manifest, [future], ["microwave"], fixture.settings, 8, return_keys=True)

    @staticmethod
    def source_row(group, seed, p=.2, r=.8, f1=.3):
        return {"candidate": group, "seed": seed, "precision": p, "recall": r, "f1": f1,
                "state_pattern": "mixed", "support_sha256": "same"}

    def test_source_selection_rejects_recall_loss_and_uses_registered_ties(self):
        rows = [self.source_row("W50", s) for s in precision.SEEDS]
        rows += [self.source_row("W10", s, .3, .77, .4) for s in precision.SEEDS]
        rows += [self.source_row("W1", s, .9, .6, .5) for s in precision.SEEDS]
        result = precision.choose_source(rows, self.plan(), ["W50", "W10", "W1"])
        self.assertEqual(result["selected"], "W10")
        rows += [self.source_row("W10_HN", s, .3, .77, .4) for s in precision.SEEDS]
        self.assertEqual(precision.choose_source(rows, self.plan(), ["W50", "W10", "W1", "W10_HN"])["selected"], "W10")
        changed = copy.deepcopy(rows)
        changed[3]["support_sha256"] = "other"
        self.assertEqual(precision.choose_source(changed[:9], self.plan(), ["W50", "W10", "W1"])["selected"], "W50")
        with self.assertRaises(DataError):
            precision.choose_source(rows[:-1], self.plan(), ["W50", "W10", "W1", "W10_HN"])

    def development_rows(self):
        rows = []
        for group, tp, fp, fn in (("W50", 354, 8482, 29), ("W10", 330, 3000, 53)):
            for seed in precision.SEEDS:
                rows.append({"candidate": group, "seed": seed, "partition": precision.DEVELOPMENT,
                             "appliance": "microwave", "view": "raw", "precision": tp / (tp + fp),
                             "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
                             "false_positive": fp, "false_negative": fn, "state_pattern": "mixed", "support_sha256": "same"})
        return rows

    def test_development_gate_needs_precision_recall_paired_gains_and_fp_reduction(self):
        rows = self.development_rows()
        self.assertTrue(precision.development_gate(rows, "W10", self.plan())["precision_exit_passed"])
        changes = [{"recall": .849}, {"precision": .07}, {"f1": .05},
                   {"false_positive": 8000}, {"support_sha256": "different"},
                   {"state_pattern": "all_off", "precision": None, "recall": 0., "f1": 0.}]
        for change in changes:
            altered = copy.deepcopy(rows)
            for r in altered:
                if r["candidate"] == "W10":
                    r.update(change)
            self.assertFalse(precision.development_gate(altered, "W10", self.plan())["precision_exit_passed"], change)
        self.assertFalse(precision.development_gate(rows, "W50", self.plan())["precision_exit_passed"])

    def test_incomplete_freeze_rejected_before_evaluation_label_access(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(run / "selection.json", {"selected": "W10"})
            write_json(run / "stage_one_selection.json", {"selected": "W10"})
            write_json(run / "freeze.json", {"groups": ["W50", "W10", "W1", "W10_HN"], "model_count": 12,
                       "cases": [], "selection_sha256": sha256(run / "selection.json"), "development_labels_parsed": False})
            write_json(run / "run.json", {"status": "frozen", "freeze_sha256": sha256(run / "freeze.json")})
            with patch.object(precision, "_registered", return_value=self.plan()), patch.object(precision, "read_partition") as read:
                with self.assertRaises(DataError):
                    precision.evaluate(run)
                read.assert_not_called()

    def test_role_changes_are_rejected_without_reading_data(self):
        # Schema fixture, independent of whether real experiment data are installed.
        protocol = {"purpose": "engineering_pilot_only", "accuracy_usage": "development_only", "sample_interval_seconds": 8,
                    "target_appliances": precision.DEVICES, "split_gap_seconds": 3600,
                    "expansion_training": {"steps_per_checkpoint": 587}, "partitions": [
                        {"partition_id": "source_train_refit_2_0", "role": "source_train", "household_id": "refit_2"},
                        {"partition_id": "source_train_refit_19_0", "role": "source_train", "household_id": "refit_19"},
                        {"partition_id": precision.SOURCE_VALIDATION, "role": "source_validation", "household_id": "refit_5"},
                        {"partition_id": precision.DEVELOPMENT, "role": "frozen_test", "household_id": "refit_17"}]}
        precision.validate_protocol(protocol)
        protocol["partitions"][0]["household_id"] = "refit_8"
        with self.assertRaises(DataError):
            precision.validate_protocol(protocol)

    def test_numeric_prediction_matches_ordinary_inference_including_missing_support(self):
        fixture = self.fixture()
        train(fixture.manifest, fixture.settings_path, fixture.model, appliances=["microwave"])
        from virtual_submeter.models.artifacts import load_model
        from virtual_submeter.models.arrays import aggregate_arrays, windows
        from virtual_submeter.models.prediction import predict_rows
        from virtual_submeter.data.protocol import read_partition
        metadata, model = load_model(fixture.model)
        data = read_partition(fixture.manifest, "frozen_test_house_3", "inference")
        rows = data["aggregate"]
        stamps, aggregate = aggregate_arrays(rows, 8)
        x, centers = windows(stamps, aggregate, fixture.settings["window_samples"], 8)
        numeric = np.full(len(rows), np.nan)
        numeric[centers] = precision._predict_values(metadata, model, x)[:, 0]
        ordinary = predict_rows(metadata, model, rows)
        values = np.asarray([r["appliances"]["microwave"]["raw_power_w"] if r["appliances"]["microwave"]["raw_power_w"] is not None else np.nan for r in ordinary])
        np.testing.assert_array_equal(numeric, values)


if __name__ == "__main__":
    unittest.main()
