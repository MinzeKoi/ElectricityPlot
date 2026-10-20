"""M5 freeze barrier, full comparison, tamper rejection and rerun evidence."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.data.common import DataError, read_json, write_json
from virtual_submeter.experiments.runner import (
    freeze_experiment, evaluate_experiment, verify_experiment, validate_plan,
)
from virtual_submeter.experiments.summary import compare_runs, stats
import test_m2_baseline as baseline_tests


class SummaryTest(unittest.TestCase):
    def test_null_scores_are_not_zeros_and_singleton_std_is_unknown(self):
        self.assertEqual(stats([None, 2]), {"mean": 2, "sample_std": None, "valid_count": 1, "missing_count": 1})
        self.assertIsNone(stats([None])["mean"])
        self.assertAlmostEqual(stats([1, 3])["sample_std"], 2 ** 0.5)


@unittest.skipUnless(baseline_tests.NUMERIC, "M5 integration requires numeric environment")
class ExperimentTest(unittest.TestCase):
    def setUp(self):
        baseline_tests.BaselineTest.setUp(self)
        protocol = read_json(self.manifest)
        protocol["data_kind"] = "synthetic_contract_fixture"
        write_json(self.manifest, protocol)
        self.plan = read_json(ROOT / "configs/m5_experiment.json")
        self.plan.update(backbone="constant_mean", budgets=[0, 5], seeds=[42, 1019])
        self.experiment = self.root / "experiment.json"
        write_json(self.experiment, self.plan)
        self.output = self.root / "m5"

    def freeze(self, output=None):
        return freeze_experiment(self.manifest, self.experiment, self.settings_path,
                                 ROOT / "configs/m3_adaptation.json", ROOT / "configs/m4_trust.json",
                                 output or self.output)

    def test_full_matrix_freeze_barrier_rerun_and_readonly_verification(self):
        original = Path.open
        def no_test_answers(path, *args, **kwargs):
            if "evaluation_labels" in path.parts:
                raise AssertionError("Test labels opened before complete experiment freeze")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", no_test_answers):
            result = self.freeze()
        self.assertEqual(result["cases"], 16)
        self.assertFalse(read_json(self.output / "run.json")["test_labels_opened"])
        self.assertEqual(verify_experiment(self.output)["phase"], "frozen")
        design = read_json(self.output / "frozen/design.json")
        def guarded_answers(path, *args, **kwargs):
            if "evaluation_labels" in path.parts:
                self.assertTrue((self.output / "freeze.json").is_file())
                self.assertEqual(read_json(self.output / "run.json")["status"], "evaluating")
                self.assertTrue(all((self.output / c["prediction"] / "predictions.json").is_file() for c in design["cases"]))
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded_answers):
            evaluate_experiment(self.output)
        def no_labels(path, *args, **kwargs):
            if any(p in path.parts for p in ("evaluation_labels", "oracle_labels", "source_labels")):
                raise AssertionError("Verification reopened labels")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", no_labels):
            self.assertEqual(verify_experiment(self.output)["status"], "m5_artifacts_verified")
        summary = read_json(self.output / "summary.json")
        self.assertEqual({r["group"] for r in summary["rows"]}, {"B0", "B1", "B2", "B3", "P1", "P2"})
        self.assertEqual(len(summary["rows"]), 60)
        self.assertEqual(len(summary["background_stress"]), 4)
        self.assertFalse(summary["formal_acceptance"])
        for row in summary["paired_differences"]:
            self.assertEqual(row["P1_minus_control"]["mae_w"], 0)
        other = self.root / "repeat"
        self.freeze(other)
        evaluate_experiment(other)
        self.assertEqual(compare_runs(self.output, other)["status"], "reproduction_matches")
        with self.assertRaises(DataError):
            compare_runs(self.output, self.output)
        with self.assertRaises(DataError):
            evaluate_experiment(self.output)
        with self.assertRaises(DataError):
            self.freeze()
        report = self.output / "summary.csv"
        report.write_text("damaged", encoding="utf-8")
        with self.assertRaises(DataError):
            verify_experiment(self.output)

    def test_frozen_artifact_or_code_change_blocks_test_access(self):
        self.freeze()
        file = self.output / "frozen/training.json"
        before = file.read_bytes()
        file.write_text("{}")
        with self.assertRaises(DataError):
            evaluate_experiment(self.output)
        file.write_bytes(before)
        with patch("virtual_submeter.experiments.runner.source_fingerprint", return_value="changed"):
            with self.assertRaisesRegex(DataError, "Code changed"):
                evaluate_experiment(self.output)
        self.assertFalse(read_json(self.output / "run.json")["test_labels_opened"])
        self.assertFalse((self.output / "evaluation").exists())

    def test_failed_evaluation_cannot_publish_success_or_retry_test_selection(self):
        self.freeze()
        with patch("virtual_submeter.evaluation.runner.evaluate", side_effect=DataError("simulated failure")):
            with self.assertRaises(DataError):
                evaluate_experiment(self.output)
        self.assertEqual(read_json(self.output / "run.json")["status"], "failed_evaluation")
        self.assertFalse((self.output / "summary.json").exists())
        with self.assertRaises(DataError):
            verify_experiment(self.output)
        with self.assertRaises(DataError):
            evaluate_experiment(self.output)

    def test_final_gates_and_invalid_matrix(self):
        protocol = read_json(self.manifest)
        for changes in ({"mode": "final"}, {"seeds": [42, 42]}, {"budgets": [5]},
                        {"seeds": [True]}, {"background_stress": "yes"}):
            plan = {**self.plan, **changes}
            with self.subTest(changes=changes), self.assertRaises(DataError):
                validate_plan(plan, protocol)
        broken = copy.deepcopy(protocol)
        broken["partitions"][-1]["start"] = broken["partitions"][-2]["end"]
        with self.assertRaises(DataError):
            validate_plan(self.plan, broken)

    def test_freeze_failure_retains_log_but_no_success_manifest(self):
        with patch("virtual_submeter.models.training.train", side_effect=DataError("simulated failure")):
            with self.assertRaises(DataError):
                self.freeze()
        self.assertEqual(read_json(self.output / "run.json")["status"], "failed_freeze")
        self.assertFalse((self.output / "freeze.json").exists())
