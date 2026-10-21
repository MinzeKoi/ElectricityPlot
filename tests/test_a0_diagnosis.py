"""A0 diagnosis support, development permissions, freeze and artifact checks."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.diagnosis import event_support, paired_summary, query_summary
from virtual_submeter.accuracy.runner import run, verify
from virtual_submeter.data.common import DataError, iso, read_json, write_json
from virtual_submeter.data.protocol import read_partition
import test_m2_baseline as baseline_tests


class DiagnosisTest(unittest.TestCase):
    def test_events_exclude_missing_boundary_and_gapped_runs(self):
        labels = [{"timestamp": iso(i * 8), "power_w": str(v) if v is not None else ""}
                  for i, v in enumerate([20, 0, 20, 20, 0, 20, None, 20, 0, 20])]
        result = event_support(labels, 10, 8)
        self.assertEqual(result["complete_threshold_events"], 1)
        self.assertEqual(result["censored_threshold_events"], 4)
        self.assertEqual(result["valid_samples"], 9)

    def test_pair_uses_common_support_and_detects_all_off_regression(self):
        before = [{"household_id": "3", "timestamp": iso(i * 8),
                   "appliances": {"kettle": {"power_w": p, "raw_power_w": p, "state": s}}}
                  for i, (p, s) in enumerate([(20, True), (20, False), (None, None)])]
        import copy
        after = copy.deepcopy(before)
        after[0]["appliances"]["kettle"]["state"] = False
        labels = [{"household_id": "3", "timestamp": iso(i * 8), "power_w": str(p)} for i, p in enumerate([20, 0, 20])]
        result = paired_summary(before, after, labels, "kettle", 10, 8)
        self.assertEqual(result["f1_delta"], -1)
        self.assertEqual(result["adapted"]["state_pattern"], "all_off")
        self.assertEqual(result["adapted"]["valid_samples"], 2)
        self.assertIsNone(result["adapted"]["precision"])
        after[0]["appliances"]["kettle"]["power_w"] = 0
        with self.assertRaises(DataError):
            paired_summary(before, after, labels, "kettle", 10, 8)

    def test_skip_and_unsure_spend_budget_but_are_not_classes(self):
        result = query_summary([{"appliance_type": "kettle", "response": r} for r in ("used", "unsure", "skipped")], ["kettle", "microwave"])
        self.assertEqual(result["kettle"]["queries"], 3)
        self.assertEqual(result["kettle"]["usable"], 1)
        self.assertTrue(result["kettle"]["single_class"])
        self.assertEqual(result["microwave"]["queries"], 0)

    def test_no_valid_support_is_unavailable_instead_of_all_off(self):
        rows = [{"household_id": "3", "timestamp": iso(0),
                 "appliances": {"kettle": {"power_w": None, "raw_power_w": None, "state": None}}}]
        labels = [{"household_id": "3", "timestamp": iso(0), "power_w": "20"}]
        result = paired_summary(rows, rows, labels, "kettle", 10, 8)
        self.assertEqual(result["adapted"]["state_pattern"], "unavailable")
        self.assertIsNone(result["adapted"]["predicted_on_fraction"])
        self.assertIsNone(result["f1_delta"])


@unittest.skipUnless(baseline_tests.NUMERIC, "A0 integration needs numeric environment")
class PipelineTest(unittest.TestCase):
    def setUp(self):
        baseline_tests.BaselineTest.setUp(self)
        baseline_tests.BaselineTest.train(self, "constant_mean")

    def test_diagnosis_consumer_rejects_unmarked_and_final_protocols(self):
        protocol = read_json(self.manifest)
        for usage, purpose in ((None, "engineering_pilot_only"), ("development_only", "final_evaluation")):
            protocol.update(accuracy_usage=usage, purpose=purpose)
            write_json(self.manifest, protocol)
            with self.assertRaises(DataError):
                read_partition(self.manifest, "target_adaptation_house_3", "development_diagnosis", True)
        protocol.update(accuracy_usage="development_only", purpose="engineering_pilot_only")
        write_json(self.manifest, protocol)
        self.assertIn("kettle", read_partition(self.manifest, "target_adaptation_house_3", "development_diagnosis", True))
        for consumer in ("selection", "adaptation"):
            with self.assertRaises(DataError):
                read_partition(self.manifest, "target_adaptation_house_3", consumer, True)

    def test_freeze_failure_cannot_open_diagnosis_labels_or_publish_success(self):
        output = self.root / "failed_a0"
        with patch("virtual_submeter.adaptation.runner.run_comparison", side_effect=DataError("freeze failure")), \
                patch("virtual_submeter.accuracy.runner.read_partition", side_effect=AssertionError("Labels opened")):
            with self.assertRaises(DataError):
                run(self.model, self.manifest, ROOT / "configs/m3_adaptation.json", output, self.root)
        self.assertEqual(read_json(output / "run.json")["status"], "failed")
        self.assertFalse((output / "diagnosis.json").exists())
        self.assertFalse((output / "freeze.json").exists())

    def test_pipeline_freezes_before_diagnosis_and_verifies_without_labels(self):
        # Narrow fixture matrix, while exercising real sessions and weak oracle.
        protocol = read_json(self.manifest)
        protocol.update(query_budgets=[0, 5], random_seeds=[42])
        write_json(self.manifest, protocol)
        from virtual_submeter.data.common import sha256
        model = read_json(self.model / "model.json")
        model["protocol_sha256"] = sha256(self.manifest)
        write_json(self.model / "model.json", model)
        output = self.root / "a0"
        original = read_partition

        def guarded(manifest, partition, consumer, include_labels=False):
            if consumer == "development_diagnosis":
                self.assertTrue((output / "freeze.json").exists())
                self.assertTrue(read_json(output / "run.json")["development_labels_opened"])
            return original(manifest, partition, consumer, include_labels)

        with patch("virtual_submeter.accuracy.runner.read_partition", guarded):
            result = run(self.model, self.manifest, ROOT / "configs/m3_adaptation.json", output, self.root)
        self.assertEqual(result["cases"], 4)
        self.assertTrue((output / "report.html").exists())
        self.assertFalse(read_json(self.manifest).get("accuracy_usage") == "development_only")
        with patch("virtual_submeter.accuracy.runner.read_partition", side_effect=AssertionError("Labels reopened")):
            self.assertEqual(verify(output)["status"], "a0_artifacts_verified")
        (output / "query_distribution.csv").write_text("tampered", encoding="utf-8")
        with self.assertRaises(DataError):
            verify(output)
        with self.assertRaises(DataError):
            run(self.model, self.manifest, ROOT / "configs/m3_adaptation.json", output, self.root)


if __name__ == "__main__":
    unittest.main()
