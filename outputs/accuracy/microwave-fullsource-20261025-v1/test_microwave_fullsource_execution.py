"""Synthetic end-to-end execution and immutable development-label barrier tests."""
import csv
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from virtual_submeter.accuracy import microwave_fullsource as f
from virtual_submeter.accuracy import microwave_fullsource_execution as e
from virtual_submeter.data.common import DataError, iso, read_json, sha256, write_json
from virtual_submeter.accuracy.microwave_diagnosis import normalization

ROOT = Path(__file__).resolve().parents[1]


class ExecutionTests(unittest.TestCase):
    def test_no_training_for_preparation_only(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(f, "verify_preparation"):
            run = Path(directory)
            write_json(run/"registration.json", {"execution_adapter_status": "preparation_only"})
            with self.assertRaises(DataError):
                e.train(run)
            self.assertFalse((run/"training_started.json").exists())

    def test_truth_not_opened_before_matrix_guard(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(e, "registered"), patch.object(f, "evaluation_guard", side_effect=DataError("incomplete")):
            run = Path(directory)
            with patch.object(e, "reference_arrays") as read_reference:
                with self.assertRaises(DataError):
                    e.evaluate(run)
                read_reference.assert_not_called()
            self.assertFalse((run/"evaluation").exists())

    def test_complete_synthetic_nine_fit_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            p = read_json(ROOT/"configs/microwave_fullsource_20261025.json")
            p.update(checkpoints=2, steps_per_checkpoint=2, cpu_threads=1)
            matrix = {"arms": f.ORDER, "seeds": p["seeds"], "fits": 9, "preparation_passed": True,
                      "supported_validation_blocks": ["block_0", "block_1"]}
            write_json(run/"plan.json", p)
            write_json(run/"matrix.json", matrix)
            write_json(run/"registration.json", {"execution_adapter_status": "ready"})
            write_json(run/"execution_authorization.json", {"execute": True})
            t = np.arange(100)*8
            y = np.zeros(100, np.float32)
            y[20:25], y[60:65] = 300, 300
            events = np.full(100, -1)
            events[20:25], events[60:65] = 0, 1
            x = np.full((100, 31), 500, np.float32)
            d = {"x": x, "y": y, "t": t, "house": np.zeros(100, np.int64), "event": events}
            np.savez_compressed(run/"train.npz", **d)
            (run/"validation").mkdir()
            for key in matrix["supported_validation_blocks"]:
                np.savez_compressed(run/f"validation/{key}.npz", **d)
            write_json(run/"normalization.json", {**normalization(x), "cpu_threads": 1})
            power = np.zeros_like(x)
            power[y >= 200, 15] = 300
            np.savez_compressed(run/"pools.npz", power=power, residual=x-power,
                                template_ok=y >= 200, background_ok=y < 200,
                                reason=np.full(100, "eligible"), source_hashes=np.full((2, 2), "synthetic"))
            rows = [{"household_id": "refit_17", "timestamp": iso(int(s)), "aggregate_power_w": "500"} for s in t]
            raw = np.full((100, 3), np.nan)
            raw[15:85] = [100, 100, 10]
            base = {"stamps": t, "aggregate": np.full(100, 500, np.float32), "raw": raw}
            thresholds = {"kettle": 2000, "microwave": 200, "washing_machine": 20}
            write_json(run/"baseline_settings.json", {"on_thresholds_w": thresholds})
            (run/"protocol").mkdir()
            truth_path = run/"protocol/microwave.csv"
            with truth_path.open("w", encoding="utf-8", newline="") as stream:
                w = csv.DictWriter(stream, fieldnames=["timestamp", "power_w"])
                w.writeheader()
                w.writerows({"timestamp": iso(int(s)), "power_w": float(v)} for s, v in zip(t, y))
            write_json(run/"baseline_identity.json", {"reference_run": str(run), "development_partition": {
                "artifacts": {"microwave": {"path": "microwave.csv", "sha256": sha256(truth_path)}}}})
            with patch.object(e, "registered", return_value=(p, matrix)), patch.object(f, "verify_preparation"), patch.object(e, "target_input", return_value=rows), patch.object(e, "reference_arrays", return_value=base):
                result = e.train(run)
                self.assertEqual(result["fits"], 9)
                with self.assertRaises(DataError):
                    e.train(run)
                frozen = e.freeze(run)
                self.assertEqual(frozen["predictions"], 9)
                with self.assertRaises(DataError):
                    e.freeze(run)
                self.assertTrue(f.evaluation_guard(run))
                complete_before = sha256(run/"freeze.json")
                result = e.evaluate(run)
                self.assertFalse(result["development_acceptance_passed"])
                with self.assertRaises(DataError):
                    e.evaluate(run)
                verified = e.verify(run)
                self.assertEqual(verified["status"], "fullsource_evaluation_verified")
                self.assertFalse(verified["target_truth_csv_reopened"])
                self.assertEqual(complete_before, sha256(run/"freeze.json"))
                first = e.report(run)
                self.assertEqual(first, e.report(run))
                # Byte tampering must fail before re-evaluation or label access.
                path = run/"predictions/natural_42.npz"
                path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaises(DataError):
                    f.evaluation_guard(run)

    def test_source_row_uses_equal_block_means(self):
        selected = {"source_gate_passed": True, "block_metrics": {
            "a": {"precision": .1, "recall": .9, "f1": .18, "tp": 9, "fp": 81, "n": 100},
            "b": {"precision": .3, "recall": .9, "f1": .45, "tp": 9, "fp": 21, "n": 1000}}}
        row = e.source_row(42, selected)
        self.assertEqual(row["precision"], .2)
        self.assertFalse(row["degenerate"])


if __name__ == "__main__":
    unittest.main()
