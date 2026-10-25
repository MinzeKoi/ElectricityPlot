"""A1 state-baseline fairness, event gates and bounded source comparison."""

import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.baseline import candidate_rank, event_detection
from virtual_submeter.accuracy.baseline_data import load_plan
from virtual_submeter.data.common import DataError, iso, read_json, write_json

try:
    import numpy as np
    NUMERIC = True
except ImportError:
    NUMERIC = False


class A1MetricsTest(unittest.TestCase):
    def test_threshold_events_distinguish_complete_censored_and_unsupported(self):
        labels = [{"timestamp": iso(i * 8), "power_w": "" if v is None else str(v)} for i, v in enumerate([0, 20, 20, 0, None, 20, 0])]
        rows = [{"timestamp": iso(i * 8), "appliances": {"kettle": {"raw_state": s}}} for i, s in enumerate([False, False, True, False, None, None, False])]
        result = event_detection(rows, labels, "kettle", 10, 8)
        self.assertEqual(result["complete_events"], 1)
        self.assertEqual(result["complete_detected"], 1)
        self.assertEqual(result["censored_fragments"], 1)
        self.assertEqual(result["censored_detected"], 0)
        self.assertEqual(result["events_without_prediction_support"], 1)

    def test_candidate_gate_rejects_all_off_or_no_complete_event_hits(self):
        metric = {"valid_samples": 10, "true_positive": 1, "predicted_on_samples": 2,
                  "f1": 0.5, "active_mae_w": 10, "on_threshold_w": 100,
                  "energy": {"absolute_error_kwh": 0.1, "truth_kwh": 1}}
        rows = [{"partition": p, "appliance": a, "raw": copy.deepcopy(metric), "events": {"complete_detected": 1}}
                for p in ("source_validation", "development") for a in ("kettle", "microwave")]
        self.assertTrue(candidate_rank(rows, ["kettle", "microwave"])["eligible"])
        rows[0]["events"]["complete_detected"] = 0
        self.assertFalse(candidate_rank(rows, ["kettle", "microwave"])["eligible"])
        rows[0]["events"]["complete_detected"] = 1
        rows[0]["raw"]["predicted_on_samples"] = 10
        self.assertFalse(candidate_rank(rows, ["kettle", "microwave"])["eligible"])

    def test_plan_rejects_extra_candidates_and_household_overlap(self):
        plan = read_json(ROOT / "configs/a1_baseline.json")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            plan["candidates"].append({"id": "fourth", "epochs": 1, "active_loss_weight": 5})
            write_json(path, plan)
            with self.assertRaises(DataError):
                load_plan(path)
            plan["candidates"].pop()
            plan["development_households"] = ["2"]
            write_json(path, plan)
            with self.assertRaises(DataError):
                load_plan(path)


@unittest.skipUnless(NUMERIC, "A1 state inference needs NumPy")
class A1StateTest(unittest.TestCase):
    def test_initial_head_is_identical_to_raw_baseline_while_legacy_remains(self):
        from virtual_submeter.models.prediction import predict_rows
        from virtual_submeter.adaptation.state import INITIAL, apply_head
        appliances = ["kettle", "microwave", "washing_machine"]
        meta = {"kind": "constant_mean", "sample_interval_seconds": 8, "appliances": appliances,
                "settings": {"window_samples": 7, "on_thresholds_w": {"kettle": 2000, "microwave": 200, "washing_machine": 20}}}
        inputs = [{"household_id": "3", "timestamp": iso(i*8), "aggregate_power_w": "100", "quality_flags": "[]", "coverage": "1"} for i in range(12)]
        legacy = predict_rows(meta, np.array([2500, 400, 40]), inputs)
        aligned = predict_rows({**meta, "state_threshold_view": "raw_power_w"}, np.array([2500, 400, 40]), inputs)
        with_initial = copy.deepcopy(aligned)
        head = {"method": "max_pool_mil_logistic_v1", "coefficients": {a: INITIAL for a in appliances}, "active_appliances": appliances}
        apply_head(with_initial, {**meta, "state_adapter": head})
        for old, new, init in zip(legacy, aligned, with_initial):
            for a in appliances:
                self.assertEqual(new["appliances"][a]["state"], init["appliances"][a]["state"])
                self.assertEqual(old["appliances"][a]["power_w"], new["appliances"][a]["power_w"])
                if new["appliances"][a]["state"] is not None:
                    self.assertTrue(new["appliances"][a]["state"])
                    self.assertFalse(old["appliances"][a]["state"])


class A1ProtocolTest(unittest.TestCase):
    def test_real_prepare_path_preserves_roles_gap_event_support_and_label_permissions(self):
        from virtual_submeter.accuracy.baseline_data import build_protocol
        from virtual_submeter.data.prepare import prepare as prepare_data
        from virtual_submeter.data.protocol import read_partition
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = read_json(ROOT / "configs/refit_channels.json")["households"]
            inputs = {}
            for house in ("2", "3", "5"):
                path = root / f"house{house}.csv"
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["Unix", "Aggregate", *[f"Appliance{i}" for i in range(1, 10)], "Issues"])
                    writer.writeheader()
                    for i in range(2701):
                        on = i % 40 > 30
                        row = {f"Appliance{j}": 0 for j in range(1, 10)}
                        for appliance, value in (("kettle", 2500), ("microwave", 400), ("washing_machine", 40)):
                            row[mapping[house][appliance][0]] = value if on else 0
                        row.update(Unix=i * 8, Aggregate=3200 if on else 500, Issues=0)
                        writer.writerow(row)
                inputs[house] = path
            prepare_data({h: inputs[h] for h in ("2", "5")}, root / "sources", ROOT / "configs/refit_channels.json")
            prepare_data({"3": inputs["3"]}, root / "development", ROOT / "configs/refit_channels.json")
            plan = read_json(ROOT / "configs/a1_baseline.json")
            plan["minimum_source_observed_days"] = 0.1
            manifest = build_protocol(root / "sources", root / "development", root / "protocol", plan, read_json(ROOT / "configs/m2_baseline.json"))
            protocol = read_json(manifest)
            self.assertEqual(protocol["data_kind"], "synthetic_contract_fixture")
            parts = protocol["partitions"]
            target = next(p for p in parts if p["role"] == "target_adaptation")
            future = next(p for p in parts if p["role"] == "frozen_test")
            from virtual_submeter.data.common import utc
            self.assertEqual(utc(future["start"]) - utc(target["end"]), 3600)
            admission = read_json(root / "protocol/admission.json")["households"]
            self.assertFalse(next(a for a in admission if a["household_id"] == "11")["included"])
            for consumer in ("training", "selection", "adaptation"):
                with self.assertRaises(DataError):
                    read_partition(manifest, future["partition_id"], consumer, True)
            self.assertIn("kettle", read_partition(manifest, future["partition_id"], "evaluation", True))
            with self.assertRaises(DataError):
                build_protocol(root / "sources", root / "development", root / "protocol", plan, read_json(ROOT / "configs/m2_baseline.json"))


if __name__ == "__main__":
    unittest.main()
