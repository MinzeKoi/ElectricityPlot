"""Preparation and acceptance boundaries; synthetic arrays only, zero fits."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from virtual_submeter.accuracy import microwave_fullsource as f
from virtual_submeter.data.common import DataError, iso, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return read_json(ROOT/"configs/microwave_fullsource_20261025.json")


def data():
    y = np.zeros(100, np.float32)
    y[10:15] = y[60:65] = 300
    events = np.full(100, -1, np.int64)
    events[10:15], events[60:65] = 0, 1
    return {"x": np.full((100, 31), 500, np.float32), "y": y, "t": np.arange(100)*8,
            "house": np.zeros(100, np.int64), "event": events}


def seed_rows(precision=.1, recall=.9, f1=.18):
    return [{"seed": s, "precision": precision, "recall": recall, "f1": f1,
             "source_gate_passed": True, "degenerate": False, "fp": 100} for s in plan()["seeds"]]


class FullSourceTests(unittest.TestCase):
    def test_partition_label_permissions_before_io(self):
        for house, role in (("refit_17", "train"), ("refit_5", "train"), ("refit_2", "validation"),
                            ("refit_8", "train"), ("refit_9", "validation"), ("refit_20", "train")):
            with patch.object(f, "sha256") as hashing:
                with self.assertRaises(DataError):
                    f.source_data({"household_id": house, "artifacts": {"aggregate": {}}}, ROOT, role)
                hashing.assert_not_called()

    def test_fixed_calendar_blocks(self):
        self.assertEqual(f.date_blocks(8, 14*86400+8), [(i*86400, (i+2)*86400) for i in range(1, 13, 2)])

    def test_sampler_replay_and_quota(self):
        a, b = (f.BatchStream(data(), "event", 42, plan()) for _ in range(2))
        for _ in range(20):
            left = a.next()
            np.testing.assert_array_equal(left, b.next())
            self.assertEqual(int((data()["y"][left] >= 200).sum()), 16)

    def test_source_common_threshold_and_no_precision_floor(self):
        d = data()
        scores = np.where(d["y"] >= 200, .9, .1)
        blocks = {"a": d, "b": d}
        point = f.select_checkpoint({"a": scores, "b": scores}, blocks, plan())
        self.assertEqual(point["threshold"], .9)
        self.assertFalse(point["limited_evidence"])
        self.assertEqual(point["mean_recall"], 1.)

    def test_low_recall_and_degenerate_rejected(self):
        d = data()
        # Positives ranked below every negative: no nondegenerate high-recall threshold.
        scores = np.where(d["y"] >= 200, .1, .9)
        self.assertIsNone(f.select_checkpoint({"a": scores, "b": scores}, {"a": d, "b": d}, plan()))
        self.assertIsNone(f.select_checkpoint({"a": np.ones(100), "b": np.ones(100)}, {"a": d, "b": d}, plan()))

    def test_source_precision_below_eight_percent_allowed(self):
        d = {"y": np.r_[np.full(10, 300), np.zeros(1990)],
             "event": np.r_[np.repeat([0, 1], 5), np.full(1990, -1)]}
        scores = np.r_[np.full(1999, .9), .1]
        point = f.select_checkpoint({"a": scores, "b": scores}, {"a": d, "b": d}, plan())
        self.assertIsNotNone(point)
        self.assertLess(point["mean_precision"], .08)
        self.assertGreaterEqual(point["mean_recall"], .85)

    def test_insufficient_source_blocks_rejected(self):
        d = data()
        self.assertIsNone(f.select_checkpoint({"a": np.linspace(0, 1, 100)}, {"a": d}, plan()))

    def test_candidate_rules_and_tie_order(self):
        rows = {"natural": seed_rows(), "event": seed_rows(.2, .9, .3), "event_background": seed_rows(.2, .9, .3)}
        self.assertEqual(f.select_source(rows, plan())["selected_arm"], "event")
        rows["event"][0]["recall"] = .8
        self.assertEqual(f.select_source(rows, plan())["selected_arm"], "event_background")
        rows["event_background"][0]["degenerate"] = True
        result = f.select_source(rows, plan())
        self.assertEqual(result["selected_arm"], "natural")
        self.assertFalse(result["improvement_candidate"])

    def test_natural_competes_and_invalid_natural_not_deployable(self):
        natural = seed_rows(.3, .9, .4)
        event = seed_rows(.31, .9, .41)
        event[2]["precision"] = .1
        result = f.select_source({"natural": natural, "event": event}, plan())
        self.assertEqual(result["selected_arm"], "natural")
        self.assertFalse(result["improvement_candidate"])
        natural[0]["source_gate_passed"] = False
        result = f.select_source({"natural": natural, "event": event}, plan())
        self.assertIsNone(result["selected_arm"])
        self.assertTrue(result["no_deployable_candidate"])

    def test_exact_natural_tie_wins(self):
        natural = seed_rows(.3, .9, .4)
        event = seed_rows(.31, .9, .4)
        event[2]["precision"] = .28
        # Exact binary fractions avoid incidental floating mean differences.
        for r in natural:
            r["precision"] = .25
        event[0]["precision"], event[1]["precision"], event[2]["precision"] = .375, .375, 0.
        result = f.select_source({"natural": natural, "event": event}, plan())
        self.assertEqual(result["selected_arm"], "natural")

    def test_development_f1_gain_inclusive_boundary(self):
        base, candidate = seed_rows(.04, .9, .1), seed_rows(.1, .9, .12)
        for row in candidate:
            row["fp"] = 60
        self.assertTrue(f.development_gate(base, candidate, plan()))
        for row in candidate:
            row["f1"] -= 1e-10
        self.assertFalse(f.development_gate(base, candidate, plan()))

    def test_development_acceptance_paired_complete_matrix(self):
        base, candidate = seed_rows(.04, .9, .077), seed_rows(.1, .9, .18)
        for row in candidate:
            row["fp"] = 60
        self.assertTrue(f.development_gate(base, candidate, plan()))
        candidate[0]["recall"] = .84
        self.assertFalse(f.development_gate(base, candidate, plan()))
        with self.assertRaises(DataError):
            f.development_gate(base, candidate[:2], plan())

    def test_common_support_and_other_devices_unchanged(self):
        original = {k: np.arange(5, dtype=float) for k in ("t", "kettle_power", "kettle_state", "washing_machine_power", "washing_machine_state", "microwave_power")}
        candidate = {k: a.copy() for k, a in original.items()}
        self.assertTrue(f.aligned_unchanged(original, candidate))
        for key in original:
            changed = {k: a.copy() for k, a in candidate.items()}
            changed[key][0] += 1
            with self.assertRaises(DataError):
                f.aligned_unchanged(original, changed)

    def test_incomplete_matrix_cannot_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_json(run/"matrix.json", {"arms": ["natural", "event"], "seeds": plan()["seeds"], "preparation_passed": True})
            write_json(run/"training_complete.json", {"completed_models": ["natural/42"]})
            write_json(run/"freeze.json", {"predictions": []})
            with patch.object(f, "verify_preparation"):
                with self.assertRaises(DataError):
                    f.evaluation_guard(run)

    def test_missing_not_filled_and_window_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            part = {"household_id": "refit_2", "start": iso(0), "end": iso(800), "artifacts": {}}
            for channel in ("aggregate", "microwave"):
                path = run/f"{channel}.csv"
                field = "aggregate_power_w" if channel == "aggregate" else "power_w"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["household_id", "timestamp", field])
                    writer.writeheader()
                    for i in range(100):
                        value = "" if i == (40 if channel == "aggregate" else 70) else 300
                        writer.writerow({"household_id": "refit_2", "timestamp": iso(i*8), field: value})
                part["artifacts"][channel] = {"path": path.name, "sha256": sha256(path)}
            result, _, _, _, _ = f.source_data(part, run, "train")
            self.assertNotIn(70*8, result["t"])
            self.assertFalse(np.any((result["t"] >= 25*8) & (result["t"] <= 55*8)))
            self.assertTrue(np.isfinite(result["x"]).all())


if __name__ == "__main__":
    unittest.main()
