"""Sampler identities, leakage gates and a tiny synthetic six-fit lifecycle."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from virtual_submeter.accuracy import microwave_sampling as s
from virtual_submeter.accuracy import microwave_diagnosis as d
from virtual_submeter.data.common import DataError, iso, read_json, sha256, utc, write_json
from virtual_submeter.models.arrays import windows

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return read_json(ROOT / "configs/microwave_sampling_20261025.json")


def fixture():
    n = 400
    y = np.zeros(n, np.float32)
    event = np.full(n, -1, np.int64)
    # Two dates; two events per date, different duration.
    for e, (a, b) in enumerate(((0, 5), (10, 50), (200, 205), (210, 250))):
        y[a:b], event[a:b] = 300, e
    t = np.r_[np.arange(200)*8, 86400+np.arange(200)*8]
    return {"y": y, "x": np.repeat(np.arange(n, dtype=np.float32)[:, None], 31, axis=1),
            "event": event, "t": t, "house": np.zeros(n, np.int64)}


class SamplingTests(unittest.TestCase):
    def test_fixed_positive_quota_event_balance_and_identity(self):
        data, p = fixture(), plan()
        a, b = s.BatchStream(data, "event", 42, p), s.BatchStream(data, "event", 42, p)
        counts = np.zeros(4, int)
        negative_counts = np.zeros(len(data["y"]), int)
        seen = set()
        for _ in range(40):
            idx = a.next()
            np.testing.assert_array_equal(idx, b.next())
            self.assertEqual(len(idx), 256)
            pos = idx[data["y"][idx] >= 200]
            self.assertEqual(len(pos), 16)
            np.add.at(counts, data["event"][pos], 1)
            np.add.at(negative_counts, idx[data["y"][idx] < 200], 1)
            seen.update(pos.tolist())
            np.testing.assert_array_equal(data["x"][idx, 15], idx.astype(np.float32))
        self.assertLessEqual(int(counts.max()-counts.min()), 1)
        self.assertEqual(len(seen), int((data["y"] >= 200).sum()))
        negatives = negative_counts[data["y"] < 200]
        self.assertLessEqual(int(negatives.max()-negatives.min()), 1)

    def test_natural_cycle_not_rescaled_to_positive_quota(self):
        data, p = fixture(), plan()
        sampler = s.BatchStream(data, "natural", 1019, p)
        idx = np.r_[sampler.next(), sampler.next()]
        self.assertEqual(len(set(idx[:400])), 400)
        self.assertEqual(int((data["y"][idx[:400]] >= 200).sum()), 90)

    def test_missing_not_off_and_complete_context_and_crossing_event(self):
        t = np.arange(300)*8
        power = np.zeros(300)
        power[98:103], power[180] = 300, np.nan
        ids, events = d.segments(t, power)
        self.assertEqual(ids[180], -1)
        aggregate = np.ones(300, np.float32)
        aggregate[200] = np.nan
        x, c = windows(t, aggregate, 31, 8)
        self.assertFalse(((c >= 185) & (c <= 215)).any())
        left = d.split_mask(t, c, ids, events, 0, 800)
        right = d.split_mask(t, c, ids, events, 800, 2400)
        self.assertFalse((ids[c[left]] == 0).any())
        self.assertFalse((ids[c[right]] == 0).any())
        self.assertLess(t[c[left][-1]]+120, t[c[right][0]]-120)

    def test_selection_guard_multiple_dates_ties_and_degeneracy(self):
        data, p = fixture(), plan()
        blocks = {"day0": data, "day1": data}
        scores = np.where(data["y"] >= 200, .9, .1)
        best = s.selection({"day0": scores, "day1": scores}, blocks, p)
        self.assertEqual(best["threshold"], .9)
        self.assertEqual(best["worst_precision"], 1.)
        for bad in (np.ones(400), np.zeros(400), 1-scores):
            self.assertIsNone(s.selection({"day0": scores, "day1": bad}, blocks, p))
        self.assertLess(s.rank(best, 1), s.rank(best, 2))
        zero = {**data, "y": np.zeros(400), "event": np.full(400, -1)}
        self.assertIsNotNone(s.selection({"day0": scores, "day1": scores, "off": np.zeros(400)},
                                        {**blocks, "off": zero}, p))
        # Low precision is reported, not vetoed by a new absolute P/F1/FPR gate.
        weak = scores.copy()
        weak[np.flatnonzero(data["y"] < 200)[:300]] = .95
        self.assertIsNotNone(s.selection({"day0": weak, "day1": weak}, blocks, p))

    def test_house_and_outer_access_rejected_before_read(self):
        with patch.object(s, "load_npz", side_effect=AssertionError("must not read")):
            for house, role in (("refit_17", "train"), ("refit_8", "train"),
                                ("refit_9", "train"), ("refit_20", "train"), ("refit_2", "outer")):
                with self.assertRaises(DataError):
                    s.source_array(Path("unused"), house, role)
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            with self.assertRaises(DataError):
                s.check_freeze(run)
            self.assertFalse((run / "evaluation").exists())

    def test_unfinished_audit_and_unregistered_recipe_rejected(self):
        with self.assertRaises(DataError):
            s.validate_manifest({"status": "pending"}, ROOT)
        p = plan()
        p["maximum_fits"] = 12
        with self.assertRaises(DataError):
            s.validate_plan(p)

    def test_zero_positive_block_reports_false_alarms(self):
        m = s.partition_metrics(np.array([.8, .1, .2, .9]), np.zeros(4), .5)
        self.assertEqual((m["fp"], m["tn"], m["off_false_positive_rate"]), (2, 2, .5))
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["ap"])

    def test_synthetic_lifecycle_freeze_inference_and_immutability(self):
        # Synthetic 6 x 2 updates: integration test, not a real-data experiment.
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            p = plan()
            p["checkpoints"], p["steps_per_checkpoint"] = 2, 1
            pp = base / "plan.json"
            write_json(pp, p)
            audit = base / "audit.json"
            write_json(audit, {"status": "synthetic_test_only"})
            manifest = {"schema": "microwave_sampling_audit_v1", "status": "ready",
                        "audit": {"path": audit.name, "sha256": sha256(audit)}, "sources": []}
            start = utc("2014-10-01T00:00:00Z")
            for house in s.HOUSES:
                t = np.concatenate([start+day*86400+offset+np.arange(160)*8
                                    for day, offset in ((0, 0), (10, 3600), (11, 3600), (12, 3600))])
                y = np.zeros(len(t))
                for j in range(4):
                    y[j*160+30:j*160+37] = 300
                    y[j*160+90:j*160+97] = 300
                artifacts = {}
                for channel in ("aggregate", "microwave"):
                    path = base / f"{house}_{channel}.csv"
                    rows = [{"household_id": house, "timestamp": iso(int(stamp)),
                             ("aggregate_power_w" if channel == "aggregate" else "power_w"):
                             str(float(100+i%1200) if channel == "aggregate" else y[i])}
                            for i, stamp in enumerate(t)]
                    with path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    artifacts[channel] = {"path": path.name, "sha256": sha256(path)}
                manifest["sources"].append({"household_id": house, "start": iso(start),
                    "end": iso(start+14*86400), "artifacts": artifacts})
            mp, run = base / "manifest.json", base / "run"
            write_json(mp, manifest)
            with patch.object(s, "validate_plan"):
                result = s.prepare(run, pp, mp)
            self.assertEqual(len(result["supported_validation_blocks"]), 4)
            with self.assertRaises(DataError):
                s.evaluate(run)
            self.assertFalse((run / "evaluation").exists())
            bad = base / "bad.json"
            write_json(bad, {"execute": False})
            with self.assertRaises(DataError):
                s.train(run, bad)
            self.assertFalse((run / "training_started.json").exists())
            auth = base / "auth.json"
            write_json(auth, {"execute": True, "maximum_fits": 6,
                "registration_sha256": sha256(run / "registration.json"),
                "audit_sha256": sha256(audit), "coordinator_execution_message": "synthetic unittest only"})
            s.train(run, auth)
            s.freeze(run)
            self.assertEqual(s.evaluate(run)["status"], "evaluated_once")
            self.assertEqual(s.verify(run)["prediction_readbacks"], 12)
            self.assertEqual(s.report(run)["status"], "reported")
            for action in (lambda: s.prepare(run, pp, mp), lambda: s.train(run, auth),
                           lambda: s.freeze(run), lambda: s.evaluate(run), lambda: s.report(run)):
                with self.assertRaises(DataError):
                    action()


if __name__ == "__main__":
    unittest.main()
