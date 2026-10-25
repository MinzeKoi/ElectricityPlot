"""Missing-context controls, synthetic fixtures only; no optimizer steps or real fits."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from virtual_submeter.accuracy import microwave_mask as f
from virtual_submeter.accuracy import microwave_mask_execution as e
from virtual_submeter.accuracy.microwave_diagnosis import normalization
from virtual_submeter.data.common import DataError, iso, read_json, sha256, write_json
from virtual_submeter.models import mask_state_classifier as m
from virtual_submeter.models.state_classifier import StateClassifier, load_state_model as old_load, predict_state_rows as old_predict

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    t = np.arange(100, dtype=np.int64)*8
    a = np.arange(100, dtype=np.float32)+100
    y = np.zeros(100, np.float32)
    y[40:44] = 300
    return t, a, y


def candidate(t, a, y, house="refit_2"):
    return f.candidate_data(t, a, y, 0, 14*86400, house)


def rows(precision=.1, recall=.9, f1=.18):
    return [{"seed": s, "precision": precision, "recall": recall, "f1": f1,
             "source_gate_passed": True, "degenerate": False, "fp": 100} for s in [42, 1019, 2026]]


class MaskTests(unittest.TestCase):
    def test_source_house_bounds_and_long_gap(self):
        t, a, y = fixture()
        for house in ("refit_5", "refit_17", "refit_8", "refit_9", "refit_20"):
            with self.assertRaises(DataError):
                candidate(t, a, y, house)
        with self.assertRaises(DataError):
            f.candidate_data(t, a, y, 0, 86400, "refit_2")
        t[50:] += 80
        _, d, _, _ = candidate(t, a, y)
        self.assertFalse(np.any((d["raw_index"] >= 35) & (d["raw_index"] <= 64)))

    def test_missing_center_label_never_becomes_off(self):
        t, a, y = fixture()
        y[41] = np.nan
        a[60] = np.nan
        _, d, stats, _ = candidate(t, a, y)
        self.assertNotIn(41, d["raw_index"])
        self.assertNotIn(60, d["raw_index"])
        self.assertTrue(np.isfinite(d["y"]).all())
        self.assertGreater(stats["rejections_and_acceptances"]["center_label_missing"], 0)

    def test_partial_adds_event_without_filling_and_mask_alignment(self):
        t, a, y = fixture()
        a[35] = np.nan
        strict, partial, stats, _ = candidate(t, a, y)
        self.assertEqual(stats["newly_supported_event_ids"], [0])
        self.assertGreater(stats["added_positive_centers"], 0)
        self.assertTrue(strict["mask"].all())
        self.assertTrue(np.array_equal(partial["mask"], np.isfinite(partial["x"])))
        for x, mask, c in zip(partial["x"], partial["mask"], partial["raw_index"]):
            np.testing.assert_array_equal(x, a[c-15:c+16])
            np.testing.assert_array_equal(mask, np.isfinite(a[c-15:c+16]))
        self.assertTrue(np.isnan(a[35]))
        self.assertEqual(sum(stats["rejections_and_acceptances"].values()), len(t))

    def test_fixed_24_point_and_single_grid_run_limit(self):
        t, a, y = fixture()
        a[[30, 32, 34, 36, 38, 40, 42]] = np.nan
        _, d, _, _ = candidate(t, a, y)
        self.assertIn(45, d["raw_index"])
        a[44] = np.nan
        _, d, _, _ = candidate(t, a, y)
        self.assertNotIn(45, d["raw_index"])
        a = np.full(100, 500, np.float32)
        a[30:32] = np.nan
        _, d, _, _ = candidate(t, a, y)
        self.assertFalse(np.any((d["raw_index"] >= 16) & (d["raw_index"] <= 45)))

    def test_standardized_sentinel_does_not_change_normalization(self):
        x = np.arange(62, dtype=np.float32).reshape(2,31)+1
        meta = normalization(x)
        partial = x.copy()
        partial[0,2] = np.nan
        before = copy.deepcopy(meta)
        z = m.tensor_input(meta, partial).numpy()
        self.assertEqual(meta, before)
        self.assertEqual(z[0,0,2], 0)
        self.assertEqual(z[0,1,2], 0)
        self.assertEqual(z[0,1,3], 1)
        self.assertAlmostEqual(float(z[0,0,3]), float((np.log1p(x[0,3])-meta["input_log_mean"])/meta["input_log_std"]), places=6)
        with self.assertRaises(DataError):
            m.tensor_input(meta, partial, np.ones_like(partial, bool))
        with self.assertRaises(DataError):
            m.predict_windows(meta, m.MaskStateClassifier(), partial)

    def test_pair_initialization_and_old_default_shape(self):
        p = read_json(ROOT/"configs/microwave_mask_20261025.json")
        a, _ = e.configure(42,p)
        b, _ = e.configure(42,p)
        for k,v in a.state_dict().items():
            self.assertTrue(torch.equal(v,b.state_dict()[k]))
        self.assertEqual(a.features[0].in_channels, 2)
        old = StateClassifier()
        self.assertEqual(old.features[0].in_channels, 1)
        self.assertEqual(sum(v.numel() for v in a.parameters())-sum(v.numel() for v in old.parameters()),40)

    def test_ordinary_readback_preserves_strict_missing_semantics(self):
        meta = {**normalization(np.full((1,31),500,np.float32)), "kind": m.KIND,
                "truth_threshold_w":200,"window_samples":31,"inference_support":"strict_complete_31",
                "decision_threshold_probability":.5}
        raw = [{"household_id":"refit_2", "timestamp":iso(i*8),
                "aggregate_power_w":"" if i == 20 else "500", "coverage":0 if i == 20 else 1} for i in range(80)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            torch.save(m.MaskStateClassifier().state_dict(),path/"weights.pt")
            meta["weights_sha256"] = sha256(path/"weights.pt")
            write_json(path/"model.json",meta)
            mm, model = m.load_state_model(path)
            predicted = m.predict_state_rows(mm,model,raw)
            legacy = old_predict(meta,StateClassifier(),raw)
            self.assertEqual([r["state"] is None for r in predicted], [r["state"] is None for r in legacy])
            self.assertIsNone(predicted[20]["state"])
            self.assertIsNotNone(predicted[50]["state"])
            with self.assertRaises(DataError):
                old_load(path)

    def test_source_selection_rejects_low_recall_degenerate_incomplete_and_tie(self):
        p = read_json(ROOT/"configs/microwave_mask_20261025.json")
        rr = {"strict_mask":rows(),"partial_mask":rows()}
        self.assertEqual(f.select_source(rr,p)["selected_arm"],"strict_mask")
        rr["partial_mask"] = rows(.12,.9,.2)
        self.assertEqual(f.select_source(rr,p)["selected_arm"],"partial_mask")
        rr["partial_mask"][0]["recall"] = .84
        self.assertEqual(f.select_source(rr,p)["selected_arm"],"strict_mask")
        rr["strict_mask"][0]["degenerate"] = True
        self.assertIsNone(f.select_source(rr,p)["selected_arm"])
        rr["strict_mask"].pop()
        with self.assertRaises(DataError):
            f.select_source(rr,p)

    def test_common_threshold_rejects_low_recall_and_all_on(self):
        p = read_json(ROOT/"configs/microwave_mask_20261025.json")
        d = {"y":np.array([300]*5+[0]*20,np.float32), "event":np.array([0,0,1,1,1]+[-1]*20)}
        blocks = {f"block_{i}":d for i in range(6)}
        self.assertIsNone(f.select_checkpoint({k:np.ones(25) for k in blocks},blocks,p))
        scores = {k:np.array([.9]*5+[.1]*20) for k in blocks}
        point = f.select_checkpoint(scores,blocks,p)
        self.assertEqual(point["mean_recall"],1)
        self.assertEqual(point["supported_blocks"],list(blocks))

    def test_no_authorization_no_training_and_no_truth_io(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            with self.assertRaises(DataError):
                e.authorize(path,None,"train")
            with patch.object(e,"authorize",side_effect=DataError("missing coordinator")), patch.object(e,"registered"), patch.object(e,"sha256") as hashing:
                with self.assertRaises(DataError):
                    e.evaluate(path)
                hashing.assert_not_called()
            with patch.object(f,"verify_preparation"), patch.object(f,"read_json",return_value={"preparation_passed":False,"fits":0}):
                with self.assertRaises(DataError):
                    f.evaluation_guard(path)

    def test_complete_freeze_matrix_required(self):
        p = read_json(ROOT/"configs/microwave_mask_20261025.json")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            write_json(path/"matrix.json",{"preparation_passed":True,"fits":6,"seeds":p["seeds"]})
            write_json(path/"training_complete.json",{"completed_models":["strict_mask/42"],"fits":1})
            write_json(path/"freeze.json",{"predictions":["strict_mask/42"]})
            with patch.object(f,"verify_preparation"):
                with self.assertRaises(DataError):
                    f.evaluation_guard(path)


if __name__ == "__main__":
    unittest.main()
