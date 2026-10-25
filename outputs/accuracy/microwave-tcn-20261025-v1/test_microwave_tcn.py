"""Synthetic tests, zero real fits; no development truth or final households."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from virtual_submeter.models import center_mask_tcn as m
from virtual_submeter.accuracy import microwave_tcn as f, microwave_tcn_execution as e
from virtual_submeter.accuracy.microwave_diagnosis import normalization
from virtual_submeter.data.common import DataError, write_json, sha256, iso

META = {"input_log_mean": 5., "input_log_std": .8, "cpu_threads": 4, "batch_size": 256}
PLAN = {"seeds": [42,1019,2026], "cpu_threads":4, "learning_rate":.001}


def source_rows(p=.2, r=.9, f1=.32):
    return [{"seed": s, "precision": p, "recall": r, "f1": f1,
             "source_gate_passed": True, "degenerate": False} for s in PLAN["seeds"]]


class TCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_unknown_outer_matches_short_including_gradients(self):
        torch.manual_seed(42)
        model = m.CenterMaskTCN()
        x = np.full((3,31), 300., np.float32); x[:,9] = np.nan
        long = np.full((3,127), np.nan, np.float32); long[:,48:79] = x
        short_out = model(m.tensor_input(META,x))
        long_out = model(m.tensor_input(META,long))
        torch.testing.assert_close(short_out, long_out, atol=1e-7, rtol=1e-6)
        short_out.sum().backward()
        grads = [v.grad.clone() for v in model.parameters()]
        model.zero_grad(); long_out.sum().backward()
        for expected, v in zip(grads, model.parameters()):
            torch.testing.assert_close(expected, v.grad, atol=1e-6, rtol=1e-5)

    def test_real_outer_reaches_center(self):
        model = m.CenterMaskTCN()
        with torch.no_grad():
            for v in model.parameters(): v.fill_(.05)
        x = np.full((1,127), 300., np.float32)
        a = model(m.tensor_input(META,x))
        x[:,0] = 3000.
        b = model(m.tensor_input(META,x))
        self.assertGreater(float((b-a).detach()), 0.)

    def test_parameters_and_same_seed_initialization(self):
        a,_ = e.configure(42, PLAN); b,_ = e.configure(42, PLAN)
        self.assertEqual(sum(v.numel() for v in a.parameters()), 5329)
        for x,y in zip(a.parameters(),b.parameters()): self.assertTrue(torch.equal(x,y))
        self.assertEqual(float(a.head[-1].bias.detach()), -3.)
        self.assertFalse(any(isinstance(v,(torch.nn.BatchNorm1d,torch.nn.Dropout)) for v in a.modules()))

    def test_gap_truncates_both_directions(self):
        t = np.arange(160)*8; a = np.full(160,300.,np.float32)
        a[[59,60,99,100]] = np.nan
        d = m.expand_context(t,a,[80],0,1280)
        self.assertTrue(np.isnan(d["x"][0,:44]).all())
        self.assertTrue(np.isnan(d["x"][0,82:]).all())
        self.assertEqual(d["left_stop"][0],"missing_run")
        self.assertEqual(d["right_stop"][0],"missing_run")

    def test_single_missing_not_interpolated_or_barrier(self):
        t = np.arange(160)*8; a = np.full(160,300.,np.float32); a[100] = np.nan
        d = m.expand_context(t,a,[80],0,1280)
        self.assertTrue(np.isnan(d["x"][0,83]))
        self.assertEqual(d["x"][0,84],300.)
        np.testing.assert_array_equal(np.packbits(d["mask"],axis=1),d["mask_identity"])

    def test_time_break_and_partition_boundary(self):
        t = np.arange(160)*8; a = np.full(160,300.,np.float32)
        t[105:] += 80
        d = m.expand_context(t,a,[80],60*8,120*8)
        self.assertEqual(d["left_stop"][0],"boundary")
        self.assertEqual(d["right_stop"][0],"time_break")
        self.assertTrue(np.isnan(d["x"][0,:43]).all())
        self.assertTrue(np.isnan(d["x"][0,88:]).all())

    def test_mask_nan_zero_watts_alignment(self):
        x = np.full((1,31),300.,np.float32); x[0,0] = np.nan; x[0,1] = 0
        z = m.tensor_input(META,x)
        self.assertEqual(float(z[0,0,0]),0.)
        self.assertEqual(float(z[0,1,0]),0.)
        self.assertEqual(float(z[0,1,1]),1.)
        self.assertNotEqual(float(z[0,0,1]),0.)
        with self.assertRaises(DataError): m.tensor_input(META,x,np.ones_like(x,bool))
        x[0,0] = np.inf
        with self.assertRaises(DataError): m.tensor_input(META,x)

    def test_normalization_not_recomputed_from_outer(self):
        core = np.arange(62,dtype=np.float32).reshape(2,31)
        meta = normalization(core)
        expected = copy.deepcopy(meta)
        outer = np.full((2,127),np.nan,np.float32); outer[:,48:79] = core
        m.tensor_input(meta,outer)
        self.assertEqual(meta,expected)

    def test_incomplete_core_is_none_outer_missing_keeps_centers(self):
        t = np.arange(100)*8; a = np.full(100,300.,np.float32); a[0] = np.nan
        rows = [{"timestamp":iso(int(s)),"household_id":"refit_5","power_w":str(v)} for s,v in zip(t,a)]
        # aggregate_arrays uses aggregate_power_w schema; use array context directly below.
        x,c = m.context_at_complete_centers({"window_samples":127}, t,a)
        self.assertNotIn(15,c)
        self.assertIn(16,c)
        self.assertEqual(len(c),69)
        self.assertTrue(np.isnan(x[0,:47]).all())

    def test_ordinary_loader_roundtrip_and_wrong_kind(self):
        from virtual_submeter.models.mask_state_classifier import load_state_model as old_load
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); model = m.CenterMaskTCN()
            torch.save(model.state_dict(),p/"weights.pt")
            meta = {**META,"kind":m.KIND,"truth_threshold_w":200,"window_samples":127,
                    "inference_support":"strict_complete_31","decision_threshold_probability":.5,
                    "weights_sha256":sha256(p/"weights.pt")}
            write_json(p/"model.json",meta)
            loaded, restored = m.load_state_model(p)
            x = np.full((2,127),np.nan,np.float32); x[:,48:79] = 300.
            np.testing.assert_array_equal(m.predict_windows(meta,model,x),m.predict_windows(loaded,restored,x))
            with self.assertRaises(DataError): old_load(p)
            x[0,63] = np.nan
            with self.assertRaises(DataError): m.predict_windows(meta,model,x)

    def test_selection_requires_new_arm_improvement_and_ties(self):
        rows = {a:source_rows() for a in ["partial_mask"]+f.ORDER}
        self.assertEqual(f.select_source(rows,PLAN)["selected_arm"],"partial_mask")
        rows["TCN31"] = source_rows(.21,.9,.33)
        rows["TCN127"] = source_rows(.21,.9,.33)
        self.assertEqual(f.select_source(rows,PLAN)["selected_arm"],"TCN31")
        rows["TCN127"] = source_rows(.22,.9,.34)
        self.assertEqual(f.select_source(rows,PLAN)["selected_arm"],"TCN127")

    def test_selection_low_recall_degenerate_no_candidate(self):
        rows = {a:source_rows() for a in ["partial_mask"]+f.ORDER}
        rows["TCN127"] = source_rows(.9,.84,.5)
        rows["TCN31"] = source_rows(.9,.9,.5)
        rows["TCN31"][0]["degenerate"] = True
        self.assertEqual(f.select_source(rows,PLAN)["selected_arm"],"partial_mask")
        rows["partial_mask"][0]["source_gate_passed"] = False
        self.assertIsNone(f.select_source(rows,PLAN)["selected_arm"])

    def test_authorization_and_freeze_required_before_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            with self.assertRaises(DataError): e.authorize(p,None,"train")
            write_json(p/"registration.json",{})
            auth = p/"auth.json"
            write_json(auth,{"coordinator_execution_message":True,"train":True,"maximum_fits":7,
                             "registration_sha256":sha256(p/"registration.json")})
            with self.assertRaises(DataError): e.authorize(p,auth,"train")
            with patch.object(f,"verify_preparation",return_value=True):
                with self.assertRaises((DataError,OSError)): f.evaluation_guard(p)

    def test_prepare_rejects_development_house_before_file_access(self):
        with patch.object(f,"sha256",side_effect=AssertionError("Unexpected file access")):
            for h in ("refit_17","refit_8","refit_9","refit_20"):
                with self.assertRaises(DataError): f.read_aggregate({"path":"truth_once.npz"},h)

    def test_existing_default_has_no_tcn_import(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("cli.py","models/state_classifier.py"):
            self.assertNotIn("center_mask_tcn",(root/"src/virtual_submeter"/name).read_text(encoding="utf-8"))


if __name__ == "__main__": unittest.main()
