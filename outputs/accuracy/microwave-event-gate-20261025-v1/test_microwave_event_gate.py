"""Small exact array tests; no real households, fitting or parameter updates."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from virtual_submeter.accuracy import microwave_event_gate as e
from virtual_submeter.data.common import DataError, write_json


class GateTests(unittest.TestCase):
    def test_single_max_does_not_propagate(self):
        p = np.full(70,.2)
        p[20] = .9
        h = e.local_max(np.arange(70)*8,p)
        np.testing.assert_array_equal(np.flatnonzero(h == .9),np.arange(5,36))
        self.assertEqual(h[36],.2)

    def test_nan_break(self):
        p = np.array([.9,np.nan,.2,.1])
        h = e.local_max(np.arange(4)*8,p)
        self.assertTrue(np.isnan(h[1]))
        self.assertEqual(h[2],.2)

    def test_timestamp_break(self):
        h = e.local_max(np.array([0,8,24,32]),np.array([.9,.1,.2,.1]))
        np.testing.assert_array_equal(h,[.9,.9,.2,.2])

    def test_block_boundary(self):
        h = e.local_max(np.arange(4)*8,np.array([.9,.1,.2,.1]),block=[4,4,5,5])
        np.testing.assert_array_equal(h,[.9,.9,.2,.2])

    def test_exact_max_against_nested_reference(self):
        rng = np.random.default_rng(18)
        t = np.arange(100)*8
        t[62:] += 8
        p = rng.random(100)
        p[25] = np.nan
        block = np.repeat([4,5],50)
        h = e.local_max(t,p,block=block)
        expected = np.full(100,np.nan)
        for i in range(100):
            if not np.isfinite(p[i]):
                continue
            values = []
            for j in range(max(0,i-15),min(100,i+16)):
                a,b = sorted((i,j))
                if np.isfinite(p[a:b+1]).all() and (np.diff(t[a:b+1])==8).all() and (block[a:b+1]==block[i]).all():
                    values.append(p[j])
            expected[i] = max(values)
        np.testing.assert_array_equal(h,expected)

    def test_low_fail_is_false_negative(self):
        p = np.array([.1,.8,.2,.9])
        h = e.local_max(np.arange(4)*8,p)
        state = e.gate_state(p,h,.5,.9)
        m = e.confusion(state,[True,True,False,False])
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn'],m['n']),(1,1,1,1,4))
        c = e.exact_curve(np.where(p>=.5,h,-1),np.array([True,True,False,False]),np.array([.9]))
        self.assertEqual(int(c['fn'][0]),1)

    def test_missing_stays_missing_low_failure_stays_off(self):
        p = np.array([.1,np.nan,.8])
        h = e.local_max(np.arange(3)*8,p)
        np.testing.assert_array_equal(e.gate_state(p,h,.5,.7),[0,-1,1])

    def test_unknown_label_peak_can_be_feature(self):
        p = np.array([.5,.95,.1])
        h = e.local_max(np.arange(3)*8,p)
        state = e.gate_state(p,h,.4,.9)
        # Only indices 0/2 have labels; unknown index 1 is never counted as off.
        self.assertEqual(e.confusion(state[[0,2]],[True,False])['tp'],1)

    def test_exact_curve_against_brute_force_with_ties(self):
        v = np.array([.1,.1,.5,.7,-1,.9])
        y = np.array([1,0,1,0,1,1],bool)
        thresholds = np.array([.1,.5,.7,.9])
        c = e.exact_curve(v,y,thresholds)
        for i,h in enumerate(thresholds):
            expected = e.confusion((v>=h).astype(np.int8),y)
            for k in ('tp','fp','fn','tn','precision','recall','f1'):
                self.assertEqual(c[k][i],expected[k])

    def test_selection_matches_exhaustive_ranking(self):
        rng = np.random.default_rng(71)
        for _ in range(25):
            blocks = {str(i):dict(values=rng.choice([.1,.3,.5,.8,.9],20),y=rng.random(20)<.4) for i in (4,5)}
            candidates = np.array([.1,.3,.5,.8,.9])
            actual,_ = e.select_threshold(blocks,candidates,.85)
            ranked = []
            for threshold in candidates:
                ms = [e.confusion((d['values']>=threshold).astype(np.int8),d['y']) for d in blocks.values()]
                if all(m['recall']>=.85 and not m['degenerate'] for m in ms):
                    ranked.append((min(m['precision'] for m in ms),np.mean([m['precision'] for m in ms]),np.mean([m['f1'] for m in ms]),threshold))
            expected = max(ranked)[-1] if ranked else None
            self.assertEqual(actual,expected)

    def test_infeasible_does_not_lower_recall(self):
        blocks = {'4':dict(values=np.array([.1,.9]),y=np.array([1,0],bool)),
                  '5':dict(values=np.array([.1,.9]),y=np.array([1,0],bool))}
        h,_ = e.select_threshold(blocks,np.array([.1,.9]),.95)
        self.assertIsNone(h)

    def test_reject_unqualified_recipe(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'recipe.json'
            write_json(p,dict(schema='event_gate_recipe_v1',eligible=False,L=None,H=None))
            with self.assertRaises(DataError):
                e.load_recipe(p)

    def test_no_radius_search(self):
        with self.assertRaises(DataError):
            e.local_max(np.arange(3)*8,np.array([.1,.2,.3]),radius=16)

    def test_nonfinite_eval_cannot_be_silently_filtered(self):
        with self.assertRaises(DataError):
            e.exact_curve(np.array([np.nan,.5]),np.array([1,0]),np.array([.4]))
        with self.assertRaises(DataError):
            e.confusion(np.array([-1,0]),[True,False])

    def test_household_and_utc_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'aggregate.csv'
            p.write_text('household_id,timestamp,aggregate_power_w\nrefit_17,2014-11-23T00:00:00Z,400\n',encoding='utf-8')
            with self.assertRaises(DataError):
                e.csv_rows(p,'refit_5')
        with self.assertRaises(DataError):
            e.align(np.array([0,8,16]),np.array([0,24]))
        np.testing.assert_array_equal(e.align(np.array([0,8,16]),np.array([0,16])),[0,2])

    def test_source_development_isolation_including_hash_reads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'run'
            root.mkdir()
            cache = Path(d)/'truth_once.npz'
            cache.write_bytes(b'no data')
            with e.isolation(root,'source_select',[cache]):
                with self.assertRaises(DataError):
                    e.sha256(cache)
            path = Path(d)/'frozen_test_refit_17_0.npz'
            path.write_bytes(b'no data')
            with e.isolation(root,'prepare',[path]):
                with self.assertRaises(DataError):
                    e.sha256(path)

    def test_unsealed_matrix_rejected_before_truth(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(e,'source_guard',return_value=({}, {}, {})):
                with self.assertRaisesRegex(DataError,'sealed'):
                    e.evaluation_guard(Path(d))

    def test_optimizer_and_backward_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            with e.isolation(Path(d),'synthetic'):
                with self.assertRaises(DataError):
                    torch.tensor(2.,requires_grad=True).backward()
                with self.assertRaises(DataError):
                    torch.optim.SGD([torch.tensor(2.,requires_grad=True)],lr=.1)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'receipt.json'
            e.fresh_json(p,{'ok':True})
            with self.assertRaises(DataError):
                e.fresh_json(p,{'ok':False})

    def test_ordinary_recipe_reads_back_complete_scores(self):
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)/'parent'
            parent.mkdir()
            (parent/'model.json').write_text('{}')
            (parent/'weights.pt').write_bytes(b'frozen synthetic identity')
            recipe = Path(d)/'recipe.json'
            write_json(recipe,dict(schema='event_gate_recipe_v1',eligible=True,radius_samples=15,interval_seconds=8,
                L=.4,H=.8,parent=str(parent),parent_hashes=e.inventory(parent),
                score_semantics='uncalibrated sigmoid state score',algorithm_sha256=e.sha256(Path(e.__file__))))
            t = np.arange(64)*8
            p = np.full(64,.2,dtype=np.float32)
            p[31:33] = [.5,.9]
            p[5] = np.nan
            with patch.object(e,'infer',return_value=(t,p,{})):
                actual = e.ordinary(recipe,[{'aggregate':'synthetic'}])
            np.testing.assert_array_equal(actual['scores'],p)
            np.testing.assert_array_equal(actual['state'],e.gate_state(p,e.local_max(t,p),.4,.8))


if __name__ == '__main__':
    unittest.main()
