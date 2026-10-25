"""Audit accounting and conservative observation contract checks (no model fits)."""
from pathlib import Path
import sys
from collections import Counter
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_microwave_data import Sample,resample,window_masks,event_runs,block_support

class MicrowaveAuditTest(unittest.TestCase):
    def test_nine_second_cadence_does_not_fabricate_coverage(self):
        samples=[Sample(t,{'aggregate':100.},{'aggregate':set()}) for t in range(0,361,9)]
        bins=list(resample(samples,8,{'raw_intervals_seconds':Counter()}))
        self.assertTrue(any(v['aggregate']['coverage']==.875 for _,v in bins))
        self.assertTrue(all(v['aggregate']['power_w'] is None for _,v in bins if v['aggregate']['coverage']<1))

    def test_exclusive_accounting_distinguishes_context_and_label_loss(self):
        t=np.arange(100)*8;agg=np.ones(100);y=np.zeros(100)
        agg[40]=np.nan;y[75]=np.nan
        context,usable,loss=window_masks(t,agg,y)
        self.assertEqual(loss['edge'].sum(),30)
        self.assertEqual(loss['aggregate_center_missing'].sum(),1)
        self.assertEqual(loss['aggregate_context_missing'].sum(),30)
        self.assertEqual(loss['label_center_missing'].sum(),1)
        self.assertEqual(usable.sum(),38)
        self.assertTrue(context[75])

    def test_label_missing_never_removes_aggregate_context(self):
        t=np.arange(100)*8;agg=np.ones(100)
        c1,u1,_=window_masks(t,agg,np.zeros(100))
        c2,u2,_=window_masks(t,agg,np.full(100,np.nan))
        np.testing.assert_array_equal(c1,c2)
        self.assertEqual(u1.sum(),70);self.assertEqual(u2.sum(),0)

    def test_whole_crossing_event_excluded_and_context_inside_block(self):
        t=np.arange(200)*8;y=np.zeros(200);y[60:100]=500
        events=event_runs(t,y);usable=np.ones(200,bool)
        keep,s=block_support(t,y,usable,events,0,800)
        self.assertTrue(keep[60:100].any())
        self.assertEqual(s['cross_boundary_event_ids_excluded'],[])
        # End exactly on boundary belongs to this block but context removes last centers.
        self.assertEqual(s['complete_observed_segments'],0)
        keep,s=block_support(t,y,usable,events,0,720)
        self.assertFalse(keep[60:100].any())
        self.assertEqual(s['cross_boundary_event_ids_excluded'],[0])
        self.assertTrue(np.all(t[keep]+128<=720))

    def test_observation_hole_breaks_threshold_segment_and_completeness(self):
        t=np.arange(200)*8;y=np.zeros(200);y[60:70]=500;y[65]=np.nan
        events=event_runs(t,y)
        self.assertEqual(len(events),2)
        self.assertFalse(any(e['observed_off_boundaries'] for e in events))

if __name__=='__main__':unittest.main()
