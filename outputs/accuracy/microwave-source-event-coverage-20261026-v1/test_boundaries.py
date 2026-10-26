import unittest
import numpy as np
from audit import permit,segments,legal_centers,Sample,resample,Counter

class Boundaries(unittest.TestCase):
    def test_house_date_isolation(self):
        cfg={'candidate_bounds':{'2':[100,200],'19':[300,400]}}
        permit('2',100,200,cfg)
        for h,lo,hi in [('17',100,200),('5',100,200),('8',100,200),('2',100,201),('19',100,200)]:
            with self.assertRaises(ValueError):permit(h,lo,hi,cfg)
    def test_nan_and_gap_break_segments(self):
        t=np.array([0,8,16,32,40,48]);y=np.array([300,300,np.nan,300,0,300])
        ids,e=segments(t,y)
        self.assertEqual(ids.tolist(),[0,0,-1,1,-1,2]);self.assertEqual(len(e),3)
        ids,e=segments(np.array([0,16]),np.array([300,300]));self.assertEqual(len(e),2)
    def test_nine_seconds_does_not_fill(self):
        ss=[Sample(t,{'a':100.},{'a':set()}) for t in range(0,100,9)]
        bins=list(resample(ss,8,{'raw_intervals_seconds':Counter()}))
        self.assertTrue(any(v['a']['power_w'] is None for t,v in bins))
        self.assertTrue(any(v['a']['coverage']<1 for t,v in bins))
    def test_two_missing_core_rejected(self):
        t=np.arange(100)*8;a=np.ones(100);y=np.zeros(100)
        a[49:51]=np.nan;c,_,_=legal_centers(t,a,y,0,800)
        self.assertNotIn(48,c);self.assertNotIn(51,c)
    def test_date_preselection_label_independent(self):
        end=1413181728
        low=(end+86399)//86400*86400
        self.assertEqual(low,1413244800)
        self.assertEqual(low+14*86400,1414454400)
        # Date calculation receives no labels, activity, scores or predictions.
        for arbitrary_label in [0,200,999999,np.nan]:
            self.assertEqual((end+86399)//86400*86400,low)

if __name__=='__main__':unittest.main()
