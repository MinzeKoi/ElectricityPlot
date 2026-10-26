import unittest
import numpy as np

from virtual_submeter.accuracy.microwave_source3_background import eligibility, role_values
from virtual_submeter.data.common import DataError


class SourcePoolTests(unittest.TestCase):
    def fixture(self):
        stamps=np.arange(0,1600,8)
        micro={int(t):0. for t in stamps}
        micro[800]=300.
        aggregate={int(t):1000. for t in stamps}
        t=np.array([800,808])
        x=np.full((2,127),1000.,np.float32)
        x[1,0]=np.nan
        d=dict(t=t,x=x,mask=np.isfinite(x),y=np.array([300,0],np.float32))
        return d,micro,aggregate,[(0,0,1600)]

    def test_complete_and_unknown_at_mask_zero(self):
        d,m,a,r=self.fixture()
        e=eligibility(d,m,a,r)
        np.testing.assert_array_equal(e['template'],[True,False])
        np.testing.assert_array_equal(e['background'],[False,True])
        m[304]=np.nan
        e=eligibility(d,m,a,r)
        self.assertFalse(e['template'][0])
        self.assertTrue(e['background'][1])
        self.assertTrue(np.isnan(e['residual'][1,0]))

    def test_unknown_observed_and_negative_residual_reject(self):
        d,m,a,r=self.fixture()
        m[816]=np.nan
        self.assertFalse(eligibility(d,m,a,r)['background'][1])
        m[816]=1200.
        self.assertFalse(eligibility(d,m,a,r)['background'][1])

    def test_full_window_role_not_center_role(self):
        d,m,a,r=self.fixture()
        e=eligibility(d,m,a,[(0,400,1600)])
        self.assertFalse(e['template'][0])

    def test_raw_w_identity_and_center_identity(self):
        d,m,a,r=self.fixture()
        d['x'][0,63]=np.log1p(1000)
        with self.assertRaises(AssertionError): eligibility(d,m,a,r)

    def test_validation_values_never_parsed(self):
        rows=[dict(household_id='refit_5',timestamp='1970-01-01T00:00:00Z',power_w='2'),
              dict(household_id='refit_5',timestamp='1970-01-01T00:00:16Z',power_w='invalid-validation-value')]
        first=role_values(rows,'refit_5',[(0,0,8)])
        rows[1]['power_w']='99999999'
        self.assertEqual(first,role_values(rows,'refit_5',[(0,0,8)]))
        self.assertEqual(first,{0:2.})

    def test_missing_timestamp_not_filled(self):
        d,m,a,r=self.fixture()
        del m[800-63*8]
        self.assertFalse(eligibility(d,m,a,r)['template'][0])


if __name__=='__main__': unittest.main()
