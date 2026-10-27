import csv, json, hashlib, sys, datetime, unittest, tempfile
from pathlib import Path
from collections import Counter
import numpy as np
RUN=Path(__file__).resolve().parent; ROOT=Path.cwd().resolve()
import audit

def hashfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):
 with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def independently_check():
 seal=json.loads((RUN/'input_seal.json').read_text(encoding='utf-8')); totals={r['partition']:r for r in readcsv(RUN/'partitions.csv')}; daily=readcsv(RUN/'daily.csv'); events=readcsv(RUN/'fragments.csv'); counts=Counter(); ids=set(); samples={}
 for r in readcsv(RUN/'window_ids.csv'):
  h=r['house']; part=r['source_role']; stamp=audit.ts(r['center_utc']); key=(h,part,stamp); assert key not in ids; ids.add(key); counts[(part,'partial_'+('on' if r['on']=='1' else 'off'))]+=1
  for view in ('strict','full'):
   if r['strict_core31' if view=='strict' else 'full127']=='1':counts[(part,view+'_'+('on' if r['on']=='1' else 'off'))]+=1
  if len(samples.setdefault(part,[]))<12 or stamp%997==0:samples[part].append(r)
 for part,r in totals.items():
  for k in ('partial_on','partial_off','strict_on','strict_off','full_on','full_off'):assert counts[(part,k)]==int(r[k]),(part,k)
  for k in ('grid','aggregate_valid','label_valid','on_centers','partial_on','partial_off','strict_on','strict_off','full_on','full_off','edge_excluded','center_aggregate_missing','center_label_missing','core_missing_excluded'):
   assert sum(int(x[k]) for x in daily if x['partition']==part)==int(r[k]),(part,k)
  es=[e for e in events if e['partition']==part]; assert len(es)==int(r['fragments']);assert sum(int(e['complete']) for e in es)==int(r['complete_fragments']);assert sum(int(e['partial_centers']) for e in es)==int(r['partial_on'])
 for h,details in seal['houses'].items():
  paths=details['paths']; tt,a=audit.loadnorm(ROOT/paths['aggregate'],'aggregate_power_w',h); ty,y=audit.loadnorm(ROOT/paths['microwave'],'power_w',h,details['channel']); assert np.array_equal(tt,ty)
  for p in details['partitions']:
   lo=audit.ts(p['start']);hi=audit.ts(p['end']);
   for r in samples[p['id']]:
    i=int(r['raw_index']); t=int(tt[i]); assert t==audit.ts(r['center_utc']) and lo<=t<hi and t-120>=lo and t+128<=hi; assert bool(float(y[i])>=200)==bool(int(r['on']))
    core=np.isfinite(a[i-15:i+16]); assert core.sum()>=24 and not any(not core[j] and not core[j+1] for j in range(30));assert bool(core.all())==bool(int(r['strict_core31']))
    expected=(t-504>=lo and t+512<=hi and np.isfinite(a[i-63:i+64]).all());assert expected==bool(int(r['full127']))
    for d,k in [(-1,'left_stop'),(1,'right_stop')]:
     miss=0; reason='radius'
     for step in range(1,64):
      j=i+d*step
      if j<0 or j>=len(tt) or not lo<=tt[j]<hi:reason='boundary';break
      if tt[j]-tt[j-d]!=8*d:reason='time_break';break
      miss=0 if np.isfinite(a[j]) else miss+1
      if miss==2:reason='missing_run';break
     assert reason==r[k]
 return len(ids),sum(map(len,samples.values()))
class Boundaries(unittest.TestCase):
 def test_weighted_and_issues(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.csv';p.write_text('Unix,Aggregate,Appliance8,Issues\n0,100,300,0\n4,200,400,1\n8,50,0,0\n16,50,0,0\n',encoding='utf-8'); v,c=audit.rawgrid(p,'Appliance8',0,16);self.assertEqual(v[0,0],150);self.assertTrue(np.isnan(v[1,0]));self.assertEqual(v[1,1],0)
 def test_no_hold_past_eight(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.csv';p.write_text('Unix,Aggregate,Appliance8,Issues\n0,100,300,0\n20,100,0,0\n',encoding='utf-8');v,c=audit.rawgrid(p,'Appliance8',0,24);self.assertTrue(np.isnan(v[0,1]));self.assertTrue(np.isnan(v[0,2]))
 def test_duplicate_timestamps_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.csv';p.write_text('Unix,Aggregate,Appliance8,Issues\n0,1,1,0\n0,1,1,0\n',encoding='utf-8');self.assertRaises(AssertionError,audit.rawgrid,p,'Appliance8',0,8)
 def test_role_bounds_and_groups(self):
  n,s=independently_check();self.assertGreater(n,0);self.assertGreater(s,0);print('Independent window checks:',n,'sampled scalar checks:',s)
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Boundaries);assert suite.countTestCases()>0; result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not result.wasSuccessful())
