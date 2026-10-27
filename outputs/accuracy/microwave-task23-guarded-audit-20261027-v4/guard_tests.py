import unittest, pathlib, socket, importlib
import numpy as np
import independent_check as ic
ROOT=pathlib.Path.cwd().resolve();RUN=ROOT/'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4'
class Tests(unittest.TestCase):
 def test_weighted_issue_and_gap(self):
  p=RUN/'fixture_weighted.csv'
  with p.open('x',encoding='utf-8') as f:f.write('Unix,Aggregate,Appliance8,Issues\n0,100,300,0\n4,200,400,1\n8,50,0,0\n16,50,0,0\n')
  a,y=ic.recompute(p,'Appliance8',0,16);self.assertEqual(a[0],150);self.assertTrue(np.isnan(y[0]));self.assertEqual(y[1],0)
  p=RUN/'fixture_gap.csv'
  with p.open('x',encoding='utf-8') as f:f.write('Unix,Aggregate,Appliance8,Issues\n0,100,300,0\n20,100,0,0\n')
  a,y=ic.recompute(p,'Appliance8',0,24);self.assertTrue(np.isnan(a[1]));self.assertTrue(np.isnan(a[2]))
 def test_core_missing_rules(self):
  t=np.arange(160)*8;y=np.zeros(160);a=np.ones(160)
  def mask():return ic.support(t,a,y,0,1280)[0]
  a[80]=np.nan;self.assertFalse(mask()[80]);a[80]=1
  a[74:76]=np.nan;self.assertFalse(mask()[80]);a[74:76]=1
  a[[65,67,69,71,73,75,77]]=np.nan;self.assertTrue(mask()[80]);a[79]=np.nan;self.assertFalse(mask()[80])
 def test_partition_and_gap(self):
  a=np.ones(160);y=np.zeros(160);t=np.arange(160)*8
  p,_,f,*_=ic.support(t,a,y,0,1280);self.assertFalse(p[14]);self.assertTrue(p[15]);self.assertFalse(f[62]);self.assertTrue(f[63])
  t2=np.arange(160)*8+4880;p2,_,f2,*_=ic.support(t2,a,y,4880,6160);self.assertFalse(p2[14]);self.assertTrue(p2[15]);self.assertFalse(f2[62])
 def test_extension_second_missing(self):
  a=np.ones(160);a[100:102]=np.nan;t=np.arange(160)*8;y=np.zeros(160);p,s,f,on,ids,b,e,complete,c,st=ic.support(t,a,y,0,1280);k=np.flatnonzero(c==80)[0];self.assertEqual(st[1][k],'missing_run');self.assertFalse(f[80])
 def test_forbidden_files_and_stat(self):
  for house in ('17','8','9','20','16'):
   p=ROOT/f'data/raw/refit_complete_20261022/households/CLEAN_House{house}.csv'
   with self.assertRaises(self.guard_error):p.open('rb')
   with self.assertRaises(self.guard_error):p.stat()
  with self.assertRaises(self.guard_error):(ROOT/'outputs/accuracy/data-expansion-20261024-v1/models/C/seed_1/packed/model.json').open('rb')
 def test_network(self):
  with self.assertRaises(self.guard_error):socket.socket()
 def test_training_optimizer_backprop_model(self):
  for module in ('torch','torch.optim','torch.autograd','virtual_submeter.models.center_mask_tcn'):
   with self.assertRaises(self.guard_error):importlib.import_module(module)
 def test_write_protection(self):
  with self.assertRaises(self.guard_error):(RUN/'audit.py').open('w')
  with self.assertRaises(self.guard_error):(ROOT/'unapproved_output').open('w')
def run(guard_error):
 Tests.guard_error=guard_error;suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);n=suite.countTestCases();assert n>0;result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful();print('Nonzero tests:',n,'OK')
