import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from virtual_submeter.accuracy import microwave_loss_correction as f


def fixture():
    # Unequal days, unequal event counts and lengths, with a reused event id in another household.
    return dict(y=np.array([300]*7+[0]*5,dtype=np.float32),house=np.array([0]*5+[1]*7),
                t=np.array([0,8,16,24,86400,0,8,16,24,32,40,48]),event=np.array([0,0,0,1,2,0,0,-1,-1,-1,-1,-1]))


class LossCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.p=json.loads((f.ROOT/'configs/microwave_loss_correction_20261025.json').read_text(encoding='utf-8'))
        self.d=fixture()

    def test_probability_and_expected_bce(self):
        prob,w,_=f.probabilities(self.d,self.p)
        self.assertAlmostEqual(prob.sum(),1)
        self.assertAlmostEqual(prob[:7].sum(),.0625)
        self.assertNotEqual(prob[0],prob[3])
        logits=torch.linspace(-3,2,12,dtype=torch.float64)
        y=torch.tensor(self.d['y']>=200,dtype=torch.float64)
        weighted=f.losses(logits,y,f.ARMS[0],torch.tensor(w))
        target=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(50.),reduction='mean')
        self.assertAlmostEqual(float((torch.tensor(prob)*weighted).sum()),float(target),places=12)
        unit=f.losses(logits,y,f.ARMS[1],torch.tensor(w))
        torch.testing.assert_close(unit,torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none'))

    def test_identity_missing_and_finite(self):
        prob,w,_=f.probabilities(self.d,self.p)
        self.assertTrue(np.isfinite(w).all() and (w>0).all())
        order=np.arange(12)[::-1]
        shuffled={k:v[order] for k,v in self.d.items()}
        np.testing.assert_array_equal(f.probabilities(shuffled,self.p)[0],prob[order])
        for key,value in [('y',np.nan),('event',-1)]:
            bad=copy.deepcopy(self.d); bad[key][0]=value
            with self.assertRaises(ValueError): f.probabilities(bad,self.p)
        bad=copy.deepcopy(self.d);bad['t'][1]=bad['t'][0]
        with self.assertRaises(ValueError): f.probabilities(bad,self.p)

    def test_complete_stream_and_initialization(self):
        a=f.BatchStream(self.d,'event',42,self.p)
        b=f.BatchStream(self.d,'event',42,self.p)
        for _ in range(3528):
            aa,bb=a.next(),b.next()
            np.testing.assert_array_equal(aa,bb)
            self.assertEqual(int((self.d['y'][aa]>=200).sum()),16)
        for arm in f.ARMS:
            m,_=f.parent.configure(42,self.p)
            if arm==f.ARMS[0]: first={k:v.clone() for k,v in m.state_dict().items()}
            else:
                for k,v in m.state_dict().items(): torch.testing.assert_close(v,first[k],rtol=0,atol=0)

    def test_both_losses_fit_validate_save_load(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)
            d=self.d.copy();d['x']=np.ones((12,127),np.float32)*100;d['mask']=np.ones((12,127),bool)
            _,w,_=f.probabilities(d,self.p)
            np.savez_compressed(run/'weights.npz',importance=w)
            stream=f.BatchStream(d,'event',42,self.p)
            np.savez_compressed(run/'sampling_42.npz',indices=np.stack([stream.next()]))
            m,_=f.parent.configure(42,self.p);torch.save(m.state_dict(),run/'initial_42.pt')
            (run/'registration.json').write_text('{}')
            p={**self.p,'checkpoints':1,'steps_per_checkpoint':1}
            meta=dict(window_samples=127,input_log_mean=1.,input_log_std=1.,cpu_threads=4,batch_size=256)
            for arm in f.ARMS:
                row=f.fit_one(run,arm,42,p,d,{'block_0':d,'block_1':d},meta)
                path=run/f'models/{arm}/seed_42'
                loaded,model=f.load_state_model(path)
                self.assertEqual(loaded['window_samples'],127)
                self.assertEqual(len(f.predict_windows(loaded,model,d['x'])),12)
                self.assertTrue((path/'epoch_1.pt').exists())

    def test_selection_matrix_recall_degenerate(self):
        rows={a:[dict(seed=s,precision=.2 if a=='TCN127' else .3,recall=.9,f1=.35 if a=='TCN127' else .4,source_gate_passed=True,degenerate=False) for s in f.SEEDS] for a in f.ORDER}
        self.assertEqual(f.select_source(rows,self.p)['selected_arm'],f.ARMS[0])
        for a in f.ARMS: rows[a][0]['recall']=.1
        self.assertEqual(f.select_source(rows,self.p)['selected_arm'],'TCN127')
        rows['TCN127'][0]['degenerate']=True
        self.assertIsNone(f.select_source(rows,self.p)['selected_arm'])
        del rows[f.ARMS[1]]
        with self.assertRaises(ValueError): f.select_source(rows,self.p)
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)
            with self.assertRaises(ValueError): f.verify_training(run,predictions=True)

    def test_development_gate_protects_recall(self):
        old=[dict(seed=s,precision=.04,recall=.9,f1=.075,fp=100,degenerate=False) for s in f.SEEDS]
        new=[dict(seed=s,precision=.10,recall=.9,f1=.18,fp=60,degenerate=False) for s in f.SEEDS]
        self.assertTrue(f.development_gate(old,new,self.p))
        new[0]['recall']=.84
        self.assertFalse(f.development_gate(old,new,self.p))
        new[0]['recall']=.9;new[0]['degenerate']=True
        self.assertFalse(f.development_gate(old,new,self.p))


if __name__=='__main__': unittest.main()
