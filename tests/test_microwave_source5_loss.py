"""Synthetic-only guards and real fit_one call with a non-updating optimizer spy."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from virtual_submeter.accuracy import microwave_source5_loss as e
from virtual_submeter.data.common import DataError,read_json,write_json,sha256


def rows(p=.2,r=.9,f=.3):
    return [dict(seed=s,precision=p,recall=r,f1=f,degenerate=False,source_gate_passed=True) for s in e.SEEDS]


class Source5LossTests(unittest.TestCase):
    def test_config_disabled_budget_fixed(self):
        p = read_json(e.ROOT/'configs/microwave_source5_loss_20261025.json')
        e.validate_config(p)
        for key,value in [('authorized',True),('maximum_new_fits',4),('pos_weight',50),('radius_samples',16)]:
            q = copy.deepcopy(p)
            q[key] = value
            with self.assertRaises(DataError): e.validate_config(q)

    def test_authorization_missing_before_any_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/'identity.json',dict(source_inputs={}))
            with patch.object(e.temporal,'fit_one',side_effect=AssertionError('fit called')):
                with self.assertRaisesRegex(DataError,'NOT authorized'): e.train(root)
            self.assertFalse((root/'training_started.json').exists())

    def test_receipt_cannot_reuse_prior_budget_or_wrong_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_json(run/'registration.json',{})
            good = dict(authorized=True,approval_origin='human',additional_new_fits=3,maximum_updates=10584,
                experiment='microwave_source5_loss_v1',registration_sha256=sha256(run/'registration.json'),
                human_authorization_quote='SYNTHETIC FIXTURE ONLY',seeds=e.SEEDS)
            path = run/'fixture.json'
            for key,value in [('additional_new_fits',0),('approval_origin','coordinator'),('experiment','task10'),('registration_sha256','wrong'),('maximum_updates',10585),('human_authorization_quote','')]:
                write_json(path,{**good,key:value})
                with self.assertRaises(DataError): e.authorize(run,path)
            write_json(path,good)
            self.assertEqual(e.authorize(run,path),good)

    def test_new_entry_passes_only_explicit_unit_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run,parent = root/'run',root/'parent'
            run.mkdir()
            (parent/'validation').mkdir(parents=True)
            write_json(run/'registration.json',{})
            write_json(run/'parent_training_plan.json',{})
            write_json(parent/'normalization.json',{})
            np.savez(parent/'train.npz',**{k:np.zeros(1) for k in e.TRAIN_FIELDS})
            for i in (4,5): np.savez(parent/f'validation/block_{i}.npz',**{k:np.zeros(1) for k in e.VAL_FIELDS})
            for seed in e.SEEDS:
                np.savez(parent/f'sampling_{seed}.npz',indices=np.zeros((1,1),int))
                torch.save({'fixture':torch.ones(1)},parent/f'initial_{seed}.pt')
            with patch.object(e,'authorize',return_value={'synthetic':True}),patch.object(e,'registered',return_value=({},dict(parent=str(parent)))),patch.object(e.temporal,'fit_one',return_value={'synthetic':True}) as fit:
                e.train.__wrapped__(run,root/'synthetic-never-used-receipt.json')
            self.assertEqual(fit.call_count,3)
            for call in fit.call_args_list:
                self.assertEqual(call.kwargs,dict(pos_weight=1.,recipe='unit_BCE',arm='unit_BCE'))

    def test_final_house_read_block_even_if_allowlisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = root/'source_train_refit_8_0.npz'
            hidden.write_bytes(b'synthetic sentinel')
            with e.access_guard(root,'freeze',[hidden]):
                with self.assertRaises(DataError): hidden.read_bytes()

    def test_prepare_blocks_model_optimizer_backward_and_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = root/'evaluation/truth_once.npz'
            hidden.parent.mkdir()
            hidden.write_bytes(b'synthetic hidden sentinel')
            with e.access_guard(root,'prepare',[hidden]) as access:
                with self.assertRaises(DataError): e.CenterMaskTCN()
                with self.assertRaises(DataError): torch.optim.Adam([torch.nn.Parameter(torch.ones(1))])
                with self.assertRaises(DataError): torch.tensor(1.,requires_grad=True).backward()
                with self.assertRaises(DataError): np.load(hidden)
            self.assertEqual(len(access['forbidden']),1)

    def test_npz_explicit_key_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'source.npz'
            np.savez(path,t=np.arange(3),forbidden_object=np.array([{}],object))
            self.assertEqual(list(e.npz(path,['t'])),['t'])
            np.testing.assert_array_equal(e.npz(path,['t'])['t'],np.arange(3))

    def test_qualification_and_frozen_tie_order(self):
        old,new = rows(),rows(p=.21,r=.86,f=.31)
        self.assertTrue(e.qualified(old,new))
        self.assertFalse(e.qualified(old,new[:2]))
        for key,value in [('recall',.84),('degenerate',True),('source_gate_passed',False)]:
            bad = copy.deepcopy(new)
            bad[0][key] = value
            self.assertFalse(e.qualified(old,bad))
        self.assertFalse(e.qualified(old,rows(p=.3,f=.29)))
        self.assertFalse(e.qualified(old,rows(p=.2,f=.31)))
        families = {a:rows() for a in e.FAMILIES}
        eligible = {a:True for a in e.FAMILIES}
        for a in e.FAMILIES:
            self.assertEqual(e.select_families(families,eligible),a)
            eligible[a] = False

    def test_gate_one_max_no_label_based_segments_low_is_fn(self):
        t = np.arange(40)*8
        p = np.full(40,.1)
        p[0] = .99
        h = e.gate.local_max(t,p)
        self.assertEqual(h[15],.99)
        self.assertEqual(h[16],.1)  # No iterative propagation.
        p[10] = np.nan
        h = e.gate.local_max(t,p)
        self.assertEqual(h[11],.1)
        state = e.gate.gate_state(p,h,.2,.9)
        m = e.gate.confusion(state[[1,2]],np.array([True,True]))
        self.assertEqual(m['fn'],2)
        self.assertEqual(state[10],-1)

    def test_gate_calibration_same_L95_H85_calls(self):
        p = np.array([.2,.8,.9])
        d = dict(p=p,h=np.array([.9,.9,.9]),ix=np.arange(3),y=np.array([False,True,True]))
        with patch.object(e.gate,'select_threshold',wraps=e.gate.select_threshold) as select:
            low,high = e.calibrate({'4':d,'5':d})
            self.assertIsNotNone(low)
            self.assertIsNotNone(high)
            self.assertEqual([c.args[2] for c in select.call_args_list],[.95,.85])
        with patch.object(e.gate,'select_threshold',return_value=(None,{})) as select:
            self.assertEqual(e.calibrate({'4':d}), (None,None))
            self.assertEqual(select.call_count,1)

    def test_shared_fit_default50_explicit50_unit_changes_loss_gradient(self):
        # Executes the actual loop, backward, serialization and source choice.
        # Spy.step records gradients without changing any parameter: zero updates.
        class Scalar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor(.1))
            def forward(self,x): return self.w.expand(len(x))
        class NoUpdate:
            def __init__(self,model): self.model,self.grads = model,[]
            def zero_grad(self,**kwargs): self.model.zero_grad(**kwargs)
            def step(self): self.grads.append(float(self.model.w.grad))
        p = {**read_json(e.temporal.f.PLAN),'checkpoints':1,'steps_per_checkpoint':1}
        x = np.ones((64,127),np.float32)
        d = dict(x=x,mask=np.isfinite(x),y=np.r_[np.full(16,300),np.zeros(48)],t=np.arange(64)*8,house=np.zeros(64,int),event=np.zeros(64,int))
        blocks = {f'block_{i}':dict(x=x[:20],y=np.r_[np.full(6,300),np.zeros(14)],t=lo+120+np.arange(20)*8,event=np.r_[0,0,0,1,1,1,np.full(14,-1)]) for i,lo,hi in e.temporal.f.VALIDATION}
        ix = np.r_[np.arange(16),np.tile(np.arange(16,64),5)][None,:]
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for name,kwargs in [('default',{}),('explicit',dict(pos_weight=50.,recipe='control_BCE50')),('unit',dict(pos_weight=1.,recipe='unit_BCE',arm='unit_BCE'))]:
                net = Scalar()
                opt = NoUpdate(net)
                dest = Path(tmp)/name
                with patch.object(e.temporal.parent,'configure',return_value=(net,opt)),patch.object(e.temporal,'predict_windows',return_value=np.r_[np.full(6,.9),np.full(14,.1)]):
                    e.temporal.fit_one(dest,42,p,d,blocks,dict(input_log_mean=0.,input_log_std=1.),ix,{'w':torch.tensor(.1)},'synthetic',**kwargs)
                with (dest/'batches.csv').open(encoding='utf-8') as stream: batch = list(__import__('csv').DictReader(stream))[0]
                results.append((float(batch['loss']),opt.grads))
                self.assertEqual(float(net.w.detach()),float(torch.tensor(.1)))
                self.assertEqual(read_json(dest/'model.json')['recipe'],kwargs.get('recipe','control_BCE50'))
            self.assertEqual(results[0],results[1])
            self.assertNotEqual(results[0][0],results[2][0])
            self.assertNotEqual(results[0][1],results[2][1])

    def test_acceptance_is_selected_only_and_thresholds_not_lowered(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_json(run/'source_selection.json',dict(selected='old_point'))
            base = [dict(seed=s,arm='W50',precision=.07,recall=.9,f1=.12,fp=100,degenerate=False) for s in e.SEEDS]
            a = [{**r,'arm':'A'} for r in base]
            new = [dict(seed=s,arm='old_point',precision=.1,recall=.86,f1=.15,fp=60,degenerate=False) for s in e.SEEDS]
            m = dict(rows=a+base+new,supports=[dict(seed=s,n=137523) for s in e.SEEDS])
            self.assertTrue(e.summary(run,m)['development_acceptance_passed'])
            new[0]['recall'] = .849
            self.assertFalse(e.summary(run,m)['development_acceptance_passed'])
            new[0]['recall'] = .86
            for r in new: r['fp'] = 71
            self.assertFalse(e.summary(run,m)['development_acceptance_passed'])

    def test_new_freeze_evaluate_bridge_on_synthetic_137523_grid(self):
        # All files are generated in a temporary directory. No project data reads.
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run,parent,old_gate,base = [root/p for p in ('run','parent','oldgate','base')]
            for p in (run,parent,old_gate,base/'protocol'): p.mkdir(parents=True)
            n = 137523
            t = e.utc('2020-01-01T00:00:00Z')+np.arange(n)*8
            y = np.zeros(n)
            y[:383] = 300
            p = np.where(y>=200,.9,.1).astype(np.float32)
            state = (p>=.5).astype(np.int8)
            raw = np.zeros((n,5),np.float32)
            raw[:,1] = y
            aggregate,truth = base/'protocol/aggregate.csv',base/'protocol/microwave.csv'
            for path,channel,values in ((aggregate,'aggregate',np.full(n,1000)),(truth,'microwave',y)):
                with path.open('w',encoding='utf-8',newline='') as stream:
                    writer = csv.DictWriter(stream,fieldnames=['household_id','timestamp','power_w'])
                    writer.writeheader()
                    writer.writerows(dict(household_id='refit_17',timestamp=e.iso(int(tt)),power_w=float(v)) for tt,v in zip(t,values))
            write_json(parent/'baseline_identity.json',dict(reference_run=str(base),development_partition=dict(artifacts=dict(aggregate=dict(path='aggregate.csv')))))
            (old_gate/'predictions').mkdir()
            for seed in e.SEEDS:
                for folder in ('reference/predictions','predictions/W50'):
                    dest = base/folder/f'seed_{seed}'
                    dest.mkdir(parents=True)
                    np.savez(dest/e.gate.PARTITION,stamps=t,raw=raw)
                for arm in ('parent','gate'): np.savez(old_gate/f'predictions/{arm}_{seed}.npz',t=t,scores=p,state=state,raw=raw)
            s = dict(selected='old_gate',eligibility={a:True for a in e.FAMILIES},recipes=[dict(seed=seed,seed_feasible=True) for seed in e.SEEDS])
            dev = {str(path):sha256(path) for path in [aggregate,*base.rglob('*.npz'),*old_gate.rglob('*.npz')]}
            identity = dict(parent=str(parent),old_gate=str(old_gate),source_inputs={},development_unlabeled=dev,truth_original=dict(path=str(truth),sha256=sha256(truth)))
            config = dict(development_support=n)
            for name,data in [('source_selection',s),('identity',identity),('registration',{})]: write_json(run/(name+'.json'),data)
            z = dict(t=t,scores=p,state=state)
            with patch.object(e,'source_guard',return_value=(config,identity,s)),patch.object(e,'ordinary',return_value=z):
                with self.assertRaises(DataError): e.evaluate(run)
                self.assertFalse((run/'evaluation').exists())
                result = e.freeze(run)
                self.assertEqual(len(result['matrix']),12)
                result = e.evaluate(run)
                self.assertEqual(result['selected_before_development'],'old_gate')
                self.assertFalse(result['development_acceptance_passed'])
                self.assertEqual(len(read_json(run/'evaluation/metrics.json')['rows']),18)
                self.assertTrue(all(r['n']==n for r in read_json(run/'evaluation/metrics.json')['rows']))
                with self.assertRaises(FileExistsError): e.evaluate(run)


if __name__ == '__main__': unittest.main()
