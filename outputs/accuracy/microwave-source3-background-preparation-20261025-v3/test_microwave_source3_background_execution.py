"""Generated data only. Optimizer spy never updates any model parameter."""
import contextlib
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from virtual_submeter.accuracy import microwave_source3_background_execution as e
from virtual_submeter.accuracy import microwave_source5_temporal_execution as fit
from virtual_submeter.data.common import DataError,read_json,write_json,sha256
from virtual_submeter.models.center_mask_tcn import CenterMaskTCN,tensor_input,load_state_model,predict_windows
from virtual_submeter.accuracy.microwave_sampling import BatchStream


def fixture():
    rng=np.random.default_rng(22)
    x=rng.uniform(500,1200,(256,127)).astype(np.float32)
    d=dict(x=x,mask=np.isfinite(x),y=np.r_[np.full(16,300),np.zeros(240)].astype(np.float32),
        house=np.r_[np.zeros(8,int),np.ones(248,int)],t=np.arange(256)*8,
        event=np.r_[np.arange(16)//4,np.full(240,-1)])
    micro=np.full((6,127),10.,np.float32); micro[:,63]=np.arange(6)*10+300
    templates=dict(micro_or_residual=micro,house=np.array([0,0,0,1,1,1]),t=np.arange(6)*8+86400,
        event=np.array([1,1,2,3,3,4]),source_index=np.arange(6),role=np.zeros(6,int),mask=np.ones((6,127),bool))
    residual=np.full((4,127),100.,np.float32); residual[0,0]=np.nan
    backgrounds=dict(micro_or_residual=residual,house=np.array([0,0,1,1]),t=np.arange(4)*8+1600,
        event=np.full(4,-1),source_index=np.arange(4)+16,role=np.zeros(4,int),mask=np.isfinite(residual))
    p={**read_json(fit.f.PLAN),'checkpoints':2,'steps_per_checkpoint':1,'pos_weight':1.}
    meta=dict(input_log_mean=5.,input_log_std=1.,batch_size=256,cpu_threads=4)
    blocks={}
    for b,lo,hi in fit.f.VALIDATION:
        blocks[f'block_{b}']=dict(x=x[:20].copy(),y=np.r_[np.full(6,300),np.zeros(14)],
            t=lo+np.arange(20)*8+120,event=np.r_[0,0,0,1,1,1,np.full(14,-1)])
    return d,templates,backgrounds,p,meta,blocks


class Spy:
    def __init__(self,model):
        self.model=model; self.initial=copy.deepcopy(model.state_dict()); self.steps=0
    def zero_grad(self,set_to_none=True): self.model.zero_grad(set_to_none=set_to_none)
    def step(self):
        self.steps+=1
        if not all(torch.equal(v,self.initial[k]) for k,v in self.model.state_dict().items()):
            raise AssertionError('A synthetic spy must never update parameters')


class AugmentationTests(unittest.TestCase):
    def test_exact_quota_formula_mask_and_no_rng_coupling(self):
        d,t,b,p,meta,blocks=fixture()
        stream=BatchStream(d,'event',42,p); twin=BatchStream(d,'event',42,p)
        np.random.seed(77); before_np=np.random.get_state(); before_torch=torch.get_rng_state().clone()
        aug=e.Augmentor(t,b,42)
        ix=stream.next(); raw=aug(d,ix,0)
        np.testing.assert_array_equal(ix,twin.next())
        np.testing.assert_array_equal(stream.next(),twin.next())
        after_np=np.random.get_state()
        self.assertEqual(before_np[0],after_np[0]); np.testing.assert_array_equal(before_np[1],after_np[1])
        self.assertEqual(before_np[2:],after_np[2:]); self.assertTrue(torch.equal(before_torch,torch.get_rng_state()))
        pos=np.flatnonzero(d['y'][ix]>=200); unchanged=np.ones(256,bool); unchanged[pos[:8]]=False
        self.assertEqual(len(aug.records),8)
        np.testing.assert_array_equal(raw['x'][unchanged],d['x'][ix[unchanged]])
        np.testing.assert_array_equal(raw['mask'][unchanged],d['mask'][ix[unchanged]])
        for r in aug.records:
            ti,bi,slot=(r[k] for k in ('template_pool_index','background_pool_index','slot'))
            m=b['mask'][bi]; expected=np.full(127,np.nan,np.float32)
            expected[m]=b['micro_or_residual'][bi,m]+t['micro_or_residual'][ti,m]
            np.testing.assert_array_equal(raw['x'][slot],expected)
            np.testing.assert_array_equal(raw['mask'][slot],m)
            self.assertIn(slot,pos[:8])
        normalized=tensor_input(meta,raw['x'],raw['mask'])
        m=raw['mask']
        np.testing.assert_array_equal(normalized[:,1].numpy(),m.astype(np.float32))
        np.testing.assert_allclose(normalized[:,0].numpy()[m],(np.log1p(raw['x'][m])-5.))

    def test_reproducible_namespace_and_failure_no_fallback(self):
        d,t,b,p,_,_=fixture(); ix=BatchStream(d,'event',42,p).next()
        a,c=e.Augmentor(t,b,42),e.Augmentor(t,b,42)
        np.testing.assert_array_equal(a(d,ix,0)['x'],c(d,ix,0)['x'])
        self.assertEqual(a.records,c.records)
        with self.assertRaises(DataError): a(d,ix,0)
        invalid=copy.deepcopy(t); invalid['micro_or_residual'][0,0]=np.nan
        with self.assertRaises(DataError): e.Augmentor(invalid,b,42)
        invalid=copy.deepcopy(b); invalid['micro_or_residual'][0,1]=-1
        with self.assertRaises(DataError): e.Augmentor(t,invalid,42)
        with self.assertRaises(DataError): c(d,np.zeros(256,int),1)

    def test_none_path_unit_and_50_loss_gradient_exact(self):
        d,t,b,p,meta,_=fixture(); ix=BatchStream(d,'event',42,p).next(); z=tensor_input(meta,d['x'],d['mask'])
        old=z[ix]; new=fit.training_input(meta,d,ix,z,batch_transform=None,update=0)
        self.assertTrue(torch.equal(old,new))
        for weight in (1.,50.):
            torch.manual_seed(42); model=CenterMaskTCN()
            y=torch.from_numpy((d['y'][ix]>=200).astype(np.float32))
            logits=model(old); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(weight))
            loss.backward(); expected=[v.grad.clone() for v in model.parameters()]
            model.zero_grad(set_to_none=True)
            current=torch.nn.functional.binary_cross_entropy_with_logits(model(new),y,pos_weight=torch.tensor(weight))
            current.backward()
            self.assertTrue(torch.equal(loss,current))
            for a,v in zip(expected,model.parameters()): self.assertTrue(torch.equal(a,v.grad))

    def test_shared_fit_augmentation_spy_and_ordinary_model_load(self):
        d,t,b,p,meta,blocks=fixture(); torch.set_num_threads(4)
        for hook in (False,True):
            torch.manual_seed(42); model=CenterMaskTCN(); spy=Spy(model)
            initial=copy.deepcopy(model.state_dict())
            stream=BatchStream(d,'event',42,p); indices=np.stack([stream.next(),stream.next()])
            aug=e.Augmentor(t,b,42) if hook else None
            with tempfile.TemporaryDirectory() as tmp,patch.object(fit.parent,'configure',return_value=(model,spy)):
                dest=Path(tmp)/'model'
                fit.fit_one(dest,42,p,d,blocks,meta,indices,initial,'a'*64,pos_weight=1.,recipe='unit_BCE',arm='aug',batch_transform=aug)
                self.assertEqual(spy.steps,2)
                self.assertTrue(all(torch.equal(initial[k],v) for k,v in model.state_dict().items()))
                saved,loaded=load_state_model(dest)
                self.assertEqual(saved['recipe'],'unit_BCE')
                for key,block in blocks.items():
                    expected=fit.load_npz(dest/f'epoch_{saved["selected"]["epoch"]}_{key}.npz')['scores']
                    np.testing.assert_array_equal(predict_windows(saved,loaded,block['x']),expected)
                exposure=fit.load_npz(dest/'exposure.npz')['count']
                np.testing.assert_array_equal(exposure,np.bincount(indices.ravel(),minlength=256))
                if aug:
                    aug.save_exposure(dest)
                    self.assertEqual(aug.template_exposure.sum(),16)
                    self.assertEqual(aug.background_exposure.sum(),16)
                    self.assertNotEqual(len(exposure),len(aug.template_exposure))

    def test_budget_gate_precedes_model_or_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'absent'
            for operation in (e.train,e.source,e.freeze,e.evaluate,e.predict):
                with patch.object(e,'registered',side_effect=AssertionError('data reached')):
                    with self.assertRaises(DataError): operation(run)
                self.assertFalse(run.exists())
            receipt=Path(tmp)/'old.json'
            write_json(receipt,dict(authorized=True,experiment='microwave_source5_loss_v1',approval_origin='human'))
            with self.assertRaises(DataError): e.train(run,receipt)
            write_json(receipt,dict(authorized=False,experiment='microwave_source3_background_v1'))
            with self.assertRaises(DataError): e.train(run,receipt)

    def test_source_relative_qualification_and_control_tie(self):
        control=[dict(seed=s,precision=.2,recall=.97,f1=.3,degenerate=False,source_gate_passed=True) for s in e.SEEDS]
        self.assertEqual(e.select_source(control,copy.deepcopy(control))['selected'],'control_guard95')
        new=[dict(r,precision=.3,f1=.4) for r in control]
        self.assertEqual(e.select_source(control,new)['selected'],'aug_guard95')
        new[0]['recall']=.91
        self.assertFalse(e.select_source(control,new)['aug_eligible'])
        self.assertFalse(e.select_source(control,new[:2])['aug_eligible'])

    def test_source95_only_fixed_algorithm(self):
        t=np.arange(80)*8; p=np.full(80,.1); p[[20,60]]=.9
        y=np.zeros(80,bool); y[[20,60]]=True
        scores={str(i):dict(t=t,p=p,h=e.gate.local_max(t,p),ix=np.arange(80),y=y) for i in (4,5)}
        low,high=e.calibrate_source95(scores)
        self.assertIsNotNone(low); self.assertIsNotNone(high)
        for d in scores.values():
            result=e.gate.confusion(e.gate.gate_state(d['p'],d['h'],low,high),d['y'])
            self.assertGreaterEqual(result['recall'],.95)
            self.assertFalse(result['degenerate'])
        with patch.object(e.guard95,'calibrate95',wraps=e.guard95.calibrate95) as fixed:
            self.assertEqual(e.calibrate_source95(scores),(low,high)); self.assertEqual(fixed.call_count,1)

    def test_source_rejection_blocks_development_before_identity(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(e,'authorize',return_value={}),patch.object(e,'phase',return_value=contextlib.nullcontext()),patch.object(e,'source_guard',return_value=dict(aug_eligible=False,recipes=[])):
            with patch.object(e,'read_json',side_effect=AssertionError('development identity reached')):
                with self.assertRaises(DataError): e.freeze(tmp)

    def test_generated_complete_matrix_evaluation_once_and_tamper_guard(self):
        # The large support here is generated, not a real household cache.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/'run'; run.mkdir(); control=root/'control'; control.mkdir(); base=root/'baseline'
            n=137523; stamps=1000000+np.arange(n)*8; truth_y=np.where(np.arange(n)%10==0,300.,0.).astype(np.float32)
            raw=np.full((n,3),300.,np.float32); corrected=raw.copy()
            for seed in e.SEEDS:
                for folder in ('reference/predictions','predictions/W50'):
                    dest=base/folder/f'seed_{seed}'/e.gate.PARTITION; dest.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest,stamps=stamps,raw=raw,**({'corrected':corrected} if folder.startswith('reference') else {}))
            agg=root/'aggregate.csv'; agg.write_text('generated fixture only\n',encoding='utf-8')
            dev={str(agg):sha256(agg)}
            for seed in e.SEEDS:
                for folder in ('reference/predictions','predictions/W50'):
                    path=base/folder/f'seed_{seed}'/e.gate.PARTITION; dev[str(path)]=sha256(path)
            truth=root/'generated_truth_cache.npz'; np.savez_compressed(truth,t=stamps,y=truth_y)
            write_json(control/'identity.json',dict(baseline=str(base),development_unlabeled=dev,truth_cache=dict(path=str(truth),sha256=sha256(truth))))
            s=dict(selected='aug_guard95',aug_eligible=True,source_rejection=None,recipes=[dict(seed=seed,feasible=True) for seed in e.SEEDS])
            write_json(run/'source_selection.json',s); write_json(run/'registration.json',{}); write_json(run/'identity.json',{})
            def generated_ordinary(run,family,seed,rows):
                p=np.where(truth_y>=200,.9,.1).astype(np.float32)
                if family=='control_guard95': p[np.arange(n)%7==0]=.9
                return dict(t=stamps,scores=p,state=(p>=.5).astype(np.int8))
            with patch.object(e.audit,'CONTROL',control),patch.object(e,'authorize',return_value={}),patch.object(e,'phase',side_effect=lambda *a,**k:contextlib.nullcontext()),patch.object(e,'source_guard',return_value=s),patch.object(e,'ordinary',side_effect=generated_ordinary),patch.object(e.gate,'csv_rows',return_value=[]):
                result=e.freeze(run)
                self.assertEqual(len(result['matrix']),12)
                self.assertFalse((run/'evaluation').exists())
                original=e.gate.npz; truth_loads=[]
                def track(path,fields):
                    if Path(path)==truth: truth_loads.append(str(path))
                    return original(path,fields)
                with patch.object(e.gate,'npz',side_effect=track):
                    evaluated=e.evaluate(run)
                    self.assertEqual(len(truth_loads),1)
                    self.assertTrue(evaluated['development_acceptance_passed'])
                    self.assertTrue(all(evaluated['acceptance_gates'].values()))
                    with self.assertRaises(DataError): e.evaluate(run)
                    self.assertEqual(len(truth_loads),1)
                saved=read_json(run/'freeze.json'); saved['matrix'].pop(); write_json(run/'freeze.json',saved)
                with self.assertRaises(DataError): e.evaluation_guard(run)


if __name__=='__main__': unittest.main()
