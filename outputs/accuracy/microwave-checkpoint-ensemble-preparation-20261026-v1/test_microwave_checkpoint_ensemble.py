"""Synthetic only: no real training, source candidate or development reads."""
import contextlib
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from virtual_submeter.accuracy import microwave_checkpoint_ensemble as e
from virtual_submeter.data.common import DataError,write_json,read_json,sha256,iso
from virtual_submeter.models.center_mask_tcn import CenterMaskTCN,predict_windows,context_at_complete_centers


def points():
    return [dict(epoch=i,selection=dict(worst_precision=.1,mean_precision=.2,mean_f1=.3,threshold=.5)) for i in range(1,13)]


def source_rows():
    control=[dict(seed=s,precision=.2,recall=.96,f1=.3,degenerate=False,source_gate_passed=True) for s in e.SEEDS]
    new=[dict(seed=s,precision=.3,recall=.96,f1=.4,degenerate=False,source_gate_passed=True) for s in e.SEEDS]
    recipes=[dict(seed=s,feasible=True,blocks=[dict(recall=.96,degenerate=False)]*2) for s in e.SEEDS]
    return control,new,recipes


class Task16(unittest.TestCase):
    def test_sort_ties_and_no_fallback(self):
        p=points(); self.assertEqual([r['epoch'] for r in e.top3(p)],[1,2,3])
        p[9]['selection']['worst_precision']=.5
        self.assertEqual([r['epoch'] for r in e.top3(p)],[10,1,2])
        for r in p[2:]: r['selection']=None
        with self.assertRaises(DataError): e.top3(p)
        p=points(); p[-1]['epoch']=1
        with self.assertRaises(DataError): e.top3(p)

    def test_mean_dtype_nan_support(self):
        v=[np.array([.1,.9,np.nan],np.float32),np.array([.2,.8,np.nan],np.float32),np.array([.3,.7,np.nan],np.float32)]
        r=e.mean_probability(v)
        self.assertEqual(r.dtype,np.float32)
        np.testing.assert_array_equal(r[:2],np.stack(v).astype(np.float64).mean(0)[:2].astype(np.float32))
        self.assertTrue(np.isnan(r[2]))
        for invalid in (v[:2],[v[0],v[1],np.zeros(3,np.float32)],[x.astype(np.float64) for x in v], [v[0],v[1],np.array([np.inf,.5,np.nan],np.float32)]):
            with self.assertRaises(DataError): e.mean_probability(invalid)

    def test_seed_isolation_and_ordinary_exact(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td); write_json(run/'checkpoint_audit.json',dict(seeds={str(s):dict(rows=points(),selected=e.top3(points())) for s in e.SEEDS}))
            calls=[]
            def infer(seed,epoch,rows):
                calls.append((seed,epoch)); return np.arange(4)*8,np.array([epoch/20,seed/10000,np.nan,.5],np.float32)
            with patch.object(e,'checkpoint_infer',infer):
                t,p=e.ensemble_infer(run,1019,[])
                self.assertEqual(calls,[(1019,1),(1019,2),(1019,3)])
                np.testing.assert_array_equal(p,e.mean_probability([infer(1019,j,[])[1] for j in (1,2,3)]))
                with self.assertRaises(DataError): e.ensemble_infer(run,999,[])

    def test_actual_synthetic_checkpoint_inference_no_fit(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); d=root/'models/unit_BCE/seed_42'; d.mkdir(parents=True)
            meta=dict(window_samples=127,input_log_mean=5.,input_log_std=1.,cpu_threads=1,batch_size=7)
            write_json(d/'model.json',meta)
            models=[]
            for epoch in (1,2,3):
                torch.manual_seed(60+epoch); model=CenterMaskTCN().eval(); models.append(model)
                torch.save(model.state_dict(),d/f'epoch_{epoch}.pt')
            write_json(root/'checkpoint_audit.json',dict(seeds={'42':dict(rows=points(),selected=e.top3(points()))}))
            write_json(root/'source_selection.json',dict(recipes=[dict(seed=42,feasible=True,L=.2,H=.3)]))
            t=np.arange(160,dtype=np.int64)*8+1600000000
            a=np.linspace(50,600,160).astype(np.float32)
            rows=[dict(household_id='synthetic',timestamp=iso(int(ti)),aggregate_power_w=str(float(ai))) for ti,ai in zip(t,a)]
            with patch.object(e,'LOSS',root),e.isolation('source_select',[d/f'epoch_{epoch}.pt' for epoch in (1,2,3)]):
                actual_t,actual=e.checkpoint_infer(42,1,rows)
                mean_t,mean=e.ensemble_infer(root,42,rows)
                ordinary=e.ordinary(root,'ensemble_guard95',42,rows)
            x,centers=context_at_complete_centers(meta,t,a)
            expected=np.full(len(t),np.nan,np.float32); expected[centers]=predict_windows(meta,models[0],x)
            np.testing.assert_array_equal(actual_t,t); np.testing.assert_array_equal(actual,expected)
            target=np.full(len(t),np.nan,np.float32)
            target[centers]=(np.stack([predict_windows(meta,m,x) for m in models]).sum(0,dtype=np.float64)/3).astype(np.float32)
            np.testing.assert_array_equal(mean_t,t); np.testing.assert_array_equal(mean,target)
            np.testing.assert_array_equal(ordinary['scores'],target)
            np.testing.assert_array_equal(ordinary['state'],e.gate.gate_state(target,e.gate.local_max(t,target),.2,.3))

    def test_authorization_first_every_stage(self):
        for fn in (e.source,e.freeze,e.evaluate,e.predict):
            with patch.object(e,'read_json',side_effect=AssertionError('Must not read before authorization')):
                with self.assertRaises(DataError): fn('nonexistent')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'old.json'; write_json(p,dict(authorized=True,experiment='microwave_recall_guard95_v1'))
            with self.assertRaises(DataError): e.authorize(Path(td),p)

    def test_access_denies_dev_final_even_allowlisted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            files=[root/'refit_17.npz',root/'house_8.json',root/'evaluation/metrics.json',root/'truth_once.npz']
            for p in files:
                p.parent.mkdir(exist_ok=True); p.write_text('{}')
            with e.isolation('prepare',files):
                for p in files:
                    with self.assertRaises(DataError): p.read_bytes()
                with self.assertRaises(DataError): CenterMaskTCN()
                with self.assertRaises(DataError): torch.optim.Adam([torch.zeros(1,requires_grad=True)])

    def test_qualification_low_recall_all_off_and_ties(self):
        c,n,r=source_rows(); self.assertTrue(e.select_source(c,n,r)['ensemble_eligible'])
        n[0]['recall']=.90
        self.assertFalse(e.select_source(c,n,r)['ensemble_eligible'])
        c,n,r=source_rows(); r[0]['blocks'][0]['recall']=.94
        self.assertFalse(e.select_source(c,n,r)['ensemble_eligible'])
        c,n,r=source_rows(); r[0]['blocks'][0]['degenerate']=True
        self.assertFalse(e.select_source(c,n,r)['ensemble_eligible'])
        c,n,r=source_rows(); n=copy.deepcopy(c)
        self.assertEqual(e.select_source(c,n,r)['selected'],'control_guard95')
        c,n,r=source_rows()
        self.assertFalse(e.select_source(c,n[:2],r)['ensemble_eligible'])

    def test_incomplete_freeze_rejected_before_truth(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td); write_json(run/'freeze.json',dict(matrix=e.matrix()[:-1]))
            with patch.object(e,'source_guard',return_value=dict(ensemble_eligible=True)):
                with self.assertRaises(DataError): e.evaluation_guard(run)
            (run/'evaluation').mkdir()
            with patch.object(e,'authorize'),patch.object(e,'phase',return_value=contextlib.nullcontext()),patch.object(e,'evaluation_guard'):
                with self.assertRaises(DataError): e.evaluate(run,'synthetic')

    def test_scalar_tamper_and_original_gates(self):
        y=np.array([1,1,0,0],bool); p=np.array([1,0,1,0],np.int8)
        self.assertEqual(e.shared.scalar_confusion(p,y),e.gate.confusion(p,y))
        self.assertNotEqual(e.shared.scalar_confusion(p,y),e.shared.scalar_confusion(np.zeros(4,np.int8),y))
        rows=[]
        for family in ('A','W50','ensemble_guard95'):
            for seed in e.SEEDS:
                rows.append(dict(arm=family,seed=seed,**e.shared.scalar_confusion(np.zeros(4,np.int8),y)))
        result=e.shared.independent_formal_summary(rows,[dict(n=137523)]*3,'ensemble_guard95')
        self.assertFalse(result['development_acceptance_passed'])
        self.assertFalse(result['acceptance_gates']['each_R_at_least_085'])
        self.assertFalse(result['acceptance_gates']['no_new_all_on_or_off'])

    def test_synthetic_source_full_stage_scalar_replay_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); run=root/'run'; run.mkdir(); temporal=root/'temporal'; (temporal/'validation').mkdir(parents=True)
            control=root/'control'; control.mkdir()
            write_json(run/'identity.json',dict(source_inputs={}))
            write_json(run/'registration.json',dict(artifacts={}))
            write_json(run/'checkpoint_audit.json',dict(seeds={}))
            write_json(control/'identity.json',dict(source=dict(aggregate=dict(path=str(root/'source.csv')))))
            raw=[]; arrays={}
            for b,lo,hi in e.loss.temporal.f.VALIDATION:
                t=lo+np.arange(20)*8+128; y=np.r_[np.ones(6),np.zeros(14)]
                p=np.r_[np.full(6,.9),np.linspace(.01,.2,14)].astype(np.float32)
                arrays[b]=(t,p)
                np.savez_compressed(temporal/f'validation/block_{b}.npz',t=t,y=y*300)
                raw += [dict(timestamp=iso(int(ti)),aggregate_power_w='100',household_id='refit_5') for ti in t]
            write_json(run/'identity.json',dict(source_inputs={str(temporal/f'validation/block_{b}.npz'):'synthetic' for b in (4,5)}))
            c,_,_=source_rows(); write_json(control/'source_selection.json',dict(rows=dict(guard95=c),recipes=[]))
            def infer(run,seed,rows,b): return arrays[b]
            with patch.object(e,'CONTROL',control),patch.object(e,'TEMPORAL',temporal),patch.object(e,'authorize'),patch.object(e,'registered'),patch.object(e,'audit_checkpoints',return_value={}),patch.object(e.gate,'csv_rows',return_value=raw),patch.object(e,'ensemble_infer',infer):
                result=e.source(run,'synthetic')
                self.assertTrue(result['ensemble_eligible'])
                self.assertTrue((run/'source_report.md').exists())
                with e.phase(run,'verify'): e.source_guard(run)
                # Re-seal an altered source summary: scalar replay still rejects it.
                selected=read_json(run/'source_selection.json'); selected['rows']['ensemble_guard95'][0]['precision']=.1
                write_json(run/'source_selection.json',selected)
                done=read_json(run/'source_complete.json'); done['source_selection_sha256']=sha256(run/'source_selection.json'); write_json(run/'source_complete.json',done)
                with e.phase(run,'verify'):
                    with self.assertRaises(DataError): e.source_guard(run)

    def test_full_synthetic_freeze_evaluate_verify_and_resealed_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); run=root/'run'; run.mkdir(); control=root/'control'; control.mkdir(); base=root/'base'
            write_json(run/'identity.json',dict(source_inputs={}))
            write_json(run/'registration.json',dict(artifacts={}))
            write_json(run/'source_started.json',dict(synthetic=True))
            c,new,_=source_rows()
            selected=dict(selected='control_guard95',ensemble_eligible=True,qualification={},recipes=[],rows=dict(control_guard95=c,ensemble_guard95=new))
            write_json(run/'source_selection.json',selected)
            write_json(control/'source_selection.json',dict(recipes=[]))
            write_json(run/'checkpoint_audit.json',dict(seeds={str(s):dict(qualified=12,selected=e.top3(points())) for s in e.SEEDS}))
            (run/'preparation_report.md').write_text('合成准备报告，执行后必须保留。',encoding='utf-8')
            n=137523; t=1600000000+np.arange(n,dtype=np.int64)*8
            y=(np.arange(n)%100==0); truth=root/'synthetic_truth.npz'; np.savez_compressed(truth,t=t,y=y*300)
            raw=np.zeros((n,3),np.float32); raw[:,0]=40; raw[:,2]=5
            raw[:,1]=np.where(y | (np.arange(n)%10==0),300,0)
            bindings={}; agg=root/'aggregate.csv'; agg.write_text('synthetic',encoding='utf-8'); bindings[str(agg)]=sha256(agg)
            for seed in e.SEEDS:
                for prefix in ('reference/predictions','predictions/W50'):
                    dest=base/f'{prefix}/seed_{seed}'; dest.mkdir(parents=True)
                    path=dest/e.gate.PARTITION
                    np.savez_compressed(path,stamps=t,raw=raw,corrected=raw)
                    bindings[str(path)]=sha256(path)
            write_json(control/'identity.json',dict(baseline=str(base),development_unlabeled=bindings,truth_cache=dict(path=str(truth),sha256=sha256(truth))))
            def ordinary(run,family,seed,rows):
                state=(y | (np.arange(n)%(20 if family=='control_guard95' else 40)==0)).astype(np.int8)
                return dict(t=t,scores=np.where(state,.9,.1).astype(np.float32),state=state)
            def cli(args,**kwargs):
                def arg(name): return args[args.index(name)+1]
                # Same public predict function invoked by the standalone script.
                e.predict(arg('--run'),'synthetic',family=arg('--family'),seed=int(arg('--seed')),aggregate=arg('--aggregate'),output=arg('--output'))
                return SimpleNamespace(returncode=0,stderr='')
            with patch.object(e,'CONTROL',control),patch.object(e,'authorize'),patch.object(e,'registered'),patch.object(e,'source_guard',return_value=selected),patch.object(e,'ordinary',ordinary),patch.object(e.gate,'csv_rows',return_value=[]),patch.object(e.subprocess,'run',cli):
                # CSV reader needs only rows, aggregate is deliberately synthetic.
                agg.write_text('timestamp,aggregate_power_w,household_id\n',encoding='utf-8'); bindings[str(agg)]=sha256(agg)
                ref=read_json(control/'identity.json'); ref['development_unlabeled']=bindings; write_json(control/'identity.json',ref)
                result=e.freeze(run,'synthetic')
                self.assertEqual(len(result['matrix']),12)
                self.assertEqual(result['selected'],'control_guard95')
                e.evaluate(run,'synthetic')
                # Development has stronger ensemble P, but source control remains selected.
                self.assertEqual(read_json(run/'source_selection.json')['selected'],'control_guard95')
                self.assertEqual(e.verify(run,'synthetic')['new_fits'],0)
                e.report(run,'synthetic'); e.report(run,'synthetic')
                self.assertIn('12格封存',(run/'execution_report.md').read_text(encoding='utf-8'))
                self.assertEqual((run/'preparation_report.md').read_text(encoding='utf-8'),'合成准备报告，执行后必须保留。')
                with self.assertRaises(DataError): e.evaluate(run,'synthetic')
                # A coherent file hash cannot conceal false acceptance gate numerics.
                summary=read_json(run/'evaluation/summary.json'); summary['formal_means']['control_guard95']['precision']+=.02
                write_json(run/'evaluation/summary.json',summary)
                done=read_json(run/'evaluation_complete.json'); done['artifacts']['summary.json']=sha256(run/'evaluation/summary.json'); write_json(run/'evaluation_complete.json',done)
                with self.assertRaises(DataError): e.verify(run,'synthetic')


if __name__=='__main__': unittest.main()
