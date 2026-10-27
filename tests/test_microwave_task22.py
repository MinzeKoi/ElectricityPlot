"""Task22 static pipeline fixtures: never call a real fit or construct an optimizer."""
import csv
import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_task22 as m
from virtual_submeter.accuracy import microwave_task22_review as review
from virtual_submeter.models.center_mask_tcn import load_state_model,predict_windows,tensor_input
from virtual_submeter.data.common import iso


class Task22Tests(unittest.TestCase):
    def test_cold_diagnostics_helpers_loaded_before_isolation(self):
        self.assertIs(m.independent_ap,review.independent_ap)
        self.assertIs(m.scalar_metric,review.scalar_metric)

    def test_missing_old_receipt_before_data_or_model(self):
        with patch.object(m,'registered',side_effect=AssertionError('premature data read')),patch.object(m,'execution_allowed',side_effect=AssertionError('premature read')):
            for name in ('train','source_select','freeze','evaluate','diagnostics'):
                with self.assertRaisesRegex(Exception,'NEW human budget'):getattr(m,name)(Path('missing'))
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            p=Path(td)/'old.json';m.fresh(p,dict(authorized=True,approval_origin='human',experiment='microwave_task21_hard_negative_v1'))
            with self.assertRaisesRegex(Exception,'Missing NEW'):m.authorize(Path(td),p)
            with self.assertRaisesRegex(Exception,'cannot overwrite'):m.prepare(Path(td))

    def test_model_width_metadata_support_and_no_autograd(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            root=Path(td)
            with m.isolation(root,[],training=False):
                for channels,count in ((16,5329),(32,19809)):
                    model=m.CenterMaskTCN(channels);self.assertEqual(sum(p.numel() for p in model.parameters()),count)
                    dest=root/str(channels);dest.mkdir();torch.save(model.state_dict(),dest/'weights.pt')
                    meta=dict(kind='microwave_center_mask_tcn_v1',truth_threshold_w=200,window_samples=127,inference_support='strict_complete_31',
                        input_log_mean=5.,input_log_std=1.,decision_threshold_probability=.5,cpu_threads=4,batch_size=256,
                        parameter_count=count,weights_sha256=m.sha256(dest/'weights.pt'))
                    if channels==32:meta['tcn_channels']=32
                    m.fresh(dest/'model.json',meta)
                    loaded,net=load_state_model(dest)
                    x=np.full((4,127),100.,np.float32);x[0,:48]=np.nan;x[1,79:]=np.nan
                    before=predict_windows(meta,model,x);np.testing.assert_array_equal(before,predict_windows(loaded,net,x))
                    z=tensor_input(meta,x,np.isfinite(x));self.assertEqual(z.shape,(4,2,127))
                    x[2,63]=np.nan
                    self.assertEqual(net(tensor_input(meta,x,np.isfinite(x))).shape,(4,))
                    with self.assertRaisesRegex(Exception,'complete core31'):predict_windows(meta,model,x)
                    if channels==32:
                        bad=dict(meta);bad.pop('tcn_channels');(dest/'model.json').write_text(json.dumps(bad),encoding='utf-8')
                        with self.assertRaises(Exception):load_state_model(dest)
                with self.assertRaisesRegex(Exception,'optimizer/autograd'):torch.optim.Adam([])
                with self.assertRaisesRegex(Exception,'optimizer/autograd'):torch.autograd.backward([])
                with self.assertRaisesRegex(Exception,'exact whitelist'):(root/'unlisted.npz').open('rb')

    def test_scalar_ties_dates_gaps_and_acceptance(self):
        t=np.array([86376,86384,86392,86400,86408,86500,86508]);p=np.array([.2,.9,.3,.4,np.nan,.8,.2],np.float32)
        np.testing.assert_array_equal(m.gate.local_max(t,p),review.local_max_independent(t,p))
        for pred in (np.array([1,0,1,0]),np.array([0,0,0,0]),np.array([1,1,1,1])):
            self.assertEqual(m.gate.confusion(pred,np.array([1,0,1,0],bool)),review.scalar_metric(pred,np.array([1,0,1,0],bool)))
        with self.assertRaisesRegex(Exception,'9-cell'):review.independent_acceptance([],[],candidate='wide32')
        y=np.array([200,0,200,0,0],np.float32);scores=np.array([.8,.8,.3,.3,.1],np.float32)
        self.assertAlmostEqual(m.diagnostic_metrics(scores,y,.5)['ap'],review.independent_ap(scores,y>=200),places=12)

    def fixture(self,root):
        run=root/'run';t10=root/'t10';base=root/'baseline';run.mkdir();(t10/'validation').mkdir(parents=True)
        p=dict(checkpoints=12,steps_per_checkpoint=294,batch_size=256,positive_per_batch=16,uniform_negative_per_batch=240,
            learning_rate=.001,cpu_threads=4,minimum_block_positive_windows=5,minimum_block_events=2,minimum_validation_recall=.85)
        meta=dict(input_log_mean=5.,input_log_std=1.,cpu_threads=4,batch_size=256)
        n=336704;d=dict(x=np.full((n,127),100.,np.float32),y=np.zeros(n,np.float32),house=np.arange(n)%3,
            t=1300000000+np.arange(n,dtype=np.int64)*8,event=np.full(n,-1,np.int64))
        d['y'][:2275]=200;d['event'][:2275]=np.arange(2275)//10;d['mask']=np.isfinite(d['x'])
        np.savez_compressed(t10/'train.npz',**d)
        source_t=[];source_p=[];vals={}
        for b,lo,hi in m.loss.temporal.f.VALIDATION:
            t=lo+86400-480+np.arange(120)*8;y=np.zeros(120,np.float32);y[10:20]=200;y[90:100]=200
            event=np.full(120,-1,np.int64);event[10:20]=10000+b*10;event[90:100]=10000+b*10+1
            v=dict(t=t,y=y,event=event,house=np.full(120,2),block=np.full(120,b),x=np.full((120,127),100.,np.float32),mask=np.ones((120,127),bool))
            np.savez_compressed(t10/f'validation/block_{b}.npz',**v);vals[f'block_{b}']=v
            source_t.extend(t.tolist());source_p.extend(np.where(y>=200,.9,.1).astype(np.float32).tolist())
        source_t=np.array(source_t);source_p=np.array(source_p,np.float32)
        m.fresh(t10/'normalization.json',meta)
        raw=root/'source.csv'
        with raw.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['timestamp','household_id','power_w']);w.writeheader()
            w.writerows(dict(timestamp=iso(int(t)),household_id='refit_5',power_w=100) for t in source_t)
        m.fresh(t10/'source5_inputs.json',{'aggregate':dict(path=str(raw),sha256=m.sha256(raw))})
        m.fresh(run/'parent_training_plan.json',p)
        config=dict(schema=m.SCHEMA,authorized=False,preparation_only=True,seeds=m.SEEDS,tcn_channels=32,control_channels=16,
            parameter_count=19809,control_parameter_count=5329,maximum_new_fits=3,maximum_updates=10584,updates_per_seed=3528,
            positive_per_batch=16,uniform_negative_per_batch=240,recipe='unit_BCE',optimizer='Adam',learning_rate=.001,cpu_threads=4)
        m.fresh(run/'config.json',config)
        proof={}
        for s in m.SEEDS:
            batches=np.stack(list(review.independent_batches(d,s,3528)))
            for name in (run/f'batches_{s}.npz',t10/f'sampling_{s}.npz'):np.savez_compressed(name,indices=batches)
            np.savez_compressed(run/f'exposure_{s}.npz',total=np.bincount(batches.ravel(),minlength=n))
            torch.save(m.make_initial(s,16),t10/f'initial_{s}.pt')
            wide=m.make_initial(s,32);torch.save(wide,run/f'initial32_{s}.pt')
            proof[str(s)]=dict(initial32_tensor_hashes={k:m.gate.hash_array(v.numpy()) for k,v in wide.items()})
        m.fresh(run/'sampling_proof.json',proof)
        m.fresh(run/'data_identity.json',dict(validation={k:{n:m.gate.hash_array(v) for n,v in b.items()} for k,b in vals.items()}))
        m.fresh(run/'source_control.json',dict(rows={'guard95':[dict(seed=s,precision=.1,recall=1.,f1=.1) for s in m.SEEDS]}))
        # Complete generated nine-cell dev support. These are static fixture points.
        nd=137523;dt=1500000000+np.arange(nd)*8;truth=np.zeros(nd,np.float32);truth[::100]=200
        raw_a=np.zeros((nd,3),np.float32);raw_a[::100,1]=200
        raw_w=raw_a.copy();raw_w[1::100,1]=200
        for s in m.SEEDS:
            for arm,rawv in (('A',raw_a),('W50',raw_w)):
                folder=base/(f'reference/predictions/seed_{s}' if arm=='A' else f'predictions/W50/seed_{s}');folder.mkdir(parents=True)
                np.savez_compressed(folder/m.gate.PARTITION,stamps=dt,raw=rawv,corrected=rawv)
        devraw=root/'synthetic_aggregate.csv'
        with devraw.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['timestamp','household_id','power_w']);w.writeheader()
            w.writerows(dict(timestamp=iso(int(t)),household_id='refit_17',power_w=100) for t in dt)
        truthpath=root/'synthetic_truth.npz';np.savez_compressed(truthpath,t=dt,y=truth)
        dev={str(devraw):m.sha256(devraw)}
        dev.update({str(p):m.sha256(p) for p in base.rglob('*.npz')})
        m.fresh(run/'future_identity.json',dict(development_unlabeled=dev,baseline=str(base),truth_cache=dict(path=str(truthpath),sha256=m.sha256(truthpath))))
        (run/'PREPARATION_REPORT.zh.md').write_text('合成静态全流程，非真实实验。',encoding='utf-8')
        inputs=[*t10.rglob('*'),raw,*list((m.ROOT/'src').rglob('*.py')),*list(Path(m.__file__).resolve().parents[1].rglob('*.py')),Path(__file__).resolve()];inputs=[p for p in inputs if p.is_file()]
        m.fresh(run/'inputs_before.json',m.snapshot(inputs))
        m.fresh(run/'registration.json',dict(schema=m.SCHEMA,artifacts=m.inventory(run),authorized=False))
        def infer(model,rows):
            if rows and rows[0]['household_id']=='refit_5':
                t=np.array([m.loss.utc(r['timestamp']) for r in rows]);return t,source_p[np.searchsorted(source_t,t)],{}
            return dt,np.where(truth>=200,.9,.1).astype(np.float32),{}
        return run,t10,p,meta,vals,infer

    def test_complete_static_train_source_freeze_evaluate_verify_report(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            root=Path(td);run,t10,p,meta,vals,infer=self.fixture(root)
            def fakefit(dest,seed,p,d,blocks,meta,indices,initial,registration,**kwargs):
                self.assertEqual(kwargs,dict(pos_weight=1.,recipe='unit_BCE',arm='wide32',channels=32))
                dest.mkdir(parents=True);torch.save(initial,dest/'initial.pt')
                m.fresh(dest/'progress.json',dict(completed_updates=3528,attempted_updates=3528))
                m.fresh(dest/'complete.json',dict(updates=3528))
                np.savez_compressed(dest/'exposure.npz',count=np.bincount(indices.ravel(),minlength=len(d['y'])))
                with (dest/'batches.csv').open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['update','positive','negative','indices_sha256']);w.writeheader()
                    w.writerows(dict(update=k+1,positive=16,negative=240,indices_sha256=m.loss.temporal.array_hash(row)) for k,row in enumerate(indices))
                scores=[{k:np.where(v['y']>=200,.9,.1).astype(np.float32) for k,v in blocks.items()} for _ in range(12)]
                for epoch in range(1,13):
                    torch.save(initial,dest/f'epoch_{epoch}.pt')
                    for k,v in blocks.items():np.savez_compressed(dest/f'epoch_{epoch}_{k}.npz',t=v['t'],scores=scores[epoch-1][k])
                selected=m.loss.temporal.choose(scores,blocks,p)
                return m.loss.temporal.write_selected(dest,seed,'wide32',selected,meta,dest/f'epoch_{selected["epoch"]}.pt',registration,3528,'unit_BCE',channels=32)
            def nofit(*a,**kw):raise AssertionError('Optimizer/autograd forbidden even in fixture')
            with patch.object(m,'T10',t10),patch.object(m,'authorize',return_value={'synthetic_only':True}),patch.object(m.loss.temporal,'fit_one',side_effect=fakefit),patch.object(m.gate,'infer',side_effect=infer),patch.object(torch.optim.Optimizer,'__init__',nofit),patch.object(torch.Tensor,'backward',nofit),patch.object(torch.autograd,'backward',nofit):
                self.assertEqual(m.train(run,'synthetic'),dict(new_fits=3,updates=10584))
                self.assertTrue(m.source_select(run,'synthetic')['source_eligible'])
                self.assertTrue(m.freeze(run,'synthetic')['frozen'])
                # Missing matrix and full-off/low recall qualifications are rejected before truth access.
                original=m.read_json(run/'freeze.json');bad=copy.deepcopy(original);bad['matrix'].pop()
                (run/'freeze.json').write_text(json.dumps(bad),encoding='utf-8')
                with self.assertRaisesRegex(Exception,'Incomplete/unbound'):m.evaluation_guard(run)
                (run/'freeze.json').write_text(json.dumps(original,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                result=m.evaluate(run,'synthetic');self.assertTrue(result['development_acceptance_passed'])
                with self.assertRaisesRegex(Exception,'One development acceptance'):m.evaluate(run,'synthetic')
                before=m.snapshot([run/k for k in m.inventory(run)])
                self.assertTrue(m.verify(run)['execution']['independent_eight_gates'])
                self.assertTrue(m.report(run)['execution']['complete_frozen_9_cell_matrix'])
                self.assertEqual(before,m.snapshot(before))
                selection=m.read_json(run/'source_selection.json')
                for alloff in (False,True):
                    bad=copy.deepcopy(selection);bad['recipes'][0]['blocks'][0]['recall']=0 if alloff else .949
                    bad['recipes'][0]['blocks'][0]['degenerate']=alloff
                    (run/'source_selection.json').write_text(json.dumps(bad),encoding='utf-8')
                    seal=m.read_json(run/'source_complete.json');seal['source_sha256']=m.sha256(run/'source_selection.json')
                    (run/'source_complete.json').write_text(json.dumps(seal),encoding='utf-8')
                    with self.assertRaisesRegex(Exception,'Source rejected'):m.source_guard(run)

if __name__=='__main__':unittest.main()
