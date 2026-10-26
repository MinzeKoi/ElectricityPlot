"""Small high-risk diagnostic tests; no training or forbidden data access."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
while not (ROOT/'pyproject.toml').is_file():
    ROOT = ROOT.parent
PATH = ROOT / 'scripts/task19_fixed_diagnostic.py'
spec = importlib.util.spec_from_file_location('task19', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FixedDiagnosticTests(unittest.TestCase):
    def test_tied_ap_and_single_classes(self):
        y=np.array([1,0,1,0],bool);p=np.array([.8,.8,.2,.2],np.float32)
        self.assertEqual(m.ap_grouped(y,p),.5)
        self.assertEqual(m.independent_ap(y,p),.5)
        for truth in (np.zeros(4,bool),np.ones(4,bool),np.zeros(0,bool)):
            prob=np.ones(len(truth),np.float32)
            a=m.metrics(truth,prob,.7);b=m.independent_metrics(truth,prob,.7)
            m.compare_metrics(a,b)
            json.dumps(a,allow_nan=False)
        self.assertIsNone(m.metrics(np.zeros(4,bool),p,.8)['recall'])

    def test_ap_permutation_invariant(self):
        rng=np.random.default_rng(23)
        for n in (1,7,83,300):
            y=rng.integers(0,2,n).astype(bool);p=rng.integers(0,4,n).astype(np.float32)/4
            a=m.ap_grouped(y,p);b=m.independent_ap(y,p)
            if a is not None: self.assertAlmostEqual(a,b,places=14)
            order=rng.permutation(n)
            self.assertEqual(a,m.ap_grouped(y[order],p[order]))

    def test_threshold_boundary_and_no_nan(self):
        y=np.array([True,False,True,False]);p=np.array([.25,.25,.24,.1],np.float32)
        a=m.metrics(y,p,.25)
        self.assertEqual([a[k] for k in ('tp','fp','fn','tn')],[1,1,1,1])
        for bad in (np.array([np.nan,0,0,0]),np.array([np.inf,0,0,0])):
            with self.assertRaises(ValueError):m.metrics(y,bad,.25)

    def sample(self):
        return dict(y=np.array([0,201,0],np.float32),x=np.ones((3,127),np.float32),
            mask=np.ones((3,127),bool),house=np.array([0,0,0]),source_role=np.array(['original14d']*3),
            t=np.array([0,8,16]),event=np.array([-1,0,-1]),raw_index=np.arange(3))

    def test_identity_duplicates_masks_and_block(self):
        d=self.sample();self.assertTrue(m.validate_dataset(d,'train').all())
        d['t'][2]=8
        with self.assertRaisesRegex(ValueError,'Duplicate'):m.validate_dataset(d,'train')
        d=self.sample();d['mask'][0,60]=False
        with self.assertRaisesRegex(ValueError,'Mask'):m.validate_dataset(d,'train')
        d=self.sample();d['house'][:]=2;d['block']=np.full(3,5)
        with self.assertRaisesRegex(ValueError,'block'):m.validate_dataset(d,'block_4')

    def test_support_and_predefined_groups_do_not_use_scores(self):
        d=self.sample();strict=m.validate_dataset(d,'train');ident=m.identities(d,strict)
        groups=list(m.group_specs(ident,'block_4'))
        date=[g for level,g,ix in groups if level=='date']
        self.assertEqual(date,['H2|original14d|1970-01-01'])
        support=m.support(ident,np.arange(3))
        self.assertEqual(support['observable_positive_fragments'],1)
        self.assertEqual(support['supported_event_ids'],1)
        self.assertEqual(m.support(ident,np.array([],dtype=int))['observable_positive_fragments'],0)

    def test_incomplete_prediction_matrix(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as path:
            run=Path(path);(run/'predictions').mkdir()
            for f in m.FAMILIES:
                for s in m.SEEDS:
                    for d in m.SETS:
                        (run/f'predictions/{f}_{s}_{d}.npz').touch()
            m.matrix_check(run)
            (run/'predictions/control_42_block_5.npz').unlink()
            with self.assertRaisesRegex(ValueError,'matrix'):m.matrix_check(run)
            with self.assertRaisesRegex(ValueError,'matrix'):m.report_text(run)

    def test_report_chooses_partition_when_dataset_row_also_exists(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as path:
            run=Path(path);(run/'predictions').mkdir()
            rows=[];info={}
            groups=('original_train','train','original_H2','original_H19','original_H5','added_H2','added_H19','block_4','block_5','validation_pooled')
            base=m.metrics(np.array([1,0],bool),np.array([.8,.2],np.float32),.25)
            for f in m.FAMILIES:
                for s in m.SEEDS:
                    info[f'{f}_{s}']=dict(threshold=.25,selected_epoch=5,original_updates=3528)
                    for d in m.SETS:(run/f'predictions/{f}_{s}_{d}.npz').touch()
                    for view in m.VIEWS:
                        for g in groups:
                            level='dataset' if g=='train' else ('aggregate' if g in ('original_train','validation_pooled') else 'partition')
                            row=dict(family=f,seed=s,group=g,view=view,level=level,observable_positive_fragments=1,**base)
                            rows.append(row)
                            if g.startswith('block_'):rows.append(row|dict(level='dataset'))
            m.fresh(run/'metrics.json',rows);m.fresh(run/'model_identity.json',info)
            text=m.report_text(run)
            self.assertIn('每种子数值',text)
            self.assertIn('House5前缀',text)

    def test_exact_interface_and_partial_forward(self):
        load,ordinary,tensor=m.runtime()
        from virtual_submeter.models.center_mask_tcn import CenterMaskTCN
        model=CenterMaskTCN().eval()
        meta=dict(window_samples=127,cpu_threads=4,batch_size=256,input_log_mean=5.370561599731445,input_log_std=.8337093591690063)
        d=self.sample();d['x'][1,49]=np.nan;d['mask'][1,49]=False
        strict=m.validate_dataset(d,'train')
        old=torch.is_grad_enabled();torch.set_grad_enabled(False)
        try:
            p=m.infer(meta,model,d,strict,ordinary,tensor)
            np.testing.assert_array_equal(p[strict],ordinary(meta,model,d['x'][strict]))
            with torch.inference_mode():q=torch.sigmoid(model(tensor(meta,d['x'][~strict],d['mask'][~strict]))).numpy()
            np.testing.assert_array_equal(p[~strict],q)
            with self.assertRaises(ValueError):ordinary(meta,model,d['x'])
        finally:torch.set_grad_enabled(old)

    def test_readback_preserves_original_incomplete_batch(self):
        for n in (259,512,519):
            ids=np.arange(n)
            first,last=m.ordinary_readback_batches(ids,'train')
            np.testing.assert_array_equal(first,ids[:256])
            self.assertEqual(len(last),n%256 or 256)
            self.assertEqual(int(last[0])%256,0)
        load,ordinary,tensor=m.runtime()
        from virtual_submeter.models.center_mask_tcn import CenterMaskTCN
        model=CenterMaskTCN().eval()
        meta=dict(window_samples=127,cpu_threads=4,batch_size=256,input_log_mean=5.370561599731445,input_log_std=.8337093591690063)
        x=np.random.default_rng(7).uniform(0,2000,(259,127)).astype(np.float32)
        p=ordinary(meta,model,x)
        for ix in m.ordinary_readback_batches(np.arange(259),'train'):
            np.testing.assert_array_equal(ordinary(meta,model,x[ix]),p[ix])

    def test_exact_path_isolation(self):
        run=m.ROOT/'tmp/task19_fake_output'
        inputs={p.resolve() for p in m.allowed_inputs()}
        m.check_open(m.T18/'train.npz','rb',0,run,inputs)
        for target,mode in ((m.ROOT/'tmp/not_whitelisted.npz','rb'),(m.T18/'train.npz','wb'),(m.ROOT/'tmp/not_whitelisted.py','r')):
            with self.assertRaises(PermissionError):m.check_open(target,mode,0,run,inputs)
        m.check_open(run/'a.json','x',0,run,inputs)
        with self.assertRaisesRegex(PermissionError,'read-only'):
            m.check_open(run/'a.json','x',0,run,inputs,readonly=True)

    def test_delivery_hash_mtime_replay(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as path:
            run=Path(path);p=run/'a.txt';p.write_text('original',encoding='utf-8')
            entries=m.snapshot([p])
            m.fresh(run/'delivery_manifest.json',dict(status='complete_fixed_diagnostic',artifacts={str(Path(k).relative_to(run)):v for k,v in entries.items()}))
            m.verify_delivery_manifest(run)
            p.write_text('altered',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash/mtime'):m.verify_delivery_manifest(run)

    def test_cache_threshold_mapping_and_nan_tamper(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as path:
            run=Path(path);(run/'predictions').mkdir();(run/'identities').mkdir()
            model=run/'model';model.mkdir()
            m.fresh(model/'model.json',{'decision_threshold_probability':.25})
            (model/'weights.pt').write_bytes(b'synthetic no weights loaded')
            data=run/'data.npz';data.write_bytes(b'synthetic identity only')
            (run/'identities/train.npz').write_bytes(b'identity')
            binding=dict(family='control',seed=42,dataset='train',threshold=.25,
                model_json_sha256=m.sha(model/'model.json'),weights_sha256=m.sha(model/'weights.pt'),
                input_sha256=m.sha(data),identity_sha256=m.sha(run/'identities/train.npz'))
            dest=run/'predictions/control_42_train.npz'
            def write_cache(bind,probability=np.array([.2,.7],np.float32),ix=np.array([0,1])):
                np.savez_compressed(dest,probability=probability,row_index=ix,binding_json=np.array(json.dumps(bind)))
                side=bind|dict(prediction_sha256=m.sha(dest),probability_sha256=m.array_sha(probability))
                dest.with_suffix('.json').write_text(json.dumps(side),encoding='utf-8')
            with patch.object(m,'model_path',return_value=model),patch.object(m,'data_path',return_value=data):
                write_cache(binding)
                m.load_prediction(run,'control',42,'train')
                write_cache(binding|dict(threshold=.5))
                with self.assertRaisesRegex(ValueError,'threshold'):m.load_prediction(run,'control',42,'train')
                write_cache(binding,ix=np.array([0,0]))
                with self.assertRaisesRegex(ValueError,'mapping'):m.load_prediction(run,'control',42,'train')
                write_cache(binding,probability=np.array([np.nan,.7],np.float32))
                with self.assertRaisesRegex(ValueError,'probabilities'):m.load_prediction(run,'control',42,'train')

    def test_live_backward_optimizer_and_io_guard(self):
        code='''import importlib.util, pathlib, torch
p=pathlib.Path(%r)
s=importlib.util.spec_from_file_location('diag',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.isolation(m.ROOT/'tmp/task19_guard_output')
for fn,args in [(torch.autograd.backward,()),(torch.autograd.grad,()),(torch.optim.Optimizer.__init__,(None,[],{}))]:
 try:fn(*args)
 except RuntimeError as e:assert 'prohibits' in str(e)
 else:raise AssertionError('guard missing')
try:(m.ROOT/'tmp/not_allowed.npz').open('rb')
except PermissionError:pass
else:raise AssertionError('read guard missing')
assert not torch.is_grad_enabled()
print('guard passed')
''' % str(PATH)
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('guard passed',result.stdout)


if __name__=='__main__':unittest.main()
