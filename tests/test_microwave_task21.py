"""Task21 high-risk synthetic checks. Never instantiate an optimizer or fit."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_task21 as m
from virtual_submeter.accuracy import microwave_task21_review as independent


class Task21Tests(unittest.TestCase):
    def sample(self,n=80):
        return dict(y=np.zeros(n,np.float32),house=np.zeros(n,np.int64),
            t=np.arange(n,dtype=np.int64)*8,source_role=np.full(n,'original14d'))

    def test_threshold_ties_missing_and_on(self):
        d=self.sample(5);d['y'][:]=[0,199,200,np.nan,0]
        prob=np.array([.5,.5,.9,.9,.49])
        hard=np.isfinite(d['y'])&(d['y']<200)&(prob>=.5)
        self.assertEqual(np.flatnonzero(hard).tolist(),[0,1])
        self.assertEqual([v.tolist() for v in m.fragments(d,hard)],[[0,1]])
        with self.assertRaisesRegex(Exception,'missing/on'):m.fragments(d,np.ones(5,bool))

    def test_boundaries_gap_date_house_role_and_31(self):
        d=self.sample();d['t'][40:]+=8;d['source_role'][60:]='other'
        hard=np.ones(80,bool);hard[3]=False
        a=m.fragments(d,hard);b=independent.independent_parts(d,hard)
        self.assertEqual([v.tolist() for v in a],[v.tolist() for v in b])
        self.assertTrue(all(len(v)<=31 for v in a))
        d=self.sample(8);d['t']+=86376;d['house'][6:]=1
        a=m.fragments(d,np.ones(8,bool))
        self.assertEqual([v.tolist() for v in a],[[0,1,2],[3,4,5],[6,7]])

    def test_singleton_and_empty(self):
        d=self.sample(1);part=m.fragments(d,np.ones(1,bool))
        np.testing.assert_array_equal(m.HardStream(d,part,42).next(50),np.zeros(50,np.int64))
        with self.assertRaisesRegex(Exception,'Empty hard pool'):m.HardStream(d,[],42).next(1)

    def test_independent_rng_queues_and_global_rng_unchanged(self):
        d=self.sample(110);d['house'][40:80]=1;d['house'][80:]=2
        d['t'][20:40]+=86400;d['t'][60:80]+=86400
        hard=np.ones(110,bool);hard[10]=False
        parts=m.fragments(d,hard)
        np.random.seed(11);before=copy.deepcopy(np.random.get_state())
        for seed in m.SEEDS:
            a=m.HardStream(d,parts,seed)
            actual=np.r_[a.next(77),a.next(490)]
            np.testing.assert_array_equal(actual,independent.independent_sequence(d,parts,seed,567))
        after=np.random.get_state()
        self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])

    def test_slots_exact_no_shuffle_and_exposure(self):
        d=self.sample(300);d['y'][:16]=200
        original=np.stack([np.roll(np.arange(256),n) for n in (0,7,121)])
        parts=m.fragments(d,d['y']<200)
        changed,slots,added=m.replace_batches(original,d['y'],m.HardStream(d,parts,42))
        for i,row in enumerate(original):
            neg=np.flatnonzero(d['y'][row]<200);pos=np.flatnonzero(d['y'][row]>=200)
            np.testing.assert_array_equal(row[pos],changed[i,pos])
            np.testing.assert_array_equal(row[neg[:120]],changed[i,neg[:120]])
            np.testing.assert_array_equal(slots[i],neg[120:])
        e=m.exposure(d,parts,original,changed,slots,added)
        np.testing.assert_array_equal(e['total'],e['hard']+e['ordinary']+e['positive'])
        self.assertEqual(int(e['hard'].sum()),360)
        self.assertAlmostEqual(e['expected_hard'].sum(),360)

    def test_wrong_indices_quota_missing(self):
        d=self.sample(300);d['y'][:16]=200;parts=m.fragments(d,d['y']<200)
        for old in (np.arange(256)[None,:]+100,np.zeros((1,256),np.int64)):
            with self.assertRaises(Exception):m.replace_batches(old,d['y'],m.HardStream(d,parts,42))
        d['y'][20]=np.nan
        with self.assertRaises(Exception):m.replace_batches(np.arange(256)[None,:],d['y'],m.HardStream(d,parts,42))

    def test_execution_auth_precedes_everything(self):
        with patch.object(m,'registered',side_effect=AssertionError('read before auth')),patch.object(m,'execution_allowed',side_effect=AssertionError('read before auth')):
            for name in ('train','source_select','freeze','evaluate'):
                with self.assertRaisesRegex(Exception,'NEW human budget'):getattr(m,name)(Path('missing'))

    def test_old_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            receipt=Path(td)/'old.json'
            receipt.write_text(json.dumps(dict(authorized=True,approval_origin='human',experiment='microwave_source5_loss_v1')),encoding='utf-8')
            with self.assertRaisesRegex(Exception,'Missing NEW'):m.authorize(Path(td),receipt)

    def test_preparation_blocks_optimizer_backward_and_readonly_writes(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            run=Path(td);file=run/'test.json';file.write_text('{}',encoding='utf-8')
            with m.isolation(run,[file],readonly=True):
                with self.assertRaisesRegex(Exception,'optimizer/autograd'):torch.optim.Adam([])
                with self.assertRaisesRegex(Exception,'optimizer/autograd'):torch.autograd.backward([])
                with self.assertRaisesRegex(Exception,'write forbidden'):file.write_text('bad',encoding='utf-8')
                with self.assertRaisesRegex(Exception,'exact whitelist'):(run/'not_allowed.npz').open('rb')

    def test_completed_directory_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            with self.assertRaisesRegex(Exception,'cannot be overwritten'):m.prepare(Path(td))

    def test_source_rejects_before_development_identity(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            with patch.object(m,'authorize',return_value={}),patch.object(m,'execution_allowed',return_value=[]),patch.object(m,'isolation',return_value=contextlib.nullcontext()),patch.object(m,'source_guard',side_effect=m.DataError('Source rejected: 0 development reads')),patch.object(m,'read_json',side_effect=AssertionError('development read')):
                with self.assertRaisesRegex(Exception,'Source rejected'):m.freeze(Path(td),'new')

    def test_training_routes_sealed_indices_without_real_fit(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            run=Path(td);indices=np.arange(256)[None,:];calls=[]
            proof={str(s):dict(hard_total=1) for s in m.SEEDS}
            def read(p):
                if Path(p).name=='sampling_proof.json':return proof
                if Path(p).name=='parent_training_plan.json':return {'checkpoints':12,'steps_per_checkpoint':294}
                return {}
            def fakefit(*args,**kw):
                calls.append((args,kw));np.testing.assert_array_equal(args[6],indices)
                self.assertEqual(kw['pos_weight'],1.);return dict(seed=args[1])
            def arrays(p,fields=None):return dict(indices=indices) if 'batches_' in str(p) else {}
            with patch.object(m,'authorize',return_value={'new':True}),patch.object(m,'execution_allowed',return_value=[]),patch.object(m,'isolation',return_value=contextlib.nullcontext()),patch.object(m,'registered'),patch.object(m,'read_json',side_effect=read),patch.object(m,'load',side_effect=arrays),patch.object(m,'fresh'),patch.object(m,'sha256',return_value='sealed'),patch.object(m,'inventory',return_value={}),patch.object(torch,'load',return_value={}),patch.object(m.loss.temporal,'fit_one',side_effect=fakefit):
                result=m.train(run,'new_receipt')
            self.assertEqual(result,dict(new_fits=3,updates=10584));self.assertEqual(len(calls),3)

    def test_incomplete_freeze_matrix_before_truth(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            with patch.object(m,'source_guard'),patch.object(m,'read_json',return_value=dict(artifacts={},matrix=[],truth_opened=False)):
                with self.assertRaisesRegex(Exception,'Incomplete/unbound'):m.evaluation_guard(Path(td))

    def test_authorization_valid_new_receipt_routes(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            run=Path(td);receipt=run/'new.json'
            a=dict(authorized=True,approval_origin='human',experiment='microwave_task21_hard_negative_v1',registration_sha256='abc',additional_new_fits=3,maximum_updates=10584,seeds=m.SEEDS,human_authorization_quote='new approval',human_authorization_reference='new user turn')
            receipt.write_text(json.dumps(a),encoding='utf-8')
            original=m.read_json
            def read(p):
                if Path(p)==receipt:return original(p)
                return {'artifacts':{}} if Path(p).name=='preparation_manifest.json' else {'passed':True}
            with patch.object(m,'sha256',return_value='abc'),patch.object(m,'read_json',side_effect=read):self.assertEqual(m.authorize(run,receipt),a)

    def test_public_ordinary_recipe_gate_and_missing_support(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            root=Path(td);model=root/'model';model.mkdir()
            (model/'model.json').write_text('{}',encoding='utf-8');(model/'weights.pt').write_bytes(b'static-fixture')
            recipe=dict(schema='event_gate_recipe_v1',eligible=True,L=.3,H=.7,radius_samples=15,interval_seconds=8,
                score_semantics='uncalibrated sigmoid state score',parent=str(model),
                parent_hashes={k:m.sha256(model/k) for k in ('model.json','weights.pt')},algorithm_sha256=m.sha256(Path(m.gate.__file__)))
            path=root/'recipe.json';m.fresh(path,recipe)
            t=np.array([0,8,16,24,32,80,88]);p=np.array([.4,.9,np.nan,.4,.5,.4,.8],np.float32)
            with patch.object(m.gate,'infer',return_value=(t,p,{})):
                out=m.gate.ordinary(path,[{'aggregate':'synthetic'}])
            np.testing.assert_array_equal(out['h'],independent.local_max_independent(t,p))
            np.testing.assert_array_equal(out['state'],np.array([1,1,-1,0,0,1,1],np.int8))
            recipe['radius_samples']=16;(root/'recipe.json').write_text(json.dumps(recipe),encoding='utf-8')
            with self.assertRaisesRegex(Exception,'Rule identity'):m.gate.ordinary(path,[])

    def test_independent_scalar_counts_and_eight_gates(self):
        rows=[]
        for seed in m.SEEDS:
            for arm,pred in (('A',[1,0,1,0]),('W50',[1,1,1,0]),('hard_negative',[1,0,1,0])):
                state=np.array(pred,np.int8);truth=np.array([1,0,1,0],bool)
                independent_row=independent.scalar_metric(state,truth)
                self.assertEqual(independent_row,m.gate.confusion(state,truth))
                rows.append(dict(seed=seed,arm=arm,**independent_row))
        supports=[dict(seed=s,n=137523) for s in m.SEEDS]
        expected=independent.independent_acceptance(rows,supports)
        with patch.object(m.loss,'read_json',return_value={'selected':'hard_negative'}):
            actual=m.loss.summary(Path('synthetic'),dict(rows=rows,supports=supports))
        self.assertEqual(expected,actual['acceptance_gates']);self.assertTrue(all(expected.values()))
        with self.assertRaisesRegex(Exception,'9-cell'):independent.independent_acceptance(rows[:-1],supports)

    def test_successful_source_adapter_no_fit(self):
        # Static scores exercise L95/H95/r15 routing and a complete three-seed
        # source selection. No network inference or optimizer is constructed.
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            run=Path(td);(run/'models').mkdir()
            y=np.array([200,0,0,200,0],np.float32);t=np.arange(5)*8
            scores=np.array([.9,.2,.1,.8,.1],np.float32)
            raw=[dict(timestamp=str(int(i))) for i in t]
            def read(p):
                n=Path(p).name
                if n=='training_complete.json':return dict(new_fits=3,updates=10584,artifacts={},rows=[dict(source_gate_passed=True)]*3)
                if n=='source5_inputs.json':return {'aggregate':{'path':'synthetic.csv'}}
                if n=='source_control.json':return {'rows':{'guard95':[dict(seed=s,precision=.1,recall=1.,f1=.1) for s in m.SEEDS]}}
                if n=='model.json':return dict(selected={'epoch':1},decision_threshold_probability=.5)
                return {}
            def arrays(p,fields=None):return dict(t=t,y=y) if 'validation' in str(p) else dict(t=t,scores=scores)
            with patch.object(m,'authorize',return_value={}),patch.object(m,'execution_allowed',return_value=[]),patch.object(m,'isolation',return_value=contextlib.nullcontext()),patch.object(m,'registered'),patch.object(m,'read_json',side_effect=read),patch.object(m,'load',side_effect=arrays),patch.object(m,'sha256',return_value='fixture'),patch.object(m.gate,'csv_rows',return_value=raw),patch.object(m.loss,'utc',side_effect=int),patch.object(m.loss.temporal.f,'VALIDATION',[(4,0,100),(5,0,100)]),patch.object(m.gate,'infer',return_value=(t,scores,{})),patch.object(m.gate,'check'):
                result=m.source_select(run,'new')
            self.assertTrue(result['source_eligible']);self.assertEqual(result['development_reads'],0)
            selected=json.loads((run/'source_selection.json').read_text(encoding='utf-8'))
            self.assertEqual(len(selected['recipes']),3)
            self.assertTrue(all((run/f'source_results/recipe_{s}.json').exists() for s in m.SEEDS))

    def test_successful_freeze_adapter_public_api_static_matrix(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            run=Path(td);(run/'source_results').mkdir()
            t=np.arange(6)*8;p=np.array([.4,.9,np.nan,.2,.3,.1],np.float32)
            raw=np.tile([2001.,100.,22.],(6,1));raw[2]=np.nan
            source=dict(recipes=[dict(seed=s,L=.3,H=.7) for s in m.SEEDS])
            def read(path):
                if Path(path)==m.T14/'identity.json':return dict(development_unlabeled={'synthetic/aggregate.csv':'fixture'},baseline='synthetic/base')
                return {}
            def arrays(path,fields=None):return dict(stamps=t,raw=raw,corrected=raw)
            def ordinary(path,rows):
                h=independent.local_max_independent(t,p)
                return dict(t=t,scores=p,h=h,state=np.where(np.isfinite(p),(p>=.3)&(h>=.7),-1).astype(np.int8))
            with patch.object(m,'authorize',return_value={}),patch.object(m,'execution_allowed',return_value=[]),patch.object(m,'isolation',return_value=contextlib.nullcontext()),patch.object(m,'source_guard',return_value=source),patch.object(m,'read_json',side_effect=read),patch.object(m,'load',side_effect=arrays),patch.object(m.loss,'check_paths'),patch.object(m.gate,'csv_rows',return_value=[]),patch.object(m.gate,'infer',return_value=(t,p,{})),patch.object(m.gate,'ordinary',side_effect=ordinary) as public,patch.object(m,'sha256',return_value='fixture'):
                result=m.freeze(run,'new')
            self.assertEqual(result,dict(frozen=True,truth_reads=0));self.assertEqual(public.call_count,3)
            files=list((run/'predictions').glob('*.npz'));self.assertEqual(len(files),9)
            with np.load(run/'predictions/hard_negative_42.npz') as z:
                np.testing.assert_array_equal(z['state'],[1,1,-1,0,0,0])
                np.testing.assert_array_equal(z['other_states'][:,[0,2]],np.where(np.isfinite(raw[:,[0,2]]),raw[:,[0,2]]>=np.array([2000,20]),-1))

    def test_successful_evaluate_adapter_static_137523_points(self):
        # Generated fixture has the formal support size; it contains no real
        # development values and checks the one-shot evaluation routing.
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            run=Path(td);n=137523;t=np.arange(n)*8
            y=np.zeros(n,np.float32);y[::100]=200
            raw=np.zeros((n,3),np.float32);raw[::100,1]=200
            base=raw.copy();base[1::100,1]=200
            state=(y>=200).astype(np.int8);other=np.zeros((n,3),np.int8);other[:,1]=state
            def read(path):
                if Path(path)==m.T14/'identity.json':return dict(truth_cache=dict(path='synthetic_truth.npz',sha256='fixture'))
                return {'selected':'hard_negative'}
            def arrays(path,fields=None):
                p=str(path)
                if p=='synthetic_truth.npz':return dict(t=t,y=y)
                if 'hard_negative_' in p:return dict(t=t,scores=state.astype(np.float32),state=state,raw=raw,corrected=raw,other_states=other)
                if '/W50_' in p.replace('\\','/'):return dict(t=t,raw=base)
                return dict(t=t,raw=raw,corrected=raw,states=other)
            with patch.object(m,'authorize',return_value={}),patch.object(m,'execution_allowed',return_value=[]),patch.object(m,'isolation',return_value=contextlib.nullcontext()),patch.object(m,'evaluation_guard'),patch.object(m,'read_json',side_effect=read),patch.object(m,'load',side_effect=arrays),patch.object(m,'sha256',return_value='fixture'),patch.object(m.loss,'read_json',return_value={'selected':'hard_negative'}):
                result=m.evaluate(run,'new')
            self.assertTrue(result['development_acceptance_passed'])
            summary=json.loads((run/'evaluation/summary.json').read_text(encoding='utf-8'))
            metrics=json.loads((run/'evaluation/metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['acceptance_gates'],independent.independent_acceptance(metrics['rows'],metrics['supports']))


if __name__=='__main__':unittest.main()
