"""Synthetic fixtures only. No real H95 calibration or development reads."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from virtual_submeter.accuracy import microwave_recall_guard95 as e
from virtual_submeter.data.common import DataError, write_json, read_json, sha256, iso


def metrics(p=.2,r=.98,f=.3):
    return [dict(seed=s,precision=p,recall=r,f1=f,degenerate=False,source_gate_passed=True) for s in e.SEEDS]


class Recall95Tests(unittest.TestCase):
    def test_disabled_no_fit(self):
        c=read_json(e.ROOT/e.CONFIG)
        self.assertFalse(c['authorized'])
        self.assertEqual((c['new_fits'],c['optimizer_updates'],c['high_minimum_recall']),(0,0,.95))

    def test_all_execution_entrypoints_reject_before_identity_or_data(self):
        for name in ('source_select','freeze','evaluate','verify','report','predict'):
            with self.subTest(name=name), patch.object(e,'registered',side_effect=AssertionError('data touched')):
                with self.assertRaisesRegex(DataError,'explicit human approval'):
                    getattr(e,name)(Path('nonexistent-synthetic-run'))

    def test_old_fit_authorization_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            write_json(p/'registration.json',{})
            write_json(p/'old.json',dict(authorized=True,approval_origin='human',experiment='microwave_source5_loss_v1',additional_new_fits=3))
            with self.assertRaises(DataError): e.authorize(p,p/'old.json')

    def test_receipt_bound_and_quote_required(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            write_json(p/'registration.json',{})
            a=dict(authorized=True,approval_origin='human',experiment='microwave_recall_guard95_v1',registration_sha256=sha256(p/'registration.json'),new_fits=0,optimizer_updates=0,seeds=e.SEEDS,
                   approved_rule='fixed_L95_H95_radius15_W50_source_guard',human_authorization_quote='SYNTHETIC TEST ONLY',human_authorization_reference='synthetic-fixture')
            write_json(p/'a.json',a)
            self.assertEqual(e.authorize(p,p/'a.json'),a)
            for key,value in [('registration_sha256','bad'),('new_fits',3),('human_authorization_quote',''),('human_authorization_reference',''),('approved_rule','H85')]:
                bad=a|{key:value}; write_json(p/'a.json',bad)
                with self.assertRaises(DataError): e.authorize(p,p/'a.json')

    def test_zero_optimizer_and_backward(self):
        with e.isolation('prepare',[]):
            for fn in (lambda: torch.tensor(1.,requires_grad=True).backward(),lambda: torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))],lr=.1),lambda: torch.autograd.backward(torch.tensor(1.,requires_grad=True))):
                with self.assertRaises(DataError): fn()

    def test_label_and_final_house_isolation_even_allowlisted(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('frozen_test_refit_17_0.npz','truth_once.npz','house_8.csv','refit_9.npz','refit_20.pt'):
                p=Path(d)/name; p.write_bytes(b'synthetic')
                with e.isolation('source_select',[p]):
                    with self.assertRaises(DataError): sha256(p)
            p=Path(d)/'frozen_test_refit_17_0_microwave.csv';p.write_text('synthetic')
            with e.isolation('evaluate',[p]):
                with self.assertRaises(DataError): sha256(p)

    def test_local_unknown_numeric_denied(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'unknown.npz';p.write_bytes(b'synthetic')
            with e.isolation('prepare',[]):
                with self.assertRaises(DataError): sha256(p)

    def test_95_parameter_is_distinct_from_85(self):
        b={'4':dict(p=np.array([.1,.5,.9]),h=np.array([.1,.8,.9]),ix=np.array([0,1,2]),y=np.array([0,1,1],bool))}
        calls=[]
        def select(blocks,candidates,recall):
            calls.append(recall)
            return (.5 if len(calls)==1 else .8),{}
        with patch.object(e.gate,'select_threshold',side_effect=select):
            self.assertEqual(e.calibrate95(b,.5),.8)
        self.assertEqual(calls,[.95,.95])

    def test_no_different_low_or_relaxation(self):
        b={'4':dict(p=np.array([.1,.8]),h=np.array([.8,.8]),ix=np.array([0,1]),y=np.array([1,0],bool))}
        with patch.object(e.gate,'select_threshold',return_value=(.2,{})):
            with self.assertRaises(DataError): e.calibrate95(b,.3)
        with patch.object(e.gate,'select_threshold',side_effect=[(.2,{}),(None,{})]) as select:
            self.assertIsNone(e.calibrate95(b,.2))
            self.assertEqual(select.call_count,2)

    def test_low_failure_counted_fn_nan_and_block_segments(self):
        p=np.array([.1,.9,np.nan,.2,.1])
        h=e.gate.local_max(np.arange(5)*8,p,block=np.array([4,4,4,5,5]))
        z=e.gate.gate_state(p,h,.5,.8)
        np.testing.assert_array_equal(z,[0,1,-1,0,0])
        self.assertEqual(e.gate.confusion(z[[0,1,3,4]],[1,1,0,0])['fn'],1)
        self.assertEqual(h[3],.2)

    def test_w50_exact_centers_no_filtering_or_rethreshold(self):
        t=np.arange(4)*8
        b={'4':dict(t=t[[0,2]],y=np.array([300,0])),'5':dict(t=t[[1,3]],y=np.array([0,300]))}
        raw=np.zeros((4,3));raw[:,1]=[200,199.9,0,250]
        z=dict(stamps=t,raw=raw)
        m=e.aligned_w50(z,b,t)
        self.assertEqual(m['4']['tp'],1);self.assertEqual(m['5']['tp'],1)
        with self.assertRaises(AssertionError): e.aligned_w50(z,b,t+8)
        raw[0,1]=np.nan
        with self.assertRaises(DataError): e.aligned_w50(z,b,t)

    def test_w50_source_guard_and_ties(self):
        w=metrics();c=metrics(p=.3,r=.95,f=.4)
        self.assertEqual(e.choose(w,c)['selected'],'guard95')
        self.assertEqual(e.choose(w,w)['selected'],'W50')
        for bad in (c[:2],metrics(p=.3,r=.90,f=.4),metrics(p=.3,r=.98,f=.2)):
            self.assertFalse(e.choose(w,bad)['guard95_eligible'])
        c[0]['source_gate_passed']=False
        self.assertFalse(e.choose(w,c)['guard95_eligible'])

    def test_infeasible_matrix_refused(self):
        with self.assertRaises(DataError): e.matrix(dict(recipes=[dict(feasible=False)]))

    def test_synthetic_source_freeze_evaluate_verify_report_predict(self):
        """Full production orchestration with synthetic arrays and mocked model only."""
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'run';run.mkdir()
            base=Path(d)/'base';parent=Path(d)/'parent'
            source=Path(d)/'source.csv'; dev=Path(d)/'aggregate.csv'
            source.write_text('fixture'); dev.write_text('fixture')
            truthpath=Path(d)/'truth_once.npz'
            n=137523;t=np.arange(n,dtype=np.int64)*8
            truth=np.zeros(n);truth[:100]=300
            np.savez_compressed(truthpath,t=t,y=truth)
            devmap={str(dev):sha256(dev)}
            raw=np.zeros((n,3));raw[:,1]=0;raw[:300,1]=300
            for seed in e.SEEDS:
                for group in ('reference/predictions','predictions/W50'):
                    p=base/group/f'seed_{seed}'/e.gate.PARTITION;p.parent.mkdir(parents=True)
                    np.savez_compressed(p,stamps=t,raw=raw,corrected=raw)
                    devmap[str(p)]=sha256(p)
            identity=dict(source_inputs={str(source):sha256(source)},development_unlabeled=devmap,source=dict(aggregate=dict(path=str(source))),parent=str(parent),baseline=str(base),
                truth_cache=dict(path=str(truthpath),sha256=sha256(truthpath)),original_L={str(s):float(np.float32(.8)) for s in e.SEEDS})
            write_json(run/'identity.json',identity);write_json(run/'registration.json',dict(artifacts={}))
            receipt=run/'synthetic-authorize.json'
            write_json(receipt,dict(authorized=True,approval_origin='human',experiment='microwave_recall_guard95_v1',registration_sha256=sha256(run/'registration.json'),new_fits=0,optimizer_updates=0,seeds=e.SEEDS,
                approved_rule='fixed_L95_H95_radius15_W50_source_guard',human_authorization_quote='SYNTHETIC FIXTURE ONLY',human_authorization_reference='synthetic fixture'))
            rows=[dict(timestamp=iso(i*8),household_id='refit_5') for i in range(100)]
            blocks={};score_hash={}
            for k,ix in [('4',np.arange(50)),('5',np.arange(50,100))]:
                bt=ix*8;p=np.full(50,.1,np.float32);p[:5]=.8
                y=np.arange(50)<5
                blocks[k]=dict(t=bt,y=y*300)
                z=dict(t=bt,p=p,h=e.gate.local_max(bt,p),ix=np.arange(50),y=y)
                score_hash[k]={f:e.gate.hash_array(z[f]) for f in z}
            proof=dict(seeds={str(s):dict(scores=score_hash,W50_mean=dict(precision=.2,recall=.98,f1=.3)) for s in e.SEEDS})
            write_json(run/'source_identity_replay.json',proof)
            def csvrows(path,house): return rows if house=='refit_5' else [dict(household_id='refit_17')]
            def infer(model,rr):
                if rr[0]['household_id']=='refit_17':
                    p=np.full(n,.1,np.float32);p[:100]=.8
                    return t,p,{}
                bt=np.array([e.utc(r['timestamp']) for r in rr]);p=np.full(len(bt),.1,np.float32);p[:5]=.8
                return bt,p,{}
            with patch.object(e,'registered',return_value=({},identity)),patch.object(e,'source_blocks',return_value=blocks),patch.object(e.gate,'csv_rows',side_effect=csvrows),patch.object(e.gate,'infer',side_effect=infer),patch.object(e.loss.temporal.f,'VALIDATION',[(4,0,400),(5,400,800)]):
                self.assertEqual(e.source_select(run,receipt)['selected'],'guard95')
                e.freeze(run,receipt)
                self.assertEqual(len(read_json(run/'freeze.json')['matrix']),9)
                # An incomplete matrix is refused before truth hash/read.
                f=read_json(run/'freeze.json');bad=copy.deepcopy(f);bad['matrix'].pop();write_json(run/'freeze.json',bad)
                with self.assertRaises(DataError): e.evaluate(run,receipt)
                self.assertFalse((run/'evaluation').exists())
                write_json(run/'freeze.json',f)
                self.assertTrue(e.evaluate(run,receipt)['development_acceptance_passed'])
                self.assertEqual(e.verify(run,receipt)['new_fits'],0)
                self.assertTrue(e.report(run,receipt)['development_acceptance_passed'])
                with self.assertRaises(FileExistsError): e.evaluate(run,receipt)
                # Public prediction consumes only a CSV and selected frozen rule.
                ordinary=Path(d)/'ordinary.csv'
                ordinary.write_text('household_id,timestamp,aggregate_power_w\nrefit_5,1970-01-01T00:00:00Z,400\n')
                e.predict(run,receipt,seed=42,aggregate=ordinary,output=Path(d)/'prediction.csv')
                self.assertTrue((Path(d)/'prediction.csv').exists())


if __name__=='__main__': unittest.main()
