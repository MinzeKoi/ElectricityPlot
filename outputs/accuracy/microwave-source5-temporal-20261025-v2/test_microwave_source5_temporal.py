"""Synthetic-only tests: no source/target household files are opened."""
import copy
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from virtual_submeter.accuracy import microwave_source5_temporal as f
from virtual_submeter.data.common import DataError, read_json, write_json


def grid(block=0):
    rows = f.TRAIN+f.VALIDATION
    _, low, high = next(r for r in rows if r[0] == block)
    # A small synthetic grid straddles the approved block start.
    t = np.arange(low-640,low+1600,8,dtype=np.int64)
    a = np.full(len(t),500.,np.float32)
    y = np.zeros(len(t),np.float32)
    y[75:85] = 300  # boundary-crossing segment: must vanish in its entirety.
    y[120:126] = 300
    y[160:166] = 300
    return t,a,y


def weights_fixture():
    return dict(y=np.array([300]*7+[0]*5,dtype=np.float32),house=np.array([0]*5+[2]*7),
                t=np.array([0,8,16,24,86400,0,8,16,24,32,40,48]),
                event=np.array([0,0,0,1,2,0,0,-1,-1,-1,-1,-1]))


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.p = read_json(f.PLAN)

    def test_exact_permit(self):
        for block,low,high in f.TRAIN:
            f.permit('refit_5','train','aggregate',block,low,high)
            for house,role,channel,b,l,h in [('refit_17','train','aggregate',block,low,high),
                    ('refit_8','train','aggregate',block,low,high),('refit_9','train','aggregate',block,low,high),
                    ('refit_20','train','aggregate',block,low,high),('refit_5','train','kettle',block,low,high),
                    ('refit_5','validation','aggregate',block,low,high),('refit_5','train','aggregate',block,low,high+8)]:
                with self.assertRaises(DataError): f.permit(house,role,channel,b,l,h)
        for b,l,h in f.VALIDATION:
            with self.assertRaises(DataError): f.permit('refit_5','train','microwave',b,l,h)
        self.assertEqual(f.VALIDATION[0][1]-f.TRAIN[-1][2],1800)

    def test_core_outer_event_and_unknown(self):
        t,a,y = grid()
        a[140] = np.nan
        d = f.build_block(t,a,y,0,'train')
        self.assertTrue((d['house']==2).all())
        self.assertTrue((d['house_name']=='refit_5').all())
        self.assertFalse(np.isin(t[75:85],d['t']).any())
        self.assertGreater(int((d['y']>=200).sum()),0)
        self.assertTrue(np.isnan(d['x']).any())
        np.testing.assert_array_equal(d['mask'],np.isfinite(d['x']))
        used = d['t'][:,None]+np.arange(-63,64)*8
        self.assertTrue((used[d['mask']] >= f.TRAIN[0][1]).all())
        self.assertTrue((used[d['mask']] < f.TRAIN[0][2]).all())
        core = d['mask'][:,48:79]
        self.assertTrue((core.sum(axis=1)>=24).all())
        self.assertFalse(((~core[:,:-1]) & (~core[:,1:])).any())
        a[139:141] = np.nan
        d = f.build_block(t,a,y,0,'train')
        self.assertFalse(np.isin(t[126:155], d['t']).any())
        # Outer radius stops before the second consecutive unknown.
        row = np.flatnonzero(d['t'] == t[123])[0]
        self.assertFalse(d['mask'][row,63+17:].any())

    def test_time_break_and_tail_boundary(self):
        t,a,y = grid()
        t[140:] += 8
        d = f.build_block(t,a,y,0,'train')
        self.assertFalse(np.isin(t[125:155],d['t']).any())
        _,low,high = f.TRAIN[3]
        t = np.arange(high-1600,high+640,8)
        a = np.ones(len(t),np.float32)*500
        y = np.zeros(len(t),np.float32)
        y[195:205] = 300
        d = f.build_block(t,a,y,3,'train')
        self.assertFalse((d['y']>=200).any())
        used = d['t'][:,None]+np.arange(-63,64)*8
        self.assertTrue((used[d['mask']]<high).all())

    def test_validation_frozen_support_and_identity(self):
        t,a,y = grid(4)
        ix = np.arange(100,240)
        frozen = dict(t=t[ix],y=y[ix],x=np.lib.stride_tricks.sliding_window_view(a,31)[ix-15])
        d = f.build_block(t,a,y,4,'validation',frozen)
        np.testing.assert_array_equal(d['t'],frozen['t'])
        np.testing.assert_array_equal(d['y'],frozen['y'])
        with self.assertRaises(DataError): f.build_block(t,a,y,4,'validation')
        frozen['t'][0] = f.VALIDATION[0][1]+112
        with self.assertRaises(DataError): f.build_block(t,a,y,4,'validation',frozen)

    def test_append_preserves_every_original_row_and_normalization(self):
        t,a,y=grid()
        added=f.build_block(t,a,y,0,'train')
        old={k:v[:12].copy() for k,v in added.items()}
        old['house'][:]=1;old['house_name'][:]='refit_19'
        norm=dict(input_log_mean=5.,input_log_std=1.)
        combined=f.append_training(old,added,norm,norm.copy())
        for k,v in old.items(): np.testing.assert_array_equal(v,combined[k][:12])
        with self.assertRaises(DataError): f.append_training(old,added,norm,dict(input_log_mean=6.,input_log_std=1.))
        added['house'][:]=1
        with self.assertRaises(DataError): f.append_training(old,added,norm,norm)

    def test_fixed_prior_expectation_not_merged_prior(self):
        d=weights_fixture()
        actual,target,w=f.fixed_prior_weights(d,self.p)
        positive=d['y']>=200
        self.assertAlmostEqual(float(target[positive].sum()),849/259806)
        self.assertNotAlmostEqual(float(target[positive].sum()),positive.mean())
        self.assertAlmostEqual(float(actual[positive].sum()),.0625)
        logits=torch.linspace(-3,2,len(positive),dtype=torch.float64)
        y=torch.tensor(positive,dtype=torch.float64)
        point=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(50.),reduction='none')
        loss=f.point_loss(logits,y,'importance_BCE50',torch.tensor(w))
        self.assertAlmostEqual(float((torch.tensor(actual)*loss).sum()),float((torch.tensor(target)*point).sum()),places=12)
        self.assertAlmostEqual(50*f.P_REF/(50*f.P_REF+1-f.P_REF),50*849/(50*849+259806-849))
        for recipe in ('unit_BCE','control_BCE50'):
            got=f.point_loss(logits,y,recipe,torch.tensor(w))
            expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(1. if recipe=='unit_BCE' else 50.),reduction='none')
            torch.testing.assert_close(got,expected)
        bad=copy.deepcopy(d);bad['y'][0]=np.nan
        with self.assertRaises(DataError): f.fixed_prior_weights(bad,self.p)
        bad=copy.deepcopy(d);bad['t'][1]=bad['t'][0]
        with self.assertRaises(DataError): f.fixed_prior_weights(bad,self.p)

    def test_two_block_checkpoint_selection_only_and_earlier_tie(self):
        blocks={}
        for b,low,high in f.VALIDATION:
            blocks[f'block_{b}']=dict(t=low+np.arange(20)*8+120,
                y=np.r_[np.full(6,300),np.zeros(14)],event=np.r_[0,0,0,1,1,1,np.full(14,-1)])
        scores={k:np.r_[np.full(6,.8),np.full(14,.2)] for k in blocks}
        point=f.reselect_checkpoints([scores]*12,blocks,self.p)
        self.assertEqual(point['epoch'],1)
        self.assertEqual(point['supported_blocks'],['block_4','block_5'])
        self.assertEqual(point['threshold'],.8)
        self.assertIsNone(f.reselect_checkpoints([{k:np.ones(20)*.5 for k in blocks}]*12,blocks,self.p))
        with self.assertRaises(DataError): f.reselect_checkpoints([scores]*11,blocks,self.p)
        with self.assertRaises(DataError): f.reselect_checkpoints([scores]*12,{**blocks,'block_0':blocks['block_4']},self.p)
        blocks['block_5']['t'][0]=f.VALIDATION[1][1]+112
        with self.assertRaises(DataError): f.reselect_checkpoints([scores]*12,blocks,self.p)

    def test_matrix_source_and_development_gates(self):
        old=[dict(seed=s,precision=.1,recall=.9,f1=.18,source_gate_passed=True,degenerate=False,fp=100) for s in f.SEEDS]
        new=[dict(seed=s,precision=.2,recall=.9,f1=.3,source_gate_passed=True,degenerate=False,fp=60) for s in f.SEEDS]
        self.assertEqual(f.select_source(dict(control=old,source5=new))['selected_arm'],'source5')
        self.assertEqual(f.select_source(dict(control=old,source5=copy.deepcopy(old)))['selected_arm'],'control')
        self.assertTrue(f.development_gate(old,new,self.p))
        new[0]['recall']=.84
        self.assertFalse(f.development_gate(old,new,self.p))
        self.assertEqual(f.select_source(dict(control=old,source5=new))['selected_arm'],'control')
        new[0]['recall']=.9;new[0]['degenerate']=True
        self.assertFalse(f.development_gate(old,new,self.p))
        with self.assertRaises(DataError): f.select_source(dict(control=old,source5=new[:2]))
        for r in old+new: r['source_gate_passed']=False
        self.assertIsNone(f.select_source(dict(control=old,source5=new))['selected_arm'])

    def test_pending_condition_zero_data_prepare_and_mutation(self):
        with self.assertRaises(DataError): f.condition_receipt({})
        receipt=dict(task8_development_passed=False,source_recipe='unit_BCE',maximum_new_fits=3,
                     coordinator_message='synthetic test only',source_thread_id='01a16d7b-478e-7780-a999-88abe71e8c94',
                     task8_source_selection_sha256='a'*64,task8_freeze_sha256='b'*64)
        self.assertEqual(f.condition_receipt(receipt),'unit_BCE')
        receipt['task8_development_passed']=True
        with self.assertRaises(DataError): f.condition_receipt(receipt)
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'prepared'
            result=f.prepare(run)
            self.assertEqual(result['actual_real_fits'],0)
            self.assertFalse(list(run.rglob('*.npz')))
            for stage in ('train','source_select','freeze','evaluate'):
                with self.assertRaises(DataError): f.pending_stage(run,stage)
            with self.assertRaises(FileExistsError): f.prepare(run)
            path=run/'configs/microwave_source5_temporal_20261025.json'
            path.write_text('{}',encoding='utf-8')
            with self.assertRaises(DataError): f.verify(run)

    def test_all_recipe_synthetic_train_save_ordinary_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            for recipe in f.RECIPES:
                result=f.synthetic_roundtrip(Path(tmp)/recipe,recipe)
                self.assertTrue(result['ordinary_load_prediction_equal'])
                self.assertEqual(result['parameter_count'],5329)
                self.assertEqual(result['real_fits'],0)


if __name__=='__main__': unittest.main()
