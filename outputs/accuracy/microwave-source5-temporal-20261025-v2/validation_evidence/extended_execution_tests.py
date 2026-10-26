"""Exercise the real adapter with generated arrays before any real fit."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from virtual_submeter.accuracy import microwave_source5_temporal_execution as e
from virtual_submeter.data.common import DataError,read_json,write_json,sha256
from virtual_submeter.models.center_mask_tcn import load_state_model,predict_windows


def tiny_fixture():
    p={**read_json(e.f.PLAN), 'checkpoints':2,'steps_per_checkpoint':1,'pos_weight':50.}
    rng=np.random.default_rng(17)
    x=rng.uniform(0,2000,(64,127)).astype(np.float32)
    d=dict(x=x,mask=np.isfinite(x),y=np.r_[np.full(16,300),np.zeros(48)].astype(np.float32),
           house=np.r_[np.zeros(32,int),np.ones(16,int),np.full(16,2)],t=np.arange(64)*8,
           event=np.r_[np.arange(16)//4,np.full(48,-1)])
    blocks={}
    for b,l,h in e.f.VALIDATION:
        blocks[f'block_{b}']=dict(x=x[:20].copy(),y=np.r_[np.full(6,300),np.zeros(14)],
            t=l+np.arange(20)*8+120,event=np.r_[0,0,0,1,1,1,np.full(14,-1)])
    norm=dict(input_log_mean=5.,input_log_std=1.,batch_size=256,cpu_threads=4)
    return p,d,blocks,norm


class ExecutionTests(unittest.TestCase):
    def test_actual_fit_selection_and_ordinary_readback(self):
        p,d,blocks,meta=tiny_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'model'
            sampler=e.BatchStream(d,'event',42,p)
            indices=np.stack([sampler.next() for _ in range(2)])
            initial,_=e.parent.configure(42,p)
            row=e.fit_one(dest,42,p,d,blocks,meta,indices,initial.state_dict(),'a'*64)
            self.assertEqual(row['seed'],42)
            info,model=load_state_model(dest)
            self.assertEqual(info['window_samples'],127)
            self.assertEqual(info['updates'],2)
            self.assertEqual(info['recipe'],'control_BCE50')
            selected=info['selected']['epoch']
            scores=[]
            for epoch in (1,2): scores.append({k:e.load_npz(dest/f'epoch_{epoch}_{k}.npz')['scores'] for k in blocks})
            self.assertEqual(e.choose(scores,blocks,p),info['selected'])
            for key,b in blocks.items(): np.testing.assert_array_equal(predict_windows(info,model,b['x']),scores[selected-1][key])
            exposure=e.load_npz(dest/'exposure.npz')['count']
            np.testing.assert_array_equal(exposure,np.bincount(indices.ravel(),minlength=64))
            self.assertEqual(read_json(dest/'progress.json'),dict(completed_updates=2,attempted_updates=2))
            with self.assertRaises(FileExistsError): e.fit_one(dest,42,p,d,blocks,meta,indices,initial.state_dict(),'a'*64)

    def test_failed_update_persists_start_no_retry(self):
        p,d,blocks,meta=tiny_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'failed'
            initial,_=e.parent.configure(42,p)
            indices=np.zeros((2,256),int)  # Invalid quota, fail before optimizer update.
            with self.assertRaises(DataError): e.fit_one(dest,42,p,d,blocks,meta,indices,initial.state_dict(),'a'*64)
            self.assertTrue((dest/'started.json').exists())
            self.assertFalse((dest/'complete.json').exists())
            self.assertEqual(read_json(dest/'progress.json'),dict(completed_updates=0,attempted_updates=1))

    def test_guard_rejects_unregistered_and_truth_before_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'allowed.npz').write_bytes(b'not actually an array')
            (root/'truth_once.npz').write_bytes(b'no truth in this synthetic fixture')
            with e.read_guard(root/'run',[]) as audit:
                with self.assertRaises(DataError): (root/'allowed.npz').read_bytes()
                with self.assertRaises(DataError): (root/'truth_once.npz').read_bytes()
            self.assertEqual(len(audit['forbidden']),2)
            with e.read_guard(root/'run',[root/'allowed.npz']) as audit:
                (root/'allowed.npz').read_bytes()
            self.assertEqual(len(audit['reads']),1)

    def test_eval_guard_complete_matrix_hash_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            for name in ('registration','training_complete','source_selection'): write_json(run/(name+'.json'),{})
            checks=dict(checks=[dict(ordinary_scores_exact=True,ordinary_state_exact=True,raw_power_other_devices_exact=True) for _ in range(6)])
            write_json(run/'inference_checks.json',checks)
            freeze=dict(predictions=[f'{a}/{s}' for a in e.f.ARMS for s in e.f.SEEDS],
                source_selection_sha256=sha256(run/'source_selection.json'),training_complete_sha256=sha256(run/'training_complete.json'),
                registration_sha256=sha256(run/'registration.json'),house17_labels_opened=False,
                artifacts={'inference_checks.json':sha256(run/'inference_checks.json')})
            write_json(run/'freeze.json',freeze)
            with patch.object(e,'registered',return_value=({},{})):
                e.evaluation_guard(run)
                freeze['predictions'].pop();write_json(run/'freeze.json',freeze)
                with self.assertRaises(DataError): e.evaluation_guard(run)

    def test_source_selection_cannot_change_after_written(self):
        row=[dict(seed=s,precision=.2,recall=.9,f1=.3,source_gate_passed=True,degenerate=False) for s in e.f.SEEDS]
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            write_json(run/'source_rows.json',dict(control=row,source5=copy.deepcopy(row)))
            self.assertEqual(e.source_select(run)['selected_arm'],'control')
            value=read_json(run/'source_rows.json');value['source5'][0]['precision']=.5;value['source5'][1]['precision']=.5
            write_json(run/'source_rows.json',value)
            with self.assertRaises(DataError): e.source_select(run)

    def test_execution_plan_only_control_recipe(self):
        p=e.execution_plan()
        self.assertEqual(p['source_recipe'],'control_BCE50')
        self.assertEqual(p['maximum_new_fits'],3)
        self.assertEqual(p['checkpoints']*p['steps_per_checkpoint'],3528)

    def test_default_protocol_still_rejects_train_validation_same_house(self):
        from virtual_submeter.data.protocol import freeze
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'manifest.json',{})
            plan=dict(schema_version=1,protocol_id='synthetic',purpose='engineering_pilot_only',
                source_train=['2','5'],source_validation=['5'],target_households=['17'],
                adaptation_fraction=.5,max_model_context_seconds=1000,
                minimum_aggregate_coverage=1,minimum_label_coverage=1)
            write_json(root/'plan.json',plan)
            with self.assertRaisesRegex(DataError,'must be disjoint'):
                freeze(root,root/'plan.json',root/'output',object())

    def test_real_scoring_exact_support_and_missing_prediction_rejected(self):
        n=137523
        t=np.arange(n,dtype=np.int64)*8
        raw=np.zeros((n,3),np.float32)
        raw[:20,1]=300
        base=dict(stamps=t,raw=raw,aggregate=np.ones(n,np.float32)*500)
        truth=np.zeros(n,np.float64);truth[:10]=300
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'predictions').mkdir()
            for arm in e.f.ARMS:
                for seed in e.f.SEEDS:
                    np.savez_compressed(run/f'predictions/{arm}_{seed}.npz',t=t,raw=raw,scores=np.ones(n,np.float32)*.1,state=(raw[:,1]>=200).astype(np.int8))
            with patch.object(e.parent,'reference_arrays',return_value=base):
                rows,support=e.score(run,truth)
                self.assertEqual(len(rows),12)
                self.assertTrue(all(v['n']==n for v in support.values()))
                candidate=next(r for r in rows if r['arm']=='source5')
                self.assertEqual((candidate['tp'],candidate['fp'],candidate['fn']),(10,10,0))
                data=e.load_npz(run/'predictions/source5_42.npz');data['scores'][30]=np.nan
                np.savez_compressed(run/'predictions/source5_42.npz',**data)
                with self.assertRaises(DataError): e.score(run,truth)

    def test_real_freeze_and_once_evaluate_on_small_synthetic_data(self):
        # Production freeze/evaluate run unchanged; only fixed large-support scorer
        # is replaced by a tiny scorer, separately checked at 137523 above.
        p,d,blocks,meta=tiny_fixture()
        t=e.f.VALIDATION[0][1]+np.arange(100)*8
        aggregate=np.ones(100,np.float32)*500
        raw=np.full((100,3),np.nan,np.float32);raw[15:-15]=0;raw[20:28,1]=300
        base=dict(stamps=t,aggregate=aggregate,raw=raw)
        inputs=[dict(timestamp=e.iso(int(stamp)),household_id='synthetic',aggregate_power_w='500') for stamp in t]
        truth=np.zeros(100);truth[20:28]=300
        def tiny_score(run,actual):
            out=[]
            mask=np.isfinite(raw[:,1])
            for arm in ['A','W50']+e.f.ARMS:
                for seed in e.f.SEEDS:
                    predicted=raw[:,1]>=200 if arm in ('A','W50') else e.load_npz(run/f'predictions/{arm}_{seed}.npz')['state']==1
                    out.append(dict(arm=arm,seed=seed,**e.parent.confusion(predicted[mask],actual[mask]>=200)))
            return out,{str(s):dict(n=70) for s in e.f.SEEDS}
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            for name,value in [('control_identity',dict(run=str(run),models={})),('plan',p),('registration',{}),('training_complete',{})]:
                write_json(run/(name+'.json'),value)
            label=run/'protocol/truth.csv';label.parent.mkdir()
            label.write_text('timestamp,power_w\n'+''.join(f'{e.iso(int(stamp))},{value}\n' for stamp,value in zip(t,truth)),encoding='utf-8')
            baseline=dict(reference_run=str(run),development_partition=dict(artifacts=dict(
                aggregate=dict(path='aggregate.csv'),microwave=dict(path='truth.csv',sha256=sha256(label)))))
            write_json(run/'baseline_identity.json',baseline)
            write_json(run/'baseline_settings.json',dict(on_thresholds_w=dict(kettle=200,washing_machine=200)))
            row=[dict(seed=s,precision=.2,recall=.9,f1=.3,source_gate_passed=True,degenerate=False) for s in e.f.SEEDS]
            selection=e.f.select_source(dict(control=row,source5=copy.deepcopy(row)))
            write_json(run/'source_selection.json',selection)
            for arm in e.f.ARMS:
                for seed in e.f.SEEDS:
                    modeldir=run/f'models/{arm}/seed_{seed}';modeldir.mkdir(parents=True)
                    model,_=e.parent.configure(seed,p)
                    torch.save(model.state_dict(),modeldir/'epoch_1.pt')
                    chosen=dict(epoch=1,threshold=.5,source_gate_passed=True,
                        block_metrics={k:e.metrics(np.r_[np.full(6,.8),np.full(14,.2)],b['y'],.5) for k,b in blocks.items()})
                    e.write_selected(modeldir,seed,arm,chosen,meta,
                                     modeldir/'epoch_1.pt',sha256(run/'registration.json'),0)
            with patch.object(e,'registered',return_value=(p,{})),patch.object(e,'verify_training',return_value=True),\
                 patch.object(e,'verify_references',return_value=None),patch.object(e.parent,'target_input',return_value=inputs),\
                 patch.object(e.parent,'reference_arrays',return_value=base),patch.object(e,'score',side_effect=tiny_score):
                result=e.freeze(run)
                self.assertEqual(result['status'],'six_predictions_frozen_before_truth')
                self.assertFalse((run/'evaluation').exists())
                result=e.evaluate(run)
                self.assertEqual(result['selected_before_development'],'control')
                self.assertEqual(read_json(run/'evaluation_complete.json')['truth_csv_parses'],1)
                with self.assertRaises(FileExistsError): e.evaluate(run)


if __name__=='__main__': unittest.main()
