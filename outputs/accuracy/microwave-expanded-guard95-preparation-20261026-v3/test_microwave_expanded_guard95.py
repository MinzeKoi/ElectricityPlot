"""Task18 synthetic verification only: no real fits/optimizer updates or real dev reads."""
import copy
import contextlib
import csv
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
import numpy as np
import torch
from virtual_submeter.accuracy import microwave_expanded_guard95 as e
from virtual_submeter.models.center_mask_tcn import KIND


@contextlib.contextmanager
def matrix_fixture_root():
    destination=os.environ.get('TASK18_SYNTHETIC_ARTIFACT_DIR')
    if destination:
        root=Path(destination).resolve()
        if root.exists(): raise ValueError('Synthetic artifacts need a new directory')
        root.mkdir(parents=True)
        yield str(root)
    else:
        with tempfile.TemporaryDirectory() as tmp: yield tmp


def receipt(run):
    p=run/'synthetic_authorization.json'
    e.fresh(p,dict(authorized=True,approval_origin='human',experiment=e.EXPERIMENT,additional_new_fits=3,maximum_updates=19404,seeds=e.SEEDS,
        registration_sha256=e.sha256(run/'registration.json'),approved_rule='expanded_guard95_fixed_L95_H95_radius15',
        human_authorization_quote='SYNTHETIC TEST ONLY - NO REAL BUDGET',human_authorization_reference='unittest temporary fixture'))
    return p


def tcn(path):
    path.mkdir(parents=True)
    model=e.CenterMaskTCN()
    with torch.no_grad():
        for v in model.parameters(): v.zero_()
        model.project.weight[0,0,0]=8
        model.head[0].weight[0,0]=1
        model.head[2].weight[0,0]=1
    torch.save(model.state_dict(),path/'weights.pt')
    e.fresh(path/'model.json',dict(kind=KIND,truth_threshold_w=200,window_samples=127,inference_support='strict_complete_31',
        input_log_mean=float(np.log1p(1000)),input_log_std=1.,decision_threshold_probability=.6,cpu_threads=4,batch_size=256,weights_sha256=e.sha256(path/'weights.pt')))


def constant(path):
    path.mkdir(parents=True)
    e.fresh(path/'model.json',dict(schema_version=1,status='trained',kind='constant_mean',appliances=['kettle','microwave','washing_machine'],
        sample_interval_seconds=8,constant_power_w=[100,250,10],settings=dict(window_samples=31,batch_size=256,cpu_threads=4,
        on_thresholds_w=dict(kettle=2000,microwave=200,washing_machine=20))))


def synthetic_fixture(root,n=137553):
    run=root/'run';run.mkdir();base=root/'baseline';parent=root/'parent';ctrl=root/'control';ctrl.mkdir()
    t=np.arange(n,dtype=np.int64)*8+1600000000
    a=np.where(np.arange(n)%80<40,1000.,2000.).astype(np.float32)
    rows=[dict(household_id='refit_17',timestamp=e.iso(int(s)),aggregate_power_w=float(v),quality_flags='[]') for s,v in zip(t,a)]
    agg=root/'synthetic_aggregate.csv';e.gate.csv_save(agg,rows)
    raw=np.tile(np.array([100,250,10],np.float32),(n,1));raw[:15]=np.nan;raw[-15:]=np.nan
    known=np.isfinite(raw[:,1]);truth=root/'synthetic_cache.npz'
    np.savez_compressed(truth,t=t,y=np.where(known,np.where(a>1500,400.,0.),np.nan).astype(np.float32))
    source_inputs={};dev={str(agg):e.sha256(agg)}
    for seed in e.SEEDS:
        for group in ('reference/models','models/W50'): constant(base/group/f'seed_{seed}')
        for group in ('reference/predictions','predictions/W50'):
            dest=base/group/f'seed_{seed}';dest.mkdir(parents=True)
            np.savez_compressed(dest/e.gate.PARTITION,stamps=t,raw=raw,corrected=raw)
            dev[str(dest/e.gate.PARTITION)]=e.sha256(dest/e.gate.PARTITION)
        tcn(run/f'models/expanded/seed_{seed}');tcn(parent/f'models/unit_BCE/seed_{seed}')
    for p in list(base.rglob('model.json'))+list(parent.rglob('*')):
        if p.is_file(): source_inputs[str(p)]=e.sha256(p)
    identity=dict(baseline=str(base),parent=str(parent),control=str(ctrl),source_inputs=source_inputs,development_unlabeled=dev,truth_cache=dict(path=str(truth),sha256=e.sha256(truth)))
    e.fresh(run/'identity.json',identity);e.fresh(run/'config.json',e.read_json(e.ROOT/e.CONFIG))
    e.fresh(run/'registration.json',dict(artifacts={n:e.sha256(run/n) for n in ('identity.json','config.json')}))
    auth=receipt(run)
    e.fresh(run/'training_started.json',dict(authorization=e.read_json(auth)))
    e.fresh(run/'training_complete.json',dict(new_fits=3,updates=19404,control_retrains=0,artifacts=e.inventory(run/'models')))
    control_recipes=[];recipes=[];old=[];new=[]
    y=np.arange(100)<10
    for seed in e.SEEDS:
        cb=[dict(block=str(b),**e.gate.confusion((np.arange(100)<50).astype(np.int8),y)) for b in (4,5)]
        nb=[dict(block=str(b),**e.gate.confusion((np.arange(100)<20).astype(np.int8),y)) for b in (4,5)]
        control_recipes.append(dict(seed=seed,L=.49,H=.49,feasible=True,blocks=cb))
        recipes.append(dict(seed=seed,L=.6,H=.6,feasible=True,blocks=nb))
        old.append(dict(seed=seed,**e.gate.mean_rows(cb),degenerate=False,source_gate_passed=True))
        new.append(dict(seed=seed,**e.gate.mean_rows(nb),degenerate=False,source_gate_passed=True))
    e.fresh(ctrl/'source_selection.json',dict(guard95_eligible=True,selected='guard95',recipes=control_recipes,rows=dict(guard95=old)))
    s=dict(source_only=True,**e.choose(old,new),rows=dict(control_guard95=old,expanded_guard95=new),recipes=recipes,control_recipes=control_recipes,
        registration_sha256=e.sha256(run/'registration.json'),training_sha256=e.sha256(run/'training_complete.json'),algorithm_sha256=e.sha256(Path(e.__file__)))
    e.fresh(run/'source_selection.json',s);e.fresh(run/'source_complete.json',dict(source_selection_sha256=e.sha256(run/'source_selection.json')))
    # This harness replaces ONLY completed-training provenance. Production parser,
    # authorization, isolation, source guards, actual ordinary models, public CSV
    # generation/readback, freeze/evaluation and scalar paths remain active.
    original_script=HERE/'run_microwave_expanded_guard95.py' if (HERE/'registration.json').exists() else e.ROOT/e.SCRIPT
    source_root=Path(e.__file__).resolve().parents[2]
    driver="import sys, runpy\nfrom pathlib import Path\nsys.dont_write_bytecode=True\nsys.path.insert(0,"+repr(str(source_root))+")\nfrom virtual_submeter.accuracy import microwave_expanded_guard95 as e\ne.training_guard=lambda run:(e.read_json(run/'config.json'),e.read_json(run/'identity.json'))\nrunpy.run_path("+repr(str(original_script))+",run_name='__main__')\n"
    (run/'run_microwave_expanded_guard95.py').write_text(driver,encoding='utf-8')
    return run,auth,identity,rows


class Task18Tests(unittest.TestCase):
    def source_fixture(self,root,flat=False):
        run,auth,identity,_=synthetic_fixture(root,n=100)
        # Genuine fixed L95/H95 calibration over synthetic saved source scores.
        rows=[];blocks={};scores={}
        for block,low,high in e.loss.temporal.f.VALIDATION:
            t=low+np.arange(100,dtype=np.int64)*8
            y=np.where(np.arange(100)<10,400.,0.).astype(np.float32)
            p=np.full(100,.5,np.float32) if flat else np.where(y>=200,.9,.1).astype(np.float32)
            blocks[str(block)]=dict(t=t,y=y);scores[str(block)]=p
            rows += [dict(household_id='refit_5',timestamp=e.iso(int(s)),aggregate_power_w=1000) for s in t]
        agg=root/'source_aggregate.csv';e.gate.csv_save(agg,rows)
        identity['source']=dict(aggregate=dict(path=str(agg),sha256=e.sha256(agg)))
        identity['source_inputs'][str(agg)]=e.sha256(agg)
        # Delete only this temporary synthetic fixture's earlier source seal.
        for name in ('source_selection.json','source_complete.json'):(run/name).unlink()
        (run/'identity.json').write_text(json.dumps(identity),encoding='utf-8')
        for seed in e.SEEDS:
            model=run/f'models/expanded/seed_{seed}'
            meta=e.read_json(model/'model.json');meta['selected']=dict(epoch=1)
            (model/'model.json').write_text(json.dumps(meta),encoding='utf-8')
            for block in (4,5):np.savez_compressed(model/f'epoch_1_block_{block}.npz',t=blocks[str(block)]['t'],scores=scores[str(block)])
        c=e.read_json(run/'training_complete.json');c['artifacts']=e.inventory(run/'models');(run/'training_complete.json').write_text(json.dumps(c),encoding='utf-8')
        reg=e.read_json(run/'registration.json');reg['artifacts']['identity.json']=e.sha256(run/'identity.json');(run/'registration.json').write_text(json.dumps(reg),encoding='utf-8')
        auth.unlink();auth=receipt(run);(run/'training_started.json').write_text(json.dumps(dict(authorization=e.read_json(auth))),encoding='utf-8')
        def infer(model,rr):
            t=np.array([e.utc(r['timestamp']) for r in rr]);key='4' if int(t[0])==e.loss.temporal.f.VALIDATION[0][1] else '5'
            return t,scores[key],{}
        return run,auth,identity,blocks,infer

    def test_actual_synthetic_source_calibration_and_source_only_rejection(self):
        for flat in (False,True):
            with self.subTest(flat=flat),tempfile.TemporaryDirectory() as tmp:
                run,auth,identity,blocks,infer=self.source_fixture(Path(tmp),flat)
                with patch.object(e,'training_guard',return_value=(e.read_json(run/'config.json'),identity)),patch.object(e.control,'source_blocks',return_value=blocks),patch.object(e.gate,'infer',side_effect=infer):
                    out=e.source(run,auth)
                    self.assertEqual(out['expanded_eligible'],not flat)
                    e.source_guard(run)
                    if flat:
                        with patch.object(e.loss,'check_paths',side_effect=AssertionError('development hashes touched')):
                            with self.assertRaisesRegex(e.DataError,'source-only'):e.freeze(run,auth)
                        report=e.report(run)['report_zh']
                        self.assertIn('无合格阈值',report);self.assertIn('Δ来源平均P',report)
                        self.assertFalse((run/'predictions').exists());self.assertFalse((run/'evaluation').exists())
                    else:
                        s=e.read_json(run/'source_selection.json')
                        self.assertTrue(all(r['L']==float(np.float32(.9)) and r['H']==float(np.float32(.9)) for r in s['recipes']))

    def test_first_gate_and_old_receipts(self):
        for stage in ('train','source','freeze','evaluate','predict'):
            with patch.object(e,'allowed_files',side_effect=AssertionError('identity touched')):
                with self.assertRaisesRegex(e.DataError,'NEW explicit human'): getattr(e,stage)(Path('missing'))
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);e.fresh(run/'registration.json',{})
            p=run/'old.json';e.fresh(p,dict(authorized=True,approval_origin='human',experiment='microwave_source5_loss_v1',additional_new_fits=3,maximum_updates=10584))
            with self.assertRaises(e.DataError):e.authorize(run,p)
            p.unlink();e.fresh(p,e.read_json(e.ROOT/e.CONFIG))
            with self.assertRaises(e.DataError):e.authorize(run,p)

    def test_receipt_all_fields_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);e.fresh(run/'registration.json',{});p=receipt(run);a=e.authorize(run,p)
            for key,value in [('maximum_updates',10584),('additional_new_fits',6),('seeds',[42]),('registration_sha256','bad'),('human_authorization_quote',''),('human_authorization_reference',''),('approved_rule','old')]:
                p.write_text(json.dumps(a|{key:value}),encoding='utf-8')
                with self.assertRaises(e.DataError):e.authorize(run,p)

    def test_exact_isolation_and_source_code_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'evaluation'/'registered.py';p.parent.mkdir();p.write_text('synthetic')
            bad=[root/n for n in ('house_8.json','refit_9.npz','CLEAN_House20.csv','house_17.json','truth_once.npz','evaluation/metrics.json','unknown.npz')]
            for b in bad:b.write_text('synthetic')
            with e.isolation('prepare',[p]+bad):
                self.assertEqual(p.read_text(),'synthetic')
                for b in bad[:-1]:
                    with self.assertRaises(e.DataError):e.sha256(b)
            with e.isolation('prepare',[]):
                with self.assertRaises(e.DataError):e.sha256(bad[-1])
                with self.assertRaises(e.DataError):torch.tensor(1.,requires_grad=True).backward()
                with self.assertRaises(e.DataError):torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])

    def test_core_missing_context_and_role_boundaries(self):
        audit=e.load_module(e.AUDIT/'audit.py','synthetic_task18_audit')
        t=np.arange(300,dtype=np.int64)*8;a=np.full(300,500,np.float32);y=np.zeros(300,np.float32);y[100:103]=400
        a[75]=np.nan;a[180:182]=np.nan;y[140]=np.nan
        d,_=e.build_added(t,a,y,0,0,2400,100,audit)
        self.assertFalse(140*8 in d['t']);self.assertFalse(180*8 in d['t'])
        self.assertTrue(np.isnan(d['x']).any());self.assertTrue((d['event'][d['y']>=200]>=100).all())
        for i,center in enumerate(d['raw_index']):
            obs=np.isfinite(d['x'][i]);indices=center+np.arange(-63,64)
            self.assertFalse(obs[(indices<0)|(indices>=len(t))].any())
        a2=a.copy();t2=t.copy();t2[220:]+=16
        gap,_=e.build_added(t2,a2,y,0,0,2416,100,audit)
        self.assertFalse(((gap['raw_index']>=205)&(gap['raw_index']<235)).any())
        y2=y.copy();y2[0]=400
        with self.assertRaisesRegex(e.DataError,'Active role boundary'):e.build_added(t,a,y2,0,0,2400,100,audit)

    def test_native_hold_gap_issues_and_nan(self):
        audit=e.load_module(e.AUDIT/'audit.py','synthetic_task18_raw_audit')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'data/raw/refit_complete_20261022/households/CLEAN_House2.csv';p.parent.mkdir(parents=True)
            rows=[dict(Unix=t,Aggregate=1000,Appliance5=('NaN' if t==24 else 400),Issues=('1' if t==16 else '0')) for t in (-4,0,8,16,24,32,52,60,80)]
            e.gate.csv_save(p,rows);e.fresh(p.with_suffix('.receipt.json'),dict(sha256=e.sha256(p)))
            with patch.object(audit,'ROOT',root),e.isolation('prepare',[p,p.with_suffix('.receipt.json')]):
                (t,a,y,ac,yc),proof=audit.raw_selected('2',0,80,'Appliance5',dict(candidate_bounds={'2':[0,80]}))
            self.assertTrue(np.isnan(y[t==16]).all());self.assertTrue(np.isnan(y[t==24]).all())
            self.assertTrue(np.isnan(a[t==40]).all());self.assertTrue(np.isnan(a[t==48]).all())
            self.assertEqual(proof['boundary_context_unix'],[-4,80])

    def test_source_low_recall_degenerate_ties(self):
        old=[dict(seed=s,precision=.2,recall=.98,f1=.33,degenerate=False,source_gate_passed=True) for s in e.SEEDS]
        new=[r|dict(precision=.3,f1=.4) for r in old]
        self.assertTrue(e.choose(old,new)['expanded_eligible'])
        self.assertEqual(e.choose(old,old)['selected'],'control_guard95')
        for key,value in [('source_gate_passed',False),('degenerate',True),('recall',.9)]:
            bad=copy.deepcopy(new);bad[0][key]=value
            self.assertFalse(e.choose(old,bad)['expanded_eligible'])

    def test_expected_exposure_day_event_center_balance(self):
        d=dict(y=np.array([400,400,400,400,0]),house=np.array([0,0,0,2,2]),t=np.array([0,8,16,86400,86408]),event=np.array([0,0,1,3,-1]))
        ex=e.expected_exposure(d,dict(checkpoints=12,steps_per_checkpoint=539))
        self.assertEqual(ex[0],6468*16/(2*2*2));self.assertEqual(ex[2],6468*16/(2*2));self.assertEqual(ex[3],6468*16/2)
        self.assertEqual(ex[4],6468*240)

    def test_train_complete_writes_hashes_under_real_isolation_no_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);cfg=e.read_json(e.ROOT/e.CONFIG);identity=dict(source_inputs={},development_unlabeled={})
            e.fresh(run/'identity.json',identity);e.fresh(run/'config.json',cfg)
            d={k:np.array([0],np.int64) for k in e.loss.TRAIN_FIELDS};d['source_role']=np.array(['synthetic'])
            np.savez_compressed(run/'train.npz',**d)
            for s in e.SEEDS:
                np.savez_compressed(run/f'sampling_{s}.npz',indices=np.zeros((6468,256),np.int64));torch.save({},run/f'initial_{s}.pt')
            e.fresh(run/'normalization.json',{});e.fresh(run/'training_plan.json',cfg)
            e.fresh(run/'registration.json',dict(artifacts=e.inventory(run)));auth=receipt(run)
            calls=[]
            def no_update_fit(dest,seed,plan,*args,**kw):
                self.assertEqual((plan['steps_per_checkpoint'],kw['pos_weight']),(539,1.))
                dest.mkdir(parents=True);torch.save({},dest/'epoch_12.pt');e.sha256(dest/'epoch_12.pt')
                calls.append(seed);return dict(seed=seed,synthetic_spy_no_fit=True)
            with patch.object(e,'registered',return_value=(cfg,identity)),patch.object(e.loss.temporal,'validation',return_value={}),patch.object(e.loss.temporal,'fit_one',side_effect=no_update_fit),patch.object(torch.optim,'Adam',side_effect=AssertionError('optimizer created')):
                out=e.train(run,auth)
            self.assertEqual(calls,e.SEEDS);self.assertEqual(out['updates'],19404)
            with self.assertRaisesRegex(e.DataError,'restart/retry'):
                with patch.object(e,'registered',return_value=(cfg,identity)):e.train(run,auth)

    def test_full_137523_twelve_cell_public_cli_freeze_once_evaluate_readonly(self):
        with matrix_fixture_root() as tmp:
            run,auth,identity,rows=synthetic_fixture(Path(tmp))
            def provenance(run):return e.read_json(run/'config.json'),e.read_json(run/'identity.json')
            with patch.object(e,'training_guard',side_effect=provenance):
                out=e.freeze(run,auth)
                self.assertEqual((out['matrix_size'],out['public_CLI_readbacks']),(12,12))
                self.assertFalse((run/'evaluation').exists())
                # Rehashed state corruption must still fail semantic gate replay.
                path=run/'predictions/expanded_guard95_42.npz';original=path.read_bytes()
                with np.load(path,allow_pickle=False) as z:bad={k:z[k] for k in z.files}
                bad['state'][100]=1-bad['state'][100];np.savez_compressed(path,**bad)
                f=e.read_json(run/'freeze.json');f['artifacts']['predictions/expanded_guard95_42.npz']=e.sha256(path)
                seal=(run/'freeze.json').read_bytes();(run/'freeze.json').write_text(json.dumps(f),encoding='utf-8')
                with self.assertRaises(AssertionError):e.evaluate(run,auth)
                self.assertFalse((run/'evaluation').exists())
                path.write_bytes(original);(run/'freeze.json').write_bytes(seal)
                missing=f['artifacts'].pop('predictions/expanded_guard95_1019.npz');(run/'freeze.json').write_text(json.dumps(f),encoding='utf-8')
                with self.assertRaisesRegex(e.DataError,'Incomplete/extra'):e.evaluate(run,auth)
                self.assertFalse((run/'evaluation').exists());(run/'freeze.json').write_bytes(seal)
                result=e.evaluate(run,auth)
                self.assertTrue(result['development_acceptance_passed'])
                self.assertEqual(len(result['acceptance_gates']),8)
                truth=Path(identity['truth_cache']['path']);truth.rename(truth.with_suffix('.sealed'))
                before={str(p):(e.sha256(p),p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
                for _ in range(2):
                    self.assertEqual(e.verify(run)['status'],'execution_verified_readonly')
                    self.assertIn('八门槛通过',e.report(run)['report_zh'])
                after={str(p):(e.sha256(p),p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
                self.assertEqual(before,after)
                with self.assertRaises(FileExistsError):e.evaluate(run,auth)
                e.fresh(run/'synthetic_verification.json',dict(status='passed',synthetic_only=True,real_fits=0,real_optimizer_updates=0,
                    matrix_size=12,support=137523,public_CLI_readbacks=12,actual_TCN_and_constant_model_inference=True,
                    provenance_override='Synthetic completed-training metadata only; no fit was run. Actual authorization/isolation/source guards/models/predict/freeze/evaluate/scalar/report paths executed.',
                    missing_matrix_rejected_before_truth=True,rehashed_state_tamper_rejected_before_truth=True,repeat_evaluate_rejected=True,
                    parent_truth_renamed_before_verify=True,verify_report_repetitions=2,hash_and_mtime_exact=True,
                    acceptance_gates=result['acceptance_gates']))


if __name__=='__main__':unittest.main()
