"""Static recovery adapter tests: synthetic data/fakefit; zero optimizer/backward."""
import csv,copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_task22_recovery as m
from virtual_submeter.data.common import iso


def fake_model(dest,seed,p,d,blocks,meta,indices,initial,registration,**kw):
    if kw!={'pos_weight':1.,'recipe':'unit_BCE','arm':'wide32','channels':32}:raise AssertionError('Recipe differs')
    dest.mkdir(parents=True,exist_ok=False);torch.save(initial,dest/'initial.pt')
    m.fresh(dest/'progress.json',dict(completed_updates=3528,attempted_updates=3528))
    np.savez_compressed(dest/'exposure.npz',count=np.bincount(indices.ravel(),minlength=len(d['y'])))
    with (dest/'batches.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['update','positive','negative','indices_sha256']);w.writeheader()
        w.writerows(dict(update=i+1,positive=16,negative=240,indices_sha256=m.loss.temporal.array_hash(row)) for i,row in enumerate(indices))
    scores=[{k:np.where(v['y']>=200,.9,.1).astype(np.float32) for k,v in blocks.items()} for _ in range(12)]
    for epoch in range(1,13):
        torch.save(initial,dest/f'epoch_{epoch}.pt')
        for k,v in blocks.items():np.savez_compressed(dest/f'epoch_{epoch}_{k}.npz',t=v['t'],scores=scores[epoch-1][k])
    selected=m.loss.temporal.choose(scores,blocks,p)
    row=m.loss.temporal.write_selected(dest,seed,'wide32',selected,meta,dest/f'epoch_{selected["epoch"]}.pt',registration,3528,'unit_BCE',channels=32)
    m.fresh(dest/'complete.json',dict(updates=3528,artifacts=m.inventory(dest)))
    return row


def make_fixture(root):
    old=root/'old';run=root/'run';t10=root/'t10';audit=root/'audit';base=root/'baseline'
    old.mkdir();run.mkdir();audit.mkdir();(t10/'validation').mkdir(parents=True)
    m.fresh(old/'registration.json',dict(synthetic_original=True));oldsha=m.sha256(old/'registration.json')
    m.fresh(audit/'audit_manifest.json',dict(synthetic=True,artifacts={}));auditsha=m.sha256(audit/'audit_manifest.json')
    plan=dict(checkpoints=12,steps_per_checkpoint=294,batch_size=256,positive_per_batch=16,uniform_negative_per_batch=240,
        learning_rate=.001,cpu_threads=4,minimum_block_positive_windows=5,minimum_block_events=2,minimum_validation_recall=.85)
    norm=dict(input_log_mean=5.,input_log_std=1.,cpu_threads=4,batch_size=256)
    y=np.r_[np.full(16,200),np.zeros(240)].astype(np.float32);n=len(y)
    d=dict(x=np.full((n,127),100.,np.float32),mask=np.ones((n,127),bool),y=y,t=1300000000+np.arange(n)*8,house=np.arange(n)%3,event=np.where(y>=200,np.arange(n)//4,-1))
    d['x'][0,48]=np.nan;d['mask']=np.isfinite(d['x'])
    np.savez_compressed(t10/'train.npz',**d);m.fresh(t10/'normalization.json',norm)
    blocks={};source_t=[];source_p=[]
    for b,lo,hi in m.loss.temporal.f.VALIDATION:
        t=lo+86400-480+np.arange(120)*8;labels=np.zeros(120,np.float32);labels[10:20]=200;labels[90:100]=200
        event=np.full(120,-1,np.int64);event[10:20]=10000+b*10;event[90:100]=10001+b*10
        v=dict(t=t,y=labels,event=event,house=np.full(120,2),block=np.full(120,b),x=np.full((120,127),100.,np.float32),mask=np.ones((120,127),bool))
        blocks[f'block_{b}']=v;np.savez_compressed(t10/f'validation/block_{b}.npz',**v)
        source_t.extend(t.tolist());source_p.extend(np.where(labels>=200,.9,.1).astype(np.float32).tolist())
    source_t=np.array(source_t);source_p=np.array(source_p,np.float32)
    raw=root/'source.csv'
    with raw.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['timestamp','household_id','power_w']);w.writeheader();w.writerows(dict(timestamp=iso(int(t)),household_id='refit_5',power_w=100) for t in source_t)
    m.fresh(t10/'source5_inputs.json',{'aggregate':dict(path=str(raw),sha256=m.sha256(raw))})
    indices=np.tile(np.arange(256,dtype=np.int64),(3528,1))
    for s in m.SEEDS:
        np.savez_compressed(run/f'batches_{s}.npz',indices=indices)
        np.savez_compressed(run/f'exposure_{s}.npz',total=np.bincount(indices.ravel(),minlength=n))
        initial=m.original.make_initial(s,32);torch.save(initial,run/f'initial32_{s}.pt')
        if s in (42,1019):fake_model(old/f'models/wide32/seed_{s}',s,plan,d,blocks,norm,indices,initial,oldsha,pos_weight=1.,recipe='unit_BCE',arm='wide32',channels=32)
    datafiles=[p for p in t10.rglob('*') if p.is_file()]+[raw]
    m.fresh(old/'inputs_before.json',m.snapshot(datafiles))
    oldfiles=[p for p in old.rglob('*') if p.is_file()]
    m.fresh(run/'old_v4_files.json',m.snapshot(oldfiles))
    m.fresh(run/'reused_model_identity.json',dict(original_registration_sha256=oldsha,models={str(s):dict(artifacts=m.inventory(old/f'models/wide32/seed_{s}')) for s in (42,1019)}))
    m.fresh(run/'interruption_identity.json',dict(audit_manifest_sha256=auditsha,failed_model_inventory_sha256='static-failed-directory',ledger={}))
    m.fresh(run/'config.json',m.RECIPE);m.fresh(run/'parent_training_plan.json',plan)
    m.fresh(run/'source_control.json',dict(rows={'guard95':[dict(seed=s,precision=.1,recall=1.,f1=.1) for s in m.SEEDS]}))
    nd=137523;dt=1500000000+np.arange(nd)*8;truth=np.zeros(nd,np.float32);truth[::100]=200
    raw_a=np.zeros((nd,3),np.float32);raw_a[::100,1]=200;raw_w=raw_a.copy();raw_w[1::100,1]=200
    for seed in m.SEEDS:
        for arm,values in (('A',raw_a),('W50',raw_w)):
            folder=base/(f'reference/predictions/seed_{seed}' if arm=='A' else f'predictions/W50/seed_{seed}');folder.mkdir(parents=True)
            np.savez_compressed(folder/m.gate.PARTITION,stamps=dt,raw=values,corrected=values)
    devraw=root/'synthetic_aggregate.csv'
    with devraw.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['timestamp','household_id','power_w']);w.writeheader();w.writerows(dict(timestamp=iso(int(t)),household_id='refit_17',power_w=100) for t in dt)
    truthfile=root/'synthetic_truth.npz';np.savez_compressed(truthfile,t=dt,y=truth)
    dev={str(devraw):m.sha256(devraw)};dev.update({str(p):m.sha256(p) for p in base.rglob('*.npz')})
    m.fresh(run/'future_identity.json',dict(development_unlabeled=dev,baseline=str(base),truth_cache=dict(path=str(truthfile),sha256=m.sha256(truthfile))))
    (run/'PREPARATION_REPORT.zh.md').write_text('全为静态夹具；0真实fit。',encoding='utf-8')
    inputs=oldfiles+datafiles+[audit/'audit_manifest.json',Path(__file__).resolve()]+list(Path(m.__file__).resolve().parents[1].rglob('*.py'))
    m.fresh(run/'inputs_before.json',m.snapshot(inputs))
    m.fresh(run/'registration.json',dict(schema=m.SCHEMA,authorized=False,artifacts=m.inventory(run),original_registration_sha256=oldsha,
        interruption_audit_sha256=auditsha,reused_model_identity_sha256=m.sha256(run/'reused_model_identity.json')))
    def infer(model,rows):
        if rows and rows[0]['household_id']=='refit_5':
            t=np.array([m.loss.utc(r['timestamp']) for r in rows]);return t,source_p[np.searchsorted(source_t,t)],{}
        return dt,np.where(truth>=200,.9,.1).astype(np.float32),{}
    return run,old,t10,audit,oldsha,auditsha,infer


class RecoveryTests(unittest.TestCase):
    def test_no_budget_rejected_before_any_data(self):
        with patch.object(m,'registered',side_effect=AssertionError('early data')),patch.object(m,'execution_allowed',side_effect=AssertionError('early data')):
            for name in ('train','source_select','freeze','evaluate','diagnostics'):
                with self.assertRaisesRegex(Exception,'NEW recovery budget'):getattr(m,name)(Path('missing'))
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            old=Path(td)/'old.json';m.fresh(old,dict(authorized=True,approval_origin='human',experiment='microwave_task22_capacity_v1'))
            with self.assertRaisesRegex(Exception,'old budgets rejected'):m.authorize(Path(td),old)
            with self.assertRaisesRegex(Exception,'cannot overwrite'):m.prepare(Path(td))

    def test_full_static_fakefit_review_and_guards(self):
        def no_training(*a,**kw):raise AssertionError('Optimizer/backward forbidden')
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td,patch.object(torch.optim.Optimizer,'__init__',no_training),patch.object(torch.optim.Optimizer,'add_param_group',no_training),patch.object(torch.Tensor,'backward',no_training),patch.object(torch.autograd,'backward',no_training),patch.object(torch.autograd,'grad',no_training):
            root=Path(td);run,old,t10,audit,oldsha,auditsha,infer=make_fixture(root)
            calls=[]
            def fake(*args,**kw):calls.append(args[1]);return fake_model(*args,**kw)
            with patch.object(m,'OLD',old),patch.object(m,'T10',t10),patch.object(m,'AUDIT',audit),patch.object(m,'ORIGINAL_SHA',oldsha),patch.object(m,'AUDIT_SHA',auditsha),patch.object(m,'authorize',return_value={'synthetic_fixture_only':True}),patch.object(m.loss.temporal,'fit_one',side_effect=fake),patch.object(m.gate,'infer',side_effect=infer):
                m.registered(run)
                with self.assertRaisesRegex(Exception,'training_complete'):m.check_complete(run)
                result=m.train(run,'synthetic');self.assertEqual(calls,[2026]);self.assertEqual(result['new_fits'],1);self.assertEqual(result['reused_fits'],2)
                with self.assertRaisesRegex(Exception,'No recovery retry'):m.train(run,'synthetic')
                complete=m.read_json(run/'training_complete.json');self.assertEqual(complete['effective_matrix_updates'],10584);self.assertEqual(complete['cumulative_attempted_updates_upper_bound'],11152)
                self.assertEqual(m.read_json(run/'models/wide32/seed_42/model.json')['registration_sha256'],oldsha)
                self.assertEqual(m.read_json(run/'models/wide32/seed_2026/model.json')['registration_sha256'],m.sha256(run/'registration.json'))
                bad=copy.deepcopy(complete);bad['rows'].pop();(run/'training_complete.json').write_text(json.dumps(bad),encoding='utf-8')
                with self.assertRaisesRegex(Exception,'Incomplete/unbound'):m.check_complete(run)
                (run/'training_complete.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                self.assertTrue(m.source_select(run,'synthetic')['source_eligible'])
                class FakeNet(torch.nn.Module):
                    def __init__(self,width):super().__init__();self.width=width
                    def forward(self,z):
                        if torch.is_grad_enabled() or torch.get_num_threads()!=4:raise AssertionError('CPU4/no-grad missing')
                        return z[:,0,63]*.1+z[:,1,48]*.4+(self.width-16)/100
                fake_loads=[]
                def fake_load(path):
                    width=16 if 'unit_BCE' in str(path) else 32;fake_loads.append(width)
                    return dict(input_log_mean=5.,input_log_std=1.,cpu_threads=4,batch_size=256,window_samples=127,decision_threshold_probability=.5),FakeNet(width).eval()
                fake_loader=patch.object(m,'load_state_model',side_effect=fake_load);fake_loader.start();self.addCleanup(fake_loader.stop)
                diag=m.diagnostics(run,'synthetic')
                self.assertEqual((diag['diagnostic_cells'],diag['prediction_caches'],diag['cpu_threads']),(24,12,4))
                with self.assertRaisesRegex(Exception,'cannot overwrite'):m.diagnostics(run,'synthetic')
                self.assertEqual(sorted(fake_loads),[16,16,16,32,32,32])
                self.assertEqual(len(list((run/'diagnostics').glob('*.npz'))),12)
                self.assertTrue(m.freeze(run,'synthetic')['frozen'])
                frozen=m.read_json(run/'freeze.json');bad=copy.deepcopy(frozen);bad['matrix'].pop();(run/'freeze.json').write_text(json.dumps(bad),encoding='utf-8')
                with self.assertRaisesRegex(Exception,'Incomplete/unbound'):m.evaluation_guard(run)
                (run/'freeze.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                self.assertTrue(m.evaluate(run,'synthetic')['development_acceptance_passed'])
                with self.assertRaisesRegex(Exception,'One development acceptance'):m.evaluate(run,'synthetic')
                before=m.snapshot([run/k for k in m.inventory(run)])
                review=m.verify(run);self.assertTrue(review['execution']['independent_eight_gates']);self.assertEqual(review['actual_new_fits'],1)
                self.assertEqual(review['execution']['diagnostics']['cells'],24);self.assertTrue(review['execution']['independent_summary_ledger_exact'])
                self.assertTrue(m.report(run)['execution']['complete_frozen_9_cell_matrix'])
                self.assertEqual(before,m.snapshot(before))
                summary_bytes=(run/'evaluation/summary.json').read_bytes();seal_bytes=(run/'evaluation_complete.json').read_bytes()
                summary=m.read_json(run/'evaluation/summary.json');evaluation_seal=m.read_json(run/'evaluation_complete.json')
                self.assertEqual(summary['actual_new_fits'],1);self.assertEqual(summary['new_fit_updates'],3528)
                self.assertEqual(summary['reused_fits'],2);self.assertEqual(summary['historic_interrupted_fits'],1)
                self.assertEqual(summary['total_fit_attempts'],4);self.assertEqual(summary['effective_models'],3)
                self.assertEqual(summary['effective_matrix_updates'],10584);self.assertEqual(summary['cumulative_attempted_updates_upper_bound'],11152)
                self.assertNotIn('new_fits',summary);self.assertNotIn('updates',summary)
                for variant in ('wrong_new_fits','missing_history'):
                    bad=copy.deepcopy(summary)
                    if variant=='wrong_new_fits':bad['actual_new_fits']=3
                    else:bad.pop('historic_interrupted_fits')
                    (run/'evaluation/summary.json').write_text(json.dumps(bad),encoding='utf-8')
                    changed=copy.deepcopy(evaluation_seal);changed['artifacts']['summary.json']=m.sha256(run/'evaluation/summary.json')
                    (run/'evaluation_complete.json').write_text(json.dumps(changed),encoding='utf-8')
                    with self.assertRaisesRegex(Exception,'Recovery summary ledger'):m.verify(run)
                (run/'evaluation/summary.json').write_bytes(summary_bytes)
                (run/'evaluation_complete.json').write_bytes(seal_bytes)
                self.assertTrue(m.verify(run)['execution']['independent_summary_ledger_exact'])
                original_selection=m.read_json(run/'source_selection.json')
                for alloff in (False,True):
                    selection=copy.deepcopy(original_selection);selection['recipes'][0]['blocks'][0]['recall']=0 if alloff else .949
                    selection['recipes'][0]['blocks'][0]['degenerate']=alloff
                    (run/'source_selection.json').write_text(json.dumps(selection),encoding='utf-8')
                    seal=m.read_json(run/'source_complete.json');seal['source_sha256']=m.sha256(run/'source_selection.json');(run/'source_complete.json').write_text(json.dumps(seal),encoding='utf-8')
                    read_real=m.read_json;reads=[]
                    def tracked(p):reads.append(str(p));return read_real(p)
                    with patch.object(m,'read_json',side_effect=tracked):
                        with self.assertRaisesRegex(Exception,'Source rejected'):m.freeze(run,'synthetic')
                    self.assertFalse(any('future_identity' in p for p in reads))
                (run/'models/wide32/seed_42/weights.pt').write_bytes(b'tampered')
                with self.assertRaisesRegex(Exception,'Incomplete/unbound'):m.check_complete(run)
                (old/'models/wide32/seed_1019/weights.pt').write_bytes(b'tampered-old')
                with self.assertRaisesRegex(Exception,'Old v4 artifact changed'):m.registered(run)

if __name__=='__main__':unittest.main()
