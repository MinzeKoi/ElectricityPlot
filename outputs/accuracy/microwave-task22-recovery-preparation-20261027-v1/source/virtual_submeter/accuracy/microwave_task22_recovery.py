"""Task22 recovery: an explicitly authorized fresh seed 2026 fit, never resume Adam."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import torch
from ..data.common import DataError,read_json,sha256,inventory
from ..models.center_mask_tcn import load_state_model,tensor_input
from . import microwave_task22 as original
from . import microwave_recovery_pipeline as pipeline
from .microwave_task22_metrics import independent_ap
from .microwave_task21_review import scalar_metric

ROOT=original.ROOT
OLD=ROOT/'outputs/accuracy/microwave-task22-preparation-20261026-v4'
AUDIT=ROOT/'outputs/accuracy/microwave-task22-interruption-audit-20261027-v1'
ORIGINAL_SHA='2ffd72773185d8be1372b0744729b7eb3daf8656f1dfd2097da47cb113037c53'
AUDIT_SHA='ef575ca1ea871756cd7056ebfce9d9c2832cddb0e99fc200a55d782d2a0889a1'
SCHEMA='microwave_task22_recovery_seed2026_v1'
SEEDS=original.SEEDS
T10,T12,T14=original.T10,original.T12,original.T14
loss,gate,guard95=original.loss,original.gate,original.guard95
require,fresh,load,snapshot,isolation=original.require,original.fresh,original.load,original.snapshot,original.isolation
CONFIG='configs/microwave_task22_recovery_20261027.disabled.json'
PROTOCOL='docs/MICROWAVE_TASK22_RECOVERY_20261027.md'
RECIPE=dict(schema=SCHEMA,authorized=False,preparation_only=True,recovery_seed=2026,new_fits=1,new_fit_updates=3528,
    reused_seeds=[42,1019],historic_interrupted_fits=1,effective_models=3,effective_matrix_updates=10584,
    historical_completed_updates_lower_bound=7623,historical_attempted_updates_upper_bound=7624,
    cumulative_completed_updates_lower_bound=11151,cumulative_attempted_updates_upper_bound=11152,total_fit_attempts=4,
    tcn_channels=32,parameter_count=19809,batch_size=256,positive_per_batch=16,uniform_negative_per_batch=240,
    checkpoints=12,steps_per_checkpoint=294,recipe='unit_BCE',optimizer='Adam',learning_rate=.001,cpu_threads=4,
    fresh_initialization=True,optimizer_resume=False,no_retry=True)


def original_data_paths():
    data=read_json(OLD/'inputs_before.json')
    return {p:v for p,v in data.items() if not Path(p).is_relative_to(ROOT/'src') and not Path(p).is_relative_to(ROOT/'scripts')
        and not Path(p).is_relative_to(ROOT/'tests') and not Path(p).is_relative_to(ROOT/'docs') and not Path(p).is_relative_to(ROOT/'configs')}


def all_inputs():
    audit=read_json(AUDIT/'audit_manifest.json')
    paths={Path(p).resolve() for p in audit['original_inputs'] if Path(p).resolve()!=ROOT/'scripts/task22_interruption_audit.py'}
    paths.update(Path(p).resolve() for p in original_data_paths())
    paths.update(p.resolve() for p in AUDIT.iterdir() if p.is_file())
    paths.update((ROOT/p).resolve() for p in (CONFIG,PROTOCOL,'scripts/run_task22_recovery.py','scripts/finalize_task22_recovery.py','tests/test_task22_recovery.py'))
    paths.update((ROOT/'src/virtual_submeter/accuracy'/p).resolve() for p in ('microwave_task22_recovery.py','microwave_recovery_pipeline.py','microwave_recovery_review.py','microwave_recovery_independent.py'))
    return sorted(paths)


def validate_config(config):require(config==RECIPE,'Recovery recipe is fixed and disabled')


def prepare(run):
    run=Path(run).resolve()
    if run.exists():
        if (run/'preparation_manifest.json').exists():return verify(run)
        raise DataError('Incomplete attempt retained; use a new version, cannot overwrite')
    paths=all_inputs()
    with isolation(run,paths) as access:
        before=snapshot(paths)
        require(sha256(OLD/'registration.json')==ORIGINAL_SHA and sha256(AUDIT/'audit_manifest.json')==AUDIT_SHA,'Original registration/audit identity differs')
        audit=read_json(AUDIT/'audit.json');gate.check(AUDIT,read_json(AUDIT/'audit_manifest.json')['artifacts'])
        audit_before=read_json(AUDIT/'audit_manifest.json')['original_inputs']
        checked={p:v for p,v in audit_before.items() if Path(p).resolve()!=ROOT/'scripts/task22_interruption_audit.py'}
        require(snapshot(checked)==checked,'Original v4 interruption evidence changed')
        require(snapshot(original_data_paths())==original_data_paths(),'Original data/model identity changed')
        require(not (OLD/'training_complete.json').exists() and not (OLD/'source_selection.json').exists() and not (OLD/'diagnostics').exists(),'Original incomplete run changed')
        validate_config(read_json(ROOT/CONFIG))
        run.mkdir(parents=True,exist_ok=False)
        for path in sorted((OLD/'source/virtual_submeter').rglob('*.py')):
            dst=run/'source'/path.relative_to(OLD/'source');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
        for name in ('microwave_task22_recovery.py','microwave_recovery_pipeline.py','microwave_recovery_review.py','microwave_recovery_independent.py'):
            shutil.copyfile(ROOT/'src/virtual_submeter/accuracy'/name,run/'source/virtual_submeter/accuracy'/name)
        for src,dst in ((ROOT/'scripts/run_task22_recovery.py',run/'run_task22_recovery.py'),(ROOT/'scripts/finalize_task22_recovery.py',run/'finalize_task22_recovery.py'),
                        (ROOT/'tests/test_task22_recovery.py',run/'test_task22_recovery.py'),(ROOT/PROTOCOL,run/'protocol.md')):shutil.copyfile(src,dst)
        for name in ('workspace_identity.json','parent_training_plan.json','source_control.json','future_identity.json'):
            shutil.copyfile(OLD/name,run/name)
        for seed in SEEDS:
            for name in (f'batches_{seed}.npz',f'exposure_{seed}.npz',f'initial32_{seed}.pt'):shutil.copyfile(OLD/name,run/name)
        fresh(run/'config.json',RECIPE)
        reuse={str(s):audit['complete_models'][str(s)] for s in (42,1019)}
        fresh(run/'reused_model_identity.json',dict(original_registration_sha256=ORIGINAL_SHA,models=reuse))
        fresh(run/'interruption_identity.json',dict(audit_manifest_sha256=AUDIT_SHA,failed_model_inventory_sha256=audit['interrupted']['inventory_sha256'],
            interrupted=audit['interrupted'],ledger=audit['ledger'],original_budget_sha256=audit['original_budget_sha256']))
        old_v4={p:v for p,v in before.items() if Path(p).is_relative_to(OLD)}
        fresh(run/'old_v4_files.json',old_v4)
        fresh(run/'inputs_before.json',before)
        fresh(run/'prepare_access.json',dict(reads=sorted(access['reads']),forbidden=access['forbidden'],new_fits=0,optimizers=0,optimizer_updates=0,
            backward_calls=0,development_reads=0,final_reads=0,original_failed_artifacts_preserved=True))
        require(snapshot(paths)==before,'Preparation modified inputs')
        fresh(run/'registration.json',dict(schema=SCHEMA,authorized=False,artifacts=inventory(run),original_registration_sha256=ORIGINAL_SHA,
            interruption_audit_sha256=AUDIT_SHA,reused_model_identity_sha256=sha256(run/'reused_model_identity.json'),new_fits=0,optimizer_updates=0))
    return dict(status='prepared_pending_review_and_new_recovery_budget',registration_sha256=sha256(run/'registration.json'),new_fits=0,optimizer_updates=0)


def registered(run):
    run=Path(run).resolve();reg=read_json(run/'registration.json')
    require(reg['schema']==SCHEMA and reg['authorized'] is False and reg['original_registration_sha256']==ORIGINAL_SHA and reg['interruption_audit_sha256']==AUDIT_SHA,'Recovery registration differs')
    gate.check(run,reg['artifacts']);validate_config(read_json(run/'config.json'))
    require(reg['reused_model_identity_sha256']==sha256(run/'reused_model_identity.json'),'Reuse identity differs')
    require(sha256(OLD/'registration.json')==ORIGINAL_SHA and sha256(AUDIT/'audit_manifest.json')==AUDIT_SHA,'Old registration/audit differs')
    old=read_json(run/'old_v4_files.json');require(snapshot(old)==old,'Old v4 artifact changed')
    names={str(p.resolve()) for p in OLD.rglob('*') if p.is_file()}
    require(names==set(old),'Old v4 file set changed')
    require(not (OLD/'training_complete.json').exists() and not (OLD/'source_selection.json').exists(),'Original interruption changed')
    require(snapshot(original_data_paths())==original_data_paths(),'Original data/model identity changed')
    return read_json(run/'config.json')


def execution_allowed(run):return list(read_json(Path(run)/'inputs_before.json'))+[Path(run)/k for k in inventory(run)]


def authorize(run,receipt):
    require(receipt is not None,'NEW recovery budget is required before data/model/fit')
    a=read_json(receipt)
    require(a.get('authorized') is True and a.get('approval_origin')=='human' and a.get('experiment')==SCHEMA,'NEW recovery receipt required; old budgets rejected')
    run=Path(run).resolve();identity=read_json(run/'interruption_identity.json')
    require(a.get('registration_sha256')==sha256(run/'registration.json') and a.get('original_registration_sha256')==ORIGINAL_SHA
        and a.get('interruption_audit_sha256')==AUDIT_SHA and a.get('reused_model_identity_sha256')==sha256(run/'reused_model_identity.json')
        and a.get('failed_model_inventory_sha256')==identity['failed_model_inventory_sha256'] and a.get('seed')==2026
        and a.get('additional_new_fits')==1 and a.get('maximum_updates')==3528 and a.get('fresh_from_original_initial') is True
        and a.get('human_authorization_quote','').strip() and a.get('human_authorization_reference','').strip(),'NEW 1fit/3528 hash-bound recovery budget required')
    gate.check(run,read_json(run/'preparation_manifest.json')['artifacts'])
    require(read_json(run/'independent_review.json')['passed'],'Recovery review required')
    require(Path(__file__).resolve()==(run/'source/virtual_submeter/accuracy/microwave_task22_recovery.py').resolve(),'Use registered recovery snapshot')
    return a


def train(run,authorization=None):
    run=Path(run).resolve();a=authorize(run,authorization)
    with isolation(run,execution_allowed(run),training=True):
        registered(run)
        require(not any((run/k).exists() for k in ('training_started.json','recovery_fit_started.json','models','training_complete.json')),'No recovery retry or overwrite')
        fresh(run/'training_started.json',dict(authorization=a,seed=2026,new_fit_limit=1,updates_limit=3528,loaded_source=str(Path(__file__).resolve())))
        identities=read_json(run/'reused_model_identity.json')['models']
        for seed in (42,1019):
            source=OLD/f'models/wide32/seed_{seed}';dest=run/f'models/wide32/seed_{seed}'
            dest.mkdir(parents=True,exist_ok=False)
            for name in sorted(identities[str(seed)]['artifacts']):
                rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe reused artifact path')
                target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
                with (source/rel).open('rb') as reader,target.open('xb') as writer:shutil.copyfileobj(reader,writer)
            require(inventory(dest)==identities[str(seed)]['artifacts'],'Copied reused model differs')
        d=load(T10/'train.npz');p=read_json(run/'parent_training_plan.json');meta=read_json(T10/'normalization.json')
        blocks={f'block_{b}':load(T10/f'validation/block_{b}.npz') for b in (4,5)}
        ix=load(run/'batches_2026.npz')['indices'];initial=torch.load(run/'initial32_2026.pt',weights_only=True,map_location='cpu')
        fresh(run/'recovery_fit_started.json',dict(seed=2026,fresh_initialization=True,original_optimizer_state_reused=False,maximum_updates=3528))
        row=loss.temporal.fit_one(run/'models/wide32/seed_2026',2026,p,d,blocks,meta,ix,initial,sha256(run/'registration.json'),
            pos_weight=1.,recipe='unit_BCE',arm='wide32',channels=32)
        rows=[loss.temporal.parent.source_row(s,read_json(run/f'models/wide32/seed_{s}/model.json')['selected']) for s in (42,1019)]+[row]
        fresh(run/'training_complete.json',dict(new_fits=1,new_fit_updates=3528,reused_fits=2,historic_interrupted_fits=1,total_fit_attempts=4,
            historical_completed_updates_lower_bound=7623,historical_attempted_updates_upper_bound=7624,cumulative_completed_updates_lower_bound=11151,
            cumulative_attempted_updates_upper_bound=11152,effective_models=3,effective_matrix_updates=10584,control_retrains=0,rows=rows,
            reused_model_identity_sha256=sha256(run/'reused_model_identity.json'),original_registration_sha256=ORIGINAL_SHA,artifacts=inventory(run/'models')))
        check_complete(run)
    return dict(new_fits=1,new_fit_updates=3528,reused_fits=2,effective_models=3,effective_matrix_updates=10584,cumulative_attempted_updates_upper_bound=11152)


def check_complete(run):
    run=Path(run);v=read_json(run/'training_complete.json')
    required=dict(new_fits=1,new_fit_updates=3528,reused_fits=2,historic_interrupted_fits=1,total_fit_attempts=4,
        historical_completed_updates_lower_bound=7623,historical_attempted_updates_upper_bound=7624,cumulative_completed_updates_lower_bound=11151,
        cumulative_attempted_updates_upper_bound=11152,effective_models=3,effective_matrix_updates=10584,control_retrains=0,original_registration_sha256=ORIGINAL_SHA)
    require(all(v.get(k)==x for k,x in required.items()) and v['reused_model_identity_sha256']==sha256(run/'reused_model_identity.json'),'Invalid actual/historical/effective recovery ledger')
    require([r['seed'] for r in v['rows']]==SEEDS and inventory(run/'models')==v['artifacts'],'Incomplete/unbound effective model matrix')
    ids=read_json(run/'reused_model_identity.json')['models']
    for seed in SEEDS:
        model=run/f'models/wide32/seed_{seed}';meta=read_json(model/'model.json');finish=read_json(model/'complete.json')
        require(read_json(model/'progress.json')==dict(completed_updates=3528,attempted_updates=3528) and finish['updates']==3528,'Incomplete effective fit')
        gate.check(model,finish['artifacts'])
        require(meta['registration_sha256']==(ORIGINAL_SHA if seed in (42,1019) else sha256(run/'registration.json'))
            and meta['seed']==seed and meta['tcn_channels']==32 and meta['parameter_count']==19809 and meta['recipe']=='unit_BCE','Model origin/width differs')
        if seed in (42,1019):require(inventory(model)==ids[str(seed)]['artifacts'],'Reused model bytes changed')
        require(v['rows'][SEEDS.index(seed)]==loss.temporal.parent.source_row(seed,meta['selected']),'Effective source row differs')
    return v


def source_select(run,authorization=None):return pipeline.source_select(Path(run).resolve(),authorization,task=sys.modules[__name__])
def source_guard(run):return pipeline.source_guard(Path(run).resolve(),task=sys.modules[__name__])
def freeze(run,authorization=None):return pipeline.freeze(Path(run).resolve(),authorization,task=sys.modules[__name__])
def evaluation_guard(run):return pipeline.evaluation_guard(Path(run).resolve(),task=sys.modules[__name__])
def evaluate(run,authorization=None):return pipeline.evaluate(Path(run).resolve(),authorization,task=sys.modules[__name__])


def diagnostics(run,authorization=None):
    run=Path(run).resolve();authorize(run,authorization);torch.set_num_threads(4)
    with isolation(run,execution_allowed(run)):
        registered(run);check_complete(run)
        require(not (run/'diagnostics').exists(),'Fixed diagnostics cannot overwrite')
        (run/'diagnostics').mkdir()
        data={'train':load(T10/'train.npz')}
        for b in (4,5):data[f'block_{b}']=load(T10/f'validation/block_{b}.npz')
        source={k:np.concatenate([data[f'block_{b}'][k] for b in (4,5)]) for k in ('x','y','t','house')}
        data={'train':data['train'],'source':source};rows=[]
        for seed in SEEDS:
            for arm,path in (('control16',T12/f'models/unit_BCE/seed_{seed}'),('wide32',run/f'models/wide32/seed_{seed}')):
                meta,net=load_state_model(path)
                for name,d in data.items():
                    with torch.inference_mode():
                        z=tensor_input(meta,d['x']);probability=np.concatenate([torch.sigmoid(net(v)).numpy() for v in z.split(meta['batch_size'])])
                    for view,mask in (('original',np.ones(len(d['y']),bool)),('strict_core31',np.isfinite(d['x'][:,48:79]).all(axis=1))):
                        p,y=probability[mask],d['y'][mask];metric=original.diagnostic_metrics(p,y,meta['decision_threshold_probability'])
                        indep=scalar_metric((p>=meta['decision_threshold_probability']).astype(np.int8),y>=200)
                        require(all(metric[k]==v for k,v in indep.items()) and abs(metric['ap']-independent_ap(p,y>=200))<2e-12,'Independent diagnostic mismatch')
                        rows.append(dict(arm=arm,seed=seed,dataset=name,view=view,support_sha256=gate.hash_array(mask),**metric))
                    np.savez_compressed(run/f'diagnostics/{arm}_{seed}_{name}.npz',probability=probability,t=d['t'],house=d['house'])
        fresh(run/'diagnostics/metrics.json',dict(rows=rows,explanatory_only=True,new_thresholds=0,new_selections=0,development_reads=0,cpu_threads=torch.get_num_threads()))
        fresh(run/'diagnostics_complete.json',dict(artifacts=inventory(run/'diagnostics')))
    return dict(diagnostic_cells=len(rows),prediction_caches=12,development_reads=0,new_selections=0,cpu_threads=4)


def verify(run):
    from .microwave_recovery_review import review_execution
    run=Path(run).resolve()
    with isolation(run,execution_allowed(run),readonly=True):
        registered(run)
        if (run/'preparation_manifest.json').exists():gate.check(run,read_json(run/'preparation_manifest.json')['artifacts'])
        result=dict(passed=True,status='recovery_prepared',new_fits=0,optimizer_updates=0,development_reads=0,final_reads=0,
            original_registration_sha256=ORIGINAL_SHA,registration_sha256=sha256(run/'registration.json'))
    if (run/'training_complete.json').exists():result['execution']=review_execution(run,task=sys.modules[__name__])
    elif (run/'training_started.json').exists():result.update(status='recovery_attempt_incomplete_no_retry',requires_manual_budget_accounting=True)
    return result


def report(run):return dict(**verify(run),report=(Path(run)/'PREPARATION_REPORT.zh.md').read_text(encoding='utf-8'))


def independent_verify(run):
    from .microwave_recovery_independent import review
    return review(Path(run).resolve())
