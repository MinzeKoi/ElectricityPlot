"""Task22 width-only preparation and receipt-gated execution; default 0 fit."""
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from ..data.common import DataError, read_json, sha256, inventory
from ..models.center_mask_tcn import CenterMaskTCN, tensor_input, predict_windows, load_state_model
from . import microwave_task21 as common
from . import microwave_capacity_pipeline as pipeline
from .microwave_sampling import BatchStream

ROOT,T10,T12,T14,SEEDS,N = common.ROOT,common.T10,common.T12,common.T14,common.SEEDS,common.N
loss,gate,guard95 = common.loss,common.gate,common.guard95
fresh,load,snapshot,isolation,require = common.fresh,common.load,common.snapshot,common.isolation,common.require
SCHEMA='microwave_task22_capacity_v1'
CONFIG='configs/microwave_task22_20261026.disabled.json'
PROTOCOL='docs/MICROWAVE_TASK22_PROPOSAL_20261026.md'


def input_paths():
    paths=[T10/n for n in ('train.npz','plan.json','normalization.json','registration.json','source5_inputs.json')]
    paths += [T10/f'validation/block_{b}.npz' for b in (4,5)]
    paths += [T10/f'{kind}_{s}.{ext}' for s in SEEDS for kind,ext in (('sampling','npz'),('dry_exposure','npz'),('initial','pt'))]
    paths += [T12/'final_delivery_manifest.json',T14/'final_delivery_manifest.json',T14/'source_selection.json',T14/'identity.json']
    for seed in SEEDS:
        model=T12/f'models/unit_BCE/seed_{seed}'
        paths += [model/n for n in ('model.json','weights.pt','initial.pt')]
        epoch=read_json(model/'model.json')['selected']['epoch']
        paths += [model/f'epoch_{epoch}_block_{b}.npz' for b in (4,5)]
    paths += [ROOT/CONFIG,ROOT/PROTOCOL,ROOT/'scripts/run_microwave_task22.py',ROOT/'scripts/finalize_microwave_task22.py',ROOT/'tests/test_microwave_task22.py']
    paths += sorted((ROOT/'src/virtual_submeter').rglob('*.py'))
    paths += [Path(v['path']) for v in read_json(T10/'source5_inputs.json').values()]
    return sorted(set(p.resolve() for p in paths))


def validate_config(c):
    require(c==dict(schema=SCHEMA,authorized=False,preparation_only=True,seeds=SEEDS,tcn_channels=32,
        control_channels=16,parameter_count=19809,control_parameter_count=5329,maximum_new_fits=3,
        maximum_updates=10584,updates_per_seed=3528,positive_per_batch=16,uniform_negative_per_batch=240,
        recipe='unit_BCE',optimizer='Adam',learning_rate=.001,cpu_threads=4), 'Task22 fixed disabled configuration differs')


def make_initial(seed,channels):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return {k:v.clone() for k,v in CenterMaskTCN(channels).state_dict().items()}


def validate_blocks(d,blocks):
    require([sum(len(v['y']) for v in blocks.values()),sum(int((v['y']>=200).sum()) for v in blocks.values())]==[34351,201], 'Source support differs')
    for b,lo,hi in loss.temporal.f.VALIDATION:
        v=blocks[f'block_{b}']
        require(np.isfinite(v['y']).all() and (v['y']>=0).all() and (v['house']==2).all()
            and (v['block']==b).all() and (v['t']-120>=lo).all() and (v['t']+128<=hi).all(), 'Source boundary/label differs')
        require(np.isfinite(v['x'][:,48:79]).all() and np.array_equal(v['mask'],np.isfinite(v['x'])), 'Strict core/mask differs')
        require(not np.intersect1d(v['t'],d['t'][d['house']==2]).size, 'Source center leakage')
        require(not np.intersect1d(v['event'][v['y']>=200],d['event'][(d['house']==2)&(d['y']>=200)]).size, 'Source event leakage')


def prepare(run):
    run=Path(run).resolve()
    if run.exists():
        if (run/'preparation_manifest.json').exists():return verify(run)
        raise DataError('Incomplete attempt retained; use a new version directory, cannot overwrite')
    paths=input_paths()
    with isolation(run,paths) as access:
        before=snapshot(paths)
        for base,name in ((T10,'registration.json'),(T12,'final_delivery_manifest.json'),(T14,'final_delivery_manifest.json')):
            artifacts=read_json(base/name)['artifacts']
            for path in paths:
                if path.is_relative_to(base) and (key:=path.relative_to(base).as_posix()) in artifacts:
                    bound=artifacts[key];require(before[str(path)]['sha256']==(bound['sha256'] if isinstance(bound,dict) else bound),'Prior artifact seal differs: '+key)
        for item in read_json(T10/'source5_inputs.json').values():require(sha256(Path(item['path']))==item['sha256'],'Source raw identity differs')
        c=read_json(ROOT/CONFIG);validate_config(c)
        d=common.validate_pool(load(T10/'train.npz'))
        blocks={f'block_{b}':load(T10/f'validation/block_{b}.npz') for b in (4,5)}
        validate_blocks(d,blocks)
        p=read_json(T10/'plan.json')
        require([p[k] for k in ('checkpoints','steps_per_checkpoint','batch_size','positive_per_batch','uniform_negative_per_batch','learning_rate','cpu_threads','minimum_validation_recall')]==[12,294,256,16,240,.001,4,.85],'Original schedule differs')
        run.mkdir(parents=True,exist_ok=False)
        for path in paths:
            if path.is_relative_to(ROOT/'src'):
                target=run/'source'/path.relative_to(ROOT/'src');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        for name in ('run_microwave_task22.py','finalize_microwave_task22.py'):shutil.copyfile(ROOT/'scripts'/name,run/name)
        shutil.copyfile(ROOT/'tests/test_microwave_task22.py',run/'test_microwave_task22.py')
        shutil.copyfile(ROOT/PROTOCOL,run/'protocol.md')
        fresh(run/'workspace_identity.json',dict(root=str(ROOT)))
        fresh(run/'config.json',c);fresh(run/'parent_training_plan.json',p)
        fresh(run/'inputs_before.json',before)
        fresh(run/'source_control.json',common.source_control())
        # Metadata only; never inspect/hash any referenced development artifact here.
        fresh(run/'future_identity.json',read_json(T14/'identity.json'))
        meta=read_json(T10/'normalization.json');proof={}
        torch.set_num_threads(4)
        for seed in SEEDS:
            old=load(T10/f'sampling_{seed}.npz',['indices'])['indices']
            require(old.shape==(3528,256),'Full original sampling matrix required')
            stream=BatchStream(d,'event',seed,p)
            for row in old:
                np.testing.assert_array_equal(row,stream.next())
                require(int((d['y'][row]>=200).sum())==16,'Original quota differs')
            counts=np.bincount(old.ravel(),minlength=N)
            np.testing.assert_array_equal(counts,load(T10/f'dry_exposure_{seed}.npz',['count'])['count'])
            np.savez_compressed(run/f'batches_{seed}.npz',indices=old)
            np.savez_compressed(run/f'exposure_{seed}.npz',total=counts)
            initial=make_initial(seed,16)
            for path in (T10/f'initial_{seed}.pt',T12/f'models/unit_BCE/seed_{seed}/initial.pt'):
                saved=torch.load(path,map_location='cpu',weights_only=True)
                require(saved.keys()==initial.keys() and all(torch.equal(v,saved[k]) for k,v in initial.items()),'Historical16 initial differs')
            wide=make_initial(seed,32);repeat=make_initial(seed,32)
            require(all(torch.equal(v,repeat[k]) for k,v in wide.items()) and sum(v.numel() for v in wide.values())==19809,'Wide initialization not repeatable')
            torch.save(wide,run/f'initial32_{seed}.pt')
            control=T12/f'models/unit_BCE/seed_{seed}';cm,net=load_state_model(control)
            for key,v in blocks.items():
                saved=load(control/f'epoch_{cm["selected"]["epoch"]}_{key}.npz')
                np.testing.assert_array_equal(v['t'],saved['t'])
                np.testing.assert_array_equal(predict_windows(cm,net,v['x']),saved['scores'])
            # Only generated inputs enter wide candidate forward checks.
            wm=CenterMaskTCN(32);wm.load_state_dict(wide)
            x=np.full((3,127),120.,np.float32);x[0,:48]=np.nan;x[1,79:]=np.nan
            z=tensor_input(meta,x,np.isfinite(x));out=wm(z)
            require(out.shape==(3,) and torch.isfinite(out).all(),'Wide masked forward differs')
            proof[str(seed)]=dict(original_batches_exact=3528,positive=16,uniform_negative=240,
                indices_sha256=gate.hash_array(old),initial16_exact=True,control_ordinary_exact=True,
                initial32_repeatable=True,initial32_tensor_hashes={k:gate.hash_array(v.numpy()) for k,v in wide.items()},
                width32_parameters=19809,masked127_forward=True)
        fresh(run/'sampling_proof.json',proof)
        fresh(run/'data_identity.json',dict(train={k:gate.hash_array(v) for k,v in d.items()},
            validation={k:{n:gate.hash_array(v) for n,v in b.items()} for k,b in blocks.items()},
            normalization_sha256=sha256(T10/'normalization.json'),support=[336704,2275,34351,201]))
        require(snapshot(paths)==before,'Preparation mutated input')
        fresh(run/'prepare_access.json',dict(reads=sorted(access['reads']),forbidden=access['forbidden'],new_fits=0,
            optimizers=0,optimizer_updates=0,backward_calls=0,development_reads=0,final_reads=0,optimizers_autograd_blocked=True))
        fresh(run/'registration.json',dict(schema=SCHEMA,artifacts=inventory(run),new_fits=0,optimizer_updates=0,authorized=False))
    return dict(status='prepared_pending_review_and_new_budget',registration_sha256=sha256(run/'registration.json'),new_fits=0,optimizer_updates=0)


def registered(run):
    run=Path(run);reg=read_json(run/'registration.json')
    require(reg['schema']==SCHEMA and not reg['authorized'],'Task22 registration differs')
    gate.check(run,reg['artifacts']);validate_config(read_json(run/'config.json'))
    # Validate immutable data/model inputs; mutable workspace code is never a runtime dependency.
    before=read_json(run/'inputs_before.json')
    data={p:v for p,v in before.items() if not Path(p).is_relative_to(ROOT/'src') and not Path(p).is_relative_to(ROOT/'scripts') and not Path(p).is_relative_to(ROOT/'tests') and p!=str(ROOT/CONFIG) and p!=str(ROOT/PROTOCOL)}
    require(snapshot(data)==data,'Registered data/model hash/mtime differs')
    return read_json(run/'config.json')


def execution_allowed(run):return list(read_json(run/'inputs_before.json'))+[run/k for k in inventory(run)]


def authorize(run,receipt):
    require(receipt is not None,'Task22 requires NEW human budget; 0 training started')
    a=read_json(receipt)
    require(a.get('authorized') is True and a.get('approval_origin')=='human' and a.get('experiment')==SCHEMA
        and a.get('registration_sha256')==sha256(Path(run)/'registration.json') and a.get('additional_new_fits')==3
        and a.get('maximum_updates')==10584 and a.get('seeds')==SEEDS and a.get('human_authorization_quote','').strip()
        and a.get('human_authorization_reference','').strip(),'Missing NEW hash-bound Task22 budget; old receipts insufficient')
    gate.check(run,read_json(Path(run)/'preparation_manifest.json')['artifacts'])
    require(read_json(Path(run)/'independent_review.json')['passed'],'Independent review required')
    # All mutation routes must run from the registered snapshot.
    expected=Path(run)/'source/virtual_submeter/accuracy/microwave_task22.py'
    require(Path(__file__).resolve()==expected.resolve(),'Execute sealed run_microwave_task22.py; mutable working-tree runtime forbidden')
    return a


def train(run,authorization=None):
    run=Path(run).resolve();a=authorize(run,authorization)
    with isolation(run,execution_allowed(run),training=True):
        registered(run)
        require(not (run/'training_started.json').exists() and not (run/'models').exists(),'No restart/retry permitted')
        fresh(run/'training_started.json',dict(authorization=a,seeds=SEEDS,maximum_updates=10584,loaded_source=str(Path(__file__).resolve())))
        d=load(T10/'train.npz');p=read_json(run/'parent_training_plan.json');meta=read_json(T10/'normalization.json')
        blocks={f'block_{b}':load(T10/f'validation/block_{b}.npz') for b in (4,5)}
        rows=[]
        for s in SEEDS:
            ix=load(run/f'batches_{s}.npz')['indices'];initial=torch.load(run/f'initial32_{s}.pt',weights_only=True,map_location='cpu')
            rows.append(loss.temporal.fit_one(run/f'models/wide32/seed_{s}',s,p,d,blocks,meta,ix,initial,
                sha256(run/'registration.json'),pos_weight=1.,recipe='unit_BCE',arm='wide32',channels=32))
        fresh(run/'training_complete.json',dict(rows=rows,new_fits=3,updates=10584,control_retrains=0,artifacts=inventory(run/'models')))
    return dict(new_fits=3,updates=10584)


def source_select(run,authorization=None):return pipeline.source_select(Path(run).resolve(),authorization,task=sys.modules[__name__])
def source_guard(run):return pipeline.source_guard(Path(run),task=sys.modules[__name__])
def freeze(run,authorization=None):return pipeline.freeze(Path(run).resolve(),authorization,task=sys.modules[__name__])
def evaluation_guard(run):return pipeline.evaluation_guard(Path(run),task=sys.modules[__name__])
def evaluate(run,authorization=None):return pipeline.evaluate(Path(run).resolve(),authorization,task=sys.modules[__name__])


def verify(run):
    from .microwave_task22_review import review_preparation,review_execution
    result=review_preparation(Path(run).resolve())
    if (Path(run)/'training_started.json').exists():result['execution']=review_execution(Path(run))
    return result


def report(run):return dict(**verify(run),report=(Path(run)/'PREPARATION_REPORT.zh.md').read_text(encoding='utf-8'))

def diagnostic_metrics(probability,y,threshold):
    """Fixed-point diagnostics, with tied scores grouped; never selects a threshold."""
    from ..models.state_selection import pr_curve
    curve=pr_curve(probability,y>=200)
    ap=float(np.sum(np.diff(np.r_[0.,curve['recall']])*curve['precision']))
    return dict(**gate.confusion((probability>=threshold).astype(np.int8),y>=200),ap=ap,threshold=float(threshold))


def diagnostics(run,authorization=None):
    """Optional post-training source/train explanation only, including failed sources."""
    run=Path(run).resolve();authorize(run,authorization)
    with isolation(run,execution_allowed(run)):
        registered(run)
        complete=read_json(run/'training_complete.json');gate.check(run/'models',complete['artifacts'])
        require(complete['new_fits']==3 and complete['updates']==10584,'Full fixed trained matrix required')
        require(not (run/'diagnostics').exists(),'Fixed diagnostics cannot overwrite')
        (run/'diagnostics').mkdir()
        datasets={'train':load(T10/'train.npz')}
        for b in (4,5):datasets[f'block_{b}']=load(T10/f'validation/block_{b}.npz')
        source={k:np.concatenate([datasets[f'block_{b}'][k] for b in (4,5)]) for k in ('x','y','t','house')}
        datasets={'train':datasets['train'],'source':source};rows=[]
        for seed in SEEDS:
            for arm,path in (('control16',T12/f'models/unit_BCE/seed_{seed}'),('wide32',run/f'models/wide32/seed_{seed}')):
                meta,net=load_state_model(path)
                for name,d in datasets.items():
                    with torch.inference_mode():
                        z=tensor_input(meta,d['x']);probability=np.concatenate([torch.sigmoid(net(v)).numpy() for v in z.split(meta['batch_size'])])
                    for view,mask in (('original',np.ones(len(d['y']),bool)),('strict_core31',np.isfinite(d['x'][:,48:79]).all(axis=1))):
                        p,y=probability[mask],d['y'][mask]
                        metric=diagnostic_metrics(p,y,meta['decision_threshold_probability'])
                        independent=scalar_metric((p>=meta['decision_threshold_probability']).astype(np.int8),y>=200)
                        require(all(metric[k]==v for k,v in independent.items()) and abs(metric['ap']-independent_ap(p,y>=200))<2e-12,'Independent diagnostic mismatch')
                        rows.append(dict(arm=arm,seed=seed,dataset=name,view=view,support_sha256=gate.hash_array(mask),**metric))
                    np.savez_compressed(run/f'diagnostics/{arm}_{seed}_{name}.npz',probability=probability,t=d['t'],house=d['house'])
        fresh(run/'diagnostics/metrics.json',dict(rows=rows,explanatory_only=True,new_thresholds=0,new_selections=0,development_reads=0))
        fresh(run/'diagnostics_complete.json',dict(artifacts=inventory(run/'diagnostics')))
    return dict(diagnostic_cells=len(rows),development_reads=0,new_selections=0)


# Resolve source dependencies before the numerical-input audit starts. Cold Python
# imports probe bytecode paths even with -B, so lazy imports inside isolation fail.
from .microwave_task22_metrics import independent_ap
from .microwave_task21_review import scalar_metric
