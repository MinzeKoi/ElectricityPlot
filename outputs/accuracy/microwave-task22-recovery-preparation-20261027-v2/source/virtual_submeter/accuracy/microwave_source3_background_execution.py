"""Prepared Task15 adapter; all real execution starts behind a new budget gate."""
import contextlib
import csv
from pathlib import Path

import numpy as np
import torch

from ..data.common import read_json, write_json, sha256, inventory, utc, iso
from . import microwave_source3_background as audit
from . import microwave_source5_loss as loss
from . import microwave_recall_guard95 as guard95
from . import microwave_event_gate as gate

require = audit.require
SEEDS = list(audit.SEEDS)
FAMILIES = ('control_guard95','aug_guard95')


class Augmentor:
    """Fixed eight positive substitutions with a local, namespaced RNG only."""
    def __init__(self, templates, backgrounds, seed):
        self.templates, self.backgrounds = templates, backgrounds
        require(len(templates['t'])>0 and len(backgrounds['t'])>0,'Empty source pools')
        require(len(np.unique(templates['house']))>=2,'At least two template households required')
        self.rng = np.random.default_rng(np.random.SeedSequence([150025,int(seed)]))
        self.groups = []
        keys = sorted(set(zip(templates['house'].tolist(),(templates['t']//86400).tolist())))
        for house,day in keys:
            ids = np.flatnonzero((templates['house']==house)&(templates['t']//86400==day))
            self.groups.append([ids[templates['event'][ids]==ev] for ev in sorted(np.unique(templates['event'][ids]))])
        self.records = []
        self.template_exposure = np.zeros(len(templates['t']),np.int64)
        self.background_exposure = np.zeros(len(backgrounds['t']),np.int64)
        self.next_update=0
        require(templates['micro_or_residual'].shape==(len(templates['t']),127),'Template shape differs')
        require(np.isfinite(templates['micro_or_residual']).all() and (templates['micro_or_residual']>=0).all(), 'Invalid full template')
        require((templates['micro_or_residual'][:,63]>=200).all(),'Template center is not on')
        b=backgrounds['micro_or_residual']; m=backgrounds['mask']
        require(np.array_equal(np.isfinite(b),m) and (b[m]>=0).all(),'Invalid residual/mask')

    def __call__(self,d,idx,update):
        require(update==self.next_update,'Augmentation cannot silently skip or replay an update')
        require(len(idx)==256,'Fixed 256 batch required')
        positives=np.flatnonzero(d['y'][idx]>=200)
        require(len(positives)==16,'Exactly sixteen original positive slots required')
        x=d['x'][idx].copy(); mask=d['mask'][idx].copy()
        for slot in positives[:8]:
            events=self.groups[int(self.rng.integers(len(self.groups)))]
            centers=events[int(self.rng.integers(len(events)))]
            ti=int(centers[int(self.rng.integers(len(centers)))])
            bi=int(self.rng.integers(len(self.backgrounds['t'])))
            bm=self.backgrounds['mask'][bi]
            raw=np.full(127,np.nan,np.float32)
            raw[bm]=self.backgrounds['micro_or_residual'][bi,bm]+self.templates['micro_or_residual'][ti,bm]
            require(np.array_equal(np.isfinite(raw),bm) and (raw[bm]>=0).all() and bm[63], 'Invalid pairing; stopping without fallback')
            x[slot]=raw; mask[slot]=bm
            self.template_exposure[ti]+=1; self.background_exposure[bi]+=1
            self.records.append(dict(update=update,slot=int(slot),original_source_index=int(idx[slot]),
                template_pool_index=ti,background_pool_index=bi,
                template_source_index=int(self.templates['source_index'][ti]),background_source_index=int(self.backgrounds['source_index'][bi]),
                template_house=int(self.templates['house'][ti]),background_house=int(self.backgrounds['house'][bi]),
                template_utc=int(self.templates['t'][ti]),background_utc=int(self.backgrounds['t'][bi]),
                synthetic_sha256=gate.hash_array(raw)))
        unchanged=np.ones(256,bool); unchanged[positives[:8]]=False
        np.testing.assert_array_equal(x[unchanged],d['x'][idx[unchanged]])
        np.testing.assert_array_equal(mask[unchanged],d['mask'][idx[unchanged]])
        self.next_update+=1
        return dict(x=x,mask=mask)

    def save_exposure(self,dest):
        gate.csv_save(Path(dest)/'augmentation_pairs.csv',self.records)
        np.savez_compressed(Path(dest)/'augmentation_exposure.npz',
            template_count=self.template_exposure,background_count=self.background_exposure,
            template_source_index=self.templates['source_index'],background_source_index=self.backgrounds['source_index'])


def load_pools(run):
    out=[]
    for kind in ('template','background'):
        houses=[]
        for hi,h in enumerate(('refit_2','refit_19','refit_5')):
            z=gate.npz(Path(run)/f'{kind}_{h}.npz',['source_index','micro_or_residual','t','role','event','mask'])
            z['house']=np.full(len(z['t']),hi,np.int64)
            houses.append(z)
        out.append({k:np.concatenate([h[k] for h in houses]) for k in houses[0]})
    return out


def authorize(run,receipt):
    # Deliberately first: no arrays, numerical hashes, model or optimizer before it.
    require(receipt is not None,'Task15 additional 3-fit augmentation budget is not authorized')
    a=read_json(Path(receipt))
    require(a.get('authorized') is True and a.get('approval_origin')=='human'
        and a.get('experiment')=='microwave_source3_background_v1'
        and a.get('additional_new_fits')==3 and a.get('maximum_updates')==10584
        and a.get('seeds')==SEEDS and a.get('approved_augmentation')=='fixed_source_template_background_8_of_16'
        and a.get('registration_sha256')==sha256(Path(run)/'registration.json')
        and bool(a.get('human_authorization_quote','').strip())
        and bool(a.get('human_authorization_reference','').strip()),
        'Need an explicit new human augmentation/3-fit budget bound to Task15 registration; historical receipts are invalid')
    return a


@contextlib.contextmanager
def phase(run,name,extra=(),allow_training=False):
    identity=read_json(Path(run)/'identity.json')
    allowed=[audit.PARENT/'train.npz']+[Path(p) for p in identity['model_inputs']]
    allowed += [Path(v['path']) for h in identity['source_inputs'].values() for v in h.values()]
    allowed += [audit.PARENT/f'sampling_{s}.npz' for s in SEEDS]+[audit.PARENT/f'initial_{s}.pt' for s in SEEDS]
    allowed += [audit.PARENT/f'validation/block_{b}.npz' for b in (4,5)]+list(extra)
    with loss.access_guard(run,'train' if allow_training else name,allowed) as state:
        try: yield
        finally:
            path=Path(run)/f'{name}_access.json'
            suffix=1
            while path.exists():
                path=Path(run)/f'{name}_access_{suffix}.json'; suffix+=1
            gate.fresh_json(path,dict(reads=sorted(state['reads']),forbidden=state['forbidden']))


def registered(run):
    a=audit.verify(run)
    require(a['feasible'],'Source pool rejected; cannot train')
    identity=read_json(Path(run)/'identity.json')
    config=read_json(Path(run)/'config.json')
    require(config['authorized'] is False and config['maximum_additional_fits_if_separately_authorized']==3
        and config['maximum_additional_updates']==10584 and config['augmentation_rng_namespace']==150025,
        'Registered disabled configuration or fixed budget differs')
    for path,digest in identity['source_metadata'].items(): require(sha256(Path(path))==digest,'Frozen control source identity changed')
    require(identity['algorithms']==dict(gate_sha256=sha256(Path(gate.__file__)),guard95_sha256=sha256(Path(guard95.__file__))), 'Fixed guard95 algorithm changed')
    # Critical executable dependencies must match the registered snapshot even if
    # an operator mistakenly launches the live workspace script after edits.
    live_root=Path(__file__).resolve().parents[1]
    live={str(p.relative_to(live_root)):sha256(p) for p in live_root.rglob('*.py')}
    frozen_root=Path(run)/'source/virtual_submeter'
    frozen={str(p.relative_to(frozen_root)):sha256(p) for p in frozen_root.rglob('*.py')}
    require(live==frozen,'Use the registered executable source snapshot')
    require(sha256(audit.PARENT/'train.npz')==identity['train_sha256'],'Training source changed')
    require(sha256(audit.PARENT/'normalization.json')==identity['normalization_sha256'],'Normalization changed')
    for path,digest in identity['model_inputs'].items(): require(sha256(Path(path))==digest,'Control model changed')
    for seed in SEEDS:
        require(sha256(audit.PARENT/f'sampling_{seed}.npz')==identity['sampling'][str(seed)]['sha256'],'Indices changed')
        require(sha256(audit.PARENT/f'initial_{seed}.pt')==identity['initial'][str(seed)],'Initialization changed')
    return config,identity


def train(run,authorization=None):
    run=Path(run); a=authorize(run,authorization)
    with phase(run,'train',allow_training=True):
        _,identity=registered(run)
        require(not (run/'training_started.json').exists() and not (run/'models').exists(),'No training restart/retry')
        gate.fresh_json(run/'training_started.json',dict(authorization=a,maximum_new_fits=3,maximum_updates=10584))
        d=gate.npz(audit.PARENT/'train.npz',loss.TRAIN_FIELDS)
        blocks={f'block_{i}':gate.npz(audit.PARENT/f'validation/block_{i}.npz',loss.VAL_FIELDS) for i in (4,5)}
        p=read_json(run/'parent_training_plan.json'); meta=read_json(audit.PARENT/'normalization.json')
        templates,backgrounds=load_pools(run); rows=[]
        for seed in SEEDS:
            indices=gate.npz(audit.PARENT/f'sampling_{seed}.npz',['indices'])['indices']
            initial=torch.load(audit.PARENT/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
            augment=Augmentor(templates,backgrounds,seed)
            dest=run/f'models/aug/seed_{seed}'
            try:
                row=loss.temporal.fit_one(dest,seed,p,d,blocks,meta,indices,initial,sha256(run/'registration.json'),
                    pos_weight=1.,recipe='unit_BCE',arm='aug',batch_transform=augment)
                rows.append(row)
            finally:
                if dest.exists(): augment.save_exposure(dest)
            require(augment.next_update==3528 and int(augment.template_exposure.sum())==28224,'Actual template exposure differs')
        gate.fresh_json(run/'new_point_rows.json',dict(rows=rows))
        gate.fresh_json(run/'training_complete.json',dict(new_fits=3,updates=10584,control_retrains=0,artifacts=inventory(run/'models')))
        return dict(status='trained_three_augmentation_models',new_fits=3,updates=10584)


def training_guard(run):
    p,identity=registered(run)
    done=read_json(Path(run)/'training_complete.json')
    require(done['new_fits']==3 and done['updates']==10584 and done['control_retrains']==0,'Training budget differs')
    gate.check(Path(run)/'models',done['artifacts'])
    return p,identity


def select_source(control,new):
    require([r['seed'] for r in control]==SEEDS,'Complete three-seed control required')
    qualified=loss.qualified(control,new)
    selected='control_guard95'
    if qualified:
        c,n=gate.mean_rows(control),gate.mean_rows(new)
        if (n['precision'],n['f1'])>(c['precision'],c['f1']): selected='aug_guard95'
    return dict(selected=selected,aug_eligible=qualified,source_rejection=None if qualified else 'Fixed source95/nondegenerate/control-relative qualification failed')


def qualification_details(control,new,recipes):
    complete=[r['seed'] for r in new]==SEEDS and [r['seed'] for r in recipes]==SEEDS
    result=dict(complete_three_seeds=complete,
        each_seed_each_block_R95=complete and all(r['feasible'] and len(r['blocks'])==2 and all(b['recall']>=.95 for b in r['blocks']) for r in recipes),
        each_seed_each_block_nondegenerate=complete and all(not b['degenerate'] for r in recipes for b in r['blocks']),
        mean_F1_not_below_control=complete and np.mean([r['f1'] for r in new])>=np.mean([r['f1'] for r in control]),
        each_seed_average_R_drop_at_most_005=complete and all(b['recall']-a['recall']>=-.05 for a,b in zip(control,new)),
        at_least_two_seed_P_improved=complete and sum(b['precision']>a['precision'] for a,b in zip(control,new))>=2)
    return {key:bool(value) for key,value in result.items()}


def calibrate_source95(scores):
    low_blocks={k:dict(values=d['p'][d['ix']],y=d['y']) for k,d in scores.items()}
    candidates=np.unique(np.concatenate([d['p'][np.isfinite(d['p'])] for d in scores.values()]))
    low,_=gate.select_threshold(low_blocks,candidates,.95)
    if low is None: return None,None
    return low,guard95.calibrate95(scores,low)


def source(run,authorization=None):
    run=Path(run); authorize(run,authorization)
    with phase(run,'source_select'):
        training_guard(run)
        require(not (run/'source_started.json').exists(),'Source selection is one-shot')
        gate.fresh_json(run/'source_started.json',dict(training_complete_sha256=sha256(run/'training_complete.json')))
        raw=gate.csv_rows(Path(read_json(run/'identity.json')['source_inputs']['refit_5']['aggregate']['path']),'refit_5')
        times=np.array([utc(r['timestamp']) for r in raw]); recipes=[]; new=[]
        source_dest=run/'source_results'; source_dest.mkdir()
        for seed in SEEDS:
            model=run/f'models/aug/seed_{seed}'; meta=read_json(model/'model.json'); scores={}
            for i,lo,hi in loss.temporal.f.VALIDATION:
                block=gate.npz(audit.PARENT/f'validation/block_{i}.npz',['t','y'])
                rr=[raw[j] for j in np.flatnonzero((times>=lo)&(times<hi))]
                t,p,_=gate.infer(model,rr); ix=gate.align(t,block['t'])
                saved=gate.npz(model/f'epoch_{meta["selected"]["epoch"]}_block_{i}.npz',['t','scores'])
                np.testing.assert_array_equal(saved['t'],block['t'])
                np.testing.assert_allclose(p[ix],saved['scores'],rtol=1e-5,atol=1e-7)
                np.testing.assert_array_equal(p[ix]>=meta['decision_threshold_probability'],saved['scores']>=meta['decision_threshold_probability'])
                scores[str(i)]=dict(t=t,p=p,h=gate.local_max(t,p),ix=ix,y=block['y']>=200)
                np.savez_compressed(source_dest/f'seed_{seed}_block_{i}.npz',t=t,p=p,ix=ix,y=block['y']>=200)
            low,high=calibrate_source95(scores) if meta['selected']['source_gate_passed'] else (None,None)
            feasible=low is not None and high is not None
            metrics=[dict(block=k,**gate.confusion(gate.gate_state(d['p'],d['h'],low,high)[d['ix']],d['y'])) for k,d in scores.items()] if feasible else []
            if feasible: new.append(dict(seed=seed,**gate.mean_rows(metrics),degenerate=any(m['degenerate'] for m in metrics),source_gate_passed=all(m['recall']>=.95 for m in metrics)))
            recipes.append(dict(seed=seed,L=low,H=high,feasible=feasible,blocks=metrics,radius_samples=15,
                parent_hashes={k:sha256(model/k) for k in ('model.json','weights.pt')}))
        control=read_json(audit.CONTROL/'source_selection.json')['rows']['guard95']
        choice=select_source(control,new)
        gate.fresh_json(run/'source_selection.json',dict(source_only=True,**choice,rows=dict(control_guard95=control,aug_guard95=new),recipes=recipes,
            qualification=qualification_details(control,new,recipes),
            registration_sha256=sha256(run/'registration.json'),training_complete_sha256=sha256(run/'training_complete.json'),
            gate_algorithm_sha256=sha256(Path(gate.__file__)),guard95_algorithm_sha256=sha256(Path(guard95.__file__))))
        gate.fresh_json(run/'source_complete.json',dict(source_selection_sha256=sha256(run/'source_selection.json'),registration_sha256=sha256(run/'registration.json'),artifacts=inventory(source_dest)))
        return choice


def source_guard(run):
    training_guard(run)
    s=read_json(Path(run)/'source_selection.json')
    complete=read_json(Path(run)/'source_complete.json')
    require(complete['source_selection_sha256']==sha256(Path(run)/'source_selection.json') and complete['registration_sha256']==sha256(Path(run)/'registration.json'),'Source completion seal differs')
    gate.check(Path(run)/'source_results',complete['artifacts'])
    require(s['source_only'] and s['registration_sha256']==sha256(Path(run)/'registration.json')
        and s['training_complete_sha256']==sha256(Path(run)/'training_complete.json')
        and s['gate_algorithm_sha256']==sha256(Path(gate.__file__)) and s['guard95_algorithm_sha256']==sha256(Path(guard95.__file__)), 'Source selection seal differs')
    require({k:s[k] for k in ('selected','aug_eligible','source_rejection')}==select_source(s['rows']['control_guard95'],s['rows']['aug_guard95']), 'Source choice differs')
    verify_source_numerics(run,s)
    return s


def scalar_confusion(state,y):
    """Independent scalar counts, following Task14's independent verifier."""
    require(len(state)==len(y) and np.isin(state,[0,1]).all(),'Invalid independent counting support')
    counts=dict(tp=0,fp=0,fn=0,tn=0)
    for pred,target in zip(state,y):
        counts['tp' if pred and target else 'fp' if pred else 'fn' if target else 'tn']+=1
    tp,fp,fn,tn=(counts[k] for k in ('tp','fp','fn','tn'))
    n=len(y); on=tp+fp
    return dict(**counts,n=n,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
        fpr=fp/max(fp+tn,1),predicted_on=on,degenerate=on in (0,n))


def verify_source_numerics(run,s):
    """Recompute source thresholds/counts/qualification; never touches development."""
    rows=[]
    require([r['seed'] for r in s['recipes']]==SEEDS,'Source recipe matrix incomplete')
    control=read_json(audit.CONTROL/'source_selection.json')['rows']['guard95']
    require(s['rows']['control_guard95']==control,'Frozen source control rows differ')
    for recipe in s['recipes']:
        seed=recipe['seed']; scores={}
        model=Path(run)/f'models/aug/seed_{seed}'
        meta=read_json(model/'model.json')
        for name,digest in recipe['parent_hashes'].items(): require(sha256(model/name)==digest,'Source recipe model binding differs')
        for block in (4,5):
            z=gate.npz(Path(run)/f'source_results/seed_{seed}_block_{block}.npz',['t','p','ix','y'])
            original=gate.npz(audit.PARENT/f'validation/block_{block}.npz',['t','y'])
            np.testing.assert_array_equal(z['t'][z['ix']],original['t'])
            np.testing.assert_array_equal(z['y'],original['y']>=200)
            scores[str(block)]={**z,'h':gate.local_max(z['t'],z['p'])}
        low,high=calibrate_source95(scores) if meta['selected']['source_gate_passed'] else (None,None)
        require((recipe['L'],recipe['H'])==(low,high),'Exact source L95/H95 replay differs')
        feasible=low is not None and high is not None
        require(recipe['feasible']==feasible,'Source feasibility differs')
        metrics=[]
        if feasible:
            for key,z in scores.items():
                prediction=((z['p'][z['ix']]>=low)&(z['h'][z['ix']]>=high)).astype(np.int8)
                metrics.append(dict(block=key,**scalar_confusion(prediction,z['y'])))
            rows.append(dict(seed=seed,**gate.mean_rows(metrics),degenerate=any(m['degenerate'] for m in metrics),source_gate_passed=all(m['recall']>=.95 for m in metrics)))
        require(metrics==recipe['blocks'],'Source block scalar confusion differs')
    require(rows==s['rows']['aug_guard95'],'Source seed scalar metrics differ')
    # Independent implementation of the fixed relative qualification and ranking.
    eligible=(len(rows)==3 and [r['seed'] for r in rows]==SEEDS and all(r['source_gate_passed'] and not r['degenerate'] for r in rows)
        and np.mean([r['f1'] for r in rows])>=np.mean([r['f1'] for r in control])
        and all(b['recall']-a['recall']>=-.05 for a,b in zip(control,rows))
        and sum(b['precision']>a['precision'] for a,b in zip(control,rows))>=2)
    selected='control_guard95'
    if eligible:
        c,n=gate.mean_rows(control),gate.mean_rows(rows)
        if (n['precision'],n['f1'])>(c['precision'],c['f1']): selected='aug_guard95'
    require(bool(eligible)==s['aug_eligible'] and selected==s['selected'],'Independent source qualification/preselection differs')
    details=qualification_details(control,rows,s['recipes'])
    require(s['qualification']==details and bool(all(details.values()))==bool(eligible),'Detailed source qualification differs')
    return dict(source_qualified=bool(eligible),selected=selected,rows=rows)


def ordinary(run,family,seed,rows):
    s=read_json(Path(run)/'source_selection.json')
    if family=='control_guard95':
        recipe=next(r for r in read_json(audit.CONTROL/'source_selection.json')['recipes'] if r['seed']==seed)
        model=audit.LOSS/f'models/unit_BCE/seed_{seed}'
    else:
        require(family=='aug_guard95','Unknown family')
        recipe=next(r for r in s['recipes'] if r['seed']==seed)
        model=Path(run)/f'models/aug/seed_{seed}'
    require(recipe['feasible'],'No feasible source recipe')
    t,p,_=gate.infer(model,rows)
    return dict(t=t,scores=p,state=gate.gate_state(p,gate.local_max(t,p),recipe['L'],recipe['H']))


def frozen_matrix():
    return [[a,s] for a in ('A','W50',*FAMILIES) for s in SEEDS]


def freeze(run,authorization=None):
    run=Path(run); authorize(run,authorization)
    # Qualify source before reading or hashing any development numerical artifact.
    with phase(run,'source_freeze_gate'):
        s=source_guard(run)
        require(s['aug_eligible'] and all(r['feasible'] for r in s['recipes']),'Source rejected: no development matrix may be generated')
    ref=read_json(audit.CONTROL/'identity.json')
    with phase(run,'freeze',extra=list(ref['development_unlabeled'])):
        require(not (run/'predictions').exists(),'Freeze is one-shot')
        loss.check_paths(ref['development_unlabeled'])
        base=Path(ref['baseline']); agg=next(Path(p) for p in ref['development_unlabeled'] if p.endswith('aggregate.csv'))
        raw=gate.csv_rows(agg,'refit_17'); dest=run/'predictions'; dest.mkdir(); checks=[]
        for seed in SEEDS:
            a=gate.npz(base/f'reference/predictions/seed_{seed}/{gate.PARTITION}',['stamps','raw','corrected'])
            w=gate.npz(base/f'predictions/W50/seed_{seed}/{gate.PARTITION}',['stamps','raw'])
            np.testing.assert_array_equal(a['stamps'],w['stamps'])
            np.testing.assert_array_equal(np.isfinite(a['raw']),np.isfinite(w['raw']))
            states=np.where(np.isfinite(a['corrected']),a['corrected']>=np.array([2000,200,20]),-1).astype(np.int8)
            np.savez_compressed(dest/f'A_{seed}.npz',t=a['stamps'],raw=a['raw'],corrected=a['corrected'],states=states)
            np.savez_compressed(dest/f'W50_{seed}.npz',t=w['stamps'],raw=w['raw'])
            for family in FAMILIES:
                z=ordinary(run,family,seed,raw); replay=ordinary(run,family,seed,raw)
                for k in z: np.testing.assert_array_equal(z[k],replay[k])
                np.testing.assert_array_equal(z['t'],a['stamps'])
                np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
                other=states.copy(); other[:,1]=z['state']
                path=dest/f'{family}_{seed}.npz'
                np.savez_compressed(path,**z,raw=a['raw'],corrected=a['corrected'],other_states=other)
                saved=gate.npz(path,['t','scores','state','raw','corrected','other_states'])
                for k in z: np.testing.assert_array_equal(saved[k],replay[k])
                np.testing.assert_array_equal(saved['raw'],a['raw']); np.testing.assert_array_equal(saved['corrected'],a['corrected'])
                np.testing.assert_array_equal(saved['other_states'][:,[0,2]],states[:,[0,2]])
                checks.append(dict(seed=seed,family=family,ordinary_exact=True,raw_A_exact=True,other_states_A_exact=True,support_exact=True))
        gate.fresh_json(run/'inference_checks.json',dict(checks=checks))
        gate.fresh_json(run/'freeze.json',dict(matrix=frozen_matrix(),source_selection_sha256=sha256(run/'source_selection.json'),
            registration_sha256=sha256(run/'registration.json'),identity_sha256=sha256(run/'identity.json'),truth_opened=False,
            artifacts={'predictions/'+k:v for k,v in inventory(dest).items()}|{'inference_checks.json':sha256(run/'inference_checks.json')}))
        return dict(status='frozen',matrix=frozen_matrix(),selected=s['selected'])


def evaluation_guard(run):
    s=source_guard(run); f=read_json(Path(run)/'freeze.json')
    require(s['aug_eligible'] and f['matrix']==frozen_matrix() and not f['truth_opened'],'Incomplete/invalid freeze')
    for name in ('source_selection','registration','identity'): require(f[name+'_sha256']==sha256(Path(run)/(name+'.json')),'Freeze seal differs')
    expected={f'predictions/{a}_{seed}.npz' for a,seed in frozen_matrix()}|{'inference_checks.json'}
    require(set(f['artifacts'])==expected,'Incomplete or extra matrix')
    gate.check(run,f['artifacts'])
    checks=read_json(Path(run)/'inference_checks.json')['checks']
    require(len(checks)==6 and all(all(r[k] for k in ('ordinary_exact','raw_A_exact','other_states_A_exact','support_exact')) for r in checks),'Missing exact inference checks')
    return s


def contiguous_rows(t,selected):
    """Every false/unknown point or non-eight-second step closes the segment."""
    indices=np.flatnonzero(selected)
    if not len(indices): return []
    breaks=np.flatnonzero((np.diff(indices)!=1)|(np.diff(t[indices])!=8))+1
    return [dict(start_utc=iso(int(t[g[0]])),end_exclusive_utc=iso(int(t[g[-1]])+8),
                 points=len(g),duration_seconds=len(g)*8) for g in np.split(indices,breaks)]


def paired_analysis(run,truth):
    """Diagnostic only; called after source selection and complete freeze."""
    summaries=[]; segments=[]
    for seed in SEEDS:
        w=gate.npz(Path(run)/f'predictions/W50_{seed}.npz',['t','raw'])
        c=gate.npz(Path(run)/f'predictions/control_guard95_{seed}.npz',['t','state'])
        a=gate.npz(Path(run)/f'predictions/aug_guard95_{seed}.npz',['t','state'])
        t=w['t']; np.testing.assert_array_equal(t,truth['t'])
        for z in (c,a): np.testing.assert_array_equal(t,z['t'])
        valid=np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1]); y=truth['y']>=200
        new=a['state']==1
        for reference,before in (('W50',w['raw'][:,1]>=200),('control_guard95',c['state']==1)):
            removed=valid&~y&before&~new; added=valid&~y&~before&new
            old_metrics=scalar_confusion(before[valid],y[valid]); new_metrics=scalar_confusion(new[valid],y[valid])
            summaries.append(dict(seed=seed,reference=reference,removed_fp=int(removed.sum()),added_fp=int(added.sum()),
                restored_tp=int((valid&y&~before&new).sum()),new_fn=int((valid&y&before&~new).sum()),
                delta_tp=new_metrics['tp']-old_metrics['tp'],delta_fp=new_metrics['fp']-old_metrics['fp'],
                delta_fn=new_metrics['fn']-old_metrics['fn'],delta_tn=new_metrics['tn']-old_metrics['tn'],
                delta_P=new_metrics['precision']-old_metrics['precision'],delta_R=new_metrics['recall']-old_metrics['recall'],
                delta_F1=new_metrics['f1']-old_metrics['f1']))
            for kind,mask in (('reference_fp',valid&~y&before),('augmentation_fp',valid&~y&new),('removed_fp',removed),('added_fp',added)):
                rows=contiguous_rows(t,mask)
                require(sum(r['points'] for r in rows)==int(mask.sum()),'Segment support changed')
                segments.extend(dict(seed=seed,reference=reference,kind=kind,**r) for r in rows)
    return dict(paired=summaries,segments=segments)


def independent_formal_summary(rows,supports,selected):
    """Recalculate all original eight gates without trusting saved booleans."""
    groups={arm:[next(r for r in rows if r['arm']==arm and r['seed']==seed) for seed in SEEDS] for arm in ('A','W50',selected)}
    means={arm:{k:float(np.mean([r[k] for r in rr])) for k in ('precision','recall','f1','fp')} for arm,rr in groups.items()}
    w,new=groups['W50'],groups[selected]; wm,nm=means['W50'],means[selected]
    gates=dict(mean_P_at_least_008=nm['precision']>=.08,each_R_at_least_085=all(r['recall']>=.85 for r in new),
        mean_F1_gain_at_least_002=nm['f1']-wm['f1']+1e-12>=.02,
        at_least_two_seed_P_F1_up=sum(b['precision']>a['precision'] and b['f1']>a['f1'] for a,b in zip(w,new))>=2,
        each_F1_drop_at_most_002=all(b['f1']-a['f1']>=-.02 for a,b in zip(w,new)),
        mean_FP_reduction_at_least_30_percent=nm['fp']<=.7*wm['fp'],
        no_new_all_on_or_off=all(0<r['tp']+r['fp']<r['n'] for r in new),
        support_137523_exact=all(r['n']==137523 for r in supports))
    return dict(acceptance_gates=gates,development_acceptance_passed=all(gates.values()),formal_means=means)


def verify_development_numerics(run):
    """Read the local evaluated cache, never reopen the parent truth cache."""
    run=Path(run); s=evaluation_guard(run)
    complete=read_json(run/'evaluation_complete.json'); gate.check(run/'evaluation',complete['artifacts'])
    require(complete['truth_cache_loads']==1 and complete['original_truth_csv_reads']==0,'Truth load count differs')
    truth=gate.npz(run/'evaluation/truth_once.npz',['t','y'])
    rows=[]; supports=[]
    for seed in SEEDS:
        a=gate.npz(run/f'predictions/A_{seed}.npz',['t','raw','corrected','states'])
        w=gate.npz(run/f'predictions/W50_{seed}.npz',['t','raw'])
        np.testing.assert_array_equal(a['t'],truth['t']); np.testing.assert_array_equal(w['t'],truth['t'])
        valid=np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1])
        np.testing.assert_array_equal(valid,np.isfinite(truth['y'])&np.isfinite(a['raw'][:,1]))
        require(valid.sum()==137523,'Independent effective support changed')
        supports.append(dict(seed=seed,n=int(valid.sum()),mask_sha256=gate.hash_array(valid),t_sha256=gate.hash_array(a['t'][valid])))
        states={arm:z['raw'][:,1]>=200 for arm,z in (('A',a),('W50',w))}
        for family in FAMILIES:
            z=gate.npz(run/f'predictions/{family}_{seed}.npz',['t','raw','corrected','other_states','scores','state'])
            np.testing.assert_array_equal(z['t'],truth['t']); np.testing.assert_array_equal(z['raw'],a['raw']); np.testing.assert_array_equal(z['corrected'],a['corrected'])
            np.testing.assert_array_equal(z['other_states'][:,[0,2]],a['states'][:,[0,2]])
            np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
            states[family]=z['state']
        for family,pred in states.items(): rows.append(dict(seed=seed,arm=family,**scalar_confusion(pred[valid],truth['y'][valid]>=200)))
    recorded=read_json(run/'evaluation/metrics.json')
    require(recorded==dict(rows=rows,supports=supports),'Independent scalar counts/P/R/F1/support differ from metrics')
    independent=independent_formal_summary(rows,supports,s['selected'])
    summary=read_json(run/'evaluation/summary.json')
    require(all(summary[k]==independent[k] for k in independent),'Independent eight-gate summary differs')
    pair=paired_analysis(run,truth)
    require(pair['paired']==read_json(run/'evaluation/paired_analysis.json')['paired'],'Independent paired FP changes differ')
    with (run/'evaluation/fp_segments.csv').open(encoding='utf-8',newline='') as stream: saved=list(csv.DictReader(stream))
    expected=[{k:str(v) for k,v in r.items()} for r in pair['segments']]
    require(saved==expected,'Independent continuous FP segments differ')
    return dict(status='independent_scalar_numeric_verification_passed',**independent,rows=rows,supports=supports,
        paired=pair['paired'],segments=len(pair['segments']),parent_truth_cache_reopened=False)


def evaluate(run,authorization=None):
    run=Path(run); authorize(run,authorization)
    # No Task12 truth-cache numerical hash or opening occurs before these seals.
    with phase(run,'evaluation_gate'):
        evaluation_guard(run)
        require(not (run/'evaluation').exists(),'Development evaluation is one-shot')
    ref=read_json(audit.CONTROL/'identity.json'); info=ref['truth_cache']
    with phase(run,'evaluate',extra=[info['path']]):
        dest=run/'evaluation'; dest.mkdir()
        gate.fresh_json(dest/'started.json',dict(freeze_sha256=sha256(run/'freeze.json')))
        require(sha256(Path(info['path']))==info['sha256'],'Task12 truth cache identity changed')
        truth=gate.npz(Path(info['path']),['t','y']); rows=[]; supports=[]
        for seed in SEEDS:
            a=gate.npz(run/f'predictions/A_{seed}.npz',['t','raw','corrected','states'])
            w=gate.npz(run/f'predictions/W50_{seed}.npz',['t','raw'])
            np.testing.assert_array_equal(a['t'],truth['t']); np.testing.assert_array_equal(w['t'],a['t'])
            mask=np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1])
            np.testing.assert_array_equal(mask,np.isfinite(truth['y'])&np.isfinite(a['raw'][:,1]))
            require(mask.sum()==137523,'Effective support changed')
            supports.append(dict(seed=seed,n=int(mask.sum()),mask_sha256=gate.hash_array(mask),t_sha256=gate.hash_array(a['t'][mask])))
            for arm,z in (('A',a),('W50',w)):
                rows.append(dict(seed=seed,arm=arm,**gate.confusion((z['raw'][mask,1]>=200).astype(np.int8),truth['y'][mask]>=200)))
            for family in FAMILIES:
                z=gate.npz(run/f'predictions/{family}_{seed}.npz',['t','raw','scores','state','corrected','other_states'])
                np.testing.assert_array_equal(z['t'],a['t']); np.testing.assert_array_equal(z['raw'],a['raw']); np.testing.assert_array_equal(z['corrected'],a['corrected'])
                np.testing.assert_array_equal(z['other_states'][:,[0,2]],a['states'][:,[0,2]])
                np.testing.assert_array_equal(np.isfinite(z['scores']),np.isfinite(a['raw'][:,1]))
                rows.append(dict(seed=seed,arm=family,**gate.confusion(z['state'][mask],truth['y'][mask]>=200)))
        # Preserve this one loaded truth for later independent verification; the
        # parent Task12 cache is not reopened by verify/report.
        np.savez_compressed(dest/'truth_once.npz',**truth)
        metrics=dict(rows=rows,supports=supports)
        gate.fresh_json(dest/'metrics.json',metrics)
        result=loss.summary(run,metrics)
        gate.fresh_json(dest/'summary.json',result)
        gate.csv_save(dest/'confusions.csv',rows)
        pair=paired_analysis(run,truth)
        gate.fresh_json(dest/'paired_analysis.json',dict(paired=pair['paired'],diagnostic_only=True,used_for_selection=False))
        gate.csv_save(dest/'paired_changes.csv',pair['paired'])
        gate.csv_save(dest/'fp_segments.csv',pair['segments'])
        gate.fresh_json(run/'evaluation_complete.json',dict(freeze_sha256=sha256(run/'freeze.json'),truth_cache_sha256=info['sha256'],original_truth_csv_reads=0,truth_cache_loads=1,artifacts=inventory(dest)))
        return result


def predict(run,authorization=None,*,family=None,seed=None,aggregate=None,output=None):
    run=Path(run); authorize(run,authorization)
    with phase(run,'predict_source_gate'):
        s=source_guard(run)
        family=family or s['selected']
        require(seed in SEEDS and family in FAMILIES,'Choose a frozen seed/family')
    require(aggregate is not None and output is not None and not Path(output).exists(),'Fresh output and aggregate required')
    with phase(run,'predict',extra=[aggregate]):
        with Path(aggregate).open(encoding='utf-8',newline='') as stream: raw=list(csv.DictReader(stream))
        z=ordinary(run,family,seed,raw)
        gate.csv_save(Path(output),[dict(timestamp=iso(int(t)),probability_on=float(p) if np.isfinite(p) else '',state=int(v) if v>=0 else '') for t,p,v in zip(z['t'],z['scores'],z['state'])])
        return dict(output=str(output),family=family,seed=seed)


def verify(run,authorization=None):
    run=Path(run)
    if not (run/'training_started.json').exists():
        # Does not load models, construct optimizers or touch development inputs.
        with phase(run,'verify_preparation'):
            registered(run)
        return dict(status='complete_preparation_verified',authorized=False,new_fits=0,updates=0,
                    registration_sha256=sha256(run/'registration.json'))
    authorize(run,authorization)
    with phase(run,'verify_execution'):
        _,identity=training_guard(run)
        from ..models.center_mask_tcn import load_state_model,predict_windows
        norm=read_json(audit.PARENT/'normalization.json')
        for seed in SEEDS:
            model_dir=run/f'models/aug/seed_{seed}'
            meta,model=load_state_model(model_dir)
            require(meta['updates']==3528 and meta['recipe']=='unit_BCE' and all(meta[k]==v for k,v in norm.items()),'Model recipe differs')
            original=torch.load(audit.PARENT/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
            actual=torch.load(model_dir/'initial.pt',weights_only=True,map_location='cpu')
            require(all(torch.equal(original[k],actual[k]) for k in original),'Initial parameters differ')
            frozen=gate.npz(audit.PARENT/f'sampling_{seed}.npz',['indices'])['indices']
            exposure=gate.npz(model_dir/'exposure.npz',['count'])['count']
            np.testing.assert_array_equal(exposure,np.bincount(frozen.ravel(),minlength=336704))
            extra=gate.npz(model_dir/'augmentation_exposure.npz',['template_count','background_count'])
            require(int(extra['template_count'].sum())==int(extra['background_count'].sum())==28224,'Actual synthetic exposure differs')
            for i in (4,5):
                val=gate.npz(audit.PARENT/f'validation/block_{i}.npz',['x','t'])
                saved=gate.npz(model_dir/f'epoch_{meta["selected"]["epoch"]}_block_{i}.npz',['t','scores'])
                np.testing.assert_array_equal(saved['t'],val['t'])
                np.testing.assert_array_equal(predict_windows(meta,model,val['x']),saved['scores'])
        if (run/'source_selection.json').exists(): source_guard(run)
        if (run/'freeze.json').exists(): evaluation_guard(run)
        if (run/'evaluation_complete.json').exists():
            c=read_json(run/'evaluation_complete.json')
            require(c['freeze_sha256']==sha256(run/'freeze.json') and c['truth_cache_loads']==1 and c['original_truth_csv_reads']==0,'Evaluation seal differs')
            gate.check(run/'evaluation',c['artifacts'])
            independent=verify_development_numerics(run)
            path=run/'independent_execution_verification.json'
            if path.exists():
                require(read_json(path)==independent,'Saved independent numerical verification differs')
            else: gate.fresh_json(path,independent)
        return dict(status='execution_verified',new_fits=3,updates=10584,control_retrains=0)


def preparation_report(run):
    run=Path(run); a=audit.verify(run)
    dest=run/'PREPARATION_REPORT.md'
    require(not dest.exists(),'Preparation report cannot be overwritten')
    lines=['# Task15 固定模板背景增强：完整实施准备', '',
        f"来源资格通过：{a['templates']} 个合法127点模板，{a['backgrounds']} 个合法背景，来自 {a['template_source_households']} 个模板家庭。",'',
        '|来源|原正中心|合法模板|合法背景|因127点未知淘汰|','|---|---:|---:|---:|---:|']
    for h in a['houses']: lines.append(f"|{h['house']}|{h['positive_centers']}|{h['templates']}|{h['backgrounds']}|{h['positive_lost_due_unknown127']}|")
    lines += ['', '本目录逐字节复用v2来源池，没有重构池或重新解析原CSV。v2初步报告中关于失败/缺少适配器的固定文案已由本报告更正和取代；v2保持原封存。', '',
        '当前真实新fit 0、更新0、House17/最终8/9/20数值读取0。authorized=false；只有模板保留的disabled回执，不存在真实授权回执。旧fit授权、Task14零fit规则授权均会被拒绝。', '',
        '固定增强：每批原16正前8slot替换；另8正及240负exact不变。源家庭/UTC日→事件→合法中心分层均匀模板，独立均匀背景。RNG=SeedSequence([150025,seed])，不触动原采样/模型随机状态。全部先原始W合成，再原log1p归一化；背景mask0仍NaN，无fallback。', '',
        '复用原fit_one、checkpoint选择、普通加载和gate算法。共享fit只增加默认None的原始W batch_transform，旧None输入路径exact。实际template/background曝光与原索引曝光分开保存。没有真实训练、来源阈值校准或开发候选产生。', '',
        '固定3528更新/seed（12×294），三seed共10584；每seed 903168个原索引slot，其中28224个合成正slot，三seed84672个合成slot。训练费时仅能参考原模型：同结构原训练成本量级，额外逐batch合成、日志与归一化会增加开销；本阶段不运行计时训练。', '',
        '授权后的唯一训练命令（此处为待执行说明，不会自动运行）：', '',
        '```powershell',
        f'.venv-m2/Scripts/python.exe -B "{run}/run_microwave_source3_background.py" train --run "{run}" --authorization <本任务新增3fit人类回执.json>',
        '```', '',
        '随后同入口依次 source、freeze、evaluate、verify、report；均使用同一明确授权。source使用固定L95/H95/radius15并与Task14无增强guard95对照；来源不合格则拒绝freeze，不读取开发数据。freeze封存A/W50/control_guard95/aug_guard95×3seed完整12格和普通推理exact证据后，evaluate一次读取已有Task12真值缓存。', '',
        '普通预测入口 predict --family control_guard95|aug_guard95 --seed 42 --aggregate <总表CSV> --output <新CSV>，使用同一授权并校验来源选择。UI默认不切。', '',
        '来源/开发验收门槛全部固定在config.json；House17若未来读取仅是重复工程开发。不能把本次池可行性解释为增强效果已验证。', '',
            '审阅证据：source_pool_audit.json、date_event_support.csv、audit_refit_*.npz、pool_manifest.csv、identity.json、frozen_replay.json、prepare_access.json、registration.json及source/完整源码快照。首次v1列名解析失败已保留，不计训练；v2池资格通过，本快照提供完整实施准备。']
    dest.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(status='complete_preparation_pending_human_augmentation_budget',templates=a['templates'],backgrounds=a['backgrounds'],
                authorized=False,new_fits=0,updates=0,report=str(dest))


def report(run,authorization=None):
    run=Path(run)
    if not (run/'training_started.json').exists():
        audit.verify(run)
        return dict(status='preparation_only',report=str(run/'PREPARATION_REPORT.md'),new_fits=0,updates=0)
    authorize(run,authorization)
    s=source_guard(run)
    result=dict(source_selection=s['selected'],source_rejection=s['source_rejection'],new_fits=3,updates=10584)
    dest=run/'EXECUTION_REPORT.md'
    require(not dest.exists(),'Report cannot be overwritten')
    lines=['# Task15 固定背景增强执行报告','',f"来源唯一预选：**{s['selected']}**。增强来源资格：{'通过' if s['aug_eligible'] else '拒绝'}。",'',
        '本次新增3个模型、10584更新；原控制模型0重训。训练只改变每批前8个正slot的来源模板与背景组合，来源标签集合未扩充。', '',
        '## 来源资格与唯一预选', '',
        '|资格条件|结果|','|---|---|']
    labels=dict(complete_three_seeds='三seed完整',each_seed_each_block_R95='每seed每块R≥95%',each_seed_each_block_nondegenerate='每seed每块非全开/全关',
        mean_F1_not_below_control='meanF1不低无增强control',each_seed_average_R_drop_at_most_005='各seed平均R最多降5个百分点',at_least_two_seed_P_improved='至少2seed精度提升')
    for k,v in s['qualification'].items(): lines.append(f"|{labels[k]}|{'通过' if v else '未通过'}|")
    lines += ['', '阈值仅由固定来源L95/H95/radius15规则决定；checkpoint仍按原source85规则。按meanP、meanF1预选，平分control优先。', '',
        '|seed|L95|H95|来源块|TP|FP|FN|TN|P|R|F1|','|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for recipe in s['recipes']:
        if not recipe['feasible']:
            lines.append(f"|{recipe['seed']}|{recipe['L']}|{recipe['H']}|无可行固定来源阈值|—|—|—|—|—|—|—|")
        for r in recipe['blocks']:
            lines.append(f"|{recipe['seed']}|{recipe['L']:.8g}|{recipe['H']:.8g}|{r['block']}|{r['tp']}|{r['fp']}|{r['fn']}|{r['tn']}|{r['precision']:.6f}|{r['recall']:.6f}|{r['f1']:.6f}|")
    lines += ['', '|seed|control P|control R|control F1|增强 P|增强 R|增强 F1|ΔP|ΔR|ΔF1|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in s['rows']['control_guard95']:
        new=next((r for r in s['rows']['aug_guard95'] if r['seed']==c['seed']),None)
        if new:
            lines.append('|'+str(c['seed'])+'|'+'|'.join(f'{r[k]:.6f}' for r in (c,new) for k in ('precision','recall','f1'))+'|'+'|'.join(f'{new[k]-c[k]:+.6f}' for k in ('precision','recall','f1'))+'|')
        else: lines.append(f"|{c['seed']}|{c['precision']:.6f}|{c['recall']:.6f}|{c['f1']:.6f}|无可行候选|—|—|—|—|—|")
    if not s['aug_eligible']:
        lines += ['', f"来源拒绝原因：{s['source_rejection']}。固定方案在来源阶段终止，未生成开发预测或读取开发真值；不会放宽阈值、换配方或按开发表现改选。"]
    elif not (run/'evaluation_complete.json').exists():
        lines += ['', '尚未完成正式开发评价。来源通过不代表跨家庭预测效果已改善。']
    else:
        complete=read_json(run/'evaluation_complete.json'); gate.check(run/'evaluation',complete['artifacts'])
        development=read_json(run/'evaluation/summary.json'); result['development']=development
        metrics=read_json(run/'evaluation/metrics.json')['rows']
        lines += ['', '## 工程开发结果', '',
            f"唯一来源预选保持 **{development['selected_before_development']}**；正式验收：{'全部通过' if development['development_acceptance_passed'] else '未全部通过'}。以下候选对照仅作诊断，不改选。", '',
            '|模型|seed|TP|FP|FN|TN|P|R|F1|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in metrics: lines.append(f"|{r['arm']}|{r['seed']}|{r['tp']}|{r['fp']}|{r['fn']}|{r['tn']}|{r['precision']:.6f}|{r['recall']:.6f}|{r['f1']:.6f}|")
        lines += ['', '|正式门槛|结果|','|---|---|']
        gates=dict(mean_P_at_least_008='meanP≥0.08',each_R_at_least_085='每seed R≥0.85',mean_F1_gain_at_least_002='meanF1比原W50至少+0.02（1e-12容差）',
            at_least_two_seed_P_F1_up='至少2seed P/F1同升',each_F1_drop_at_most_002='每seed F1降≤0.02',mean_FP_reduction_at_least_30_percent='meanFP至少下降30%',
            no_new_all_on_or_off='无全开/全关',support_137523_exact='137523有效点及原mask exact')
        for k,v in development['acceptance_gates'].items(): lines.append(f"|{gates[k]}|{'通过' if v else '未通过'}|")
        pairs=read_json(run/'evaluation/paired_analysis.json')['paired']
        lines += ['', '## 增强相对control与原W50的配对变化', '',
            '|seed|参照|移除FP|新增FP|恢复TP|新增FN|ΔP|ΔR|ΔF1|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in pairs: lines.append(f"|{r['seed']}|{r['reference']}|{r['removed_fp']}|{r['added_fp']}|{r['restored_tp']}|{r['new_fn']}|{r['delta_P']:+.6f}|{r['delta_R']:+.6f}|{r['delta_F1']:+.6f}|")
        lines += ['', '连续片段详见evaluation/fp_segments.csv，包含原参照FP、增强FP、移除FP和新增FP。未知标签/无效mask或时间断裂均分段；不将片段当独立运行周期，也不用于调参或改选。', '',
            '原A全部raw/corrected、其他电器原始状态、各组有效支持及普通推理回读已封存exact检查。独立verify会逐点标量重数TP/FP/FN/TN、重算P/R/F1、来源资格及全部八门槛，并与metrics/summary和配对片段表核对；不会重新读取父真值缓存。']
    lines += ['', 'House17仅重复工程开发，不称独立泛化。最终8/9/20隔离，UI默认未切换，普通推理只用总表。']
    dest.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result['report']=str(dest)
    return result
