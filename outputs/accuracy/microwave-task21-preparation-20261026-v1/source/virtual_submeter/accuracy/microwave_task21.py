"""Fixed negative-fragment replacement, isolated from the original batch RNG.

The preparation route never calls configure/fit_one and forbids optimizers and
autograd. The training route accepts only a new receipt bound to registration.
"""
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch

from ..data.common import DataError, read_json, sha256, inventory
from ..models.center_mask_tcn import CenterMaskTCN
from . import microwave_source5_loss as loss
from . import microwave_recall_guard95 as guard95
from . import microwave_event_gate as gate
from .microwave_sampling import BatchStream

ROOT = loss.ROOT
T10 = ROOT/'outputs/accuracy/microwave-source5-temporal-20261025-v2'
T12 = ROOT/'outputs/accuracy/microwave-source5-loss-preparation-20261025-v4'
T14 = ROOT/'outputs/accuracy/microwave-recall-guard95-preparation-20261025-v2'
T19 = ROOT/'outputs/accuracy/microwave-fit-generalization-diagnostic-task19-20261026-v4'
CONFIG = 'configs/microwave_task21_20261026.disabled.json'
PROTOCOL = 'docs/MICROWAVE_TASK21_PROPOSAL_20261026.md'
SEEDS = [42, 1019, 2026]
N = 336704
require = gate.require


def fresh(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def load(path, fields=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if fields is None else fields)}


def snapshot(paths):
    return {str(Path(p).resolve()): dict(sha256=sha256(Path(p)), size=Path(p).stat().st_size,
            mtime_ns=Path(p).stat().st_mtime_ns) for p in paths}


@contextlib.contextmanager
def isolation(run, allowed, *, readonly=False, training=False):
    """Exact workspace file allowlist, including JSON and local output reads.

    Runtime package reads are allowed; protected experiment data never inherit
    a directory-wide permission. A verify/report cannot write even inside run.
    """
    run = Path(run).resolve()
    allowed = {Path(p).resolve() for p in allowed}
    state = dict(active=True, reads=set(), forbidden=[], written=set())
    runtime = Path(sys.prefix).resolve()
    def audit(event, args):
        if not state['active']: return
        if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.getaddrinfo'):
            raise DataError('Task21 network/subprocess forbidden')
        if event != 'open' or isinstance(args[0], int): return
        p = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing:
            if readonly or not p.is_relative_to(run):
                raise DataError('Task21 write forbidden: '+str(p))
            state['written'].add(p)
            return
        forbidden_house = re.search(r'(?:house|refit)[_-](?:8|9|17|20)(?:[_/\\.]|$)', str(p).lower())
        protected=p.suffix.lower() in ('.npz','.npy','.pt','.csv','.parquet','.pkl','.h5','.hdf5')
        forbidden = (p.is_relative_to(ROOT) or protected) and not p.is_relative_to(runtime) and p not in allowed and p not in state['written']
        if forbidden_house and p not in allowed: forbidden = True
        if forbidden:
            state['forbidden'].append(str(p))
            raise DataError('Task21 read outside exact whitelist: '+str(p))
        if p in allowed: state['reads'].add(str(p))
    def no_fit(*a, **k): raise DataError('Task21 preparation: 0fit/0update; optimizer/autograd forbidden')
    sys.addaudithook(audit)
    with contextlib.ExitStack() as stack:
        if not training:
            for obj, name in ((torch.optim.Optimizer,'__init__'), (torch.optim.Optimizer,'add_param_group'),
                              (torch.Tensor,'backward'), (torch.autograd,'backward'), (torch.autograd,'grad')):
                stack.enter_context(patch.object(obj, name, no_fit))
            stack.enter_context(torch.no_grad())
        try: yield state
        finally: state['active'] = False


def input_paths():
    # Enumerate filenames, never recurse through an old experiment directory.
    paths = [T10/n for n in ('train.npz','plan.json','normalization.json','registration.json','source5_inputs.json')]
    paths += [T10/f'validation/block_{b}.npz' for b in (4,5)]
    paths += [T10/f'{kind}_{s}.{ext}' for s in SEEDS for kind,ext in (('sampling','npz'),('dry_exposure','npz'),('initial','pt'))]
    paths += [T12/'final_delivery_manifest.json', T14/'final_delivery_manifest.json',
              T14/'source_selection.json', T14/'source_identity_replay.json', T14/'identity.json',
              T19/'delivery_manifest.json', T19/'data_identity.json', T19/'identities/train.npz']
    for s in SEEDS:
        paths += [T12/f'models/unit_BCE/seed_{s}/{n}' for n in ('model.json','weights.pt','initial.pt')]
        paths += [T19/f'predictions/control_{s}_train.{ext}' for ext in ('npz','json')]
    paths += [ROOT/CONFIG, ROOT/PROTOCOL, ROOT/'scripts/run_microwave_task21.py',
              ROOT/'tests/test_microwave_task21.py']
    paths += sorted((ROOT/'src/virtual_submeter').rglob('*.py'))
    # Source raw paths are taken only from Task10's source-specific identity.
    paths += [Path(v['path']) for v in read_json(T10/'source5_inputs.json').values()]
    return sorted(set(p.resolve() for p in paths))


def validate_config(c):
    require(c == read_json(ROOT/CONFIG) and c['authorized'] is False and c['preparation_only'] is True,
            'Fixed disabled configuration changed')
    require(c['schema']=='microwave_task21_hard_negative_v1' and c['seeds']==SEEDS
            and c['maximum_new_fits']==3 and c['maximum_updates']==10584
            and c['updates_per_seed']==3528 and c['hard_negative_per_batch']==120
            and c['ordinary_negative_per_batch']==120 and c['positive_per_batch']==16
            and c['batch_size']==256 and c['subsegment_centers']==31
            and c['interval_seconds']==8 and c['numpy_version']==np.__version__
            and c['rng_namespace']=='task21-hard-negative-v1|' and c['bit_generator']=='PCG64', 'Task21 recipe/RNG changed')


def validate_pool(d):
    require(len(d['y'])==N and int((d['y']>=200).sum())==2275, 'Original training support differs')
    require(np.isfinite(d['y']).all() and (d['y']>=0).all(), 'Missing/invalid labels cannot enter sampling')
    require(d['x'].shape==(N,127) and np.array_equal(np.isfinite(d['x']),d['mask']), 'Window/mask differs')
    require(set(d['house'].tolist())=={0,1,2} and (d['t']%8==0).all(), 'Training house/grid differs')
    d['source_role'] = np.where(d['house']==2, 'house5_prefix', 'original14d')
    keys = np.rec.fromarrays([d['house'],d['t']], names='house,t')
    require(len(np.unique(keys))==N, 'Duplicate training identities')
    for h in (0,1,2): require((np.diff(d['t'][d['house']==h])>0).all(), 'Training time order differs')
    return d


def cache_prefix(seed, d):
    """Decode only permitted original-prefix fields; bind full archive hash.

    npz is compressed: decoding probability/row_index decompresses a full member,
    but only [:N] values participate in validation, extraction or any computation.
    No expanded input, model, report, or expanded-cache file is opened.
    """
    pth = T19/f'predictions/control_{seed}_train.npz'
    meta = read_json(T12/f'models/unit_BCE/seed_{seed}/model.json')
    with np.load(pth, allow_pickle=False) as z:
        require(set(z.files)=={'probability','row_index','binding_json'}, 'Unexpected cache fields')
        binding = json.loads(str(z['binding_json']))
        probability = z['probability'][:N].copy()
        rows = z['row_index'][:N].copy()
    side = read_json(pth.with_suffix('.json'))
    require({k:side[k] for k in binding}==binding and side['prediction_sha256']==sha256(pth), 'Probability sidecar binding differs')
    require(binding['family']=='control' and binding['dataset']=='train' and binding['seed']==seed,
            'Wrong seed/family/dataset cache')
    require(binding['model_json_sha256']==sha256(T12/f'models/unit_BCE/seed_{seed}/model.json')
            and binding['weights_sha256']==sha256(T12/f'models/unit_BCE/seed_{seed}/weights.pt')
            and binding['identity_sha256']==sha256(T19/'identities/train.npz')
            and binding['input_sha256']==read_json(T19/'data_identity.json')['train']['file_sha256']
            and binding['threshold']==meta['decision_threshold_probability'], 'Cache model/identity/threshold differs')
    require(np.array_equal(rows,np.arange(N)) and probability.shape==(N,)
            and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(), 'Probability row mapping/range differs')
    with np.load(T19/'identities/train.npz', allow_pickle=False) as z:
        for k in ('t','house','event','raw_index'):
            np.testing.assert_array_equal(z[k][:N],d[k])
        np.testing.assert_array_equal(z['positive'][:N],d['y']>=200)
        np.testing.assert_array_equal(z['row_index'][:N],np.arange(N))
        # Task19's original pool used this literal source role for all three houses.
        roles = z['source_role'][:N]
        require(set(roles.tolist())=={'original14d'}, 'Task19 original-prefix role differs')
    return probability, binding


def fragments(d, hard):
    require(hard.dtype==np.bool_ and hard.shape==d['y'].shape, 'Hard mask shape/dtype differs')
    require(np.all(np.isfinite(d['y'][hard]) & (d['y'][hard]<200)), 'Hard pool includes missing/on labels')
    out, current = [], []
    ids = np.flatnonzero(hard)
    order = np.lexsort((d['t'][ids], d['source_role'][ids], d['house'][ids]))
    for i in ids[order].tolist():
        if current:
            j = current[-1]
            same = (d['house'][i]==d['house'][j] and d['source_role'][i]==d['source_role'][j]
                    and d['t'][i]//86400==d['t'][j]//86400 and d['t'][i]-d['t'][j]==8)
            if not same or len(current)==31:
                out.append(current); current=[]
        current.append(i)
    if current: out.append(current)
    return [np.asarray(v,np.int64) for v in out]


def hard_rng(seed):
    value = int.from_bytes(hashlib.sha256(('task21-hard-negative-v1|'+str(seed)).encode()).digest()[:8], 'big')
    rng = np.random.default_rng(value)
    require(type(rng.bit_generator).__name__=='PCG64', 'Unexpected bit generator')
    return rng


class HardStream:
    """Sorted pools, lazy permutations, pop from end; one dedicated RNG per seed.

    House → UTC date → subsegment → center. Queues persist across all batches;
    subsegment pool order is the canonical fragments() order. Every refill calls
    permutation(len(pool)), including singleton pools. There is no batch shuffle.
    """
    def __init__(self, d, parts, seed):
        self.parts, self.rng, self.queues, self.groups = parts, hard_rng(seed), {}, {}
        for k, ix in enumerate(parts):
            h, day = int(d['house'][ix[0]]), int(d['t'][ix[0]]//86400)
            self.groups.setdefault(h,{}).setdefault(day,[]).append(k)

    def pick(self, key, pool):
        if not self.queues.get(key): self.queues[key] = self.rng.permutation(len(pool)).tolist()
        return pool[self.queues[key].pop()]

    def next(self, n):
        require(bool(self.groups), 'Empty hard pool: no new fit, retain frozen control')
        out = []
        for _ in range(n):
            h = self.pick(('house',), sorted(self.groups))
            day = self.pick(('date',h), sorted(self.groups[h]))
            part = self.pick(('part',h,day), self.groups[h][day])
            out.append(self.pick(('center',part), self.parts[part]))
        return np.asarray(out,np.int64)


def replace_batches(original, y, stream):
    require(original.ndim==2 and original.shape[1]==256 and original.dtype.kind in 'iu'
            and np.all((original>=0)&(original<len(y))) and np.isfinite(y).all(), 'Invalid original batches/labels')
    changed, slots, additions = original.copy(), [], []
    for b, row in enumerate(original):
        neg = np.flatnonzero(y[row]<200)
        require(len(neg)==240 and int((y[row]>=200).sum())==16, 'Original batch quota differs')
        selected = neg[120:]
        new = stream.next(120)
        require(np.all(np.isfinite(y[new]) & (y[new]<200)), 'Replacement includes missing/on labels')
        changed[b, selected] = new
        slots.append(selected); additions.append(new)
    return changed, np.asarray(slots,np.int64), np.asarray(additions,np.int64)


def exposure(d, parts, original, changed, slots, additions):
    total = np.bincount(changed.ravel(),minlength=len(d['y']))
    hard = np.bincount(additions.ravel(),minlength=len(d['y']))
    retained = original.copy()
    keep = np.ones(original.shape,bool); keep[np.arange(len(original))[:,None],slots]=False
    ordinary = np.bincount(retained[keep & (d['y'][original]<200)],minlength=len(d['y']))
    positive = np.bincount(retained[keep & (d['y'][original]>=200)],minlength=len(d['y']))
    expected = np.zeros(len(d['y']),np.float64)
    groups = {}
    for part in parts:
        h,day = int(d['house'][part[0]]),int(d['t'][part[0]]//86400)
        groups.setdefault(h,{}).setdefault(day,[]).append(part)
    for h, days in groups.items():
        for day, ps in days.items():
            for part in ps: expected[part] = additions.size/(len(groups)*len(days)*len(ps)*len(part))
    return dict(total=total,hard=hard,ordinary=ordinary,positive=positive,expected_hard=expected)


def exposure_groups(d, parts, e):
    rows = []
    groups = [('house',str(h),np.flatnonzero(d['house']==h)) for h in sorted(set(d['house'].tolist()))]
    groups += [('date',f'{h}|{day}',np.flatnonzero((d['house']==h)&(d['t']//86400==day)))
               for h in sorted(set(d['house'].tolist())) for day in sorted(set((d['t'][d['house']==h]//86400).tolist()))]
    groups += [('subsegment',str(k),ix) for k,ix in enumerate(parts)]
    for level,key,ix in groups:
        rows.append(dict(level=level,key=key,n=len(ix),actual_total=int(e['total'][ix].sum()),
                         actual_hard=int(e['hard'][ix].sum()),expected_hard=float(e['expected_hard'][ix].sum()),
                         ordinary=int(e['ordinary'][ix].sum()),positive=int(e['positive'][ix].sum())))
    return rows


def csv_write(path, rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def bind_old_inputs(before):
    """Validate selected source files against old seals; no mixed score reports."""
    checks = [(T10,read_json(T10/'registration.json')['artifacts']),
              (T12,read_json(T12/'final_delivery_manifest.json')['artifacts']),
              (T14,read_json(T14/'final_delivery_manifest.json')['artifacts']),
              (T19,read_json(T19/'delivery_manifest.json')['artifacts'])]
    for root, manifest in checks:
        for p,item in before.items():
            path = Path(p)
            if path.is_relative_to(root):
                rel = path.relative_to(root).as_posix()
                if rel in manifest:
                    expected=manifest[rel]['sha256'] if isinstance(manifest[rel],dict) else manifest[rel]
                    require(expected==item['sha256'], 'Old seal differs: '+rel)
    for v in read_json(T10/'source5_inputs.json').values():
        require(before[str(Path(v['path']).resolve())]['sha256']==v['sha256'], 'Raw source binding changed')


def source_control():
    s = read_json(T14/'source_selection.json')
    require(s['source_only'] and s['selected']=='guard95' and s['guard95_eligible'], 'Frozen control not source-qualified')
    require([r['seed'] for r in s['recipes']]==SEEDS, 'Control source matrix incomplete')
    for r in s['recipes']:
        require(r['feasible'] and [v['block'] for v in r['blocks']]==['4','5'], 'Control block matrix incomplete')
        for b in r['blocks']:
            n = b['tp']+b['fp']+b['fn']+b['tn']
            require(n==b['n'] and b['recall']==b['tp']/(b['tp']+b['fn'])
                    and b['precision']==b['tp']/(b['tp']+b['fp'])
                    and b['f1']==2*b['tp']/(2*b['tp']+b['fp']+b['fn'])
                    and b['recall']>=.95 and not b['degenerate'], 'Control scalar metrics differ')
    return s


def prepare(run):
    run = Path(run).resolve()
    require(not run.exists(), 'New preparation directory required; completed directories cannot be overwritten')
    paths = input_paths()
    with isolation(run, paths) as access:
        before = snapshot(paths)
        bind_old_inputs(before)
        c = read_json(ROOT/CONFIG); validate_config(c)
        d = validate_pool(load(T10/'train.npz'))
        p = read_json(T10/'plan.json')
        require([p[k] for k in ('checkpoints','steps_per_checkpoint','batch_size','positive_per_batch','uniform_negative_per_batch','learning_rate')]
                ==[12,294,256,16,240,.001], 'Original schedule differs')
        vals = {b:load(T10/f'validation/block_{b}.npz',['y','t','house','block']) for b in (4,5)}
        require([sum(len(v['y']) for v in vals.values()),sum(int((v['y']>=200).sum()) for v in vals.values())]==[34351,201], 'Source support differs')
        for b,v in vals.items():
            require(np.isfinite(v['y']).all() and (v['house']==2).all() and (v['block']==b).all(), 'Source identity differs')
            require(not set(v['t'].tolist()).intersection(d['t'][d['house']==2].tolist()), 'Train/validation center leakage')
        control = source_control()
        run.mkdir(parents=True,exist_ok=False)
        (run/'source').mkdir()
        # Snapshot only source code, not a copy of any old experiment framework/output.
        for path in paths:
            if path.is_relative_to(ROOT/'src'):
                target=run/'source'/path.relative_to(ROOT/'src')
                target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
        for src,dst in ((ROOT/'scripts/run_microwave_task21.py',run/'run_microwave_task21.py'),
                        (ROOT/'tests/test_microwave_task21.py',run/'test_microwave_task21.py'),
                        (ROOT/PROTOCOL,run/'protocol.md')): shutil.copyfile(src,dst)
        fresh(run/'workspace_identity.json',dict(root=str(ROOT)))
        fresh(run/'config.json',c); fresh(run/'parent_training_plan.json',p)
        fresh(run/'source_control.json',control)
        fresh(run/'whitelist.json',dict(files=[str(v) for v in paths],development_and_final_reads=0,
              expanded_data_reads=0,cache_prefix_rows=N,full_cache_file_hash_bound=True))
        fresh(run/'inputs_before.json',before)
        fresh(run/'pool_identity.json',dict(fields={k:gate.hash_array(v) for k,v in d.items()},
              train_sha256=sha256(T10/'train.npz'),normalization_sha256=sha256(T10/'normalization.json'),
              validation={str(b):{k:gate.hash_array(a) for k,a in v.items()} for b,v in vals.items()}))
        proof = {}
        for seed in SEEDS:
            old = load(T10/f'sampling_{seed}.npz',['indices'])['indices']
            require(old.shape==(3528,256), 'Original matrix incomplete')
            stream = BatchStream(d,'event',seed,p)
            for row in old: np.testing.assert_array_equal(row,stream.next())
            np.testing.assert_array_equal(np.bincount(old.ravel(),minlength=N),load(T10/f'dry_exposure_{seed}.npz',['count'])['count'])
            # Deliberately construct only a network for identity, never configure (which creates Adam).
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = CenterMaskTCN()
                initial = torch.load(T10/f'initial_{seed}.pt',map_location='cpu',weights_only=True)
                unit = torch.load(T12/f'models/unit_BCE/seed_{seed}/initial.pt',map_location='cpu',weights_only=True)
                require(model.state_dict().keys()==initial.keys()==unit.keys(), 'Initialization key mismatch')
                require(all(torch.equal(v,initial[k]) and torch.equal(v,unit[k]) for k,v in model.state_dict().items()), 'Initialization identity differs')
                del model
            probability,binding=cache_prefix(seed,d)
            hard = np.isfinite(d['y']) & (d['y']<200) & (probability>=binding['threshold'])
            counts = [int((hard & (d['house']==h)).sum()) for h in (0,1,2)]
            require(counts==c['expected_hard_by_house'][str(seed)], 'Hard count differs; investigate without tuning threshold')
            parts = fragments(d,hard)
            if not parts:
                proof[str(seed)] = dict(fallback='frozen_control_no_new_fit',hard_counts=counts)
                continue
            changed,slots,added=replace_batches(old,d['y'],HardStream(d,parts,seed))
            center_part=np.full(N,-1,np.int64)
            for k,ix in enumerate(parts): center_part[ix]=k
            ids=np.flatnonzero(hard)
            np.savez_compressed(run/f'hard_windows_{seed}.npz',row_index=ids,probability=probability[ids],
                y=d['y'][ids],t=d['t'][ids],house=d['house'][ids],source_role=d['source_role'][ids],
                utc_day=d['t'][ids]//86400,subsegment=center_part[ids],raw_index=d['raw_index'][ids])
            csv_write(run/f'fragments_{seed}.csv',[dict(subsegment=k,house=int(d['house'][ix[0]]),
                source_role=str(d['source_role'][ix[0]]),utc_day=int(d['t'][ix[0]]//86400),
                start=int(d['t'][ix[0]]),end=int(d['t'][ix[-1]]),n=len(ix)) for k,ix in enumerate(parts)])
            np.savez_compressed(run/f'batches_{seed}.npz',indices=changed,replacement_slots=slots,hard_indices=added)
            e=exposure(d,parts,old,changed,slots,added)
            np.savez_compressed(run/f'exposure_{seed}.npz',row_index=np.arange(N),house=d['house'],t=d['t'],source_role=d['source_role'],subsegment=center_part,**e)
            csv_write(run/f'exposure_groups_{seed}.csv',exposure_groups(d,parts,e))
            duplicate = [len(row)-len(np.unique(row)) for row in changed]
            shares=e['hard'][ids]/added.size
            proof[str(seed)]=dict(hard_counts=counts,hard_total=len(ids),subsegments=len(parts),binding=binding,
                original_batches_replayed=3528,positive_slots_exact=True,ordinary_negative_slots_exact=True,
                initialization_exact=True,initial_file_sha256=sha256(T10/f'initial_{seed}.pt'),
                initial_tensor_hashes={k:gate.hash_array(v.numpy()) for k,v in initial.items()},
                rng_seed=int.from_bytes(hashlib.sha256(('task21-hard-negative-v1|'+str(seed)).encode()).digest()[:8],'big'),
                inactive_houses=[h for h in (0,1,2) if not counts[h]],
                hard_presentations=int(added.size),ordinary_negative_presentations=3528*120,positive_presentations=3528*16,
                duplicate_presentations=sum(duplicate),duplicate_rate=sum(duplicate)/changed.size,
                batches_with_duplicates=sum(v>0 for v in duplicate),maximum_duplicates_in_batch=max(duplicate),
                hard_top_center_share=float(shares.max()),hard_hhi=float(np.square(shares).sum()),
                full_index_sha256=gate.hash_array(changed),hard_expected_sum=float(e['expected_hard'].sum()))
        fresh(run/'sampling_proof.json',proof)
        require(snapshot(paths)==before, 'Preparation mutated an input')
        fresh(run/'inputs_after.json',before)
        fresh(run/'prepare_access.json',dict(reads=sorted(access['reads']),forbidden=access['forbidden'],
            new_fits=0,optimizer_updates=0,network_construction='initialization identity only; forked RNG; no configure',
            optimizers_and_autograd_blocked=True,development_reads=0,final_reads=0))
        fresh(run/'registration.json',dict(schema=c['schema'],artifacts=inventory(run),authorized=False,
            proposed_new_fits=3,proposed_updates=10584,new_fits=0,optimizer_updates=0))
    return dict(status='prepared_pending_independent_verification_and_new_human_budget',new_fits=0,
                optimizer_updates=0,registration_sha256=sha256(run/'registration.json'))


def registered(run):
    reg=read_json(run/'registration.json')
    gate.check(run,reg['artifacts'])
    c=read_json(run/'config.json'); validate_config(c)
    before=read_json(run/'inputs_before.json')
    require(snapshot(before)==before, 'Registered input hash/mtime changed')
    return c


def verify(run):
    # Independent verifier does not call the production fragmenter/sampler/exposure.
    from .microwave_task21_review import review
    run=Path(run).resolve()
    result=review(run)
    if (run/'training_started.json').exists():
        from .microwave_task21_review import review_execution
        result['execution']=review_execution(run)
    return result


def report(run):
    result=verify(run)
    return dict(**result,report=(Path(run)/'PREPARATION_REPORT.zh.md').read_text(encoding='utf-8'))


def authorize(run,receipt):
    # FIRST operation in every execution route: no data/model/optimizer/output access.
    require(receipt is not None,'Task21 requires a NEW human budget bound to registration; 0 training started')
    a=read_json(receipt)
    require(a.get('authorized') is True and a.get('approval_origin')=='human'
            and a.get('experiment')=='microwave_task21_hard_negative_v1'
            and a.get('registration_sha256')==sha256(Path(run)/'registration.json')
            and a.get('additional_new_fits')==3 and a.get('maximum_updates')==10584
            and a.get('seeds')==SEEDS and a.get('human_authorization_quote','').strip()
            and a.get('human_authorization_reference','').strip(), 'Missing NEW hash-bound Task21 budget; old receipts/proposal are insufficient')
    # Preparation must have passed independent review and a sealed complete delivery.
    gate.check(run,read_json(Path(run)/'preparation_manifest.json')['artifacts'])
    require(read_json(Path(run)/'independent_review.json')['passed'], 'Independent preparation review required')
    return a


def execution_allowed(run):
    paths=list(read_json(run/'inputs_before.json'))
    paths += [str(run/k) for k in inventory(run)]
    return paths


def train(run,authorization=None):
    run=Path(run).resolve(); a=authorize(run,authorization)
    with isolation(run,execution_allowed(run),training=True):
        registered(run)
        require(not (run/'training_started.json').exists() and not (run/'models').exists(),'No restart/retry permitted')
        proof=read_json(run/'sampling_proof.json')
        active=[s for s in SEEDS if not proof[str(s)].get('fallback')]
        fresh(run/'training_started.json',dict(authorization=a,seeds=active,maximum_updates=len(active)*3528))
        d=load(T10/'train.npz');p=read_json(run/'parent_training_plan.json');meta=read_json(T10/'normalization.json')
        blocks={f'block_{b}':load(T10/f'validation/block_{b}.npz') for b in (4,5)}
        rows=[]
        for s in active:
            indices=load(run/f'batches_{s}.npz',['indices'])['indices']
            initial=torch.load(T10/f'initial_{s}.pt',weights_only=True,map_location='cpu')
            rows.append(loss.temporal.fit_one(run/f'models/hard_negative/seed_{s}',s,p,d,blocks,meta,indices,initial,
                sha256(run/'registration.json'),pos_weight=1.,recipe='unit_BCE',arm='hard_negative'))
        fresh(run/'training_complete.json',dict(rows=rows,new_fits=len(active),updates=len(active)*3528,
              control_retrains=0,artifacts=inventory(run/'models') if active else {}))
    return dict(new_fits=len(active),updates=len(active)*3528)


def source_select(run,authorization=None):
    run=Path(run).resolve(); authorize(run,authorization)
    with isolation(run,execution_allowed(run)):
        registered(run)
        complete=read_json(run/'training_complete.json');gate.check(run/'models',complete['artifacts'])
        require(complete['new_fits']==3 and complete['updates']==10584,'No full candidate matrix; retain control, no development')
        raw=gate.csv_rows(Path(read_json(T10/'source5_inputs.json')['aggregate']['path']),'refit_5')
        times=np.array([loss.utc(r['timestamp']) for r in raw])
        recipes,rows=[],[]
        for s in SEEDS:
            model=run/f'models/hard_negative/seed_{s}';meta=read_json(model/'model.json');scores={}
            for b,lo,hi in loss.temporal.f.VALIDATION:
                v=load(T10/f'validation/block_{b}.npz',['t','y'])
                t,p,_=gate.infer(model,[raw[j] for j in np.flatnonzero((times>=lo)&(times<hi))])
                ix=gate.align(t,v['t']); saved=load(model/f'epoch_{meta["selected"]["epoch"]}_block_{b}.npz')
                np.testing.assert_array_equal(v['t'],saved['t']);np.testing.assert_allclose(p[ix],saved['scores'],rtol=1e-5,atol=1e-7)
                scores[str(b)]=dict(t=t,p=p,h=gate.local_max(t,p),ix=ix,y=v['y']>=200)
            low,_=gate.select_threshold({k:dict(values=v['p'][v['ix']],y=v['y']) for k,v in scores.items()},
                np.unique(np.concatenate([v['p'][np.isfinite(v['p'])] for v in scores.values()])),.95)
            high=guard95.calibrate95(scores,low) if low is not None else None
            metrics=[dict(block=k,**gate.confusion(gate.gate_state(v['p'],v['h'],low,high)[v['ix']],v['y'])) for k,v in scores.items()] if high is not None else []
            recipe=dict(seed=s,L=low,H=high,feasible=high is not None,blocks=metrics,
                schema='event_gate_recipe_v1',eligible=high is not None,radius_samples=15,interval_seconds=8,
                parent=str(model),parent_hashes={n:sha256(model/n) for n in ('model.json','weights.pt')},
                algorithm_sha256=sha256(Path(gate.__file__)),score_semantics='uncalibrated sigmoid state score')
            if not (run/'source_results').exists():(run/'source_results').mkdir()
            fresh(run/f'source_results/recipe_{s}.json',recipe)
            for b,v in scores.items():np.savez_compressed(run/f'source_results/scores_{s}_{b}.npz',**v)
            recipes.append(recipe)
            if metrics:rows.append(dict(seed=s,**gate.mean_rows(metrics),degenerate=any(v['degenerate'] for v in metrics),source_gate_passed=all(v['recall']>=.95 for v in metrics)))
        eligible=all(v['source_gate_passed'] for v in complete['rows']) and loss.qualified(read_json(run/'source_control.json')['rows']['guard95'],rows)
        fresh(run/'source_selection.json',dict(source_only=True,selected='hard_negative' if eligible else 'control_guard95',
            eligible=eligible,recipes=recipes,rows=rows,training_sha256=sha256(run/'training_complete.json')))
        fresh(run/'source_complete.json',dict(source_sha256=sha256(run/'source_selection.json'),registration_sha256=sha256(run/'registration.json'),artifacts=inventory(run/'source_results')))
    return dict(source_eligible=eligible,development_reads=0)


def source_guard(run):
    registered(run)
    s=read_json(run/'source_selection.json')
    seal=read_json(run/'source_complete.json')
    require(seal['source_sha256']==sha256(run/'source_selection.json') and seal['registration_sha256']==sha256(run/'registration.json'), 'Source seal differs')
    gate.check(run/'source_results',seal['artifacts'])
    require(s['training_sha256']==sha256(run/'training_complete.json') and s['source_only']
            and s['eligible'] and s['selected']=='hard_negative'
            and [r['seed'] for r in s['recipes']]==SEEDS and all(r['feasible'] for r in s['recipes'])
            and loss.qualified(read_json(run/'source_control.json')['rows']['guard95'],s['rows']),
            'Source rejected/incomplete: 0 development reads')
    require(all(v['source_gate_passed'] for v in read_json(run/'training_complete.json')['rows']),'Original85 checkpoint guard failed')
    gate.check(run/'models',read_json(run/'training_complete.json')['artifacts'])
    return s


def freeze(run,authorization=None):
    run=Path(run).resolve(); authorize(run,authorization)
    # Source guard precedes even constructing/hash-checking development allowlist.
    with isolation(run,execution_allowed(run)):s=source_guard(run)
    identity=read_json(T14/'identity.json')
    dev=identity['development_unlabeled'];base=Path(identity['baseline'])
    with isolation(run,execution_allowed(run)+list(dev)):
        require(not (run/'predictions').exists(),'One-shot freeze; no retry')
        loss.check_paths(dev)
        raw=gate.csv_rows(next(Path(p) for p in dev if p.endswith('aggregate.csv')),'refit_17')
        (run/'predictions').mkdir()
        checks=[]
        for seed in SEEDS:
            a=load(base/f'reference/predictions/seed_{seed}/{gate.PARTITION}',['stamps','raw','corrected'])
            w=load(base/f'predictions/W50/seed_{seed}/{gate.PARTITION}',['stamps','raw'])
            np.testing.assert_array_equal(a['stamps'],w['stamps'])
            states=np.where(np.isfinite(a['corrected']),a['corrected']>=np.array([2000,200,20]),-1).astype(np.int8)
            np.savez_compressed(run/f'predictions/A_{seed}.npz',t=a['stamps'],raw=a['raw'],corrected=a['corrected'],states=states)
            np.savez_compressed(run/f'predictions/W50_{seed}.npz',t=w['stamps'],raw=w['raw'])
            recipe=next(r for r in s['recipes'] if r['seed']==seed)
            t,p,_=gate.infer(run/f'models/hard_negative/seed_{seed}',raw)
            # Public event-gate API loads the standalone recipe and validates model/hash bindings.
            ordinary=gate.ordinary(run/f'source_results/recipe_{seed}.json',raw)
            np.testing.assert_array_equal(t,ordinary['t']);np.testing.assert_array_equal(p,ordinary['scores'])
            np.testing.assert_array_equal(t,a['stamps']);np.testing.assert_array_equal(np.isfinite(p),np.isfinite(a['raw'][:,1]))
            st=gate.gate_state(p,gate.local_max(t,p),recipe['L'],recipe['H']);other=states.copy();other[:,1]=st
            np.testing.assert_array_equal(st,ordinary['state'])
            np.savez_compressed(run/f'predictions/hard_negative_{seed}.npz',t=t,scores=p,state=st,raw=a['raw'],corrected=a['corrected'],other_states=other)
            checks.append(dict(seed=seed,ordinary_exact=True,public_api='microwave_event_gate.ordinary',
                recipe_sha256=sha256(run/f'source_results/recipe_{seed}.json'),
                raw_A_exact=True,other_states_A_exact=True,support_exact=True))
        fresh(run/'inference_checks.json',dict(checks=checks))
        fresh(run/'freeze.json',dict(matrix=[[a,s] for a in ('A','W50','hard_negative') for s in SEEDS],
            registration_sha256=sha256(run/'registration.json'),source_sha256=sha256(run/'source_selection.json'),
            artifacts={'predictions/'+k:v for k,v in inventory(run/'predictions').items()}|{'inference_checks.json':sha256(run/'inference_checks.json')},truth_opened=False))
    return dict(frozen=True,truth_reads=0)


def evaluation_guard(run):
    source_guard(run)
    f=read_json(run/'freeze.json')
    expected={f'predictions/{a}_{s}.npz' for a in ('A','W50','hard_negative') for s in SEEDS}|{'inference_checks.json'}
    require(set(f['artifacts'])==expected and f['matrix']==[[a,s] for a in ('A','W50','hard_negative') for s in SEEDS]
            and f['truth_opened'] is False and f['registration_sha256']==sha256(run/'registration.json')
            and f['source_sha256']==sha256(run/'source_selection.json'), 'Incomplete/unbound frozen development matrix')
    gate.check(run,f['artifacts'])
    checks=read_json(run/'inference_checks.json')['checks']
    require([v['seed'] for v in checks]==SEEDS and all(v['ordinary_exact'] and v['raw_A_exact']
        and v['other_states_A_exact'] and v['support_exact'] and v['public_api']=='microwave_event_gate.ordinary'
        and v['recipe_sha256']==sha256(run/f'source_results/recipe_{v["seed"]}.json') for v in checks),'Public inference verification matrix incomplete')


def evaluate(run,authorization=None):
    run=Path(run).resolve();authorize(run,authorization)
    with isolation(run,execution_allowed(run)):evaluation_guard(run)
    identity=read_json(T14/'identity.json');truth=identity['truth_cache']
    with isolation(run,execution_allowed(run)+[truth['path']]):
        require(not (run/'evaluation').exists(),'One development acceptance only')
        (run/'evaluation').mkdir();fresh(run/'evaluation/started.json',dict(freeze_sha256=sha256(run/'freeze.json')))
        require(sha256(Path(truth['path']))==truth['sha256'],'Truth cache binding changed')
        z=load(truth['path'],['t','y']);rows=[];supports=[]
        np.savez_compressed(run/'evaluation/truth_once.npz',**z)
        for seed in SEEDS:
            a=load(run/f'predictions/A_{seed}.npz');w=load(run/f'predictions/W50_{seed}.npz');c=load(run/f'predictions/hard_negative_{seed}.npz')
            for v in (a,w,c):np.testing.assert_array_equal(v['t'],z['t'])
            mask=np.isfinite(z['y'])&np.isfinite(w['raw'][:,1])
            np.testing.assert_array_equal(mask,np.isfinite(z['y'])&np.isfinite(a['raw'][:,1]))
            np.testing.assert_array_equal(np.isfinite(c['scores']),np.isfinite(a['raw'][:,1]))
            require(mask.sum()==137523,'Original effective support differs')
            np.testing.assert_array_equal(c['raw'],a['raw']);np.testing.assert_array_equal(c['corrected'],a['corrected'])
            np.testing.assert_array_equal(c['other_states'][:,[0,2]],a['states'][:,[0,2]])
            for arm,v in (('A',a),('W50',w),('hard_negative',c)):
                state=v['state'][mask] if arm=='hard_negative' else (v['raw'][mask,1]>=200).astype(np.int8)
                rows.append(dict(arm=arm,seed=seed,**gate.confusion(state,z['y'][mask]>=200)))
            supports.append(dict(seed=seed,n=int(mask.sum()),mask_sha256=gate.hash_array(mask)))
        metrics=dict(rows=rows,supports=supports)
        result=loss.summary(run,metrics)
        fresh(run/'evaluation/metrics.json',metrics);fresh(run/'evaluation/summary.json',result)
        fresh(run/'evaluation_complete.json',dict(freeze_sha256=sha256(run/'freeze.json'),artifacts=inventory(run/'evaluation')))
    return result
