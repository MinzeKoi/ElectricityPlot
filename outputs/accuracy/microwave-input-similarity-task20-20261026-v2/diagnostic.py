"""Task20 preregistered zero-fit input diagnostic. All writes are exclusive."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').is_file():
    ROOT = ROOT.parent
T18 = ROOT / 'outputs/accuracy/microwave-expanded-guard95-preparation-20261026-v3'
T19 = ROOT / 'outputs/accuracy/microwave-fit-generalization-diagnostic-task19-20261026-v4'
RUN = ROOT / 'outputs/accuracy/microwave-input-similarity-task20-20261026-v2'
SETS = ('train', 'block_4', 'block_5')
FAMILIES = ('control', 'expanded')
SEEDS = (42, 1019, 2026)
HOUSES = (2, 19, 5)
SOURCE = T18 / 'source/virtual_submeter/models/center_mask_tcn.py'

def require(ok, why):
    if not ok:
        raise ValueError(why)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def arrsha(a):
    return hashlib.sha256(str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes()).hexdigest()

def fresh(p, v):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def csvwrite(p, rows):
    require(bool(rows), 'Empty CSV '+str(p))
    with Path(p).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def data_path(s):
    return T18 / ('train.npz' if s == 'train' else 'validation/'+s+'.npz')

def whitelist():
    paths = [T19/n for n in ('delivery_manifest.json', 'data_identity.json', 'model_identity.json', 'pre_registration.json')]
    paths += [T19/'identities'/f'{s}.npz' for s in SETS]
    paths += [T19/'predictions'/f'{f}_{k}_{s}.{ext}' for f in FAMILIES for k in SEEDS for s in SETS for ext in ('json', 'npz')]
    paths += [data_path(s) for s in SETS]+[T18/'normalization.json', SOURCE]
    return paths

def snapshot(paths):
    return {str(p.resolve()): dict(sha256=sha(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in paths}

def isolate(readonly=False):
    allowed = {p.resolve() for p in whitelist()} | {Path(__file__).resolve()}
    def audit(event, a):
        if event == 'open' and not isinstance(a[0], int):
            p = Path(os.fsdecode(a[0])).resolve()
            writing = (isinstance(a[1], str) and any(c in a[1] for c in 'wax+')) or (isinstance(a[2], int) and a[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if p.is_relative_to(RUN):
                if writing and readonly:
                    raise PermissionError('Completed delivery is read-only')
            elif writing:
                raise PermissionError('Only fresh Task20 outputs are writable')
            elif p.is_relative_to(ROOT) and p not in allowed and not p.is_relative_to(Path(sys.prefix)):
                raise PermissionError('Outside exact input whitelist: '+str(p))
        elif event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):
            raise PermissionError('No network or subprocess')
    sys.addaudithook(audit)

def register():
    RUN.mkdir(exist_ok=False)
    isolate()
    cfg = dict(task='Task20_input_similarity', authorization=dict(source_chat='01a16d7b-478e-7780-a999-88abe71e8c94', source_call='call_611a3e30a1774a7f83103e302608c8e0', human_quote='允许新开并执行这项零训练诊断'),
        allowed_paths=[str(p.resolve()) for p in whitelist()], new_fits=0, optimizer_updates=0, supervised_probes=0, threshold_searches=0, checkpoint_searches=0, output_changes=0,
        core='mask[:,48:79].all; unchanged 127 channels', transform='frozen float32 (log1p(x)-input_log_mean)/input_log_std; missing power=0 and mask=0',
        stable_id='H{physical_house}|{source_role}|{UTC_unix_seconds}|raw{raw_index}; SHA256 UTF-8; tie ID lexicographic',
        reference='strict31 train only; control row<336704; expanded all615922; per family/house/role/label SHA order max256 positive or4096 negative; independent of prediction',
        query='per family/seed/dataset/house/role/UTCday/TP,FP,FN,TN; strict31; SHA order max32; preserve empty strata',
        eligibility='same household, different UTCday, absolute center gap>=1016 seconds; no fallback',
        distance='float64 direct RMS of frozen float32 transformed power; no centering/rescaling/filling; masks only for support and exact duplicate identity',
        common127='filter already frozen strict31 query/reference lists by mask.all; common31 and common127 identical candidates; no refill; also compute d127 on common31-selected pairs',
        duplicate='all strict31 allowed datasets jointly; canonical full127 float32 power +127 float32 mask; SHA256 then elementwise exact confirmation; report global and same-house conflict scope',
        nearest_tie='minimum RMS then stableID lexicographic', query_chunk=8, reference_chunk=256,
        limits=dict(max_queries=100000, max_pairs=600000, max_curves=16, curve_per_family_category=2, max_duplicate_members=651000),
        curves='family/category each first2 query stableID SHA order with available opposite label reference for errors/same label for correct; event/contiguous-off-fragment dedup across datasets; show original127 and masks; no score ordering',
        groups='family/seed/dataset/house/role/UTCday/category, plus role and category rollups; no outcome-based rule revision',
        clarifications=['No pre-existing AGENTS.md found in workspace or inspected ancestors/docs.', 'Only whitelist members of Task19 manifest are hashed; forbidden or unrelated artifacts are never traversed or hashed.', 'Model identity and frozen thresholds come from sealed Task19 bindings; no model weights or other epochs read.', 'Full127 comparison filters the frozen sample without resampling; zero eligible references yields explicit null.', 'Zero/zero distance ratio is null (undefined); nonzero/zero is null with denominator_zero status.', 'v1 stopped at input loading: validation NPZ has no source_role field. v2 assigns the fixed Task19 validation role before identity validation. No sampling or distance results existed; v1 remains untouched.'],
        exclusions='No House17 or final8/9/20 values/hash/shape; no raw data; no other scores/epochs',
        interpretation='added dates untrained only for control; House5 suffix participated in selection; not blind; windows/fragments not independent physical events')
    fresh(RUN/'config.json', cfg)
    with (RUN/'diagnostic.py').open('x', encoding='utf-8', newline='\n') as f:
        f.write(Path(__file__).read_text(encoding='utf-8'))
    before = snapshot(whitelist())
    fresh(RUN/'inputs_before.json', before)
    man = read(T19/'delivery_manifest.json')
    require(man['status']=='complete_fixed_diagnostic' and len(man['artifacts'])==56, 'Task19 manifest incomplete')
    require(man['artifacts']['DELIVERY_REPORT.zh.md']['sha256']=='fe5fd289e3129ef7eeb3f99c6e717d93d018aac7335483085edfd3e46a03d58c', 'Task19 report binding differs')
    for p in whitelist():
        if p.is_relative_to(T19) and p.name!='delivery_manifest.json':
            require(before[str(p.resolve())]==man['artifacts'][str(p.relative_to(T19))], 'Task19 manifest member mismatch '+str(p))
    fresh(RUN/'registration.json', dict(config_sha256=sha(RUN/'config.json'), source_sha256=sha(RUN/'diagnostic.py'), input_snapshot_sha256=sha(RUN/'inputs_before.json'), status='rules_frozen_before_data_analysis'))
    print('REGISTERED', flush=True)

def transformed(x, mask, norm):
    require(np.array_equal(np.isfinite(x), mask), 'Mask differs from finite input')
    require(not np.isinf(x).any() and np.all(x[mask]>=0), 'Invalid nonnegative power')
    z = np.zeros(x.shape, np.float32)
    z[mask] = (np.log1p(x[mask])-norm['input_log_mean'])/norm['input_log_std']
    z[z==0] = 0  # canonicalize signed zero for exact numeric equality hashing
    return z

def load_data():
    norm = read(T18/'normalization.json')
    ds = []
    di = read(T19/'data_identity.json')
    offset = 0
    for s in SETS:
        with np.load(data_path(s), allow_pickle=False) as f:
            keys = ('x','mask','y','t','house','event','raw_index') + (('source_role',) if s=='train' else ())
            d = {k:f[k] for k in keys}
        require(sha(data_path(s))==di[s]['file_sha256'], 'Data binding changed')
        require(d['x'].shape==(len(d['y']),127), 'Input shape')
        require(set(np.unique(d['house'])).issubset({0,1,2}) and np.all(d['t']%8==0), 'House/time grid mismatch')
        require(np.isfinite(d['y']).all() and np.all(d['y']>=0), 'Invalid target power')
        if s!='train':
            d['source_role'] = np.full(len(d['y']), 'house5_validation_'+s, dtype='U32')
        for k in d:
            require(arrsha(d[k])==di[s]['fields'][k], 'Original field hash '+s+'/'+k)
        with np.load(T19/'identities'/f'{s}.npz', allow_pickle=False) as f:
            for k in ('t','house','source_role','event','raw_index'):
                require(np.array_equal(f[k], d[k]), 'Identity mapping '+s+'/'+k)
            require(np.array_equal(f['row_index'],np.arange(len(d['y']))) and np.array_equal(f['positive'],d['y']>=200) and np.array_equal(f['strict_core31'],d['mask'][:,48:79].all(1)), 'Identity support differs')
        d['z'] = transformed(d['x'],d['mask'],norm)
        d['dataset'] = np.full(len(d['y']),s,dtype='U8')
        d['row'] = np.arange(len(d['y']))
        ds.append(d)
        offset += len(d['y'])
    d = {k:np.concatenate([v[k] for v in ds]) for k in ds[0]}
    del ds
    d['strict'] = d['mask'][:,48:79].all(1)
    d['full'] = d['mask'].all(1)
    d['day'] = d['t']//86400
    d['positive'] = d['y']>=200
    ids = [f'H{HOUSES[int(h)]}|{r}|{int(t)}|raw{int(raw)}' for h,r,t,raw in zip(d['house'],d['source_role'],d['t'],d['raw_index'])]
    require(len(set(ids))==len(ids), 'Duplicate stable window identity')
    d['id'] = np.array(ids)
    d['hash'] = np.array([hashlib.sha256(v.encode()).hexdigest() for v in ids])
    # Observable fragments use household/role/time continuity and label, not model score.
    order = np.lexsort((d['t'],d['source_role'],d['house']))
    boundary = np.r_[True,(np.diff(d['t'][order])!=8)|(d['house'][order][1:]!=d['house'][order][:-1])|(d['source_role'][order][1:]!=d['source_role'][order][:-1])|(d['positive'][order][1:]!=d['positive'][order][:-1])]
    d['fragment'] = np.empty(len(ids),np.int64)
    d['fragment'][order] = np.cumsum(boundary)
    return d

def load_probs(d):
    identity = read(T19/'model_identity.json')
    thresholds = read(T19/'pre_registration.json')['thresholds']
    probs = {}
    require(set(identity)=={f'{f}_{s}' for f in FAMILIES for s in SEEDS}, 'Incomplete model matrix')
    for f in FAMILIES:
        for seed in SEEDS:
            parts = []
            for s in SETS:
                pth = T19/'predictions'/f'{f}_{seed}_{s}.npz'
                side = read(pth.with_suffix('.json'))
                with np.load(pth, allow_pickle=False) as z:
                    require(set(z.files)=={'probability','row_index','binding_json'}, 'Cache schema')
                    p = z['probability']
                    b = json.loads(str(z['binding_json']))
                    require(np.array_equal(z['row_index'],np.arange(len(p))), 'Cache row order')
                require((b['family'],b['seed'],b['dataset'])==(f,seed,s), 'Cache identity')
                require(b['input_sha256']==sha(data_path(s)) and b['identity_sha256']==sha(T19/'identities'/f'{s}.npz'), 'Cache data binding')
                for k in ('threshold','model_json_sha256','weights_sha256'):
                    require(b[k]==identity[f'{f}_{seed}'][k], 'Frozen model identity')
                require(b['threshold']==thresholds[f'{f}_{seed}'], 'Threshold registration')
                require(side==b|dict(prediction_sha256=sha(pth),probability_sha256=arrsha(p)), 'Cache sidecar')
                require(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and len(p)==int((d['dataset']==s).sum()), 'Cache support')
                parts.append(p)
            probs[(f,seed)] = np.concatenate(parts)
    return probs, thresholds

def rank(d, ix):
    return sorted(map(int,ix), key=lambda i:(d['hash'][i],d['id'][i]))

def freeze_samples(d, probs, thresholds):
    support, refs, queries = [], [], []
    for f in FAMILIES:
        eligible = (d['dataset']=='train') & (d['row']<(336704 if f=='control' else 615922))
        for h in sorted(set(d['house'][eligible].tolist())):
            for role in sorted(set(d['source_role'][eligible & (d['house']==h)].tolist())):
                for label in (False,True):
                    ix = np.flatnonzero(eligible & (d['house']==h)&(d['source_role']==role)&(d['positive']==label))
                    strict = ix[d['strict'][ix]]
                    chosen = rank(d,strict)[:256 if label else 4096]
                    support.append(dict(kind='reference',family=f,seed='',dataset='train',house=HOUSES[h],role=role,day='',category=str(int(label)),original_n=len(ix),strict_n=len(strict),full_n=int(d['full'][strict].sum()),sample_n=len(chosen),sample_full_n=int(d['full'][chosen].sum())))
                    refs.extend(dict(family=f,index=i,stable_id=str(d['id'][i]),sha256=str(d['hash'][i]),label=int(label)) for i in chosen)
        for seed in SEEDS:
            p = probs[(f,seed)]
            pred = p>=thresholds[f'{f}_{seed}']
            cats = np.where(d['positive'],np.where(pred,'TP','FN'),np.where(pred,'FP','TN'))
            groups = defaultdict(list)
            for i in range(len(d['y'])):
                groups[(str(d['dataset'][i]),int(d['house'][i]),str(d['source_role'][i]),int(d['day'][i]))].append(i)
            for (dataset,h,role,day), ids in sorted(groups.items()):
                ids = np.array(ids)
                for cat in ('TP','FP','FN','TN'):
                    ix=ids[cats[ids]==cat]
                    strict=ix[d['strict'][ix]]
                    chosen=rank(d,strict)[:32]
                    support.append(dict(kind='query',family=f,seed=seed,dataset=dataset,house=HOUSES[h],role=role,day=day,category=cat,original_n=len(ix),strict_n=len(strict),full_n=int(d['full'][strict].sum()),sample_n=len(chosen),sample_full_n=int(d['full'][chosen].sum())))
                    queries.extend(dict(family=f,seed=seed,index=i,stable_id=str(d['id'][i]),sha256=str(d['hash'][i]),category=cat,probability=float(p[i]),threshold=thresholds[f'{f}_{seed}']) for i in chosen)
    require(len(queries)<=100000,'Query output ceiling')
    csvwrite(RUN/'support_and_coverage.csv',support)
    fresh(RUN/'reference_sample.json',refs)
    fresh(RUN/'query_sample.json',queries)
    fresh(RUN/'samples_frozen.json',dict(status='frozen_before_distances', support_sha256=sha(RUN/'support_and_coverage.csv'), reference_sha256=sha(RUN/'reference_sample.json'), query_sha256=sha(RUN/'query_sample.json'), queries=len(queries),references=len(refs)))
    print('SAMPLES_FROZEN',len(queries),len(refs),flush=True)
    return refs, queries

def duplicate_scan(d):
    buckets=defaultdict(list)
    for i in np.flatnonzero(d['strict']):
        z=np.stack((d['z'][i],d['mask'][i].astype(np.float32)))
        buckets[hashlib.sha256(z.tobytes()).digest()].append(int(i))
    conflicts=[]
    repeated=confirmed=conflict_windows=same_house_groups=cross_day_groups=0
    for key,ids in buckets.items():
        if len(ids)<2:
            continue
        # Hash is merely an index: validate every candidate against canonical channels.
        first=ids[0]
        for j in ids[1:]:
            require(np.array_equal(d['z'][first],d['z'][j]) and np.array_equal(d['mask'][first],d['mask'][j]), 'SHA collision requires explicit partition; refusing')
            confirmed+=1
        repeated+=1
        if len(set(d['positive'][ids].tolist()))<2:
            continue
        conflict_windows+=len(ids)
        within=any(len(set(d['positive'][[i for i in ids if d['house'][i]==h]].tolist()))>1 for h in set(d['house'][ids].tolist()))
        cross=len(set(d['day'][ids].tolist()))>1
        same_house_groups+=int(within)
        cross_day_groups+=int(cross)
        for i in ids:
            conflicts.append(dict(input_sha256=key.hex(),index=i,stable_id=str(d['id'][i]),house=HOUSES[int(d['house'][i])],role=str(d['source_role'][i]),dataset=str(d['dataset'][i]),row=int(d['row'][i]),utc=int(d['t'][i]),day=int(d['day'][i]),y_watts=float(d['y'][i]),positive=bool(d['positive'][i]),mask127=''.join(map(str,d['mask'][i].astype(int))),missing_count=int((~d['mask'][i]).sum()),group_n=len(ids),same_house_conflict=within,cross_date=cross))
    require(len(conflicts)<=651000,'Duplicate output ceiling')
    fresh(RUN/'exact_conflict_members.json',conflicts)
    summary=dict(strict_windows=int(d['strict'].sum()),unique_effective_inputs=len(buckets),repeated_groups=repeated,elementwise_confirmations=confirmed,conflict_groups=len(set(v['input_sha256'] for v in conflicts)),conflict_windows=conflict_windows,same_house_conflict_groups=same_house_groups,cross_date_conflict_groups=cross_day_groups,missing_pattern_distribution={str(k):sum(v['missing_count']==k for v in conflicts) for k in sorted(set(v['missing_count'] for v in conflicts))})
    fresh(RUN/'exact_duplicates.json',summary)
    print('DUPLICATES',summary,flush=True)

def eligible(d,q,refs):
    return (d['house'][refs]==d['house'][q]) & (d['day'][refs]!=d['day'][q]) & (np.abs(d['t'][refs]-d['t'][q])>=1016)

def nearest(d, qs, refs, common):
    """Bounded direct differences, never a full query/reference matrix."""
    refs=np.array(sorted(map(int,refs),key=lambda i:d['id'][i]),dtype=np.int64)
    if common:
        refs=refs[d['full'][refs]]
    out={}
    for start in range(0,len(qs),8):
        q=np.array(qs[start:start+8],dtype=np.int64)
        modes=('common31','common127') if common else ('strict31',)
        best={mode:np.full(len(q),np.inf) for mode in modes}
        bid={mode:np.full(len(q),-1,np.int64) for mode in modes}
        counts=np.zeros(len(q),np.int64)
        for rs in range(0,len(refs),256):
            r=refs[rs:rs+256]
            ok=(d['house'][q,None]==d['house'][r][None,:])&(d['day'][q,None]!=d['day'][r][None,:])&(np.abs(d['t'][q,None]-d['t'][r][None,:])>=1016)
            if common:
                ok &= d['full'][q,None]
            counts+=ok.sum(1)
            diff=d['z'][q,None,:].astype(np.float64)-d['z'][r][None,:,:].astype(np.float64)
            for mode in modes:
                squared=diff[:,:,48:79]**2 if mode!='common127' else diff**2
                distances=np.sqrt(squared.mean(2))
                distances[~ok]=np.inf
                j=distances.argmin(1)
                val=distances[np.arange(len(q)),j]
                improve=val<best[mode]
                best[mode][improve]=val[improve]
                bid[mode][improve]=r[j[improve]]
        for i,qi in enumerate(q):
            for mode in modes:
                out[(int(qi),mode)]=(int(bid[mode][i]),float(best[mode][i]) if bid[mode][i]>=0 else None,int(counts[i]))
    return out

def describe(d,i,prefix):
    if i<0:
        return {prefix+k:None for k in ('index','stable_id','dataset','row','house','role','utc','day','raw_index','event','fragment','y_watts','positive','mask127','full127')}
    return {prefix+'index':i,prefix+'stable_id':str(d['id'][i]),prefix+'dataset':str(d['dataset'][i]),prefix+'row':int(d['row'][i]),prefix+'house':HOUSES[int(d['house'][i])],prefix+'role':str(d['source_role'][i]),prefix+'utc':int(d['t'][i]),prefix+'day':int(d['day'][i]),prefix+'raw_index':int(d['raw_index'][i]),prefix+'event':int(d['event'][i]),prefix+'fragment':int(d['fragment'][i]),prefix+'y_watts':float(d['y'][i]),prefix+'positive':bool(d['positive'][i]),prefix+'mask127':''.join(map(str,d['mask'][i].astype(int))),prefix+'full127':bool(d['full'][i])}

def pairs(d,refs,queries,probs):
    cache={}
    for f in FAMILIES:
        for h in range(3):
            qs=sorted({v['index'] for v in queries if v['family']==f and d['house'][v['index']]==h})
            for label in (0,1):
                rr=[v['index'] for v in refs if v['family']==f and v['label']==label and d['house'][v['index']]==h]
                a=nearest(d,qs,rr,False)
                a.update(nearest(d,qs,rr,True))
                for (q,mode),v in a.items():
                    cache[(f,q,mode,label)]=v
            print('DISTANCES',f,HOUSES[h],len(qs),flush=True)
    rows=[]
    for q in queries:
        i=q['index']; f=q['family']; seed=q['seed']
        for mode in ('strict31','common31','common127'):
            pos=cache[(f,i,mode,1)][1]; neg=cache[(f,i,mode,0)][1]
            ratio=pos/neg if pos is not None and neg is not None and neg>0 else None
            status='defined' if ratio is not None else ('missing_reference' if pos is None or neg is None else 'denominator_zero')
            for label in (0,1):
                j,dist,count=cache[(f,i,mode,label)]
                row=dict(family=f,seed=seed,category=q['category'],mode=mode,reference_label=label,query_probability=q['probability'],threshold=q['threshold'],query_sha256=q['sha256'],distance=dist,positive_over_negative_ratio=ratio,ratio_status=status,eligible_reference_n=count,reference_probability=float(probs[(f,seed)][j]) if j>=0 else None,time_gap_seconds=abs(int(d['t'][i])-int(d['t'][j])) if j>=0 else None,paired_distance127=float(np.sqrt(np.mean((d['z'][i].astype(np.float64)-d['z'][j].astype(np.float64))**2))) if j>=0 and mode in ('common31','common127') else None)
                row.update(describe(d,i,'query_')); row.update(describe(d,j,'reference_'))
                rows.append(row)
    require(len(rows)<=600000,'Pair output ceiling')
    csvwrite(RUN/'pairs.csv',rows)
    return rows

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        for level in ('date','role','category'):
            keys=['family','seed','category','mode','reference_label']
            if level in ('date','role'):
                keys+=['query_dataset','query_house','query_role']
            if level=='date':
                keys+=['query_day']
            groups[(level,tuple((k,r[k]) for k in keys))].append(r)
    out=[]
    for (level,keys),rr in sorted(groups.items()):
        vals=[r['distance'] for r in rr if r['distance'] is not None]
        ratios=[r['positive_over_negative_ratio'] for r in rr if r['positive_over_negative_ratio'] is not None]
        out.append(dict(level=level,group=json.dumps(dict(keys),ensure_ascii=False,sort_keys=True),sample_n=len(rr),paired_n=len(vals),null_n=len(rr)-len(vals),distance_q25=float(np.quantile(vals,.25)) if vals else None,distance_median=float(np.median(vals)) if vals else None,distance_q75=float(np.quantile(vals,.75)) if vals else None,ratio_median=float(np.median(ratios)) if ratios else None))
    csvwrite(RUN/'group_distributions.csv',out)
    return out

def curves(d,rows):
    import html
    folder=RUN/'curves'; folder.mkdir(exist_ok=False)
    choices=[]
    for f in FAMILIES:
        for cat in ('FP','TP','FN','TN'):
            label=1 if cat in ('FP','TP') else 0
            rr=sorted([r for r in rows if r['family']==f and r['category']==cat and r['mode']=='strict31' and r['reference_label']==label and r['reference_index'] is not None],key=lambda r:(r['query_sha256'],r['query_stable_id'],r['seed']))
            seen=set()
            for r in rr:
                i=r['query_index']
                fragment=(r['query_house'],r['query_role'],'on',r['query_event']) if d['positive'][i] else (r['query_house'],r['query_role'],'off',r['query_fragment'])
                if fragment in seen:
                    continue
                seen.add(fragment); choices.append(r)
                if len(seen)==2:
                    break
    index=[]
    for number,r in enumerate(choices):
        i,j=r['query_index'],r['reference_index']
        vals=np.r_[d['x'][i][d['mask'][i]],d['x'][j][d['mask'][j]]]
        ymax=max(1,float(vals.max()))
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="450" viewBox="0 0 1100 450"><rect width="1100" height="450" fill="white"/>']
        svg.append(f'<text x="40" y="28" font-size="16">{r["family"]} seed {r["seed"]} {r["category"]}: nearest in frozen reference library</text>')
        for y,text in ((52,'query '+r['query_stable_id']),(74,'reference '+r['reference_stable_id']),(96,f'RMS31={r["distance"]:.6g}; original W; gaps preserved; +/-504s context; core shaded')):
            svg.append(f'<text x="40" y="{y}" font-size="12">{html.escape(text)}</text>')
        svg.append('<rect x="436.2" y="125" width="237.6" height="265" fill="#eef1f5"/>')
        for k,color,name in ((i,'#d55e00','query'),(j,'#0072b2','reference')):
            segments=[]; points=[]
            for a in range(127):
                if d['mask'][k,a]:
                    points.append(f'{60+a*980/126:.2f},{390-float(d["x"][k,a])/ymax*250:.2f}')
                elif points:
                    segments.append(points); points=[]
            if points:segments.append(points)
            for pts in segments:
                svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.7" points="'+ ' '.join(pts)+'"/>')
            svg.append(f'<text x="{60 if name=="query" else 260}" y="420" fill="{color}" font-size="14">{name}, label={int(d["positive"][k])}, y={d["y"][k]:.1f}W</text>')
        svg.append(f'<text x="60" y="120" font-size="12">power axis 0..{ymax:.1f} W</text></svg>')
        path=folder/f'{number:02d}_{r["family"]}_{r["category"]}.svg'
        with path.open('x',encoding='utf-8') as f:f.write('\n'.join(svg))
        index.append(dict(file=str(path.relative_to(RUN)),**r))
    fresh(RUN/'curve_index.json',index)

def analyze():
    isolate()
    reg=read(RUN/'registration.json')
    require(sha(RUN/'config.json')==reg['config_sha256'] and sha(Path(__file__))==reg['source_sha256'],'Frozen rules changed')
    require(snapshot(whitelist())==read(RUN/'inputs_before.json'),'Inputs changed before analysis')
    d=load_data(); probs,thresholds=load_probs(d)
    refs,queries=freeze_samples(d,probs,thresholds)
    duplicate_scan(d)
    rows=pairs(d,refs,queries,probs)
    groups=summarize(rows)
    curves(d,rows)
    after=snapshot(whitelist())
    require(after==read(RUN/'inputs_before.json'),'Inputs changed')
    fresh(RUN/'inputs_after_analysis.json',after)
    fresh(RUN/'analysis_complete.json',dict(pair_rows=len(rows),group_rows=len(groups),null_pairs=sum(r['distance'] is None for r in rows),strict_windows=int(d['strict'].sum()),full127_windows=int(d['full'].sum()),total_windows=len(d['y']),new_fits=0,optimizer_updates=0))
    print('ANALYSIS_COMPLETE',flush=True)

def verify():
    isolate(readonly=True)
    before=snapshot(whitelist())
    require(before==read(RUN/'inputs_before.json'),'Inputs no longer frozen')
    m=read(RUN/'delivery_manifest.json')
    actual={str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
    require(actual==set(m['artifacts']),'Sealed artifact inventory differs')
    outputs_before=snapshot([RUN/n for n in m['artifacts']]+[RUN/'delivery_manifest.json'])
    for name,identity in m['artifacts'].items():
        p=RUN/name
        require(dict(sha256=sha(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)==identity,'Sealed output changed '+name)
    require(snapshot(whitelist())==before,'Read-only verification changed input')
    require(snapshot([RUN/n for n in m['artifacts']]+[RUN/'delivery_manifest.json'])==outputs_before,'Read-only verification changed output')
    print(json.dumps(dict(status='passed_read_only',artifacts=len(m['artifacts']),new_fits=0,all_hash_mtime_unchanged=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','analyze','verify'))
    a=p.parse_args()
    {'register':register,'analyze':analyze,'verify':verify}[a.mode]()
