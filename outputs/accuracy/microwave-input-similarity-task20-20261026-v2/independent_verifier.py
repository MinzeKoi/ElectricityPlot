"""Independent Task20 reimplementation: no main diagnostic analysis helpers used."""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').is_file():ROOT=ROOT.parent
RUN=ROOT/'outputs/accuracy/microwave-input-similarity-task20-20261026-v2'
T18=ROOT/'outputs/accuracy/microwave-expanded-guard95-preparation-20261026-v3'
T19=ROOT/'outputs/accuracy/microwave-fit-generalization-diagnostic-task19-20261026-v4'
SETS=('train','block_4','block_5'); FAMILIES=('control','expanded'); SEEDS=(42,1019,2026); HOUSES=(2,19,5)

def check(ok, why):
    if not ok:raise AssertionError(why)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def snap(paths):return {str(p):dict(sha256=sha(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}

def fresh(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def isolation(paths):
    allowed=set(paths)|{Path(__file__).resolve(),(ROOT/'scripts/task20_input_similarity.py').resolve()}
    def audit(event,a):
        if event=='open' and not isinstance(a[0],int):
            p=Path(os.fsdecode(a[0])).resolve()
            writing=(isinstance(a[1],str) and any(x in a[1] for x in 'wax+')) or (isinstance(a[2],int) and a[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_APPEND|os.O_TRUNC))
            if writing:
                check(p.is_relative_to(RUN),'Independent verifier writes only fresh run artifacts')
                check(not (RUN/'delivery_manifest.json').exists(),'Completed delivery is immutable')
            elif p.is_relative_to(ROOT) and p not in allowed and not p.is_relative_to(RUN) and not p.is_relative_to(Path(sys.prefix)):
                raise PermissionError('Independent verifier outside whitelist '+str(p))
        elif event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):
            raise PermissionError('No external execution or network')
    sys.addaudithook(audit)

def close(a,b,why):
    check(math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=2e-12),why+': '+str((a,b)))

def load():
    norm=read(T18/'normalization.json'); data=[]; probabilities=defaultdict(list)
    for dataset in SETS:
        path=T18/('train.npz' if dataset=='train' else 'validation/'+dataset+'.npz')
        with np.load(path,allow_pickle=False) as f:
            keys=('x','mask','y','t','house','event','raw_index')+(('source_role',) if dataset=='train' else ())
            q={k:f[k] for k in keys}
        n=len(q['y'])
        if dataset!='train':q['source_role']=np.full(n,'house5_validation_'+dataset,dtype='U32')
        check(q['x'].shape==(n,127) and q['mask'].shape==(n,127),'Input geometry')
        check(np.array_equal(np.isfinite(q['x']),q['mask']),'No altered missing support')
        with np.load(T19/'identities'/f'{dataset}.npz',allow_pickle=False) as f:
            for k in ('t','house','source_role','event','raw_index'):check(np.array_equal(q[k],f[k]),'Independent identity '+dataset+'/'+k)
            check(np.array_equal(f['row_index'],np.arange(n)),'Independent row identity')
            check(np.array_equal(f['positive'],q['y']>=200),'Original200W labels')
        z=np.zeros(q['x'].shape,np.float32)
        observed=np.isfinite(q['x'])
        z[observed]=(np.log1p(q['x'][observed])-norm['input_log_mean'])/norm['input_log_std']
        z[z==0]=0
        q['z']=z; q['dataset']=np.full(n,dataset,dtype='U8');q['row']=np.arange(n)
        data.append(q)
        for f in FAMILIES:
            for seed in SEEDS:
                with np.load(T19/'predictions'/f'{f}_{seed}_{dataset}.npz',allow_pickle=False) as c:
                    check(set(c.files)=={'probability','row_index','binding_json'},'Complete frozen cache schema')
                    check(np.array_equal(c['row_index'],np.arange(n)),'Prediction row identity')
                    probabilities[(f,seed)].append(c['probability'])
    d={k:np.concatenate([q[k] for q in data]) for k in data[0]};del data
    d['positive']=d['y']>=200;d['strict']=d['mask'][:,48:79].all(1);d['full']=d['mask'].all(1);d['day']=d['t']//86400
    d['id']=[f'H{HOUSES[int(h)]}|{r}|{int(t)}|raw{int(raw)}' for h,r,t,raw in zip(d['house'],d['source_role'],d['t'],d['raw_index'])]
    d['hash']=[hashlib.sha256(s.encode()).hexdigest() for s in d['id']]
    p={k:np.concatenate(v) for k,v in probabilities.items()}
    # Independently identify observable contiguous same-label fragments.
    order=sorted(range(len(d['y'])),key=lambda i:(d['house'][i],d['source_role'][i],d['t'][i]))
    fragment=np.zeros(len(order),np.int64);last=None;num=0
    for i in order:
        if last is None or d['house'][i]!=d['house'][last] or d['source_role'][i]!=d['source_role'][last] or d['t'][i]-d['t'][last]!=8 or d['positive'][i]!=d['positive'][last]:num+=1
        fragment[i]=num;last=i
    d['fragment']=fragment
    return d,p

def assert_sampling(d,p,queries,refs,support):
    thresholds=read(T19/'pre_registration.json')['thresholds']
    reference_groups=defaultdict(list)
    for i in range(len(d['y'])):
        if d['dataset'][i]=='train':
            for f in FAMILIES:
                if f=='expanded' or d['row'][i]<336704:
                    reference_groups[(f,int(d['house'][i]),str(d['source_role'][i]),bool(d['positive'][i]))].append(i)
    expected_refs=[];expected_queries=[];expected_support=[]
    def ranked(ix):return sorted(ix,key=lambda i:(d['hash'][i],d['id'][i]))
    for key,original in sorted(reference_groups.items()):
        f,h,role,label=key;strict=[i for i in original if d['strict'][i]];chosen=ranked(strict)[:256 if label else 4096]
        expected_refs.extend((f,i) for i in chosen)
        expected_support.append(dict(kind='reference',family=f,seed='',dataset='train',house=str(HOUSES[h]),role=role,day='',category=str(int(label)),original_n=str(len(original)),strict_n=str(len(strict)),full_n=str(sum(d['full'][i] for i in strict)),sample_n=str(len(chosen)),sample_full_n=str(sum(d['full'][i] for i in chosen))))
    base=defaultdict(list)
    for i in range(len(d['y'])):base[(str(d['dataset'][i]),int(d['house'][i]),str(d['source_role'][i]),int(d['day'][i]))].append(i)
    for f in FAMILIES:
        for seed in SEEDS:
            thr=thresholds[f'{f}_{seed}']
            cat=np.where(d['positive'],np.where(p[(f,seed)]>=thr,'TP','FN'),np.where(p[(f,seed)]>=thr,'FP','TN'))
            for (dataset,h,role,day),base_ids in sorted(base.items()):
                for category in ('TP','FP','FN','TN'):
                    original=[i for i in base_ids if cat[i]==category];strict=[i for i in original if d['strict'][i]];chosen=ranked(strict)[:32]
                    expected_queries.extend((f,seed,i,category) for i in chosen)
                    expected_support.append(dict(kind='query',family=f,seed=str(seed),dataset=dataset,house=str(HOUSES[h]),role=role,day=str(day),category=category,original_n=str(len(original)),strict_n=str(len(strict)),full_n=str(sum(d['full'][i] for i in strict)),sample_n=str(len(chosen)),sample_full_n=str(sum(d['full'][i] for i in chosen))))
    check(sorted(expected_refs)==sorted((r['family'],r['index']) for r in refs),'Every frozen reference sample independently reproduced')
    check(sorted(expected_queries)==sorted((r['family'],r['seed'],r['index'],r['category']) for r in queries),'Every frozen query sample independently reproduced')
    check(sorted(json.dumps(v,sort_keys=True) for v in expected_support)==sorted(json.dumps(v,sort_keys=True) for v in support),'All support and sampled coverage independently reproduced')
    for r in refs+queries:
        i=r['index'];check(r['stable_id']==d['id'][i] and r['sha256']==d['hash'][i],'Sample stable identity/hash')
    for r in queries:
        check(r['probability']==float(p[(r['family'],r['seed'])][r['index']]),'Sample frozen probability')
        check(r['threshold']==thresholds[f'{r["family"]}_{r["seed"]}'],'Sample frozen threshold')

def independent_nearest(d,i,refs,common):
    # Different implementation: one query at a time, bounded256 references;
    # exclude ineligible before distance, and explicit (distance,stableID) min.
    rr=sorted([j for j in refs if d['house'][j]==d['house'][i] and d['day'][j]!=d['day'][i] and abs(int(d['t'][j])-int(d['t'][i]))>=1016 and (not common or (d['full'][j] and d['full'][i]))],key=lambda j:d['id'][j])
    modes=('common31','common127') if common else ('strict31',)
    best={mode:(math.inf,'',None) for mode in modes}
    for start in range(0,len(rr),256):
        j=rr[start:start+256]
        diff=d['z'][j].astype(np.float64)-d['z'][i].astype(np.float64)
        for mode in modes:
            delta=diff if mode=='common127' else diff[:,48:79]
            distances=np.sqrt(np.sum(delta*delta,axis=1)/delta.shape[1])
            a=int(np.argmin(distances));candidate=(float(distances[a]),d['id'][j[a]],j[a])
            if candidate[:2]<best[mode][:2]:best[mode]=candidate
    return {mode:(v[2],None if v[2] is None else v[0],len(rr)) for mode,v in best.items()}

def assert_pairs(d,p,queries,refs,delivered):
    pools=defaultdict(list)
    for r in refs:pools[(r['family'],int(d['house'][r['index']]),r['label'])].append(r['index'])
    nearest={};unique=sorted({(q['family'],q['index']) for q in queries})
    for count,(f,i) in enumerate(unique):
        for label in (0,1):
            rr=pools[(f,int(d['house'][i]),label)]
            values=independent_nearest(d,i,rr,False);values.update(independent_nearest(d,i,rr,True))
            for mode,v in values.items():nearest[(f,i,mode,label)]=v
        if count%2000==0:print('INDEPENDENT_NEAREST',count,'/',len(unique),flush=True)
    expected={(q['family'],q['seed'],q['index'],mode,label) for q in queries for mode in ('strict31','common31','common127') for label in (0,1)}
    actual={(r['family'],int(r['seed']),int(r['query_index']),r['mode'],int(r['reference_label'])) for r in delivered}
    check(expected==actual and len(expected)==len(delivered),'Pair matrix complete and unique')
    max_scalar_error=0.;nonnull=0
    thresholds=read(T19/'pre_registration.json')['thresholds']
    def identity(r,prefix,i):
        if i is None:
            for field in ('index','stable_id','dataset','row','house','role','utc','day','raw_index','event','fragment','y_watts','positive','mask127','full127'):check(r[prefix+field]=='','Null reference '+field)
            return
        for field,value in dict(index=i,stable_id=d['id'][i],dataset=str(d['dataset'][i]),row=int(d['row'][i]),house=HOUSES[int(d['house'][i])],role=str(d['source_role'][i]),utc=int(d['t'][i]),day=int(d['day'][i]),raw_index=int(d['raw_index'][i]),event=int(d['event'][i]),fragment=int(d['fragment'][i]),positive=bool(d['positive'][i]),mask127=''.join(str(int(x)) for x in d['mask'][i]),full127=bool(d['full'][i])).items():check(r[prefix+field]==str(value),'Delivered identity '+prefix+field)
        close(r[prefix+'y_watts'],d['y'][i],'Original label watts')
    for r in delivered:
        f=r['family'];seed=int(r['seed']);i=int(r['query_index']);mode=r['mode'];label=int(r['reference_label'])
        j,dist,count=nearest[(f,i,mode,label)]
        identity(r,'query_',i);identity(r,'reference_',j)
        check(r['query_sha256']==d['hash'][i],'Delivered query hash')
        close(r['query_probability'],p[(f,seed)][i],'Delivered frozen probability')
        close(r['threshold'],thresholds[f'{f}_{seed}'],'Delivered frozen threshold')
        pred=p[(f,seed)][i]>=thresholds[f'{f}_{seed}'];cat=('TP' if pred else 'FN') if d['positive'][i] else ('FP' if pred else 'TN')
        check(r['category']==cat,'Delivered error category')
        check(int(r['eligible_reference_n'])==count,'Eligible reference count')
        if j is None:
            check(r['distance']=='' and r['time_gap_seconds']=='' and r['reference_probability']=='' and r['paired_distance127']=='','Missing reference all values null')
        else:
            nonnull+=1
            close(r['distance'],dist,'Independent nearest distance')
            close(r['reference_probability'],p[(f,seed)][j],'Reference frozen probability')
            check(int(r['time_gap_seconds'])==abs(int(d['t'][i])-int(d['t'][j])),'Time gap')
            columns=range(127) if mode=='common127' else range(48,79)
            scalar=math.sqrt(math.fsum((float(d['z'][i,k])-float(d['z'][j,k]))**2 for k in columns)/len(columns))
            max_scalar_error=max(max_scalar_error,abs(scalar-float(r['distance'])))
            close(r['distance'],scalar,'Independent scalar all delivered pair distance')
            if mode!='strict31':
                scalar127=math.sqrt(math.fsum((float(d['z'][i,k])-float(d['z'][j,k]))**2 for k in range(127))/127)
                close(r['paired_distance127'],scalar127,'Paired127 scalar')
            else:check(r['paired_distance127']=='','Strict31 no unobserved127 distance')
        pos=nearest[(f,i,mode,1)][1];neg=nearest[(f,i,mode,0)][1]
        if pos is not None and neg is not None and neg>0:
            close(r['positive_over_negative_ratio'],pos/neg,'Distance ratio');check(r['ratio_status']=='defined','Ratio status')
        else:check(r['positive_over_negative_ratio']=='' and r['ratio_status']==('missing_reference' if pos is None or neg is None else 'denominator_zero'),'Undefined ratio status')
    return dict(delivered_rows=len(delivered),nonnull_pair_rows=nonnull,independently_recomputed_unique_query_family=len(unique),max_scalar_distance_abs_error=max_scalar_error)

def assert_duplicates(d):
    hashes=defaultdict(list)
    for i in np.flatnonzero(d['strict']):
        channels=np.concatenate((d['z'][i],d['mask'][i].astype(np.float32)))
        hashes[hashlib.sha256(channels.tobytes()).hexdigest()].append(int(i))
    summary=read(RUN/'exact_duplicates.json');members=read(RUN/'exact_conflict_members.json')
    expected={};repeated=confirmed=same_house=cross_date=0
    for digest,ix in hashes.items():
        if len(ix)<2:continue
        repeated+=1;confirmed+=len(ix)-1
        for j in ix[1:]:check(np.array_equal(d['z'][ix[0]],d['z'][j]) and np.array_equal(d['mask'][ix[0]],d['mask'][j]),'Duplicate full double channel confirmation')
        if len(set(d['positive'][ix].tolist()))<2:continue
        expected[digest]=ix
        same_house+=int(any(len(set(d['positive'][[i for i in ix if d['house'][i]==h]].tolist()))==2 for h in set(d['house'][ix].tolist())))
        cross_date+=int(len(set(d['day'][ix].tolist()))>1)
    check(summary['strict_windows']==int(d['strict'].sum()) and summary['unique_effective_inputs']==len(hashes) and summary['repeated_groups']==repeated and summary['elementwise_confirmations']==confirmed,'Duplicate total support')
    check(summary['conflict_groups']==len(expected) and summary['conflict_windows']==sum(map(len,expected.values())) and summary['same_house_conflict_groups']==same_house and summary['cross_date_conflict_groups']==cross_date,'Duplicate conflicts support')
    check(sorted((k,i) for k,ix in expected.items() for i in ix)==sorted((r['input_sha256'],r['index']) for r in members),'All conflict members independently reproduced')
    return dict(full_double_channel_windows=int(d['strict'].sum()),conflict_groups=len(expected),elementwise_confirmations=confirmed)

def synthetic_tests():
    spec=importlib.util.spec_from_file_location('task20_synthetic_target',ROOT/'scripts/task20_input_similarity.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    tests=[]
    def passed(name,func):func();tests.append(dict(test=name,status='passed'))
    def scalar_chunk():
        rng=np.random.default_rng(3007);z=rng.normal(size=(520,127)).astype(np.float32)
        d=dict(z=z,house=np.zeros(520,int),day=np.arange(520),t=np.arange(520)*86400,full=np.ones(520,bool),id=np.array([f'id{i:04}' for i in range(520)]))
        out=m.nearest(d,[0,1],list(range(2,520)),True)
        for q in (0,1):
            for mode,cols in (('common31',range(48,79)),('common127',range(127))):
                expected=min((math.sqrt(math.fsum((float(z[q,k])-float(z[j,k]))**2 for k in cols)/len(cols)),d['id'][j],j) for j in range(2,520))
                check(out[(q,mode)][0]==expected[2],'Chunked nearest stable scalar identity');close(out[(q,mode)][1],expected[0],'Chunked scalar distance')
    passed('chunked_distance_matches_independent_scalar_across_reference_chunk_boundary',scalar_chunk)
    def cross_chunk_tie():
        z=np.ones((520,127),np.float32);z[0]=0
        ids=np.array([f'id{i:04}' for i in range(520)])
        d=dict(z=z,house=np.zeros(520,int),day=np.arange(520),t=np.arange(520)*86400,full=np.ones(520,bool),id=ids)
        out=m.nearest(d,[0],list(reversed(range(1,520))),False)
        check(out[(0,'strict31')]==(1,1.,519),'Stable global ID across chunks and reversed input')
        d['day'][:]=0
        check(m.nearest(d,[0],list(range(1,520)),False)[(0,'strict31')]==(-1,None,0),'All-ineligible multi-chunk matrix remains null')
    passed('global_tie_across_chunks_and_all_ineligible_null',cross_chunk_tie)
    def leakage():
        d=dict(z=np.zeros((6,127),np.float32),house=np.array([0,0,0,1,0,0]),day=np.array([1,1,2,2,2,2]),t=np.array([100000,100008,100016,200000,200000,200008]),full=np.array([True,True,True,True,True,False]),id=np.array(['q','same_day','near','wrong_house','z','a']))
        check(m.nearest(d,[0],[1,2,3],False)[(0,'strict31')]==(-1,None,0),'No same-day,overlap,cross-house fallback')
        check(m.nearest(d,[0],[4,5],False)[(0,'strict31')][0]==5,'Exact ties stableID')
        check(m.nearest(d,[0],[4,5],True)[(0,'common31')][0]==4,'Incomplete reference excluded common127')
        d['full'][0]=False
        check(m.nearest(d,[0],[4,5],True)[(0,'common127')]==(-1,None,0),'Incomplete query cannot get common127 distance')
    passed('same_day_overlap_house_missing_and_stable_tie_guards',leakage)
    def transform():
        x=np.ones((2,127),np.float32);x[1,0]=np.nan;mask=np.isfinite(x);z=m.transformed(x,mask,dict(input_log_mean=1.,input_log_std=2.))
        check(z[1,0]==0 and not mask[1,0],'Missing channel kept with mask')
        check(not np.array_equal(np.stack((z[0],mask[0])),np.stack((z[1],mask[1]))),'Full input duplicate includes mask')
        a=z[0].copy();b=a.copy();b[0]+=1;check(np.array_equal(a[48:79],b[48:79]) and not np.array_equal(a,b),'Core equality never full127 equality')
        for bad in (-1.,np.inf):
            q=x.copy();q[0,0]=bad
            try:m.transformed(q,np.isfinite(q),dict(input_log_mean=1.,input_log_std=2.))
            except ValueError:pass
            else:raise AssertionError('Invalid power accepted')
    passed('full127_mask_duplicate_missing_and_invalid_input_guards',transform)
    def original_transform():
        import ast
        import types
        source=T18/'source/virtual_submeter/models/center_mask_tcn.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tensor_input')
        scope=dict(np=np,torch=types.SimpleNamespace(from_numpy=lambda a:a),DataError=ValueError)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
        rng=np.random.default_rng(901);x=rng.uniform(0,8000,(64,127)).astype(np.float32);x[rng.random(x.shape)<.08]=np.nan
        norm=read(T18/'normalization.json');mask=np.isfinite(x)
        frozen=scope['tensor_input'](norm,x,mask)
        check(np.array_equal(frozen[:,0],m.transformed(x,mask,norm)) and np.array_equal(frozen[:,1],mask.astype(np.float32)),'Original source tensor_input exact power and mask')
    passed('exact_original_frozen_tensor_input_conversion',original_transform)
    def matrix():
        saved=m.read
        try:
            m.read=lambda path: {'thresholds':{}} if Path(path).name=='pre_registration.json' else {}
            try:m.load_probs({})
            except ValueError as ex:check('Incomplete model matrix' in str(ex),'Reject matrix before score reads')
            else:raise AssertionError('Missing model matrix accepted')
        finally:m.read=saved
    passed('incomplete_model_matrix_refused_before_cache_reads',matrix)
    def exclusive():
        try:m.fresh(RUN/'config.json',{})
        except FileExistsError:pass
        else:raise AssertionError('Existing config overwritten')
        try:RUN.mkdir(exist_ok=False)
        except FileExistsError:pass
        else:raise AssertionError('Existing run reused')
    passed('exclusive_file_and_directory_no_overwrite',exclusive)
    return tests

def main():
    check((RUN/'analysis_complete.json').is_file(),'Analysis must complete first')
    check(not (RUN/'delivery_manifest.json').exists(),'Cannot write sealed run')
    cfg=read(RUN/'config.json');paths=[Path(p).resolve() for p in cfg['allowed_paths']]
    isolation(paths)
    before=snap(paths);check(before==read(RUN/'inputs_before.json'),'Independent before hash/mtime binding')
    registration=read(RUN/'registration.json')
    check(sha(RUN/'config.json')==registration['config_sha256'] and sha(RUN/'diagnostic.py')==registration['source_sha256'],'Preregistered sources immutable')
    frozen=read(RUN/'samples_frozen.json')
    for file,key in (('support_and_coverage.csv','support_sha256'),('reference_sample.json','reference_sha256'),('query_sample.json','query_sha256')):check(sha(RUN/file)==frozen[key],'Samples frozen before distances')
    tests=synthetic_tests();print('INDEPENDENT_SYNTHETIC_PASSED',len(tests),flush=True)
    d,p=load();queries=read(RUN/'query_sample.json');refs=read(RUN/'reference_sample.json')
    assert_sampling(d,p,queries,refs,rows(RUN/'support_and_coverage.csv'));print('INDEPENDENT_SAMPLING_PASSED',flush=True)
    pair_result=assert_pairs(d,p,queries,refs,rows(RUN/'pairs.csv'));print('INDEPENDENT_PAIRS_PASSED',flush=True)
    duplicate_result=assert_duplicates(d);print('INDEPENDENT_DUPLICATES_PASSED',flush=True)
    after=snap(paths);check(before==after,'Independent after hash/mtime unchanged')
    with (RUN/'independent_verifier.py').open('x',encoding='utf-8',newline='\n') as f:f.write(Path(__file__).read_text(encoding='utf-8'))
    fresh(RUN/'independent_synthetic_tests.json',tests)
    fresh(RUN/'independent_inputs_after.json',after)
    fresh(RUN/'independent_verification.json',dict(status='passed',implementation='Separate implementation; independent sampling support, all nearest reference candidates re-evaluated, scalar math.fsum on every delivered non-null pair; no reuse of main analysis helpers',pair_result=pair_result,duplicate_result=duplicate_result,synthetic_tests=len(tests),all_input_hash_size_mtime_unchanged=True,input_files=len(paths),verifier_sha256=sha(RUN/'independent_verifier.py'),new_fits=0,optimizer_updates=0))
    print('INDEPENDENT_COMPLETE',json.dumps(pair_result),flush=True)

if __name__=='__main__':main()
