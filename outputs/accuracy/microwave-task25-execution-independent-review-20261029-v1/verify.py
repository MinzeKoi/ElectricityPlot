import csv, hashlib, json
from pathlib import Path
import numpy as np
root=Path.cwd(); run=root/'outputs/accuracy/microwave-task25-preparation-20261028-v2'; parent=root/'outputs/accuracy/microwave-task24-index-draft-20261027-v1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf8'))
reg=read(run/'registration.json'); authpath=root/'outputs/accuracy/microwave-task25-preparation-delivery-20261028-v1/human-budget-20261028.json'; auth=read(authpath)
assert sha(run/'registration.json')==auth['registration_sha256']=='a59008deaf622bb139321e41a81cfb47f2848f4c82db7f71a4ac117240f6dadc'
assert sha(authpath)=='42b65e5373b10ccee37dacd40e844ae902a003fc3eacf46c9642a418fa046127'
assert auth['authorized'] and auth['additional_new_fits']==3 and auth['maximum_updates']==10584 and auth['seeds']==[42,1019,2026] and not auth['automatic_retry']
for n,h in auth['review_sha256'].items(): assert sha(run/n)==h
for n,h in reg['artifacts'].items(): assert sha(run/n)==h
for n,h in reg['inputs'].items(): assert sha(Path(n))==h
complete=read(run/'training_complete.json'); selection=read(run/'source_selection.json'); srccomplete=read(run/'source_complete.json')
assert (complete['new_fits'],complete['updates'],complete['control_retrains'])==(3,10584,0)
for n,h in complete['artifacts'].items(): assert sha(run/'models'/n)==h
for n,h in srccomplete['artifacts'].items(): assert sha(run/'source_results'/n)==h
assert selection['training_sha256']==sha(run/'training_complete.json') and srccomplete['source_sha256']==sha(run/'source_selection.json')
with np.load(parent/'source_index.npz') as z: y=z['y']; house=z['house']
control=read(Path(reg['training_sources']['source_control']))['rows']['guard95']; task24=read(Path(reg['training_sources']['task24_selection']))['rows']; assert task24==selection['task24_frozen_rows']
results=[]
for i,seed in enumerate((42,1019,2026)):
    model=run/f'models/unit_BCE/seed_{seed}'
    assert read(model/'started.json')=={'seed':seed,'arm':'task25','maximum_updates':3528}
    assert read(model/'progress.json')=={'completed_updates':3528,'attempted_updates':3528}
    assert read(model/'complete.json')['updates']==3528
    with np.load(run/f'batches_{seed}.npz') as z: batches=z['indices']
    with np.load(parent/f'batches_{seed}.npz') as z: old=z['indices']
    with (model/'batches.csv').open(newline='',encoding='utf8') as f: ledger=list(csv.DictReader(f))
    assert len(ledger)==3528
    for j,row in enumerate(ledger):
        assert int(row['update'])==j+1 and int(row['positive'])==16 and int(row['negative'])==240
        assert row['indices_sha256']==hashlib.sha256(np.ascontiguousarray(batches[j]).tobytes()).hexdigest()
    neg=y[old]<200; assert np.array_equal(batches[neg],old[neg]); positive=y[batches]>=200
    assert np.all(positive.sum(axis=1)==16)
    assert np.array_equal((positive & (house[batches]<3)).sum(axis=1),np.tile([11,11,10],1176))
    counts=np.bincount(batches.ravel(),minlength=len(y))
    with np.load(model/'exposure.npz') as z: assert np.array_equal(counts,z['count'])
    assert (int(counts[(y>=200)&(house<3)].sum()),int(counts[(y>=200)&(house>=3)].sum()),int(counts[y<200].sum()))==(37632,18816,846720)
    recipe=selection['recipes'][i]; assert recipe==read(run/f'source_results/recipe_{seed}.json') and recipe['seed']==seed
    blocks=[]
    for b,claimed in zip((4,5),recipe['blocks']):
        with np.load(run/f'source_results/scores_{seed}_{b}.npz') as z:
            p,h,ix,truth=z['p'],z['h'],z['ix'],z['y']; pred=((p[ix]>=recipe['L'])&(h[ix]>=recipe['H'])); assert np.isfinite(p[ix]).all() and np.isfinite(h[ix]).all()
            tp=int((pred&truth).sum()); fp=int((pred&~truth).sum()); fn=int((~pred&truth).sum()); tn=int((~pred&~truth).sum())
        assert (tp,fp,fn,tn)==tuple(claimed[k] for k in ('tp','fp','fn','tn'))
        m={'precision':tp/(tp+fp),'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn)}
        for k,v in m.items(): assert abs(v-claimed[k])<1e-12
        assert m['recall']>=.95 and 0<tp+fp<tp+fp+fn+tn
        blocks.append(m)
    mean={k:sum(m[k] for m in blocks)/2 for k in ('precision','recall','f1')}
    for k,v in mean.items(): assert abs(v-selection['rows'][i][k])<1e-12
    results.append({'seed':seed,'updates':len(ledger),'old_positive':37632,'new_positive':18816,'negative':846720,'L':recipe['L'],'H':recipe['H'],'blocks':blocks,'mean':mean})
means=lambda rows:{k:sum(r[k] for r in rows)/3 for k in ('precision','recall','f1')}
rows=selection['rows']; qualified=(means(rows)['f1']>=means(control)['f1'] and all(b['recall']-a['recall']>=-.05 for a,b in zip(control,rows)) and sum(b['precision']>a['precision'] for a,b in zip(control,rows))>=2)
rank=(means(rows)['precision'],means(rows)['f1'])>(means(control)['precision'],means(control)['f1'])
assert not qualified and not rank and selection['eligible']==False and selection['selected']=='control_guard95'
print(json.dumps({'passed':True,'registration_sha256':sha(run/'registration.json'),'authorization_sha256':sha(authpath),'training_sha256':sha(run/'training_complete.json'),'source_sha256':sha(run/'source_selection.json'),'registered_artifacts':len(reg['artifacts']),'registered_inputs':len(reg['inputs']),'model_artifacts':len(complete['artifacts']),'source_artifacts':len(srccomplete['artifacts']),'seeds':results,'candidate_mean':means(rows),'control_mean':means(control),'task24_mean':means(task24),'precision_wins':sum(b['precision']>a['precision'] for a,b in zip(control,rows)),'qualified':qualified,'rank_wins':rank,'selected':selection['selected']},ensure_ascii=False,sort_keys=True,indent=2))
