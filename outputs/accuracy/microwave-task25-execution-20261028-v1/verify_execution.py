import hashlib, json, pathlib, statistics
root=pathlib.Path.cwd(); run=root/'outputs/accuracy/microwave-task25-preparation-20261028-v2'; dest=root/'outputs/accuracy/microwave-task25-execution-20261028-v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
reg=read(run/'registration.json'); train=read(run/'training_complete.json'); sel=read(run/'source_selection.json'); done=read(run/'source_complete.json'); control=read(pathlib.Path(reg['training_sources']['source_control']))['rows']['guard95']; rows=sel['rows']
assert sha(run/'registration.json')=='a59008deaf622bb139321e41a81cfb47f2848f4c82db7f71a4ac117240f6dadc'
assert sha(root/'outputs/accuracy/microwave-task25-preparation-delivery-20261028-v1/human-budget-20261028.json')=='42b65e5373b10ccee37dacd40e844ae902a003fc3eacf46c9642a418fa046127'
assert (train['new_fits'],train['updates'],train['control_retrains'])==(3,10584,0)
for name,digest in train['artifacts'].items(): assert sha(run/'models'/name)==digest
assert done['source_sha256']==sha(run/'source_selection.json') and done['registration_sha256']==sha(run/'registration.json')
for name,digest in done['artifacts'].items(): assert sha(run/'source_results'/name)==digest
assert sel['training_sha256']==sha(run/'training_complete.json') and not sel['eligible'] and sel['selected']=='control_guard95'
assert all(r['source_gate_passed'] and not r['degenerate'] for r in rows)
assert all(block['recall']>=.95 for recipe in sel['recipes'] for block in recipe['blocks'])
assert statistics.mean(r['f1'] for r in rows)<statistics.mean(r['f1'] for r in control)
assert sum(b['precision']>a['precision'] for a,b in zip(control,rows))==1
assert statistics.mean(r['precision'] for r in rows)<statistics.mean(r['precision'] for r in control)
assert not any((run/n).exists() for n in ('future_identity.json','freeze.json','inference_checks.json','predictions','evaluation'))
assert read(dest/'source.stdout.log')=={'source_eligible':False,'development_reads':0}
assert (dest/'source.stderr.log').stat().st_size==0 and (dest/'train.stderr.log').stat().st_size==0
print(json.dumps({'passed':True,'training_sha256':sha(run/'training_complete.json'),'source_sha256':sha(run/'source_selection.json'),'source_complete_sha256':sha(run/'source_complete.json'),'model_files_verified':len(train['artifacts']),'source_files_verified':len(done['artifacts']),'selected':'control_guard95','development_stage_absent':True},ensure_ascii=False,sort_keys=True))
