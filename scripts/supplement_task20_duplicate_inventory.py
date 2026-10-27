"""Inventory all exact repeats, including equal-label repeats; no rule changes."""
import sys
sys.dont_write_bytecode=True
import importlib.util
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

source=Path(__file__).resolve().with_name('task20_input_similarity.py')
source_payload=Path(__file__).read_text(encoding='utf-8')
spec=importlib.util.spec_from_file_location('task20_inventory_support',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.isolate()
assert not (m.RUN/'delivery_manifest.json').exists()
assert m.snapshot(m.whitelist())==m.read(m.RUN/'inputs_before.json')
d=m.load_data()
groups=defaultdict(list)
for i in np.flatnonzero(d['strict']):
    channels=np.stack((d['z'][i],d['mask'][i].astype(np.float32)))
    groups[hashlib.sha256(channels.tobytes()).hexdigest()].append(int(i))
inventory=[]
for digest,ix in sorted(groups.items()):
    if len(ix)<2:continue
    first=ix[0]
    for i in ix:
        assert np.array_equal(d['z'][i],d['z'][first]) and np.array_equal(d['mask'][i],d['mask'][first])
    inventory.append(dict(input_sha256=digest,n=len(ix),positive_n=int(d['positive'][ix].sum()),negative_n=int((~d['positive'][ix]).sum()),households=sorted({m.HOUSES[int(d['house'][i])] for i in ix}),utc_days=sorted(set(map(int,d['day'][ix]))),cross_date=len(set(d['day'][ix].tolist()))>1,missing_count=int((~d['mask'][first]).sum()),mask127=''.join(map(str,d['mask'][first].astype(int))),members=[m.describe(d,i,'') for i in ix]))
expected=m.read(m.RUN/'exact_duplicates.json')
assert len(inventory)==expected['repeated_groups']
result=dict(status='all_repeats_elementwise_confirmed',groups=inventory,note='Supplement completes the preregistered duplicate date/mask inventory; same frozen definition, no sample or distance rule revision.')
if (m.RUN/'exact_repeat_groups.json').exists():
    assert m.read(m.RUN/'exact_repeat_groups.json')==result
else:
    m.fresh(m.RUN/'exact_repeat_groups.json',result)
with (m.RUN/'duplicate_inventory_source_v2.py').open('x',encoding='utf-8',newline='\n') as f:f.write(source_payload)
assert m.snapshot(m.whitelist())==m.read(m.RUN/'inputs_before.json')
print('REPEAT_INVENTORY',[(g['n'],g['positive_n'],g['missing_count'],g['cross_date']) for g in inventory],flush=True)
