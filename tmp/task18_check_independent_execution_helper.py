from pathlib import Path
import importlib.util
import sys
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'outputs/accuracy/microwave-expanded-guard95-preparation-20261026-v3'
sys.path.insert(0,str(run/'source'))
from virtual_submeter.accuracy import microwave_expanded_guard95 as e
spec=importlib.util.spec_from_file_location('independent_task18',ROOT/'scripts/verify_microwave_expanded_guard95_execution.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(18)
pairs=[(rng.random(201).astype(np.float32),rng.random(201)<.2),(rng.random(159).astype(np.float32),rng.random(159)<.3)]
c=np.unique(np.concatenate([p for p,y in pairs]));checks=[]
for recall in (.85,.95):
    a=m.threshold(pairs,c,recall)
    b,_=e.gate.select_threshold({str(i):dict(values=p,y=y) for i,(p,y) in enumerate(pairs)},c,recall)
    assert (None if a is None else a['threshold'])==b
    checks.append(dict(recall=recall,threshold_exact=True))
t=np.arange(100,dtype=np.int64)*8;t[70:]+=16
p=rng.random(100).astype(np.float32);p[40:43]=np.nan
np.testing.assert_array_equal(m.maximum(t,p),e.gate.local_max(t,p))
y=rng.random(100)<.2
for state in (np.zeros(100,np.int8),np.ones(100,np.int8),(rng.random(100)<.3).astype(np.int8)):
    assert m.confusion(state,y)==e.gate.confusion(state,y)
e.fresh(run/'independent_execution_helper_synthetic_checks.json',dict(status='passed',synthetic_only=True,curves=checks,
    independent_max_missing_and_gap_exact=True,scalar_confusions_all_on_off_and_mixed_exact=True,real_fits=0,real_optimizer_updates=0))
print('Independent helper synthetic checks passed')
