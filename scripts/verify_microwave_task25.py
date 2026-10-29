"""Read-only Task25 preparation verification; no fit or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/accuracy/microwave-task25-preparation-20261028-v2'
T24=ROOT/'outputs/accuracy/microwave-task24-recovery-preparation-20261027-v1'
sys.path.insert(0,str(RUN/'source'))
from virtual_submeter.accuracy.microwave_task21 import isolation


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''): h.update(chunk)
    return h.hexdigest()


def verify():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    reg=read(RUN/'registration.json'); parent=read(T24/'registration.json')
    paths=[RUN/'registration.json',T24/'registration.json']+[Path(p) for p in reg['inputs']]+[RUN/p for p in reg['artifacts']]
    with isolation(RUN,paths,readonly=True) as access:
        assert reg['schema']=='microwave_task25_positive_share_v1' and reg['status']=='sealed_pending_review' and not reg['authorized']
        assert reg['maximum_new_fits']==3 and reg['maximum_updates']==10584 and reg['seeds']==[42,1019,2026]
        for path,digest in reg['inputs'].items(): assert sha(Path(path))==digest,path
        for path,digest in reg['artifacts'].items(): assert sha(RUN/path)==digest,path
        with np.load(reg['training_sources']['index'],allow_pickle=False) as z:
            y,house=z['y'],z['house']
        assert len(y)==1393867 and int((y>=200).sum())==4389
        results={}
        for seed in (42,1019,2026):
            with np.load(parent['training_sources']['batches'][str(seed)],allow_pickle=False) as z: before=z['indices']
            with np.load(RUN/f'batches_{seed}.npz',allow_pickle=False) as z: after=z['indices']
            positive=y[before]>=200
            assert np.array_equal(y[after]>=200,positive) and np.array_equal(after[~positive],before[~positive])
            old=((y[after]>=200)&(house[after]<=2)).sum(axis=1)
            assert np.array_equal(old,np.tile([11,11,10],1176))
            assert np.all(old.reshape(12,294).sum(axis=1)==3136)
            with np.load(RUN/f'exposure_{seed}.npz',allow_pickle=False) as z: count=z['count']
            np.testing.assert_array_equal(count,np.bincount(after.ravel(),minlength=len(y)))
            assert int(count[(y>=200)&(house<=2)].sum())==37632 and int(count[(y>=200)&(house>=3)].sum())==18816
            assert int(count[y<200].sum())==846720
            results[str(seed)]={'old_positive':37632,'new_positive':18816,'negative':846720,'unique_centers':int((count>0).sum()),'negative_slots_exact':True}
        assert not any((RUN/name).exists() for name in ('training_started.json','models','training_complete.json','source_selection.json','future_identity.json','predictions','evaluation'))
        assert not access['forbidden']
    return {'status':'prepared_pending_independent_review_and_budget','registration_sha256':sha(RUN/'registration.json'),
            'inputs_verified':len(reg['inputs']),'artifacts_verified':len(reg['artifacts']),'seeds':results,
            'new_fits':0,'optimizer_updates':0,'candidate_predictions':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['verify','report']);args=parser.parse_args()
    result=verify()
    if args.command=='report': result['report_note']='Fixed Task25 grouped-positive comparison; no budget authorized.'
    print(json.dumps(result,ensure_ascii=False,indent=2))
