"""Independent array-only audit verification, without importing model or gate code."""
import sys
sys.dont_write_bytecode=True
import csv
import hashlib
import json
from collections import deque
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/accuracy/microwave-source5-loss-preparation-20261025-v4'
OUT=ROOT/'outputs/accuracy/microwave-gate-recall-audit-20261025-v1'

def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,keys):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in keys}
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def deque_max(t,p):
    result=np.full_like(p,np.nan)
    q=deque(); left=0
    while left<len(p):
        if not np.isfinite(p[left]): left+=1; continue
        right=left+1
        while right<len(p) and np.isfinite(p[right]) and t[right]==t[right-1]+8: right+=1
        q.clear(); added=left-1
        for j in range(left,right):
            while added<min(right-1,j+15):
                added+=1
                while q and p[q[-1]]<=p[added]: q.pop()
                q.append(added)
            while q[0]<j-15: q.popleft()
            result[j]=p[q[0]]
        left=right
    return result

def main():
    started=time.perf_counter()
    selection=js(RUN/'source_selection.json'); expected=rows(OUT/'decomposition.csv')
    truth=load(RUN/'evaluation/truth_once.npz',['t','y'])
    peaks=rows(OUT/'high_fail_peak_distances.csv'); fn=rows(OUT/'false_negatives.csv')
    checked=[]
    for seed in [42,1019,2026]:
        recipe=next(r for r in selection['recipes'] if r['seed']==seed); L,H=recipe['L'],recipe['H']
        for part in ['development','source_4','source_5']:
            if part=='development':
                z=load(RUN/f'predictions/new_gate_{seed}.npz',['t','scores','state'])
                point=load(RUN/f'predictions/new_point_{seed}.npz',['t','state'])
                w=load(RUN/f'predictions/W50_{seed}.npz',['t','raw'])
                t,p,y=z['t'],z['scores'],truth['y']
                assert np.array_equal(t,truth['t']) and np.array_equal(t,w['t']) and np.array_equal(t,point['t'])
                mask=np.isfinite(y)&np.isfinite(w['raw'][:,1]); old=point['state']==1
                assert mask.sum()==137523 and (y[mask]>=200).sum()==383
            else:
                z=load(OUT/f'source_{seed}_{part[-1]}.npz',['t','p','y'])
                t,p,y=z['t'],z['p'],z['y']; mask=np.isfinite(y)
                threshold=js(RUN/f'models/unit_BCE/seed_{seed}/model.json')['decision_threshold_probability']
                old=p>=threshold
            h=deque_max(t,p); on=(p>=L)&(h>=H)
            if part=='development': np.testing.assert_array_equal(on[mask],z['state'][mask]==1)
            pos=mask&(y>=200); neg=mask&(y<200)
            got=dict(tp=int((pos&on).sum()),fp=int((neg&on).sum()),fn=int((pos&~on).sum()),tn=int((neg&~on).sum()),nonfinite=int((pos&~np.isfinite(p)).sum()),low_fail=int((pos&np.isfinite(p)&(p<L)).sum()),high_fail=int((pos&(p>=L)&(h<H)).sum()),added_fn=int((pos&old&~on).sum()),recovered_tp=int((pos&~old&on).sum()))
            e=next(r for r in expected if int(r['seed'])==seed and r['part']==part)
            assert all(got[k]==int(e[k]) for k in got)
            f=[r for r in fn if int(r['seed'])==seed and r['part']==part]
            assert {int(r['index']) for r in f}==set(np.flatnonzero(pos&~on))
            for r in f:
                j=int(r['index']); assert float(r['p'])==float(p[j]) and float(r['h'])==float(h[j])
            current=[r for r in peaks if int(r['seed'])==seed and r['part']==part]
            assert {int(r['index']) for r in current}==set(np.flatnonzero(pos&(p>=L)&(h<H)))
            # Walk boundaries from each FN, independent of replay's component IDs.
            bounds={}
            for r in current:
                j=int(r['index']); a=b=j
                while a>0 and p[a-1]>=L and np.isfinite(p[a-1]) and t[a]-t[a-1]==8: a-=1
                while b+1<len(p) and p[b+1]>=L and np.isfinite(p[b+1]) and t[b+1]-t[b]==8: b+=1
                assert b-a+1==int(r['low_segment_samples'])
                assert float(p[a:b+1].max())==float(r['low_segment_max_p'])
                same=np.flatnonzero(p[a:b+1]>=H)+a
                assert bool(len(same))==(r['same_low_segment_has_high']=='True')
                if len(same): assert int(np.min(np.abs(t[same]-t[j])))==int(r['nearest_same_low_high_distance_seconds'])
                # Cache full finite spans; the search itself never uses labels.
                key=next((key for key in bounds if key[0]<=j<=key[1]),None)
                if key is None:
                    aa=bb=j
                    while aa>0 and np.isfinite(p[aa-1]) and t[aa]-t[aa-1]==8: aa-=1
                    while bb+1<len(p) and np.isfinite(p[bb+1]) and t[bb+1]-t[bb]==8: bb+=1
                    key=(aa,bb); bounds[key]=np.flatnonzero(p[aa:bb+1]>=H)+aa
                highs=bounds[key]
                assert bool(len(highs))==(r['full_segment_has_high']=='True')
                if len(highs): assert int(np.min(np.abs(t[highs]-t[j])))==int(r['nearest_full_high_distance_seconds'])
            checked.append(dict(seed=seed,part=part,independent_confusions=True,deque_max_frozen_states=True,fn_points_exact=True,peak_distance_and_component_boundaries=True,**got))
    manifest=js(RUN/'final_delivery_manifest.json')
    assert all(sha(RUN/k)==v for k,v in manifest['artifacts'].items())
    modules=['virtual_submeter/accuracy/microwave_event_gate.py','virtual_submeter/accuracy/microwave_source5_temporal.py','virtual_submeter/models/center_mask_tcn.py','virtual_submeter/models/arrays.py']
    source_identity={m:sha(ROOT/'src'/m)==sha(RUN/'source'/m) for m in modules}
    assert all(source_identity.values())
    ordinary=js(OUT/'ordinary_independent_recheck.json')
    assert ordinary['ordinary_infer_full_aggregate_scores_exact']
    result=dict(passed=True,checked=checked,parent_283_artifacts_still_exact=True,live_modules_match_parent_sealed_source=source_identity,ordinary_seed1019_scores_exact=True,elapsed_seconds=time.perf_counter()-started)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
