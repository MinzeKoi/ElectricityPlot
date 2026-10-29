"""Task25 preparation only: fixed grouped-positive sampling and sealed adapter."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
T24 = ROOT / 'outputs/accuracy/microwave-task24-recovery-preparation-20261027-v1'
RUN = ROOT / 'outputs/accuracy/microwave-task25-preparation-20261028-v1'
sys.path.insert(0, str(T24 / 'source'))
from virtual_submeter.accuracy.microwave_task21 import isolation
from virtual_submeter.accuracy.microwave_sampling import BatchStream
from virtual_submeter.data.common import inventory
from prepare_microwave_task24_index import sha


def main():
    assert not RUN.exists()
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    old = read(T24 / 'registration.json')
    training = read(T24 / 'training_complete.json')
    source = read(T24 / 'source_complete.json')
    assert training['new_fits'] == 3 and training['updates'] == 10584 and source['source_sha256'] == sha(T24 / 'source_selection.json')
    sources = old['training_sources']
    original_batches = dict(sources['batches'])
    inputs = [Path(p) for p in old['inputs']]
    inputs += [T24 / n for n in ('registration.json','training_complete.json','source_selection.json','source_complete.json','run_microwave_task24_draft.py')]
    inputs += [T24 / 'models' / name for name in training['artifacts']]
    inputs += [T24 / 'source_results' / name for name in source['artifacts']]
    inputs += [ROOT / 'docs/MICROWAVE_TASK25_PROPOSAL_20261028.md', ROOT / 'docs/F1_TASK24_FINDINGS_20261028.md',
               ROOT / 'docs/F1_TASK24_HOUSE_EXPOSURE_20261028.csv', ROOT / 'docs/F1_TASK24_FRAGMENT_EXPOSURE_20261028.csv', Path(__file__).resolve()]
    code = sorted((T24 / 'source').rglob('*.py'))
    paths = sorted(set(inputs + code))
    with isolation(RUN, paths) as access:
        for p, digest in old['inputs'].items(): assert sha(Path(p)) == digest
        for name, digest in training['artifacts'].items(): assert sha(T24 / 'models' / name) == digest
        for name, digest in source['artifacts'].items(): assert sha(T24 / 'source_results' / name) == digest
        with np.load(sources['index'], allow_pickle=False) as z:
            d = {k: z[k] for k in ('y','t','house','event')}
        assert len(d['y']) == 1393867 and int((d['y'] >= 200).sum()) == 4389
        assert set(np.unique(d['house'])) == set(range(7))
        plan = read(Path(sources['plan']))
        RUN.mkdir(parents=True)
        for p in code:
            dest = RUN / 'source' / p.relative_to(T24 / 'source'); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
        shutil.copyfile(ROOT / 'docs/MICROWAVE_TASK25_PROPOSAL_20261028.md', RUN / 'protocol.md')
        shutil.copyfile(Path(__file__), RUN / 'prepare_microwave_task25.py')
        (RUN / 'workspace_identity.json').write_text(json.dumps({'root':str(ROOT)})+'\n', encoding='utf-8')
        summary = {}
        for seed in (42,1019,2026):
            with np.load(original_batches[str(seed)], allow_pickle=False) as z: original = z['indices']
            assert original.shape == (3528,256)
            streams = [BatchStream(d,'event',np.random.SeedSequence([seed,25,g]),plan) for g in (0,1)]
            for group, stream in enumerate(streams):
                stream.groups = {k:v for k,v in stream.groups.items() if (k[0] <= 2) == (group == 0)}
            chooser = np.random.default_rng(np.random.SeedSequence([seed,25,2]))
            batches = original.copy()
            for step, row in enumerate(batches):
                positions = np.flatnonzero(d['y'][row] >= 200)
                assert len(positions) == 16
                old_count = (11,11,10)[step % 3]
                groups = chooser.permutation(np.r_[np.zeros(old_count,dtype=np.int8),np.ones(16-old_count,dtype=np.int8)])
                for pos, group in zip(positions, groups):
                    stream = streams[int(group)]
                    day = stream.take('days', sorted(stream.groups), 1)[0]
                    event = stream.take(('events',day), sorted(stream.groups[day]), 1)[0]
                    row[pos] = stream.take(('centers',day,event), stream.groups[day][event], 1)[0]
            positive = d['y'][original] >= 200
            assert np.array_equal(batches[~positive], original[~positive])
            assert np.array_equal(d['y'][batches] >= 200, positive)
            counts = np.bincount(batches.ravel(), minlength=len(d['y']))
            assert int(counts[(d['y']>=200)&(d['house']<=2)].sum()) == 37632
            assert int(counts[(d['y']>=200)&(d['house']>=3)].sum()) == 18816
            np.savez_compressed(RUN / f'batches_{seed}.npz', indices=batches)
            np.savez_compressed(RUN / f'exposure_{seed}.npz', count=counts)
            frame = pd.DataFrame({'house':d['house'], 'day':d['t']//86400, 'event':d['event'], 'on':d['y']>=200, 'exposure':counts})
            for key, cols in [('house',['house','on']),('day',['house','day','on']),('fragment',['house','day','event'])]:
                sample = frame if key != 'fragment' else frame[frame['on']]
                sample.groupby(cols,sort=True,as_index=False).agg(centers=('exposure','size'),
                    unique_exposed=('exposure',lambda x:int((x>0).sum())), presentations=('exposure','sum')).to_csv(RUN / f'{key}_exposure_{seed}.csv',index=False)
            summary[str(seed)] = {'old_positive':37632,'new_positive':18816,'negative':846720,
                'unique_centers':int((counts>0).sum()),'negative_slots_exact':True}
        fresh_sources = dict(sources)
        fresh_sources['batches'] = {str(s):str((RUN / f'batches_{s}.npz').resolve()) for s in (42,1019,2026)}
        fresh_sources['task24_selection'] = str((T24 / 'source_selection.json').resolve())
        adapter = (T24 / 'run_microwave_task24_draft.py').read_text(encoding='utf-8')
        adapter = adapter.replace('microwave_task24_multihouse_v1','microwave_task25_positive_share_v1')
        adapter = adapter.replace('run_microwave_task24_draft.py','run_microwave_task25.py').replace('Task24 needs','Task25 needs').replace('exact Task24 budget','exact Task25 budget')
        adapter = adapter.replace("arm='task24'", "arm='task25'")
        needle = "            batches[seed] = indices\n"
        insert = "            old_positive = ((d['y'][indices] >= 200) & (d['house'][indices] <= 2)).sum(axis=1)\n            if not np.array_equal(old_positive, np.tile([11,11,10], 1176)):\n                raise ValueError('Task25 positive quota differs')\n"
        assert adapter.count(needle) == 1
        adapter = adapter.replace(needle,insert+needle)
        needle = "        selection = {'source_only': True, 'selected': 'unit_BCE' if eligible else 'control_guard95',\n"
        replacement = "        task24 = json.loads(Path(sources['task24_selection']).read_text(encoding='utf-8'))\n" + needle
        assert adapter.count(needle) == 1
        adapter = adapter.replace(needle,replacement).replace("'training_sha256': sha(run / 'training_complete.json')}\n", "'training_sha256': sha(run / 'training_complete.json'), 'task24_frozen_rows': task24['rows']}\n")
        adapter = adapter.replace("required = [sources['source_control'], sources['source5_aggregate']]", "required = [sources['source_control'], sources['source5_aggregate'], sources['task24_selection']]")
        (RUN / 'run_microwave_task25.py').write_text(adapter,encoding='utf-8')
        (RUN / 'sampling_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        all_inputs = {str(p.resolve()):sha(p) for p in inputs}
        for seed in (42,1019,2026): all_inputs[str((RUN / f'batches_{seed}.npz').resolve())] = sha(RUN / f'batches_{seed}.npz')
        registration = {'schema':'microwave_task25_positive_share_v1','status':'sealed_pending_review','authorized':False,
            'seeds':[42,1019,2026],'maximum_new_fits':3,'maximum_updates':10584,'training_sources':fresh_sources,
            'future_identity_source':old['future_identity_source'],'inputs':all_inputs,'artifacts':inventory(RUN),
            'counts':{'centers':1393867,'positive':4389},'task24_training_sha256':sha(T24/'training_complete.json'),
            'task24_source_sha256':sha(T24/'source_selection.json'), 'actual_fits':0,'optimizer_updates':0,'candidate_predictions':0}
        (RUN / 'registration.json').write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        assert not access['forbidden']
    print(json.dumps({'status':'prepared_pending_independent_review','registration_sha256':sha(RUN/'registration.json'),'summary':summary},indent=2))


if __name__ == '__main__': main()
