"""Task24 source-only sampling draft. This program cannot train or predict."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from virtual_submeter.accuracy.microwave_sampling import BatchStream
from virtual_submeter.accuracy.microwave_task21 import isolation

OLD = ROOT / 'outputs/accuracy/microwave-source5-temporal-20261025-v2'
AUDIT = ROOT / 'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4'
RUN = ROOT / 'outputs/accuracy/microwave-task24-index-draft-20261027-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not RUN.exists(), 'Existing draft is never overwritten'
    paths = [OLD / n for n in ('train.npz', 'plan.json', 'normalization.json', 'registration.json')]
    paths += [AUDIT / n for n in ('delivery_seal.json', 'input_seal.json', 'window_ids.csv', 'fragments.csv')]
    paths += [ROOT / 'docs/MICROWAVE_TASK24_PROPOSAL_20261027.md', Path(__file__).resolve(),
              ROOT / 'src/virtual_submeter/accuracy/microwave_sampling.py']
    with isolation(RUN, paths) as access:
        seal = json.loads((AUDIT / 'delivery_seal.json').read_text(encoding='utf-8'))
        prior = {str(ROOT / x['path']): x['sha256'] for x in seal['files']}
        for name in ('input_seal.json', 'window_ids.csv', 'fragments.csv'):
            p = AUDIT / name
            assert sha(p) == prior[str(p)], f'Task23 delivery identity differs: {name}'
        reg = json.loads((OLD / 'registration.json').read_text(encoding='utf-8'))
        for name in ('train.npz', 'plan.json', 'normalization.json'):
            assert sha(OLD / name) == reg['artifacts'][name], f'Task12 predecessor identity differs: {name}'
        plan = json.loads((OLD / 'plan.json').read_text(encoding='utf-8'))
        assert [plan[k] for k in ('checkpoints', 'steps_per_checkpoint', 'batch_size', 'positive_per_batch', 'uniform_negative_per_batch')] == [12, 294, 256, 16, 240]
        with np.load(OLD / 'train.npz', allow_pickle=False) as z:
            original = {k: z[k] for k in ('y', 't', 'event', 'house', 'raw_index')}
        assert len(original['y']) == 336704 and int((original['y'] >= 200).sum()) == 2275
        assert np.isfinite(original['y']).all() and (original['y'] >= 0).all()
        windows = pd.read_csv(AUDIT / 'window_ids.csv', dtype={'house': str, 'fragment_id': str})
        assert len(windows) == 1057163 and int(windows['on'].sum()) == 2114
        assert set(windows['house']) == {'10', '12', '15', '18'} and windows['partial_core31'].eq(1).all()
        assert not windows.duplicated(['house', 'center_utc']).any()
        assert windows['fragment_id'].notna().equals(windows['on'].eq(1))
        t = pd.to_datetime(windows['center_utc'], utc=True).astype('int64').to_numpy() // 10**9
        h = windows['house'].map({'10': 3, '12': 4, '15': 5, '18': 6}).to_numpy(dtype=np.int64)
        event, fragments = pd.factorize(windows['fragment_id'], sort=True)
        d = {'y': np.concatenate([original['y'], windows['on'].to_numpy(dtype=np.float32) * 200]),
             't': np.concatenate([original['t'], t]), 'house': np.concatenate([original['house'], h]),
             'event': np.concatenate([original['event'], event.astype(np.int64)])}
        assert len(d['y']) == 1393867 and int((d['y'] >= 200).sum()) == 4389
        RUN.mkdir(parents=True)
        np.savez_compressed(RUN / 'source_index.npz', **d,
            raw_index=np.concatenate([original['raw_index'], windows['grid_index'].to_numpy(dtype=np.int64)]))
        pd.DataFrame({'event': np.arange(len(fragments)), 'fragment_id': fragments}).to_csv(RUN / 'fragment_mapping.csv', index=False)
        identity = {str(p.relative_to(ROOT)): sha(p) for p in paths}
        (RUN / 'input_identity.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
        summaries = {}
        for seed in (42, 1019, 2026):
            stream = BatchStream(d, 'event', seed, plan)
            batches = np.stack([stream.next() for _ in range(3528)])
            assert batches.shape == (3528, 256)
            assert np.all((d['y'][batches] >= 200).sum(axis=1) == 16)
            count = np.bincount(batches.ravel(), minlength=len(d['y']))
            assert int(count.sum()) == 903168 and int(count[d['y'] >= 200].sum()) == 56448
            np.savez_compressed(RUN / f'batches_{seed}.npz', indices=batches)
            np.savez_compressed(RUN / f'exposure_{seed}.npz', count=count)
            frame = pd.DataFrame({'house': d['house'], 'day': d['t'] // 86400,
                                  'event': d['event'], 'on': d['y'] >= 200, 'exposure': count})
            for key, cols in [('house', ['house', 'on']), ('day', ['house', 'day', 'on']),
                              ('fragment', ['house', 'day', 'event'])]:
                sample = frame if key != 'fragment' else frame[frame['on']]
                sample.groupby(cols, sort=True, as_index=False).agg(centers=('exposure', 'size'),
                    unique_exposed=('exposure', lambda x: int((x > 0).sum())),
                    presentations=('exposure', 'sum')).to_csv(RUN / f'{key}_exposure_{seed}.csv', index=False)
            summaries[str(seed)] = {'updates': 3528, 'positive_presentations': 56448,
                                    'negative_presentations': 846720, 'unique_centers': int((count > 0).sum())}
        result = {'status': 'index_draft_only_incomplete', 'reason': 'Window materialization, adapter, tests and independent reviews remain',
                  'actual_fits': 0, 'optimizer_updates': 0, 'backward_calls': 0, 'candidate_predictions': 0,
                  'support': {'total': len(d['y']), 'on': int((d['y'] >= 200).sum())}, 'seeds': summaries,
                  'forbidden_reads': access['forbidden']}
        (RUN / 'draft_summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
