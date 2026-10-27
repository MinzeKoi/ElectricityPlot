"""Materialize only audited source windows, with no training or predictions."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from virtual_submeter.models.center_mask_tcn import expand_context
from virtual_submeter.accuracy.microwave_task21 import isolation
from prepare_microwave_task24_index import sha

AUDIT = ROOT / 'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4'
DRAFT = ROOT / 'outputs/accuracy/microwave-task24-index-draft-20261027-v1'
RUN = ROOT / 'outputs/accuracy/microwave-task24-windows-draft-20261027-v1'
sys.path.insert(0, str(AUDIT))
import audit


def main():
    assert not RUN.exists()
    seal = json.loads((AUDIT / 'input_seal.json').read_text(encoding='utf-8'))
    delivery = json.loads((AUDIT / 'delivery_seal.json').read_text(encoding='utf-8'))
    identities = {r['path']: r['sha256'] for r in delivery['files']}
    inputs = [AUDIT / n for n in ('input_seal.json', 'delivery_seal.json', 'window_ids.csv')]
    inputs += [DRAFT / 'source_index.npz', DRAFT / 'draft_summary.json', Path(__file__).resolve(), AUDIT / 'audit.py',
               ROOT / 'src/virtual_submeter/models/center_mask_tcn.py']
    for h in ('10', '12', '15', '18'):
        inputs.append(ROOT / seal['houses'][h]['paths']['slice'])
    with isolation(RUN, inputs) as access:
        for p in inputs:
            key = p.relative_to(ROOT).as_posix()
            if key in identities:
                assert sha(p) == identities[key], key
        windows = pd.read_csv(AUDIT / 'window_ids.csv', dtype={'house': str, 'fragment_id': str})
        with np.load(DRAFT / 'source_index.npz', allow_pickle=False) as z:
            index = {k: z[k] for k in z.files}
        assert len(index['y']) == 1393867
        RUN.mkdir(parents=True)
        outputs = []
        for h, house in [('10', 3), ('12', 4), ('15', 5), ('18', 6)]:
            info = seal['houses'][h]
            start = int(pd.Timestamp(info['selection']['start']).timestamp())
            end = int(pd.Timestamp(info['selection']['end']).timestamp())
            t = np.arange(start, end, 8, dtype=np.int64)
            grid, coverage = audit.rawgrid(ROOT / info['paths']['slice'], info['channel'], start, end)
            a, y = grid
            for part in info['partitions']:
                rows = windows[(windows['house'] == h) & (windows['source_role'] == part['id'])]
                ids = rows['grid_index'].to_numpy(dtype=np.int64)
                low = int(pd.Timestamp(part['start']).timestamp())
                high = int(pd.Timestamp(part['end']).timestamp())
                assert ((t[ids] >= low) & (t[ids] < high)).all()
                assert np.array_equal(t[ids], pd.to_datetime(rows['center_utc'], utc=True).astype('int64').to_numpy() // 10**9)
                assert np.isfinite(y[ids]).all() and (y[ids] >= 0).all()
                assert np.array_equal(y[ids] >= 200, rows['on'].to_numpy(dtype=bool))
                x = expand_context(t, a, ids, low, high)
                assert np.array_equal(x['left_stop'], rows['left_stop'].to_numpy())
                assert np.array_equal(x['right_stop'], rows['right_stop'].to_numpy())
                core = x['mask'][:, 48:79]
                assert (core.sum(axis=1) >= 24).all() and not ((~core[:, :-1]) & (~core[:, 1:])).any()
                target = RUN / f'{part["id"]}.npz'
                np.savez_compressed(target, **x, y=y[ids].astype(np.float32), t=t[ids], raw_index=ids,
                                    house=np.full(len(ids), house, dtype=np.int64))
                outputs.append({'partition': part['id'], 'windows': len(ids), 'on': int((y[ids] >= 200).sum()),
                                'sha256': sha(target)})
                print(part['id'], len(ids), flush=True)
        assert sum(x['windows'] for x in outputs) == 1057163 and sum(x['on'] for x in outputs) == 2114
        result = {'status': 'source_window_draft_incomplete', 'outputs': outputs,
                  'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                  'actual_fits': 0, 'optimizer_updates': 0, 'backward_calls': 0, 'candidate_predictions': 0,
                  'forbidden_reads': access['forbidden'],
                  'remaining': 'Bind index to windows, old controls and code; build budget-gated execution adapter; tests and independent review.'}
        (RUN / 'window_summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Source windows prepared; this draft is not an execution authorization.')


if __name__ == '__main__':
    main()
