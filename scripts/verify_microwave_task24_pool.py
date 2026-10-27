"""Verify Task24 draft pool alignment; never constructs a model or trains."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from virtual_submeter.accuracy.microwave_task21 import isolation
from prepare_microwave_task24_index import sha

OLD = ROOT / 'outputs/accuracy/microwave-source5-temporal-20261025-v2'
CONTROL = ROOT / 'outputs/accuracy/microwave-source5-loss-preparation-20261025-v4'
GUARD = ROOT / 'outputs/accuracy/microwave-recall-guard95-preparation-20261025-v2'
INDEX = ROOT / 'outputs/accuracy/microwave-task24-index-draft-20261027-v1'
WINDOWS = ROOT / 'outputs/accuracy/microwave-task24-windows-draft-20261027-v1'
AUDIT = ROOT / 'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4'
RUN = ROOT / 'outputs/accuracy/microwave-task24-pool-proof-draft-20261027-v2'


def main():
    assert not RUN.exists()
    ws = json.loads((WINDOWS / 'window_summary.json').read_text(encoding='utf-8'))
    names = [v['partition'] for v in ws['outputs']]
    paths = [INDEX / 'source_index.npz', INDEX / 'input_identity.json', WINDOWS / 'window_summary.json',
             AUDIT / 'window_ids.csv', AUDIT / 'delivery_seal.json', OLD / 'train.npz',
             OLD / 'plan.json', OLD / 'normalization.json', OLD / 'registration.json',
             CONTROL / 'final_delivery_manifest.json', GUARD / 'final_delivery_manifest.json',
             GUARD / 'source_selection.json', Path(__file__).resolve()]
    paths += [WINDOWS / f'{n}.npz' for n in names]
    paths += [OLD / f'validation/block_{n}.npz' for n in (4, 5)]
    paths += [OLD / f'initial_{s}.pt' for s in (42, 1019, 2026)]
    paths += [CONTROL / f'models/unit_BCE/seed_{s}/{n}' for s in (42, 1019, 2026) for n in ('model.json', 'weights.pt', 'initial.pt')]
    paths += [INDEX / f'batches_{s}.npz' for s in (42, 1019, 2026)]
    with isolation(RUN, paths) as access:
        for entry in ws['outputs']:
            assert sha(WINDOWS / f'{entry["partition"]}.npz') == entry['sha256']
        oldreg = json.loads((OLD / 'registration.json').read_text(encoding='utf-8'))['artifacts']
        for p in paths:
            if p.is_relative_to(OLD) and p != OLD / 'registration.json':
                assert sha(p) == oldreg[p.relative_to(OLD).as_posix()]
        manifest = json.loads((CONTROL / 'final_delivery_manifest.json').read_text(encoding='utf-8'))['artifacts']
        for p in paths:
            if p.is_relative_to(CONTROL) and p != CONTROL / 'final_delivery_manifest.json':
                val = manifest[p.relative_to(CONTROL).as_posix()]
                assert sha(p) == (val['sha256'] if isinstance(val, dict) else val)
        selection = json.loads((GUARD / 'source_selection.json').read_text(encoding='utf-8'))
        guard_manifest = json.loads((GUARD / 'final_delivery_manifest.json').read_text(encoding='utf-8'))['artifacts']
        item = guard_manifest['source_selection.json']
        assert sha(GUARD / 'source_selection.json') == (item['sha256'] if isinstance(item, dict) else item)
        assert selection['source_only'] and selection['selected'] == 'guard95' and selection['guard95_eligible']
        rows = pd.read_csv(AUDIT / 'window_ids.csv', dtype={'house': str})
        with np.load(INDEX / 'source_index.npz', allow_pickle=False) as z:
            index = {k: z[k] for k in z.files}
        with np.load(OLD / 'train.npz', allow_pickle=False) as z:
            for key in ('y', 't', 'house', 'event', 'raw_index'):
                np.testing.assert_array_equal(index[key][:336704], z[key])
            assert z['x'].shape == (336704, 127)
            assert np.array_equal(np.isfinite(z['x']), z['mask'])
        offset = 336704
        for name in names:
            select = rows['source_role'].eq(name)
            expected = rows.loc[select]
            with np.load(WINDOWS / f'{name}.npz', allow_pickle=False) as z:
                n = len(z['y']); end = offset + n
                assert n == len(expected) and z['x'].shape == (n, 127)
                for key in ('t', 'house', 'raw_index'):
                    np.testing.assert_array_equal(z[key], index[key][offset:end])
                np.testing.assert_array_equal(z['y'] >= 200, index['y'][offset:end] >= 200)
                assert np.array_equal(np.isfinite(z['x']), z['mask'])
                assert np.isfinite(z['y']).all() and (z['y'] >= 0).all()
                offset = end
        assert offset == 1393867
        for seed in (42, 1019, 2026):
            with np.load(INDEX / f'batches_{seed}.npz', allow_pickle=False) as z:
                b = z['indices']
                assert b.shape == (3528, 256) and b.min() >= 0 and b.max() < offset
                assert ((index['y'][b] >= 200).sum(axis=1) == 16).all()
        RUN.mkdir(parents=True)
        result = {'status': 'draft_only_not_authorized', 'aligned_rows': offset,
                  'index_y_semantics': 'old rows retain source power; added rows are 0/200 class indicators only; use actual y from window shards',
                  'old_controls': 'three frozen Task12 unit_BCE models and Task14 guard95 identities bound',
                  'actual_fits': 0, 'optimizer_updates': 0, 'backward_calls': 0, 'candidate_predictions': 0,
                  'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
                  'forbidden_reads': access['forbidden'],
                  'remaining': 'execution adapter and budget gate, nonzero tests, independent full review, final seal'}
        (RUN / 'pool_proof.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Pool identity verified; draft remains unauthorized.')


if __name__ == '__main__':
    main()
