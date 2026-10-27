"""Compare Task24 windows to Task23 independent raw-slice grid recomputation."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

root = Path.cwd().resolve()
task23 = root / 'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4'
run = Path(sys.argv[1]).resolve()
seal = json.loads((task23 / 'input_seal.json').read_text(encoding='utf-8'))
history = runpy.run_path(str(task23 / 'independent_check.py'))
hashes = {entry['path']: entry['sha256'] for entry in seal['inputs']}
total = 0
for house, info in seal['houses'].items():
    path = info['paths']['slice']
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == hashes[path]
    lo, hi = (history['ts'](info['selection'][k]) for k in ('start', 'end'))
    a, y = history['recompute'](root / path, info['channel'], lo, hi)
    for part in info['partitions']:
        low, high = (history['ts'](part[k]) for k in ('start', 'end'))
        first, last = (low - lo) // 8, (high - lo) // 8
        aa, yy = a[first:last], y[first:last]
        tt = np.arange(low, high, 8, dtype=np.int64)
        partial, _, _, _, _, _, _, _, centers, _ = history['support'](tt, aa, yy, low, high)
        with np.load(run / (part['id'] + '.npz'), allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        np.testing.assert_array_equal(d['t'], tt[centers])
        np.testing.assert_allclose(d['y'], yy[centers], rtol=1e-6, atol=1e-5)
        np.testing.assert_array_equal(d['raw_index'], centers + first)
        assert partial.sum() == len(centers)
        expected = np.full((len(centers), 127), np.nan, dtype=np.float32)
        expected[:, 63] = aa[centers]
        for direction, field in ((-1, 'left_stop'), (1, 'right_stop')):
            active = np.ones(len(centers), dtype=bool)
            previous_missing = np.zeros(len(centers), dtype=bool)
            reason = np.full(len(centers), 'radius', dtype='U16')
            for step in range(1, 64):
                ix = centers + direction * step
                boundary = (ix < 0) | (ix >= len(aa))
                safe = np.clip(ix, 0, len(aa) - 1)
                missing = ~np.isfinite(aa[safe])
                stop_gap = active & ~boundary & previous_missing & missing
                reason[active & boundary] = 'boundary'
                reason[stop_gap] = 'missing_run'
                active &= ~boundary & ~stop_gap
                take = active & ~missing
                expected[take, 63 + direction * step] = aa[safe[take]]
                previous_missing = missing
            np.testing.assert_array_equal(d[field], reason)
        mask = np.isfinite(expected)
        np.testing.assert_array_equal(d['mask'], mask)
        np.testing.assert_allclose(d['x'], expected, rtol=1e-6, atol=1e-5, equal_nan=True)
        np.testing.assert_array_equal(d['mask_identity'], np.packbits(mask, axis=1))
        outer = mask.sum(axis=1) - mask[:, 48:79].sum(axis=1)
        np.testing.assert_array_equal(d['outer_observed'], outer)
        np.testing.assert_array_equal(d['outer_fraction'], outer.astype(np.float32) / 96)
        total += len(centers)
        print(part['id'], len(centers), flush=True)
assert total == 1057163
print('Independent raw grid and all C127 windows passed:', total, flush=True)
