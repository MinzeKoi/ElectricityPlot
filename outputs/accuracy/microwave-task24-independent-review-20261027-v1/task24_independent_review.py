"""Read-only Task24 index and sampling review; never imports experiment code."""
from collections import deque
import csv
import json
from pathlib import Path
import sys

import numpy as np

root = Path.cwd().resolve()
run = Path(sys.argv[1]).resolve()
old_path = root / 'outputs/accuracy/microwave-source5-temporal-20261025-v2/train.npz'
windows_path = root / 'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4/window_ids.csv'
with np.load(run / 'source_index.npz', allow_pickle=False) as z:
    data = {k: z[k] for k in z.files}
with np.load(old_path, allow_pickle=False) as z:
    for key in ('y', 't', 'house', 'event'):
        np.testing.assert_array_equal(data[key][:336704], z[key])
assert len(data['y']) == 1393867
assert np.isfinite(data['y']).all()
assert int((data['y'] >= 200).sum()) == 4389
assert len(set(zip(data['house'].tolist(), data['t'].tolist()))) == len(data['y'])
assert (data['event'][data['y'] >= 200] >= 0).all()
with (run / 'fragment_mapping.csv').open(encoding='utf-8-sig', newline='') as f:
    mapping = {row['fragment_id']: int(row['event']) for row in csv.DictReader(f)}
houses = {10: 3, 12: 4, 15: 5, 18: 6}
i = 336704
with windows_path.open(encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        if row['partial_core31'] != '1':
            continue
        t = int(np.datetime64(row['center_utc'].replace('Z', ''), 's').astype(np.int64))
        assert int(data['house'][i]) == houses[int(row['house'])]
        assert int(data['t'][i]) == t
        assert (data['y'][i] >= 200) == (row['on'] == '1')
        if row['on'] == '1':
            assert int(data['event'][i]) == mapping[row['fragment_id']]
        i += 1
assert i == len(data['y'])
groups = {}
for i in np.flatnonzero(data['y'] >= 200):
    day = (int(data['house'][i]), int(data['t'][i] // 86400))
    groups.setdefault(day, {}).setdefault(int(data['event'][i]), []).append(int(i))
negatives = np.flatnonzero(data['y'] < 200)
out = {}
for seed in (42, 1019, 2026):
    with np.load(run / f'batches_{seed}.npz', allow_pickle=False) as z:
        batches = z['indices']
    assert batches.shape == (3528, 256)
    rng = np.random.Generator(np.random.PCG64(seed))
    queues = {}
    def take(key, pool, n):
        values = []
        for _ in range(n):
            if key not in queues or not queues[key]:
                queues[key] = deque(pool[int(j)] for j in rng.permutation(len(pool))[::-1])
            values.append(queues[key].popleft())
        return values
    for row in batches:
        selected = []
        for _ in range(16):
            day = take('date', sorted(groups), 1)[0]
            event = take(('event', day), sorted(groups[day]), 1)[0]
            selected += take(('center', day, event), groups[day][event], 1)
        selected += take('negative', negatives, 240)
        np.testing.assert_array_equal(row, rng.permutation(np.asarray(selected, np.int64)))
    counts = np.bincount(batches.ravel(), minlength=len(data['y']))
    with np.load(run / f'exposure_{seed}.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(counts, z['count'])
    assert counts[data['y'] >= 200].sum() == 56448
    assert counts[data['y'] < 200].sum() == 846720
    for name, fields, mask in (
        ('house', ('house', 'on'), np.ones(len(counts), dtype=bool)),
        ('day', ('house', 'day', 'on'), np.ones(len(counts), dtype=bool)),
        ('fragment', ('house', 'day', 'event'), data['y'] >= 200),
    ):
        columns = {'house': data['house'], 'day': data['t'] // 86400,
                   'on': data['y'] >= 200, 'event': data['event']}
        keys = np.stack([columns[field][mask].astype(np.int64) for field in fields], axis=1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        centers = np.bincount(inverse)
        seen = np.bincount(inverse, weights=(counts[mask] > 0))
        presented = np.bincount(inverse, weights=counts[mask])
        expected = {tuple(map(int, key)): (int(centers[j]), int(seen[j]), int(presented[j]))
                    for j, key in enumerate(unique)}
        with (run / f'{name}_exposure_{seed}.csv').open(encoding='utf-8-sig', newline='') as f:
            actual = {}
            for row in csv.DictReader(f):
                key = tuple(int(row[k] == 'True') if k == 'on' else int(row[k]) for k in fields)
                assert key not in actual
                actual[key] = tuple(int(row[k]) for k in ('centers', 'unique_exposed', 'presentations'))
        assert actual == expected, f'{name} exposure differs for seed {seed}'
    out[seed] = {'unique_centers': int((counts > 0).sum()), 'presentations': int(counts.sum())}
print(json.dumps({'passed': True, 'source_windows': len(data['y']), 'replay': out}, ensure_ascii=False))
