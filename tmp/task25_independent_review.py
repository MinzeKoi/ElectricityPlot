"""Independent deterministic Task25 replay and exposure verification."""
from collections import deque
import csv
import json
from pathlib import Path
import sys
import numpy as np

root = Path.cwd().resolve()
run = Path(sys.argv[1]).resolve()
parent = root / 'outputs/accuracy/microwave-task24-index-draft-20261027-v1'
with np.load(parent / 'source_index.npz', allow_pickle=False) as z:
    d = {k: z[k] for k in ('y', 't', 'house', 'event')}
assert len(d['y']) == 1393867 and int((d['y'] >= 200).sum()) == 4389
groups = [{}, {}]
for i in np.flatnonzero(d['y'] >= 200):
    house = int(d['house'][i])
    group = 0 if house < 3 else 1
    day = (house, int(d['t'][i] // 86400))
    groups[group].setdefault(day, {}).setdefault(int(d['event'][i]), []).append(int(i))
results = {}
for seed in (42, 1019, 2026):
    with np.load(parent / f'batches_{seed}.npz', allow_pickle=False) as z:
        old = z['indices']
    with np.load(run / f'batches_{seed}.npz', allow_pickle=False) as z:
        new = z['indices']
    assert old.shape == new.shape == (3528, 256)
    rngs = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 25, i]))) for i in range(3)]
    queues = [{}, {}]
    def take(g, key, pool):
        if key not in queues[g] or not queues[g][key]:
            queues[g][key] = deque(pool[int(j)] for j in rngs[g].permutation(len(pool))[::-1])
        return queues[g][key].popleft()
    old_total = new_total = 0
    for step, (parent_row, row) in enumerate(zip(old, new)):
        slots = np.flatnonzero(d['y'][parent_row] >= 200)
        negatives = d['y'][parent_row] < 200
        assert len(slots) == 16
        np.testing.assert_array_equal(parent_row[negatives], row[negatives])
        assert (d['y'][row[slots]] >= 200).all()
        n_old = (11, 11, 10)[step % 3]
        plan = rngs[2].permutation(np.array([0] * n_old + [1] * (16 - n_old), dtype=np.int8))
        expected = []
        for group in plan:
            group = int(group)
            day = take(group, 'days', sorted(groups[group]))
            event = take(group, ('events', day), sorted(groups[group][day]))
            expected.append(take(group, ('centers', day, event), groups[group][day][event]))
        np.testing.assert_array_equal(row[slots], expected)
        old_total += int((d['house'][row[slots]] < 3).sum())
        new_total += int((d['house'][row[slots]] >= 3).sum())
    assert (old_total, new_total) == (37632, 18816)
    counts = np.bincount(new.ravel(), minlength=len(d['y']))
    with np.load(run / f'exposure_{seed}.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(counts, z['count'])
    for name, fields, mask in (
        ('house', ('house', 'on'), np.ones(len(counts), dtype=bool)),
        ('day', ('house', 'day', 'on'), np.ones(len(counts), dtype=bool)),
        ('fragment', ('house', 'day', 'event'), d['y'] >= 200),
    ):
        columns = {'house': d['house'], 'day': d['t'] // 86400,
                   'on': d['y'] >= 200, 'event': d['event']}
        keys = np.stack([columns[field][mask].astype(np.int64) for field in fields], axis=1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        centers = np.bincount(inverse)
        seen = np.bincount(inverse, weights=(counts[mask] > 0))
        shown = np.bincount(inverse, weights=counts[mask])
        expected = {tuple(map(int, key)): (int(centers[j]), int(seen[j]), int(shown[j]))
                    for j, key in enumerate(unique)}
        with (run / f'{name}_exposure_{seed}.csv').open(encoding='utf-8-sig', newline='') as f:
            actual = {}
            for row in csv.DictReader(f):
                key = tuple(int(row[k] == 'True') if k == 'on' else int(row[k]) for k in fields)
                assert key not in actual
                actual[key] = tuple(int(row[k]) for k in ('centers', 'unique_exposed', 'presentations'))
        assert expected == actual, f'{name} exposure differs for seed {seed}'
    results[seed] = {'old_positive': old_total, 'new_positive': new_total,
                     'negative_presentations': int(counts[d['y'] < 200].sum()),
                     'unique_centers': int((counts > 0).sum())}
print(json.dumps({'passed': True, 'seeds': results}))
