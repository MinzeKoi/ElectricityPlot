"""Audit sealed source scores at the fixed thresholds; no candidate inference."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path.cwd().resolve()
run = root / 'outputs/accuracy/microwave-task24-recovery-preparation-20261027-v1'
reg = json.loads((run / 'registration.json').read_text(encoding='utf-8'))
selection_path = run / 'source_selection.json'
selection = json.loads(selection_path.read_text(encoding='utf-8'))
seal = json.loads((run / 'source_complete.json').read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert seal['source_sha256'] == sha(selection_path)
assert seal['registration_sha256'] == sha(run / 'registration.json')
assert selection['training_sha256'] == sha(run / 'training_complete.json')
for name, digest in seal['artifacts'].items():
    assert sha(run / 'source_results' / name) == digest
assert not any((run / name).exists() for name in ('future_identity.json', 'predictions', 'evaluation', 'evaluation_complete.json'))
control = json.loads(Path(reg['training_sources']['source_control']).read_text(encoding='utf-8'))['rows']['guard95']
actual = []
for recipe, saved in zip(selection['recipes'], selection['rows']):
    seed = recipe['seed']
    assert json.loads((run / f'source_results/recipe_{seed}.json').read_text(encoding='utf-8')) == recipe
    assert recipe['feasible'] and recipe['radius_samples'] == 15 and recipe['interval_seconds'] == 8
    metrics = []
    for block, row in zip((4, 5), recipe['blocks']):
        with np.load(run / f'source_results/scores_{seed}_{block}.npz', allow_pickle=False) as z:
            t, p, h, ix, y = (z[k] for k in ('t', 'p', 'h', 'ix', 'y'))
        with np.load(reg['training_sources']['validation'][f'block_{block}'], allow_pickle=False) as z:
            np.testing.assert_array_equal(t[ix], z['t'])
            np.testing.assert_array_equal(y, z['y'] >= 200)
        expected = np.full(len(p), np.nan, dtype=p.dtype)
        finite = np.isfinite(p)
        breaks = np.r_[True, (~finite[1:]) | (~finite[:-1]) | (np.diff(t) != 8)]
        starts = np.flatnonzero(finite & breaks)
        for start in starts:
            end = start + 1
            while end < len(p) and finite[end] and t[end] - t[end - 1] == 8:
                end += 1
            for i in range(start, end):
                expected[i] = np.max(p[max(start, i-15):min(end, i+16)])
        np.testing.assert_array_equal(h, expected)
        predicted = (p[ix] >= recipe['L']) & (expected[ix] >= recipe['H'])
        tp = int((predicted & y).sum()); fp = int((predicted & ~y).sum())
        fn = int((~predicted & y).sum()); tn = int((~predicted & ~y).sum())
        assert (tp, fp, fn, tn) == (row['tp'], row['fp'], row['fn'], row['tn'])
        result = {'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
                  'f1': 2 * tp / max(2 * tp + fp + fn, 1)}
        assert all(abs(result[k] - row[k]) < 1e-14 for k in result)
        assert result['recall'] >= .95 and predicted.any() and not predicted.all()
        metrics.append(result)
    average = {k: float(np.mean([r[k] for r in metrics])) for k in ('precision', 'recall', 'f1')}
    assert all(abs(average[k] - saved[k]) < 1e-14 for k in average)
    actual.append({'seed': seed, **average})
assert [x['seed'] for x in actual] == [x['seed'] for x in control] == [42, 1019, 2026]
mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
relative = {'mean_f1_not_lower': mean(actual, 'f1') >= mean(control, 'f1'),
            'each_recall_drop_at_most_005': all(b['recall'] - a['recall'] >= -.05 for a,b in zip(control, actual)),
            'two_precision_improvements': sum(b['precision'] > a['precision'] for a,b in zip(control, actual)) >= 2}
eligible = all(relative.values())
candidate_wins = (mean(actual, 'precision'), mean(actual, 'f1')) > (mean(control, 'precision'), mean(control, 'f1'))
assert selection['eligible'] == (eligible and candidate_wins)
assert selection['selected'] == ('unit_BCE' if eligible and candidate_wins else 'control_guard95')
print(json.dumps({'passed': True, 'source_sha256': sha(selection_path), 'selected': selection['selected'],
                  'candidate': actual, 'control': control, 'relative': relative,
                  'candidate_means': {k: mean(actual, k) for k in ('precision', 'recall', 'f1')},
                  'control_means': {k: mean(control, k) for k in ('precision', 'recall', 'f1')},
                  'candidate_wins_ranking': candidate_wins, 'development_artifacts_absent': True}))
