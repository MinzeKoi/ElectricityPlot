"""Read-only verification of completed Task24 training and source decision."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/accuracy/microwave-task24-recovery-preparation-20261027-v1'
sys.path.insert(0, str(RUN / 'source'))
from virtual_submeter.accuracy.microwave_task21 import isolation


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    registration = read(RUN / 'registration.json')
    training = read(RUN / 'training_complete.json')
    source = read(RUN / 'source_complete.json')
    sources = registration['training_sources']
    paths = [RUN / n for n in ('registration.json', 'training_complete.json', 'source_selection.json', 'source_complete.json')]
    paths += [RUN / 'models' / n for n in training['artifacts']]
    paths += [RUN / 'source_results' / n for n in source['artifacts']]
    paths += [Path(sources['source_control'])]
    paths += [Path(v) for v in sources['batches'].values()]
    paths += [ROOT / f'outputs/accuracy/microwave-task24-index-draft-20261027-v1/exposure_{s}.npz' for s in (42, 1019, 2026)]
    with isolation(RUN, paths, readonly=True) as access:
        assert sha(RUN / 'registration.json') == '6d6754ee5e97ad69a619c107a2441b759048299cb2785214af01ada44b4961f7'
        assert training['new_fits'] == 3 and training['updates'] == 10584 and training['control_retrains'] == 0
        assert source['registration_sha256'] == sha(RUN / 'registration.json')
        assert source['source_sha256'] == sha(RUN / 'source_selection.json')
        for relative, digest in training['artifacts'].items():
            assert sha(RUN / 'models' / relative) == digest, relative
        for relative, digest in source['artifacts'].items():
            assert sha(RUN / 'source_results' / relative) == digest, relative
        ledger = []
        for seed in (42, 1019, 2026):
            model = RUN / f'models/unit_BCE/seed_{seed}'
            progress = read(model / 'progress.json')
            assert progress == {'completed_updates': 3528, 'attempted_updates': 3528}
            assert len(read(model / 'history.json')['rows']) == 12
            with np.load(sources['batches'][str(seed)], allow_pickle=False) as z:
                batches = z['indices']
            assert batches.shape == (3528, 256)
            with (model / 'batches.csv').open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 3528
            for i, (row, batch) in enumerate(zip(rows, batches), 1):
                assert (int(row['update']), int(row['positive']), int(row['negative'])) == (i, 16, 240)
                assert row['indices_sha256'] == hashlib.sha256(np.ascontiguousarray(batch).tobytes()).hexdigest()
            with np.load(model / 'exposure.npz', allow_pickle=False) as actual, np.load(ROOT / f'outputs/accuracy/microwave-task24-index-draft-20261027-v1/exposure_{seed}.npz', allow_pickle=False) as expected:
                np.testing.assert_array_equal(actual['count'], expected['count'])
                assert int(actual['count'].sum()) == 903168
            ledger.append({'seed': seed, 'attempted_updates': 3528, 'completed_updates': 3528,
                           'checkpoints': 12, 'positive_presentations': 56448, 'negative_presentations': 846720})
        selection = read(RUN / 'source_selection.json')
        control = read(Path(sources['source_control']))['rows']['guard95']
        candidates = selection['rows']
        assert selection['source_only'] and selection['selected'] == 'control_guard95' and selection['eligible'] is False
        assert selection['training_sha256'] == sha(RUN / 'training_complete.json')
        assert [v['seed'] for v in candidates] == [42, 1019, 2026]
        assert all(r['feasible'] and all(b['recall'] >= .95 and not b['degenerate'] for b in r['blocks']) for r in selection['recipes'])
        mean = lambda rows, key: float(np.mean([r[key] for r in rows]))
        improvements = sum(b['precision'] > a['precision'] for a, b in zip(control, candidates))
        relative = {'candidate_mean_precision': mean(candidates, 'precision'), 'control_mean_precision': mean(control, 'precision'),
            'candidate_mean_f1': mean(candidates, 'f1'), 'control_mean_f1': mean(control, 'f1'), 'precision_improved_seeds': improvements}
        assert improvements == 1 and relative['candidate_mean_f1'] < relative['control_mean_f1']
        assert relative['candidate_mean_precision'] < relative['control_mean_precision']
        assert not any((RUN / p).exists() for p in ('future_identity.json', 'predictions', 'evaluation'))
        assert not access['forbidden']
    return {'status': 'source_rejected_control_retained', 'prior_v4_entry_attempts': 1, 'prior_v4_fits': 0,
            'recovery_fits': 3, 'recovery_updates': 10584, 'control_retrains': 0, 'ledger': ledger,
            'source_relative_comparison': relative, 'source_selection_sha256': source['source_sha256'],
            'training_complete_sha256': sha(RUN / 'training_complete.json'),
            'development_stage_called': False, 'development_artifacts_present': False,
            'scope_note': 'Execution path, logs and absent artifacts only; not OS-level proof of zero reads.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'report'])
    args = parser.parse_args()
    result = verify()
    if args.command == 'report':
        result['report_note'] = 'Source rejected; retain control_guard95 and stop before development.'
    print(json.dumps(result, ensure_ascii=False, indent=2))
