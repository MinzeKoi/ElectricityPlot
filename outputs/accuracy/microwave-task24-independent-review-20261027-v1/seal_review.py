"""Seal explicit review inputs and outputs without inspecting other experiment data."""
import hashlib
import json
from pathlib import Path

root = Path.cwd().resolve()
run = root / 'outputs/accuracy/microwave-task24-independent-review-20261027-v1'
index = root / 'outputs/accuracy/microwave-task24-index-draft-20261027-v1'
windows = root / 'outputs/accuracy/microwave-task24-windows-draft-20261027-v1'
task23 = root / 'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4'
paths = [run / n for n in ('task24_independent_review.py', 'task24_independent_windows.py', 'seal_review.py',
                          'index.stdout.log', 'index.stderr.log', 'windows.stdout.log', 'windows.stderr.log')]
paths += [index / n for n in ('source_index.npz', 'fragment_mapping.csv')]
for seed in (42, 1019, 2026):
    paths += [index / f'{prefix}_{seed}.{suffix}' for prefix, suffix in
              (('batches', 'npz'), ('exposure', 'npz'), ('house_exposure', 'csv'),
               ('day_exposure', 'csv'), ('fragment_exposure', 'csv'))]
paths += [windows / f'source_train_refit_{h}_{p}.npz' for h in (10, 12, 15, 18) for p in (0, 1)]
paths += [task23 / n for n in ('input_seal.json', 'window_ids.csv', 'independent_check.py')]
paths += [root / 'outputs/accuracy/microwave-source5-temporal-20261025-v2/train.npz']
entries = []
for path in paths:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    entries.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
receipt = {'scope': 'index and windows drafts only; overall preparation incomplete',
           'index_exit': 0, 'windows_exit': 0, 'input_files': entries,
           'limits': ['House10 processed inaccessible in this session',
                      'Raw-grid recomputation uses Task23 independent_check; same underlying data',
                      'Float32 windows checked rtol=1e-6 atol=1e-5',
                      'No training, candidate inference, development or final data reads',
                      'Initial copied script run failed because root path changed; corrected and rerun with exit 0'],
           'results': {'total_centers': 1393867, 'on_centers': 4389,
                       'new_windows': 1057163, 'seeds': [42, 1019, 2026]}}
with (run / 'receipt.json').open('x', encoding='utf-8') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
