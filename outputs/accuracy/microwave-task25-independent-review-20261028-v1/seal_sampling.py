"""Seal the reviewed sampling draft, code, and logs."""
import hashlib
import json
from pathlib import Path

root = Path.cwd().resolve()
audit = root / 'outputs/accuracy/microwave-task25-independent-review-20261028-v1'
draft = root / 'outputs/accuracy/microwave-task25-preparation-20261028-v1'
parent = root / 'outputs/accuracy/microwave-task24-index-draft-20261027-v1'
files = [audit / n for n in ('independent_sampling.py', 'seal_sampling.py', 'sampling.stdout', 'sampling.stderr')]
files += [parent / 'source_index.npz'] + [parent / f'batches_{s}.npz' for s in (42, 1019, 2026)]
for seed in (42, 1019, 2026):
    files += [draft / f'{kind}_{seed}.{ext}' for kind, ext in
              (('batches','npz'), ('exposure','npz'), ('house_exposure','csv'), ('day_exposure','csv'), ('fragment_exposure','csv'))]
result = {'scope': 'sampling draft, not full implementation review', 'exit_code': 0,
          'files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
with (audit / 'sampling_receipt.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
