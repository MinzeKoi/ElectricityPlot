"""Bind pre-existing source files and replay fixed batch identities without fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
parent = root / 'outputs/accuracy/microwave-source5-temporal-20261025-v2'
comparison = root / 'outputs/accuracy/microwave-task22-recovery-preparation-20261027-v3'
registration = json.loads((parent / 'registration.json').read_text(encoding='utf-8-sig'))['artifacts']
names = ['train.npz', 'normalization.json', 'plan.json', 'validation/block_4.npz', 'validation/block_5.npz']
names += [f'{kind}_{seed}.{ext}' for seed in (42, 1019, 2026) for kind, ext in [('sampling', 'npz'), ('initial', 'pt')]]
result = {}
for name in names:
    path = parent / name
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != registration[name]:
        raise ValueError('source identity mismatch: ' + name)
    result[name] = digest
control = root / 'outputs/accuracy/microwave-source5-loss-preparation-20261025-v4'
control_manifest = json.loads((control / 'final_delivery_manifest.json').read_text(encoding='utf-8-sig'))['artifacts']
comparison_manifest = json.loads((comparison / 'registration.json').read_text(encoding='utf-8-sig'))['artifacts']
for seed in (42, 1019, 2026):
    control_name = f'models/unit_BCE/seed_{seed}/initial.pt'
    with (control / control_name).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != control_manifest[control_name] or digest != result[f'initial_{seed}.pt']:
        raise ValueError('control initialization mismatch: ' + str(seed))
    batch_name = f'batches_{seed}.npz'
    with (comparison / batch_name).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != comparison_manifest[batch_name]:
        raise ValueError('comparison batch identity mismatch: ' + str(seed))
    result['control/' + control_name] = control_manifest[control_name]
    result['comparison/' + batch_name] = digest
    with np.load(parent / f'sampling_{seed}.npz', allow_pickle=False) as a, np.load(comparison / f'batches_{seed}.npz', allow_pickle=False) as b:
        original, replay = a['indices'], b['indices']
        if original.shape != (3528, 256) or not np.array_equal(original, replay):
            raise ValueError('batch replay mismatch: ' + str(seed))
output = {'parent': str(parent), 'sha256': result, 'replay': 'all 3528 x 256 indices per seed exact against Task22 replay', 'new_fits': 0, 'updates': 0}
path = Path(__file__).parent / 'source_bindings_v2.json'
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
