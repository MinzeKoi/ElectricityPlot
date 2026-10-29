"""Read-only accounting audit of completed training; no model inference."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

root = Path.cwd().resolve()
run = root / 'outputs/accuracy/microwave-task24-recovery-preparation-20261027-v1'
reg = json.loads((run / 'registration.json').read_text(encoding='utf-8'))
complete_path = run / 'training_complete.json'
complete = json.loads(complete_path.read_text(encoding='utf-8'))
assert (complete['new_fits'], complete['updates'], complete['control_retrains']) == (3, 10584, 0)
assert [r['seed'] for r in complete['rows']] == [42, 1019, 2026]
for name, digest in complete['artifacts'].items():
    path = run / 'models' / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
out = {}
for seed in (42, 1019, 2026):
    model = run / f'models/unit_BCE/seed_{seed}'
    started = json.loads((model / 'started.json').read_text(encoding='utf-8'))
    progress = json.loads((model / 'progress.json').read_text(encoding='utf-8'))
    done = json.loads((model / 'complete.json').read_text(encoding='utf-8'))
    assert started['seed'] == seed and started['maximum_updates'] == 3528
    assert progress == {'completed_updates': 3528, 'attempted_updates': 3528}
    assert done['updates'] == 3528
    for name, digest in done['artifacts'].items():
        assert hashlib.sha256((model / name).read_bytes()).hexdigest() == digest
    with np.load(reg['training_sources']['batches'][str(seed)], allow_pickle=False) as z:
        batches = z['indices']
    with (model / 'batches.csv').open(encoding='utf-8', newline='') as f:
        records = list(csv.DictReader(f))
    assert len(records) == 3528
    for step, (row, indices) in enumerate(zip(records, batches), 1):
        assert int(row['update']) == step and int(row['positive']) == 16 and int(row['negative']) == 240
        assert row['indices_sha256'] == hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()
    with np.load(model / 'exposure.npz', allow_pickle=False) as z:
        exposure = z['count']
    np.testing.assert_array_equal(exposure, np.bincount(batches.ravel(), minlength=len(exposure)))
    original = torch.load(reg['training_sources']['initial'][str(seed)], map_location='cpu', weights_only=True)
    initial = torch.load(model / 'initial.pt', map_location='cpu', weights_only=True)
    assert original.keys() == initial.keys() and all(torch.equal(original[k], initial[k]) for k in original)
    assert len(list(model.glob('epoch_*.pt'))) == 12
    assert len(json.loads((model / 'history.json').read_text(encoding='utf-8'))['rows']) == 12
    out[seed] = {'updates': 3528, 'batch_records': len(records), 'presentations': int(exposure.sum()),
                 'unique_centers': int((exposure > 0).sum())}
print(json.dumps({'passed': True, 'training_sha256': hashlib.sha256(complete_path.read_bytes()).hexdigest(),
                  'sealed_model_files': len(complete['artifacts']), 'seeds': out}))
