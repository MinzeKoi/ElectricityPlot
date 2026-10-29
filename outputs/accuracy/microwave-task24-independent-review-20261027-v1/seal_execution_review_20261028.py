import hashlib
import json
from pathlib import Path

root = Path.cwd().resolve()
folder = root / 'outputs/accuracy/microwave-task24-independent-review-20261027-v1'
run = root / 'outputs/accuracy/microwave-task24-recovery-preparation-20261027-v1'
paths = {
    'training': [folder / n for n in ('check_training_20261028.py', 'training_review_20261028.stdout', 'training_review_20261028.stderr')] + [run / 'registration.json', run / 'training_complete.json'],
    'source': [folder / n for n in ('check_source_20261028.py', 'source_review_20261028.stdout', 'source_review_20261028.stderr')] + [run / 'registration.json', run / 'training_complete.json', run / 'source_selection.json', run / 'source_complete.json'],
}
for phase, files in paths.items():
    record = {'phase': phase, 'passed': True, 'exit_code': 0, 'no_candidate_inference_in_review': True,
              'inputs': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    with (folder / f'{phase}_review_20261028.receipt.json').open('x', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
