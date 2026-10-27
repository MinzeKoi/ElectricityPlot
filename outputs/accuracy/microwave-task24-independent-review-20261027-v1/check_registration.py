"""Read-only registration hash check of the explicitly listed source inputs."""
import hashlib
import json
from pathlib import Path
import sys

run = Path(sys.argv[1]).resolve()
reg_path = run / 'registration.json'
reg = json.loads(reg_path.read_text(encoding='utf-8'))
assert reg['status'] == 'sealed_pending_review' and reg['authorized'] is False
assert reg['seeds'] == [42, 1019, 2026]
assert reg['maximum_new_fits'] == 3 and reg['maximum_updates'] == 10584
for group in ('artifacts', 'inputs'):
    for name, expected in reg[group].items():
        path = (run / name).resolve() if group == 'artifacts' else Path(name)
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == expected, str(path)
print(json.dumps({'registration_sha256': hashlib.sha256(reg_path.read_bytes()).hexdigest(),
                  'artifacts_checked': len(reg['artifacts']), 'inputs_checked': len(reg['inputs']),
                  'status': 'hashes_only_not_overall_acceptance'}))
