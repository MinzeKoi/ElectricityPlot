"""Read-only verification of a Task24 registration and its source identities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))
from virtual_submeter.accuracy.microwave_task21 import isolation


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(run):
    run = Path(run).resolve()
    registration = run / 'registration.json'
    reg = json.loads(registration.read_text(encoding='utf-8'))
    paths = [registration] + [Path(p) for p in reg['inputs']] + [run / p for p in reg['artifacts']]
    with isolation(run, paths, readonly=True) as access:
        assert reg['schema'] == 'microwave_task24_multihouse_v1' and reg['status'] == 'sealed_pending_review'
        assert reg['authorized'] is False and reg['seeds'] == [42, 1019, 2026]
        assert (reg['maximum_new_fits'], reg['maximum_updates']) == (3, 10584)
        assert reg['counts'] == {'centers': 1393867, 'positive': 4389}
        for name, digest in reg['inputs'].items():
            assert sha(Path(name)) == digest, name
        for name, digest in reg['artifacts'].items():
            assert sha(run / name) == digest, name
        sources = reg['training_sources']
        required = [sources[k] for k in ('old_pool', 'index', 'plan', 'normalization', 'source_control', 'source5_aggregate')]
        required += sources['window_shards'] + list(sources['validation'].values()) + list(sources['batches'].values()) + list(sources['initial'].values())
        assert all(str(Path(p).resolve()) in reg['inputs'] for p in required)
        assert len(sources['window_shards']) == 8 and set(sources['validation']) == {'block_4', 'block_5'}
        assert set(sources['batches']) == set(sources['initial']) == {'42', '1019', '2026'}
        assert not any((run / p).exists() for p in ('training_started.json', 'training_complete.json', 'models', 'source_selection.json', 'future_identity.json', 'predictions'))
        assert not access['forbidden']
    return {'status': 'verified_pending_independent_reviews_and_budget', 'registration_sha256': sha(registration),
            'inputs_verified': len(reg['inputs']), 'artifacts_verified': len(reg['artifacts']),
            'centers': 1393867, 'positive_centers': 4389, 'new_fits': 0, 'updates': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify', 'report'])
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.run)
    if args.command == 'report':
        result['report'] = (args.run / 'MICROWAVE_TASK24_PREPARATION_DRAFT_20261027.md').read_text(encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
