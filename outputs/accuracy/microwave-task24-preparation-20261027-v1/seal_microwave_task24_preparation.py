"""Create a new, budget-disabled Task24 registration for independent review."""
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from virtual_submeter.accuracy.microwave_task21 import isolation
from virtual_submeter.data.common import inventory
from prepare_microwave_task24_index import sha

OLD = ROOT / 'outputs/accuracy/microwave-source5-temporal-20261025-v2'
CONTROL = ROOT / 'outputs/accuracy/microwave-source5-loss-preparation-20261025-v4'
GUARD = ROOT / 'outputs/accuracy/microwave-recall-guard95-preparation-20261025-v2'
INDEX = ROOT / 'outputs/accuracy/microwave-task24-index-draft-20261027-v1'
WINDOWS = ROOT / 'outputs/accuracy/microwave-task24-windows-draft-20261027-v1'
PROOF = ROOT / 'outputs/accuracy/microwave-task24-pool-proof-draft-20261027-v2'
AUDIT = ROOT / 'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4'
REVIEW = ROOT / 'outputs/accuracy/microwave-task24-independent-review-20261027-v1'
RUN = ROOT / 'outputs/accuracy/microwave-task24-preparation-20261027-v1'


def main():
    assert not RUN.exists(), 'Preparation directories are never overwritten'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    oldreg = read(OLD / 'registration.json')['artifacts']
    source5 = read(OLD / 'source5_inputs.json')
    ws = read(WINDOWS / 'window_summary.json')
    pool = read(PROOF / 'pool_proof.json')
    guard_manifest = read(GUARD / 'final_delivery_manifest.json')['artifacts']
    identity_hash = guard_manifest['identity.json']
    if isinstance(identity_hash, dict):
        identity_hash = identity_hash['sha256']
    shards = [WINDOWS / f'{r["partition"]}.npz' for r in ws['outputs']]
    batches = {str(s): INDEX / f'batches_{s}.npz' for s in (42, 1019, 2026)}
    initials = {str(s): OLD / f'initial_{s}.pt' for s in (42, 1019, 2026)}
    validation = {f'block_{b}': OLD / f'validation/block_{b}.npz' for b in (4, 5)}
    sources = {'old_pool': OLD / 'train.npz', 'index': INDEX / 'source_index.npz', 'window_shards': shards,
        'plan': OLD / 'plan.json', 'normalization': OLD / 'normalization.json', 'validation': validation,
        'batches': batches, 'initial': initials, 'source_control': GUARD / 'source_selection.json',
        'source5_aggregate': Path(source5['aggregate']['path'])}
    inputs = [OLD / n for n in ('registration.json', 'source5_inputs.json', 'train.npz', 'plan.json', 'normalization.json')]
    inputs += list(validation.values()) + list(initials.values()) + list(batches.values()) + shards
    inputs += [INDEX / 'source_index.npz', INDEX / 'fragment_mapping.csv', INDEX / 'draft_summary.json',
               INDEX / 'input_identity.json', WINDOWS / 'window_summary.json', PROOF / 'pool_proof.json',
               AUDIT / 'delivery_seal.json', AUDIT / 'input_seal.json', AUDIT / 'window_ids.csv',
               GUARD / 'source_selection.json', GUARD / 'final_delivery_manifest.json', CONTROL / 'final_delivery_manifest.json',
               Path(source5['aggregate']['path']), REVIEW / 'receipt.json']
    inputs += [INDEX / f'{prefix}_{s}.{ext}' for s in (42, 1019, 2026)
               for prefix, ext in (('exposure', 'npz'), ('house_exposure', 'csv'), ('day_exposure', 'csv'), ('fragment_exposure', 'csv'))]
    inputs += [CONTROL / f'models/unit_BCE/seed_{s}/{n}' for s in (42, 1019, 2026) for n in ('model.json', 'weights.pt', 'initial.pt')]
    seal = read(AUDIT / 'input_seal.json')
    inputs += [ROOT / seal['houses'][h]['paths']['slice'] for h in ('10', '12', '15', '18')]
    inputs += [ROOT / path for path in pool['input_sha256']]
    inputs = sorted(set(inputs))
    package = sorted((ROOT / 'src/virtual_submeter').rglob('*.py'))
    scripts = [ROOT / 'scripts/run_microwave_task24_draft.py', Path(__file__).resolve()]
    tests = [ROOT / 'tests' / f'test_microwave_task24_{name}.py' for name in ('budget_draft', 'source_draft', 'acceptance')]
    docs = [ROOT / 'docs/MICROWAVE_TASK24_PROPOSAL_20261027.md', ROOT / 'docs/MICROWAVE_TASK24_PREPARATION_DRAFT_20261027.md']
    log = ROOT / 'tmp/task24_nontraining_tests.log'
    paths = sorted(set(inputs + package + scripts + tests + docs + [log]))
    with isolation(RUN, paths) as access:
        for path in inputs:
            if path.is_relative_to(OLD) and path != OLD / 'registration.json':
                assert sha(path) == oldreg[path.relative_to(OLD).as_posix()]
        assert sha(Path(source5['aggregate']['path'])) == source5['aggregate']['sha256']
        for row, path in zip(ws['outputs'], shards):
            assert sha(path) == row['sha256']
        for path, digest in pool['input_sha256'].items():
            assert sha(ROOT / path) == digest, path
        RUN.mkdir(parents=True)
        for path in package:
            target = RUN / 'source' / path.relative_to(ROOT / 'src')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for path in scripts + docs + [log]:
            shutil.copyfile(path, RUN / path.name)
        for path in tests:
            target = RUN / 'tests' / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
        (RUN / 'workspace_identity.json').write_text(json.dumps({'root': str(ROOT)}) + '\n', encoding='utf-8')
        def strings(obj):
            if isinstance(obj, Path): return str(obj.resolve())
            if isinstance(obj, dict): return {k: strings(v) for k, v in obj.items()}
            if isinstance(obj, list): return [strings(v) for v in obj]
            return obj
        registration = {'schema': 'microwave_task24_multihouse_v1', 'status': 'sealed_pending_review', 'authorized': False,
            'seeds': [42, 1019, 2026], 'maximum_new_fits': 3, 'maximum_updates': 10584,
            'training_sources': strings(sources), 'future_identity_source': {'path': str((GUARD / 'identity.json').resolve()), 'sha256': identity_hash},
            'inputs': {str(p.resolve()): sha(p) for p in inputs}, 'artifacts': inventory(RUN),
            'counts': {'centers': 1393867, 'positive': 4389},
            'preparation': {'actual_fits': 0, 'updates': 0, 'backward': 0, 'candidate_predictions': 0,
                'development_or_final_numerical_reads': 0, 'source_window_origin': 'Task23 audited raw slices'}}
        (RUN / 'registration.json').write_text(json.dumps(registration, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        assert not access['forbidden']
    print(json.dumps({'status': 'sealed_pending_review', 'registration_sha256': sha(RUN / 'registration.json'),
                      'new_fits': 0, 'updates': 0}, indent=2))


if __name__ == '__main__':
    main()
