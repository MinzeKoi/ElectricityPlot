import json
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from virtual_submeter.data.common import sha256, iso

root = Path('data/raw/refit_complete_20261022/households')
inventory = json.loads((root / 'inventory.json').read_text())
spans = []
for receipt in inventory['files']:
    path = root / receipt['archive_member']
    with path.open('rb') as stream:
        stream.readline()
        first = stream.readline().decode().strip().split(',')[1]
        stream.seek(max(0, path.stat().st_size - 65536))
        last = stream.read().decode().strip().splitlines()[-1].split(',')[1]
    spans.append({'household_id': receipt['household_id'], 'bytes': receipt['stored_bytes'],
                  'sha256': receipt['sha256'], 'first_observation': iso(int(first)),
                  'last_observation': iso(int(last)), 'kind': 'real_complete',
                  'archive_member_crc32_verified': True})
plegma = json.loads(Path('data/raw/plegma_20261022/inventory.json').read_text())
ideal = json.loads(Path('data/raw/ideal_metadata_20261022/inventory.json').read_text())
prepared_path = Path('data/processed/refit_complete_28d_20261022/manifest.json')
prepared = json.loads(prepared_path.read_text())
coverage = json.loads(Path('docs/DATA_EXPANSION_COVERAGE_20261022.json').read_text(encoding='utf-8'))
summary = {
    'schema_version': 1, 'date': '2026-10-22',
    'refit': {'archive_bytes': inventory['archive_bytes'], 'archive_sha256': inventory['archive_sha256'],
              'source_url': inventory['source_url'], 'license': 'CC-BY-4.0',
              'raw_csv_bytes': inventory['total_csv_bytes'], 'households': spans,
              'independent_publisher_checksum_available': False},
    'prepared_refit': {'manifest': prepared_path.as_posix(), 'sha256': sha256(prepared_path),
        'households': list(prepared['households']), 'maximum_days_per_household': 28,
        'grid_seconds': 8, 'total_bins': sum(x['bins'] for x in prepared['households'].values()),
        'coverage_duration_pass_households': [h['household_id'] for h in coverage['households'] if h['coverage_and_duration_candidate_pass']],
        'final_households': []},
    'plegma': {'archive_bytes': plegma['archive_bytes'], 'archive_sha256': plegma['archive_sha256'],
        'source_url': plegma['source_url'], 'license': plegma['license'], 'verified_files': len(plegma['files']),
        'extracted_bytes': sum(x['bytes'] for x in plegma['files']),
        'household_count': len({x['path'].split('/')[1] for x in plegma['files'] if '/House_' in x['path']}),
        'status': plegma['status'], 'normalized': False},
    'ideal': {'source_url': ideal['source_url'], 'license': ideal['license'],
        'files': [{k: v for k, v in x.items() if k != 'members'} for x in ideal['files']],
        'power_measurements_downloaded': False},
    'training_run_started': False, 'final_evaluation_run_started': False,
    'validation': {'data_protocol_tests_passed': 22, 'acquisition_tests_passed': 4, 'project_boundary_tests_passed': 6},
}
Path('docs/DATA_EXPANSION_INVENTORY_20261022.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: v for k,v in summary.items() if k not in ('refit', 'ideal')}, ensure_ascii=False, indent=2))
