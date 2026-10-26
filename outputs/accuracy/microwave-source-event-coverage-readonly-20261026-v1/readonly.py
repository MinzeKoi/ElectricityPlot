"""Repeatable read-only Task17 snapshot verification/report. Writes no files.

Usage: python -B readonly.py verify | report [--link]
Uses sealed numeric summaries; never rescans or interprets raw household data.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'microwave-source-event-coverage-20261026-v1'
ROOT = RUN.parents[2]
MANIFEST_SHA256 = 'ee8f7149bba453f3f83598b4a5d7b13e7ef59c8309ac971cb393b1540decaa98'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(name):
    with (RUN / name).open(encoding='utf-8') as stream:
        return json.load(stream)

def rows(name):
    with (RUN / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def snapshot():
    # Only this exact sealed artifact directory is enumerated, never data directories.
    return {p.name: {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns,
                     'size': p.stat().st_size}
            for p in sorted(RUN.iterdir()) if p.is_file()}

def verify():
    require(sha(RUN / 'delivery_manifest.json') == MANIFEST_SHA256, 'Manifest identity changed')
    manifest = read('delivery_manifest.json')
    expected_files = set(manifest['files']) | {'delivery_manifest.json'}
    require({p.name for p in RUN.iterdir() if p.is_file()} == expected_files, 'Sealed file inventory changed')
    for name, expected in manifest['files'].items():
        require(Path(name).name == name, 'Invalid manifest filename')
        require(sha(RUN / name) == expected, 'Sealed hash changed: ' + name)
    for relative, expected in manifest['source_dependency_hashes'].items():
        dependency = (ROOT / relative).resolve()
        require(dependency.is_relative_to(ROOT / 'src'), 'Dependency outside source directory')
        require(sha(dependency) == expected, 'Source dependency changed')
    summary = read('summary.json')
    independent = read('independent_verification.json')
    cfg = read('config.json')
    proposal = read('followup_proposal.disabled.json')
    require(independent['status'] == 'passed', 'Independent numerical verification did not pass')
    require(cfg['allowed_numeric_houses'] == [2, 19] and cfg['training_interfaces'] == [], 'Scope changed')
    require(not proposal['authorized'] and not proposal['training_enabled'], 'Proposal enabled')
    require(all(summary[k] == 0 for k in ('models_fit', 'optimizer_steps', 'inference', 'calibration',
                'development_evaluations', 'house17_reads', 'final_house_reads')), 'Audit-only boundary changed')
    daily = rows('daily_counts.csv')
    events = rows('segment_details.csv')
    checks = []
    require(len(summary['houses']) == len(independent['checks']) == 4, 'Expected four source/role checks')
    require({r['house'] for r in daily + events} == {'2', '19'}, 'Unexpected household in snapshot')
    for item in summary['houses']:
        h, role = item['house'], item['role']
        matches = [r for r in independent['checks'] if (r['house'], r['role']) == (h, role)]
        require(len(matches) == 1 and matches[0]['passed'], 'Missing independent check')
        for field, value in matches[0]['independent_counts'].items():
            require(item[field] == value, f'Independent count differs: {h}/{role}/{field}')
        days = [r for r in daily if (r['house'], r['role']) == (h, role)]
        segments = [r for r in events if (r['house'], r['role']) == (h, role)]
        for field in ('aggregate_known', 'label_known', 'positive_label_bins',
                      'eligible_centers', 'eligible_positive_centers'):
            require(sum(int(r[field]) for r in days) == item[field], 'Daily count differs: ' + field)
        require(sum(int(r['observable_segments_starting_date']) for r in days) == item['observable_segments'], 'Daily segment count differs')
        require(len(segments) == item['observable_segments'], 'Segment detail count differs')
        require(sum(int(r['samples']) for r in segments) == item['positive_label_bins'], 'Segment sample total differs')
        require(sum(int(r['eligible_positive_centers']) for r in segments) == item['eligible_positive_centers'], 'Segment positive total differs')
        require(sum(int(r['eligible_positive_centers']) > 0 for r in segments) == item['supported_segments'], 'Supported segments differ')
        require(sum(r['full_eligible'] == 'True' for r in segments) == item['fully_supported_complete_segments'], 'Complete segments differ')
        checks.append({'house': h, 'role': role, 'passed': True})
    candidate = [r for r in summary['houses'] if r['role'] == 'candidate_audit_only_not_training']
    require(sum(r['eligible_centers'] for r in candidate) == 279218, 'Addition count changed')
    require(sum(r['eligible_positive_centers'] for r in candidate) == 719, 'Addition positives changed')
    require(sum(r['supported_segments'] for r in candidate) == 160, 'Addition segments changed')
    return {'status': 'passed', 'mode': 'sealed_snapshot_readonly', 'sealed_files_verified': len(expected_files),
            'checks': checks, 'raw_data_rescanned': False, 'files_written': 0,
            'limits': 'Reconciles sealed independent verification, daily and segment counts; does not rerun raw numerical audit.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('verify', 'report'))
    parser.add_argument('--link', action='store_true', help='report: print only an absolute Markdown report link')
    args = parser.parse_args()
    before = snapshot()
    if args.stage == 'verify':
        result = verify()
        require(snapshot() == before, 'Sealed hash/mtime/size changed during verification')
        print(json.dumps(result, ensure_ascii=False))
    else:
        require(sha(RUN / 'delivery_manifest.json') == MANIFEST_SHA256, 'Manifest identity changed')
        require(sha(RUN / 'REPORT.zh.md') == read('delivery_manifest.json')['files']['REPORT.zh.md'], 'Report hash changed')
        text = ('[Task17 中文审计报告](<' + (RUN / 'REPORT.zh.md').as_posix() + '>)') if args.link else (RUN / 'REPORT.zh.md').read_text(encoding='utf-8')
        require(snapshot() == before, 'Sealed hash/mtime/size changed during report read')
        print(text)

if __name__ == '__main__':
    main()
