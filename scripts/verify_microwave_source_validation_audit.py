"""Verify audit receipts and time/support closure; never opens source or target labels."""
import argparse
import hashlib
import json
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'outputs/accuracy/microwave-source-validation-audit-20261025-v4')
    args = p.parse_args()
    run = args.run.resolve()
    receipt = json.loads((run/'receipt.json').read_text(encoding='utf-8'))
    for name, digest in receipt['artifacts'].items():
        assert Path(name).name == name
        assert hashlib.sha256((run/name).read_bytes()).hexdigest() == digest
    script = ROOT/'scripts/audit_microwave_source_validation.py'
    assert hashlib.sha256(script.read_bytes()).hexdigest() == receipt['script_sha256']
    spec = importlib.util.spec_from_file_location('source_validation_audit', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    d = json.loads((run/'audit.json').read_text(encoding='utf-8'))
    proposal = json.loads((run/'proposal.disabled.json').read_text(encoding='utf-8'))
    assert proposal == d['proposal']
    assert proposal['disabled'] and not proposal['house5_training_authorized']
    assert not proposal['compatible_with_original_cross_household_protocol']
    assert proposal['default_protocol_guard_must_remain_unchanged']
    assert len(proposal['required_human_approved_exceptions']) == 2
    assert d['fits'] == 0 and not d['house17_data_read'] and not d['final_house_data_read']
    assert not d['data_roles_changed'] and not d['house5_training_arrays_created']
    assert all(d['checks'].values())
    assert all(x['path'] in module.ALLOWED for x in d['access_log'])
    assert not any('/evaluation/' in x['path'] or '/predictions/' in x['path'] for x in d['access_log'])
    rows = d['statistics']
    for h in module.SOURCE:
        total = next(r for r in rows if r['house']==h and r['block']=='original_14d')
        daily = [r for r in rows if r['house']==h and r['block'].startswith('UTC_')]
        assert sum(r['grid_points'] for r in daily)==total['grid_points']==151200
        assert sum(r['observed_on_centers'] for r in daily)==total['observed_on_centers']
    for r in rows:
        assert module.stamp(r['end'])-module.stamp(r['start']) == r['grid_points']*8
        assert sum(r['on_power_bins'].values())==r['observed_on_centers']
        for mode in ('strict','partial'):
            s=r[mode]
            assert s['positive_centers']+s['negative_centers']==s['windows']
            if s['windows']:
                assert module.stamp(r[f'first_{mode}_center'])-120>=module.stamp(r['start'])
                assert module.stamp(r[f'last_{mode}_center'])+128<=module.stamp(r['end'])
    blocks = proposal['house5_training_block_intervals_after_approval_only']
    assert blocks[0]['start']=='2014-11-15T00:00:00Z'
    assert blocks[-1]['end']=='2014-11-22T23:30:00Z'
    assert proposal['retained_validation_blocks'][0]['start']=='2014-11-23T00:00:00Z'
    assert module.stamp(proposal['retained_validation_blocks'][0]['start'])-module.stamp(blocks[-1]['end'])==1800
    assert proposal['importance_reference_prior']['value']==849/259806
    print(json.dumps({'verified':True,'artifact_hashes':len(receipt['artifacts']),
                      'source_input_allowlist_entries':len(d['access_log']),
                      'statistic_rows':len(rows),'fits':0,'source_or_target_label_reads_by_verifier':0},indent=2))


if __name__=='__main__':
    main()
