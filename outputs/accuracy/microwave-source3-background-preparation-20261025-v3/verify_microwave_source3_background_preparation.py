"""Independent metadata/small-pool audit; never loads household development data."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from virtual_submeter.data.common import read_json,sha256,write_json


def check(run):
    run=Path(run)
    registration=read_json(run/'registration.json')
    assert registration['authorized'] is False
    assert registration['actual_real_fits']==registration['actual_updates']==0
    for name,digest in registration['artifacts'].items(): assert sha256(run/name)==digest,name
    assert not any((run/name).exists() for name in ('training_started.json','models','source_selection.json','freeze.json','evaluation'))
    audit=read_json(run/'source_pool_audit.json'); identity=read_json(run/'identity.json')
    assert audit['feasible'] and audit['templates']==214 and audit['backgrounds']==268549
    rows=[]
    for house in audit['houses']:
        h=house['house']
        with np.load(run/f'audit_{h}.npz',allow_pickle=False) as z:
            assert int(z['template'].sum())==house['templates']
            assert int(z['background'].sum())==house['backgrounds']
            assert (z['known_count'][z['template']]==127).all()
            assert not (z['template_outside_full_role']&z['template']).any()
            assert not (z['background_observed_unknown_or_negative_label']&z['background']).any()
            assert not (z['background_invalid_residual']&z['background']).any()
        with np.load(run/f'template_{h}.npz',allow_pickle=False) as z:
            micro=z['micro_or_residual']; t=z['t']; role=z['role']; event=z['event']
            assert micro.shape==(house['templates'],127)
            assert np.isfinite(micro).all() and (micro>=0).all() and (micro[:,63]>=200).all()
            for block,low,high in identity['roles'][h]:
                take=role==block
                assert (t[take]-504>=low).all() and (t[take]+512<=high).all()
            rows.append(dict(house=h,template_windows=len(t),observable_template_event_ids=sorted(set(event.tolist())),
                template_event_count=len(set(event.tolist())),template_center_power_range=[float(micro[:,63].min()),float(micro[:,63].max())]))
    accesses=[]
    for path in run.glob('*access*.json'):
        a=read_json(path); assert not a['forbidden']
        for value in a['reads']:
            lower=value.lower().replace('\\','/')
            assert not any(term in lower for term in ('refit_17','house_17','truth_once','frozen_test','predictions/guard95_'))
        accesses.append(path.name)
    replay=read_json(run/'frozen_replay.json')
    assert not replay['source_x_reopened'] and not replay['raw_csv_reopened']
    assert all(r['frozen_indices_exact_replay'] and r['initial_control_exact'] and r['batches']==3528 for r in replay['seeds'].values())
    return dict(status='independent_preparation_verified',registration_sha256=sha256(run/'registration.json'),
        source_pool=dict(templates=214,backgrounds=268549,template_houses=3),template_support=rows,
        access_logs_checked=accesses,new_real_fits=0,real_updates=0,development_numerical_reads=0,final_house_reads=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--output',type=Path)
    args=p.parse_args(); result=check(args.run)
    if args.output:
        assert not args.output.exists(); write_json(args.output,result)
    print(result)
