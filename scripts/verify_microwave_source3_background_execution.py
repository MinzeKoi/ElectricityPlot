"""Public CLI readback before truth; independent numbers after one evaluation."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np


def main(run,authorization,mode):
    sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_source3_background_execution as e
    e.authorize(run,authorization)
    if mode=='source':
        with e.phase(run,'independent_source_final'):
            e.training_guard(run)
            s=e.source_guard(run)
            independent=e.verify_source_numerics(run,s)
            e.require(not s['aug_eligible'],'This terminal mode is for source rejection only')
            e.require(not any((run/p).exists() for p in ('predictions','freeze.json','evaluation','ordinary_pretruth_seal.json')),
                      'Source rejection must not create development artifacts')
            exposures=[]
            for seed in e.SEEDS:
                model=run/f'models/aug/seed_{seed}'
                arrays=e.gate.npz(model/'augmentation_exposure.npz',['template_count','background_count'])
                with (model/'augmentation_pairs.csv').open(encoding='utf-8',newline='') as stream: pairs=list(csv.DictReader(stream))
                e.require(len(pairs)==28224,'Incorrect synthetic slot exposure')
                indices=e.gate.npz(e.audit.PARENT/f'sampling_{seed}.npz',['indices'])['indices']
                with np.load(e.audit.PARENT/'train.npz',allow_pickle=False) as z: labels=z['y']
                for update in range(3528):
                    rows=pairs[update*8:(update+1)*8]
                    expected=np.flatnonzero(labels[indices[update]]>=200)[:8]
                    np.testing.assert_array_equal([int(r['slot']) for r in rows],expected)
                    e.require(all(int(r['update'])==update and int(r['original_source_index'])==int(indices[update,int(r['slot'])]) for r in rows),
                              'Actual pair/original index trace differs')
                template=np.bincount([int(r['template_pool_index']) for r in pairs],minlength=len(arrays['template_count']))
                background=np.bincount([int(r['background_pool_index']) for r in pairs],minlength=len(arrays['background_count']))
                np.testing.assert_array_equal(template,arrays['template_count'])
                np.testing.assert_array_equal(background,arrays['background_count'])
                exposure=dict(seed=seed,updates=3528,synthetic_slots=len(pairs),unique_template_windows=int((template>0).sum()),
                    original_index_trace_exact=True,actual_template_background_counts_exact=True,
                    model_sha256=e.sha256(model/'weights.pt'))
                exposures.append(exposure)
            logs=[]
            for path in run.glob('*access*.json'):
                log=e.read_json(path)
                e.require(not log['forbidden'],'Forbidden source read attempt')
                for name in log['reads']:
                    normalized=name.lower().replace('\\','/')
                    e.require(not any(token in normalized for token in ('refit_17','house_17','truth_once','frozen_test','predictions/guard95_')),
                              'Unexpected development numerical access')
                logs.append(path.name)
            result=dict(status='independent_source_rejection_and_exposure_verification_passed',
                source_verification=independent,exposure=exposures,access_logs_checked=logs,
                new_fits=3,updates=10584,control_retrains=0,development_numerical_reads=0,development_predictions=0,
                truth_cache_reads=0,final_house_reads=0,default_changed=False,
                registration_sha256=e.sha256(run/'registration.json'),source_selection_sha256=e.sha256(run/'source_selection.json'))
            e.gate.fresh_json(run/'independent_source_execution_verification.json',result)
            return result
    if mode=='pretruth':
        require=e.require
        require(not (run/'evaluation').exists(),'Ordinary CLI must be sealed before evaluation')
        require(not (run/'ordinary_pretruth_seal.json').exists(),'Ordinary pretruth verification is one-shot')
        # This metadata is read only after a complete, source-qualified freeze.
        with e.phase(run,'ordinary_source_gate'):
            e.evaluation_guard(run)
        identity=e.read_json(e.audit.CONTROL/'identity.json')
        aggregate=next(Path(p) for p in identity['development_unlabeled'] if p.endswith('aggregate.csv'))
        outputs=[run/f'ordinary_{arm}_{seed}.csv' for arm in e.FAMILIES for seed in e.SEEDS]
        checks=[]
        with e.phase(run,'ordinary_pretruth',extra=[aggregate,*outputs]):
            for arm in e.FAMILIES:
                for seed in e.SEEDS:
                    output=run/f'ordinary_{arm}_{seed}.csv'
                    require(not output.exists(),'Fresh public-CLI output required')
                    cmd=[sys.executable,'-B',str(run/'run_microwave_source3_background.py'),'predict','--run',str(run),
                         '--authorization',str(authorization),'--family',arm,'--seed',str(seed),
                         '--aggregate',str(aggregate),'--output',str(output)]
                    completed=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
                    require(completed.returncode==0,'Public CLI failed without retry: '+completed.stderr[-3000:])
                    frozen=e.gate.npz(run/f'predictions/{arm}_{seed}.npz',['t','scores','state'])
                    with output.open(encoding='utf-8',newline='') as stream: raw=list(csv.DictReader(stream))
                    actual=dict(t=np.array([e.utc(r['timestamp']) for r in raw],np.int64),
                        scores=np.array([float(r['probability_on']) if r['probability_on'] else np.nan for r in raw]),
                        state=np.array([int(r['state']) if r['state'] else -1 for r in raw],np.int8))
                    for field in actual: np.testing.assert_array_equal(actual[field],frozen[field])
                    checks.append(dict(family=arm,seed=seed,rows=len(raw),timestamp_probability_state_exact=True,
                        csv_sha256=e.sha256(output),prediction_sha256=e.sha256(run/f'predictions/{arm}_{seed}.npz')))
                    print(f'public CLI exact: {arm} seed={seed} rows={len(raw)}',flush=True)
            result=dict(status='all_six_public_CLI_predictions_exact_before_truth',truth_opened=False,
                freeze_sha256=e.sha256(run/'freeze.json'),checks=checks,new_fits=3,updates=10584)
            e.gate.fresh_json(run/'ordinary_pretruth_seal.json',result)
        return result
    # Do not read the parent cache. The approved evaluation already saved its
    # local unique truth copy; all numerical checks work from that sealed copy.
    seal=e.read_json(run/'ordinary_pretruth_seal.json')
    e.require(not seal['truth_opened'] and seal['freeze_sha256']==e.sha256(run/'freeze.json'),'Ordinary pretruth seal differs')
    outputs=[run/f"ordinary_{r['family']}_{r['seed']}.csv" for r in seal['checks']]
    with e.phase(run,'independent_post',extra=outputs):
        for row in seal['checks']:
            path=run/f"ordinary_{row['family']}_{row['seed']}.csv"
            e.require(e.sha256(path)==row['csv_sha256'],'Ordinary CSV changed')
            e.require(e.sha256(run/f"predictions/{row['family']}_{row['seed']}.npz")==row['prediction_sha256'],'Frozen prediction changed')
        source=e.source_guard(run)
        source_result=e.verify_source_numerics(run,source)
        result=e.verify_development_numerics(run)
        result.update(source_verification=source_result,ordinary_CLI_checks=6,
            registration_sha256=e.sha256(run/'registration.json'),freeze_sha256=e.sha256(run/'freeze.json'),
            original_truth_csv_reads=0,parent_truth_cache_reopened=False)
        e.gate.fresh_json(run/'independent_post_execution_verification.json',result)
        return {k:v for k,v in result.items() if k not in ('rows','paired','source_verification','formal_means','supports')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['source','pretruth','post'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    args=p.parse_args()
    print(main(args.run.resolve(),args.authorization.resolve(),args.mode))
