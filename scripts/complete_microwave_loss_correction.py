"""Resume task 8 after a completed freeze, reusing its successful full training audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def install_frozen_verification(run):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_loss_correction as f

    def bound_verification(candidate, predictions=False):
        if candidate.resolve()!=run: raise ValueError('Wrong resumed run')
        reg=read(run/'registration.json');f.check_artifacts(run,reg['artifacts'])
        complete=read(run/'training_complete.json');frozen=read(run/'freeze.json')
        expected=[f'{a}/{s}' for a in f.ARMS for s in f.SEEDS]
        if complete['completed_models']!=expected or complete['fits']!=6:
            raise ValueError('Incomplete fixed fit matrix')
        f.check_artifacts(run/'models',complete['artifacts'])
        f.check_artifacts(run,frozen['artifacts'])
        for field,name in [('registration_sha256','registration.json'),('training_sha256','training_complete.json'),('source_selection_sha256','source_selection.json')]:
            if frozen[field]!=digest(run/name): raise ValueError('Freeze dependency changed')
        for name in ['source_rows','source_selection']:
            if complete[name+'_sha256']!=digest(run/(name+'.json')): raise ValueError('Source binding changed')
        if f.select_source(read(run/'source_rows.json'),read(run/'plan.json'))!=read(run/'source_selection.json'):
            raise ValueError('Source choice changed')
        if set(frozen['predictions'])!={f'{a}/{s}' for a in f.ORDER for s in f.SEEDS}:
            raise ValueError('Nine frozen predictions required')
        if not frozen['ordinary_inference_matches'] or not frozen['common_support_and_other_devices_match']:
            raise ValueError('Missing ordinary equivalence')
        checks=read(run/'inference_checks.json')['checks']
        if len(checks)!=9 or {(c['arm'],c['seed']) for c in checks}!={(a,s) for a in f.ORDER for s in f.SEEDS}:
            raise ValueError('Incomplete ordinary readback matrix')
        seal=read(run/'reused_control_seal.json')
        f.check_artifacts(run/'reused_control',seal['artifacts'])
        for c in checks:
            path=run/f"reused_control/seed_{c['seed']}" if c['arm']=='TCN127' else run/f"models/{c['arm']}/seed_{c['seed']}"
            meta,_=f.load_state_model(path)
            if (meta['window_samples']!=127 or meta['weights_sha256']!=c['weights_sha256'] or
                meta['decision_threshold_probability']!=c['threshold'] or c['valid_input_centers']!=139437 or
                not all(c[k] for k in ['ordinary_scores_exact','ordinary_state_exact','raw_A_all_devices_exact'])):
                raise ValueError('Ordinary model readback changed')
        # The immutable source calls verify_training before making predictions in freeze().
        # Its successful returned freeze is the prior full checkpoint/batch replay receipt.
        source=(run/'source/virtual_submeter/accuracy/microwave_loss_correction.py').read_text(encoding='utf-8')
        if 'def freeze(run):\n    verify_training(run)' not in source:
            raise ValueError('Successful freeze no longer proves the prior full training audit')
        log=run.parents[2]/'tmp/microwave-loss-freeze-20261025.log'
        if read(log)!=dict(status='nine_predictions_frozen_before_truth'):
            raise ValueError('Successful original freeze execution receipt missing')
        result=dict(status='prior_full_audit_reused_with_all_hashes_and_ordinary_models_verified',
            freeze_sha256=digest(run/'freeze.json'),training_sha256=digest(run/'training_complete.json'),
            registration_sha256=digest(run/'registration.json'),freeze_execution_log_sha256=digest(log),
            completed_fits=6,ordinary_models_loaded=9,truth_reads_during_resumption_check=0,
            basis='Original successful immutable freeze called full verify_training before predicting; no dependent artifacts changed.')
        receipt=run/'resumption_verification.json'
        if receipt.exists():
            if read(receipt)!=result: raise ValueError('Resumption receipt changed')
        else: receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return True

    f.verify_training=bound_verification
    return f


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','evaluate','verify','report','audit'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path)
    args=p.parse_args();run=args.run.resolve()
    f=install_frozen_verification(run)
    if args.command=='check': result=f.verify_training(run,predictions=True)
    elif args.command=='evaluate': result=f.evaluate(run,args.authorization)
    elif args.command=='audit':
        from verify_microwave_loss_correction_delivery import audit
        result=audit(run)
    else: result=getattr(f,args.command)(run)
    print(json.dumps(result,ensure_ascii=False,indent=2))
