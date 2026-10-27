"""Run an authorized sealed Task21 stage once, with explicit UTF-8 logging.

This driver never modifies the sealed recipe, source, batches or preparation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','train','source_select','freeze','evaluate','verify','report'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--authorization',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();auth=args.authorization.resolve()
    sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_task21 as t
    if args.stage=='preflight':
        t.require(t.sha256(run/'registration.json')=='e3fc4e93396b34205e47f32ecd8b6e9445ccb25386e0c70b1a6a35cbf429860c','Wrong authorized registration')
        t.require(not (run/'training_started.json').exists() and not (run/'models').exists(),'A previous training start exists; never retry')
        a=t.authorize(run,auth);t.registered(run)
        proof=t.read_json(run.with_name(run.name+'.readonly-receipt.json'))
        t.require(proof['passed'] and proof['passes']==2 and proof['all_input_delivery_hash_mtime_unchanged'],'Read-only preparation proof incomplete')
        t.require(t.snapshot(proof['snapshots'])==proof['snapshots'],'Preparation input/delivery changed since read-only verification')
        t.require(proof['preparation_manifest_sha256']==t.sha256(run/'preparation_manifest.json'),'Preparation manifest differs')
        result=dict(passed=True,registration_sha256=t.sha256(run/'registration.json'),
            preparation_manifest_sha256=t.sha256(run/'preparation_manifest.json'),authorization_sha256=t.sha256(auth),
            authorization=a,previous_training_started=False,previous_models=False,new_fits=0,optimizer_updates=0)
        t.fresh(run/'execution_preflight.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    t.require((run/'execution_preflight.json').exists(),'Preflight required')
    t.require(t.read_json(run/'execution_preflight.json')['authorization_sha256']==t.sha256(auth),'Receipt changed after preflight')
    logfile=run/f'execution_{args.stage}.log'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    command=[str(ROOT/'.venv-m2/Scripts/python.exe'),'-B',str(run/'run_microwave_task21.py'),args.stage,'--run',str(run)]
    if args.stage in ('train','source_select','freeze','evaluate'):command+=['--authorization',str(auth)]
    # STDERR is deliberately merged; p.stderr is None and never concatenated.
    with logfile.open('x',encoding='utf-8',newline='\n') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='strict')
        for line in p.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        returncode=p.wait()
    if returncode:
        t.fresh(run/f'execution_{args.stage}_failure.json',dict(stage=args.stage,returncode=returncode,
            retry_permitted=False,log_sha256=t.sha256(logfile)))
        raise SystemExit(returncode)


if __name__=='__main__':main()
