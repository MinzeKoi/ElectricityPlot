"""Execute one human-authorized Task22 v4 snapshot stage once, with CPU4/UTF8 logs."""
import argparse,json,os
from pathlib import Path
import subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='2ffd72773185d8be1372b0744729b7eb3daf8656f1dfd2097da47cb113037c53'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','train','source_select','diagnostics','freeze','evaluate','verify','report']);p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
 a=p.parse_args();run=a.run.resolve();auth=a.authorization.resolve();sys.path.insert(0,str(run/'source'))
 import torch
 torch.set_num_threads(4)
 from virtual_submeter.accuracy import microwave_task22 as t
 t.require(t.sha256(run/'registration.json')==EXPECTED,'Wrong Task22 v4 registration')
 if a.stage=='preflight':
  t.require(not (run/'training_started.json').exists() and not (run/'models').exists(),'Existing start/model; never retry training')
  budget=t.authorize(run,auth);t.registered(run)
  readonly=t.read_json(run.with_name(run.name+'.readonly-receipt.json'))
  t.require(readonly['passed'] and readonly['passes']==2 and len(readonly['calls'])==12 and readonly['all_input_delivery_hash_mtime_unchanged'],'Readonly proof incomplete')
  t.require(t.snapshot(readonly['snapshots'])==readonly['snapshots'],'Input/preparation hash/mtime changed')
  t.require(readonly['preparation_manifest_sha256']==t.sha256(run/'preparation_manifest.json'),'Preparation manifest changed')
  supplement=ROOT/'outputs/accuracy/microwave-task22-diagnostics-supplement-20261026-v3'
  proof=t.read_json(supplement.with_name(supplement.name+'.readonly-receipt.json'))
  t.require(proof['passed'] and proof['source_registration_sha256']==EXPECTED and proof['native_diagnostic_cells']==24 and proof['prediction_caches']==12 and proof['cpu_threads']==4,'Diagnostics supplement incomplete')
  t.require(proof['supplement_manifest_sha256']==t.sha256(supplement/'supplement_manifest.json'),'Supplement manifest changed')
  t.gate.check(supplement,t.read_json(supplement/'supplement_manifest.json')['artifacts'])
  result=dict(passed=True,registration_sha256=EXPECTED,authorization_sha256=t.sha256(auth),authorization=budget,
   source_file=str(Path(t.__file__).resolve()),snapshot_source_exact=True,previous_training_started=False,previous_models=False,
   process_check='PowerShell Get-Process python* returned no processes immediately before launch; no Task22 markers',cpu_threads=torch.get_num_threads(),
   preparation_manifest_sha256=t.sha256(run/'preparation_manifest.json'),supplement_receipt_sha256=t.sha256(supplement.with_name(supplement.name+'.readonly-receipt.json')),
   new_fits=0,optimizers=0,optimizer_updates=0,backward_calls=0)
  t.fresh(run/'execution_preflight.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return
 t.require(t.read_json(run/'execution_preflight.json')['authorization_sha256']==t.sha256(auth),'Receipt differs from preflight')
 env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 script=run/'run_microwave_task22.py';argv=[str(script),a.stage,'--run',str(run)]
 if a.stage in ('train','source_select','diagnostics','freeze','evaluate'):argv+=['--authorization',str(auth)]
 code='import json,os,runpy,sys,torch;torch.set_num_threads(4);assert torch.get_num_threads()==4;print(json.dumps(dict(stage='+repr(a.stage)+',cpu_threads=torch.get_num_threads(),OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS"))),flush=True);sys.argv='+repr(argv)+';runpy.run_path('+repr(str(script))+',run_name="__main__")'
 logpath=run/f'execution_{a.stage}.log'
 with logpath.open('x',encoding='utf-8',newline='\n') as log:
  child=subprocess.Popen([str(ROOT/'.venv-m2/Scripts/python.exe'),'-B','-u','-c',code],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='strict')
  for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
  returncode=child.wait()
 if returncode:
  t.fresh(run/f'execution_{a.stage}_failure.json',dict(stage=a.stage,returncode=returncode,retry_permitted=False,log_sha256=t.sha256(logpath)))
  raise SystemExit(returncode)
if __name__=='__main__':main()
