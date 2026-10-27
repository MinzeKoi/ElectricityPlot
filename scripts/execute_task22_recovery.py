"""Single-use Task22 recovery stages; snapshot execution, CPU4, PID and UTF-8 logs."""
import argparse,json,os,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='30dad580c6711ff021f3607dcbf6dc0b0a73e8276aee01204869534ddaf3d0f9'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','train','source_select','diagnostics','freeze','evaluate']);p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
 a=p.parse_args();run=a.run.resolve();auth=a.authorization.resolve();sys.path.insert(0,str(run/'source'))
 import torch
 torch.set_num_threads(4)
 from virtual_submeter.accuracy import microwave_task22_recovery as task
 task.require(task.sha256(run/'registration.json')==EXPECTED,'Wrong recovery registration')
 if a.stage=='preflight':
  task.require(not any((run/k).exists() for k in ('training_started.json','recovery_fit_started.json','models','training_complete.json','execution_train.log','execution_train_process.json')),'A recovery start already exists; never retry')
  receipt=task.authorize(run,auth);task.registered(run)
  proof=task.read_json(run.with_name(run.name+'.readonly-receipt.json'))
  task.require(proof['passed'] and proof['passes']==2 and len(proof['calls'])==16 and proof['all_input_delivery_hash_mtime_unchanged'] and proof['preparation_manifest_sha256']==task.sha256(run/'preparation_manifest.json'),'Preparation readonly proof differs')
  task.require(task.snapshot(proof['snapshots'])==proof['snapshots'],'Preparation/input hash or mtime changed')
  evidence=ROOT/'outputs/accuracy/microwave-task22-recovery-test-evidence-20261027-v1'
  ev=task.read_json(evidence.with_name(evidence.name+'.readonly-receipt.json'))
  task.require(ev['passed'] and ev['passes']==2 and ev['v3_registration_sha256']==EXPECTED and all(x['ok'] and x['returncode']==0 and x['tests']>0 and x['stderr_bytes']>0 for x in ev['checks']),'Static evidence incomplete')
  task.require(ev['evidence_manifest_sha256']==task.sha256(evidence/'evidence_manifest.json'),'Static evidence seal differs')
  task.gate.check(evidence,task.read_json(evidence/'evidence_manifest.json')['artifacts'])
  task.require(task.snapshot(ev['snapshots'])==ev['snapshots'],'Static evidence inputs changed')
  old=task.read_json(run/'old_v4_files.json');task.require(task.snapshot(old)==old,'Old v4 files changed')
  result=dict(passed=True,registration_sha256=EXPECTED,authorization_sha256=task.sha256(auth),authorization=receipt,
    loaded_source=str(Path(task.__file__).resolve()),old_v4_file_count=len(old),old_v4_files_hash_mtime_exact=True,
    previous_recovery_started=False,previous_recovery_models=False,process_check='Immediately before preflight PowerShell Get-Process python* returned none; no start or model marker.',
    old_progress=task.read_json(task.OLD/'models/wide32/seed_2026/progress.json'),old_failed_epoch1_preserved=True,
    preparation_readonly_receipt_sha256=task.sha256(run.with_name(run.name+'.readonly-receipt.json')),
    static_evidence_receipt_sha256=task.sha256(evidence.with_name(evidence.name+'.readonly-receipt.json')),
    driver_pid=os.getpid(),cpu_threads=torch.get_num_threads(),new_fits=0,optimizer_updates=0)
  task.fresh(run/'execution_preflight.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return
 task.require(task.read_json(run/'execution_preflight.json')['authorization_sha256']==task.sha256(auth),'Budget changed after preflight')
 if a.stage=='train':task.require(not any((run/k).exists() for k in ('training_started.json','recovery_fit_started.json','models','execution_train.log','execution_train_process.json')),'Recovery already started; never retry')
 env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 script=run/'run_task22_recovery.py';argv=[str(script),a.stage,'--run',str(run),'--authorization',str(auth)]
 code='import json,os,runpy,sys,torch;torch.set_num_threads(4);assert torch.get_num_threads()==4;print(json.dumps(dict(stage='+repr(a.stage)+',pid=os.getpid(),parent_pid=os.getppid(),cpu_threads=torch.get_num_threads(),OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS"))),flush=True);sys.argv='+repr(argv)+';runpy.run_path('+repr(str(script))+',run_name="__main__")'
 logpath=run/f'execution_{a.stage}.log';processpath=run/f'execution_{a.stage}_process.json'
 task.require(not logpath.exists() and not processpath.exists(),'Stage attempt already exists; preserve and do not retry')
 with logpath.open('x',encoding='utf-8',newline='\n') as log:
  child=subprocess.Popen([str(ROOT/'.venv-m2/Scripts/python.exe'),'-B','-u','-c',code],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='strict')
  task.fresh(processpath,dict(stage=a.stage,child_pid=child.pid,driver_pid=os.getpid(),snapshot_script=str(script),authorization_sha256=task.sha256(auth)))
  for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
  rc=child.wait()
 task.fresh(run/f'execution_{a.stage}_finished.json',dict(stage=a.stage,child_pid=child.pid,driver_pid=os.getpid(),returncode=rc,log_sha256=task.sha256(logpath),retry_permitted=False))
 if rc:raise SystemExit(rc)
if __name__=='__main__':main()
