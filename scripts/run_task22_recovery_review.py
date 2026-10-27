"""Run one recovery verification stage in a CPU4 child and retain PID/streams."""
import argparse,json,os,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify','report','independent_verify','numeric_audit']);p.add_argument('--run',type=Path,required=True)
 a=p.parse_args();run=a.run.resolve();sys.path.insert(0,str(run/'source'))
 from virtual_submeter.accuracy import microwave_task22_recovery as task
 task.require((run/'training_complete.json').exists() and (run/'source_selection.json').exists() and (run/'diagnostics_complete.json').exists(),'Complete effective matrix, source and fixed diagnostics required')
 script=ROOT/'scripts/verify_task22_recovery_execution.py' if a.stage=='numeric_audit' else run/'run_task22_recovery.py'
 argv=[str(script)]+([] if a.stage=='numeric_audit' else [a.stage])+['--run',str(run)]
 env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 code='import json,os,runpy,sys,torch;torch.set_num_threads(4);assert torch.get_num_threads()==4;print(json.dumps(dict(stage='+repr(a.stage)+',pid=os.getpid(),parent_pid=os.getppid(),cpu_threads=torch.get_num_threads(),OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS"))),flush=True);sys.argv='+repr(argv)+';runpy.run_path('+repr(str(script))+',run_name="__main__")'
 logpath=run/f'execution_{a.stage}.log';processpath=run/f'execution_{a.stage}_process.json'
 task.require(not logpath.exists() and not processpath.exists(),'Review attempt exists; retain it')
 with logpath.open('x',encoding='utf-8',newline='\n') as log:
  child=subprocess.Popen([str(ROOT/'.venv-m2/Scripts/python.exe'),'-B','-u','-c',code],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='strict')
  task.fresh(processpath,dict(stage=a.stage,child_pid=child.pid,driver_pid=os.getpid(),script=str(script)))
  for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
  rc=child.wait()
 task.fresh(run/f'execution_{a.stage}_finished.json',dict(stage=a.stage,child_pid=child.pid,driver_pid=os.getpid(),returncode=rc,log_sha256=task.sha256(logpath),audit_new_fits=0,audit_optimizer_updates=0))
 if rc:raise SystemExit(rc)
if __name__=='__main__':main()
