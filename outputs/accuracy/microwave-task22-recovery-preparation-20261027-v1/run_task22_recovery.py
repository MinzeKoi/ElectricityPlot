"""Registered Task22 recovery entry; requires distinct 1-fit authorization."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_task22_recovery as task
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','verify','report','train','source_select','freeze','evaluate','diagnostics','independent_verify']);p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path)
 a=p.parse_args()
 if a.command in ('train','source_select','freeze','evaluate','diagnostics'):result=getattr(task,a.command)(a.run,a.authorization)
 else:result=getattr(task,a.command)(a.run)
 print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
