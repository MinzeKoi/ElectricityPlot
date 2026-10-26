"""Task18 independent preparation and explicitly budget-gated execution."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_expanded_guard95 as e

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','train','source','freeze','evaluate','verify','report','predict'])
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--authorization',type=Path)
    p.add_argument('--arm',choices=e.ARMS);p.add_argument('--seed',type=int,choices=e.SEEDS)
    p.add_argument('--aggregate',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();run=a.run.resolve()
    if a.command=='prepare': result=e.prepare(run)
    elif a.command=='predict': result=e.predict(run,a.authorization,arm=a.arm,seed=a.seed,aggregate=a.aggregate,output=a.output)
    elif a.command=='report': result=e.report(run,a.authorization,output=a.output)
    else: result=getattr(e,a.command)(run,a.authorization)
    print(json.dumps(result,ensure_ascii=False,indent=2))
