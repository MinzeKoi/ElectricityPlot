"""Independent task 8 entry point; sealed copies use only their own source tree."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE/'registration.json').exists() else HERE.parent
sys.path.insert(0,str(ROOT/('source' if ROOT==HERE else 'src')))
from virtual_submeter.accuracy import microwave_loss_correction as experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','freeze','evaluate','verify','report'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/microwave_loss_correction_20261025.json')
    parser.add_argument('--authorization',type=Path)
    args = parser.parse_args()
    if args.command=='prepare':
        result = experiment.prepare(args.run,args.plan)
    elif args.command in ['train','evaluate']:
        result = getattr(experiment,args.command)(args.run,args.authorization)
    else:
        result = getattr(experiment,args.command)(args.run)
    print(json.dumps(result,ensure_ascii=False,indent=2))
