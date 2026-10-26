"""Disabled independent H95 experiment; all execution requires human rule approval."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_recall_guard95 as experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','verify_preparation','source_select','freeze','evaluate','verify','report','predict'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--seed',type=int,choices=experiment.SEEDS)
    parser.add_argument('--aggregate',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.command in ('prepare','verify_preparation'):
        result = getattr(experiment,args.command)(args.run)
    elif args.command == 'predict':
        result = experiment.predict(args.run,args.authorization,seed=args.seed,aggregate=args.aggregate,output=args.output)
    else: result = getattr(experiment,args.command)(args.run,args.authorization)
    print(json.dumps(result,ensure_ascii=False,indent=2))
