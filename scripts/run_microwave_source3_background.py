"""Task15 source-only preparation and explicitly gated future execution."""
import argparse
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_source3_background as task
from virtual_submeter.accuracy import microwave_source3_background_execution as execute

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','complete_prepare','train','source','freeze','evaluate','verify','report','predict'])
    parser.add_argument('--run',required=True)
    parser.add_argument('--authorization')
    parser.add_argument('--pool-run')
    parser.add_argument('--family',choices=execute.FAMILIES)
    parser.add_argument('--seed',type=int,choices=execute.SEEDS)
    parser.add_argument('--aggregate')
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.stage=='prepare':
        print(task.prepare(args.run))
    elif args.stage=='complete_prepare':
        print(task.complete_prepare(args.run,args.pool_run))
    elif args.stage=='predict':
        print(execute.predict(args.run,args.authorization,family=args.family,seed=args.seed,aggregate=args.aggregate,output=args.output))
    else:
        print(getattr(execute,args.stage)(args.run,args.authorization))
