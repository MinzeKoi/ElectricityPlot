"""Preparation defaults to disabled; train requires a separate human budget receipt."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_source5_loss as experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','train','source_select','freeze','evaluate','verify','report','predict'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--family', choices=experiment.FAMILIES)
    parser.add_argument('--seed', type=int, choices=experiment.SEEDS)
    parser.add_argument('--aggregate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'train':
        result = experiment.train(args.run,args.authorization)
    elif args.command == 'predict':
        result = experiment.predict(args.run,args.family,args.seed,args.aggregate,args.output)
    else:
        result = getattr(experiment,args.command)(args.run)
    print(json.dumps(result,ensure_ascii=False,indent=2))
