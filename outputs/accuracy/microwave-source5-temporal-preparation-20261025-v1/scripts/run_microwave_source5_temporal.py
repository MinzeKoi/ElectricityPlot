"""Independent House5 temporal preparation. Real stages remain condition-locked."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from virtual_submeter.accuracy import microwave_source5_temporal as experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','train','source_select','freeze','evaluate','verify','report','synthetic'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--recipe', choices=experiment.RECIPES, default='unit_BCE')
    args = parser.parse_args()
    if args.command == 'synthetic':
        result = experiment.synthetic_roundtrip(args.run,args.recipe)
    elif args.command in ('prepare','verify','report'):
        result = getattr(experiment,args.command)(args.run)
    else:
        result = experiment.pending_stage(args.run,args.command,args.authorization)
    print(json.dumps(result,ensure_ascii=False,indent=2))
