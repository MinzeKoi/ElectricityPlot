"""Task21: preparation is 0fit; training requires a new hash-bound human budget."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_task21 as task

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'report', 'train', 'source_select', 'freeze', 'evaluate'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    if args.command in ('train', 'source_select', 'freeze', 'evaluate'):
        result = getattr(task, args.command)(args.run, args.authorization)
    else:
        result = getattr(task, args.command)(args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
