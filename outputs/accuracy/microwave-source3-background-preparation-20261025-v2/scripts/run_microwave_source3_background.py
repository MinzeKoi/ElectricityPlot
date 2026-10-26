"""Source-only Task15 audit; future stages fail closed."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from virtual_submeter.accuracy import microwave_source3_background as task

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','train','source','freeze','evaluate','verify','report','predict'])
    parser.add_argument('--run',required=True)
    parser.add_argument('--authorization')
    args=parser.parse_args()
    if args.stage in ('prepare','verify','report'):
        print(getattr(task,args.stage)(args.run))
    else:
        task.pending(args.run,args.stage,args.authorization)
