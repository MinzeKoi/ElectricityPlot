"""Task16 independent, disabled, zero-training pipeline."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here/'source' if (here/'registration.json').exists() else here.parent/'src'))
from virtual_submeter.accuracy import microwave_checkpoint_ensemble as task

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare','source','freeze','evaluate','verify','report','predict'])
    p.add_argument('--run', required=True)
    p.add_argument('--authorization')
    p.add_argument('--family', choices=task.FAMILIES)
    p.add_argument('--seed', type=int, choices=task.SEEDS)
    p.add_argument('--aggregate')
    p.add_argument('--output')
    a = p.parse_args()
    if a.stage == 'predict':
        print(task.predict(a.run,a.authorization,family=a.family,seed=a.seed,aggregate=a.aggregate,output=a.output))
    elif a.stage == 'prepare':
        print(task.prepare(a.run))
    else:
        print(getattr(task,a.stage)(a.run,a.authorization))
