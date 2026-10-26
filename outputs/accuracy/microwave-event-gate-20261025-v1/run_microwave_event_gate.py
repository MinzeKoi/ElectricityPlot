"""Frozen event gate; no training entry point."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_event_gate as e

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare','source_select','freeze','evaluate','verify','report','predict'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--recipe', type=Path)
    p.add_argument('--aggregate', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.command == 'predict':
        result = e.predict(a.recipe, a.aggregate, a.output)
    else:
        result = getattr(e, a.command)(a.run.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
