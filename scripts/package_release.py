"""Seal a completed live confirmation session for portable replay."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.release import build


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-run", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.session.resolve(), args.output.resolve(), ROOT,
                           args.historical_run.resolve() if args.historical_run else None), ensure_ascii=False, indent=2))
