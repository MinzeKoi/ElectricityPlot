"""Run a small three-household A5 engineering example (never a real blind test)."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.accuracy.final_demo import run
from virtual_submeter.data.common import DataError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.output, ROOT)
    except (DataError, OSError, ImportError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "a5_synthetic_failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
