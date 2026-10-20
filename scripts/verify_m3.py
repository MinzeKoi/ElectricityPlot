"""Verify persisted query/model/prediction identities without test-label reads."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_submeter.adaptation.verification import verify
from virtual_submeter.data.common import DataError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.run)
    except (DataError, OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(json.dumps({"status": "verification_failed", "error": str(error)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
