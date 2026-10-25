"""Verify an existing diagnosis through its sealed source, independent of cwd."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("command", choices=["verify"])
parser.add_argument("--run", required=True, type=Path)
args = parser.parse_args()
run = args.run.resolve()
sys.path.insert(0, str(run / "source"))
from virtual_submeter.accuracy import microwave_diagnosis

print(json.dumps(microwave_diagnosis.verify(run), ensure_ascii=False, indent=2))
