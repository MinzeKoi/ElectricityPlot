"""Compatibility repair for v2's list-valued source rows; no selection changes."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
run = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
sys.path.insert(0, str(run/'source'))
from virtual_submeter.accuracy import microwave_regression_calibration as experiment

repair = run/'execution_repair/receipt.json'
receipt = json.loads(repair.read_text(encoding='utf-8'))
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if (digest(Path(__file__)) != receipt['wrapper_sha256']
        or digest(run/'registration.json') != receipt['registration_sha256']
        or digest(run/'source_complete.json') != receipt['source_complete_sha256']):
    raise RuntimeError('Execution repair receipt mismatch')
original = experiment.read_json


def compatible_read(path):
    if Path(path).name == 'source_rows.json':
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(value, list):
            raise RuntimeError('Source rows must be a list')
        return value
    return original(path)


experiment.read_json = compatible_read
if __name__ == '__main__':
    experiment.main()
