import sys, runpy
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,'D:\\system position\\document\\真正的文档\\AIC\\outputs\\accuracy\\microwave-expanded-guard95-preparation-20261026-v3\\source')
from virtual_submeter.accuracy import microwave_expanded_guard95 as e
e.training_guard=lambda run:(e.read_json(run/'config.json'),e.read_json(run/'identity.json'))
runpy.run_path('D:\\system position\\document\\真正的文档\\AIC\\outputs\\accuracy\\microwave-expanded-guard95-preparation-20261026-v3\\run_microwave_expanded_guard95.py',run_name='__main__')
