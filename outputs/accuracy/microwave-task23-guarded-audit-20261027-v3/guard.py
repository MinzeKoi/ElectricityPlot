import sys, os, pathlib, socket, importlib, importlib.abc, unittest, csv, json, hashlib, datetime, math, io
import numpy as np
import numpy.ma
ROOT=pathlib.Path.cwd().resolve(); RUN=ROOT/'outputs/accuracy/microwave-task23-guarded-audit-20261027-v3'
# Load only audit-owned source and standard/numpy libraries before activating the hook; no household data is loaded here.
sys.path.insert(0,str(RUN)); import audit, independent_check, guard_tests, report_builder, verify
H=['10','12','15','18']; P=ROOT/'data/processed/refit_expansion_additional_20261024_v1'; S=ROOT/'data/raw/refit_expansion_additional_20261024_v1_slices'; R=ROOT/'data/raw/refit_complete_20261022/households'; OLD=ROOT/'outputs/accuracy/data-expansion-20261024-v1'
readable={P/'manifest.json',S/'selection.json',ROOT/'configs/refit_channels_complete_20261022.json',ROOT/'configs/r0_roles_20261022_v2.json',OLD/'admission.json',OLD/'protocols/C/protocol.json',OLD/'registration.json',OLD/'plan.json',ROOT/'src/virtual_submeter/accuracy/data_expansion.py',ROOT/'src/virtual_submeter/data/prepare.py',ROOT/'src/virtual_submeter/accuracy/microwave_mask.py',ROOT/'src/virtual_submeter/models/center_mask_tcn.py'}
for h in H:
 readable.update([R/f'CLEAN_House{h}.csv',R/f'CLEAN_House{h}.receipt.json',S/f'CLEAN_House{h}.csv',S/f'CLEAN_House{h}.slice.json',P/f'aggregate/house_{h}.csv',P/f'labels/house_{h}/microwave.csv'])
source=['audit.py','guard.py','independent_check.py','guard_tests.py','report_builder.py','verify.py']; artifacts=['input_seal.json','window_ids.csv','partitions.csv','daily.csv','fragments.csv','numerical_checks.json','报告.md','guard-tests.log','seal.log','compute.log','independent.log','delivery_seal.json','fixture_weighted.csv','fixture_gap.csv']; readable.update(RUN/x for x in source+artifacts)
readable={os.path.normcase(os.path.abspath(p)) for p in readable}; write_by_action={'seal':{'input_seal.json'},'compute':{'window_ids.csv','partitions.csv','daily.csv','fragments.csv','numerical_checks.json'},'test':{'fixture_weighted.csv','fixture_gap.csv'},'report_build':{'报告.md'},'deliver':{'delivery_seal.json'}}
action=sys.argv[1];writeable={os.path.normcase(os.path.abspath(RUN/x)) for x in write_by_action.get(action,set())}; original_stat=os.stat
class GuardError(PermissionError): pass
def pathkey(path):
 if isinstance(path,int):return None
 return os.path.normcase(os.path.abspath(os.fsdecode(path)))
def access(path,write=False):
 key=pathkey(path)
 if key is None:raise GuardError('file descriptor access denied')
 if key not in (writeable if write else readable):raise GuardError('path denied: '+key)
 if write and original_stat_exists(key):raise GuardError('overwrite denied: '+key)
def original_stat_exists(path):
 try: original_stat(path);return True
 except FileNotFoundError:return False
def checked_stat(path,*args,**kw):
 access(path);return original_stat(path,*args,**kw)
def hook(event,args):
 if event=='open':
  path,mode,flags=args;writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)));access(path,writing)
 elif event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.spawn','ctypes.dlopen','os.exec','os.fork'):
  raise GuardError('operation denied: '+event)
 elif event=='import' and str(args[0]).split('.')[0] in {'torch','tensorflow','jax','virtual_submeter'}:
  raise GuardError('training/model import denied: '+str(args[0]))
class BlockImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','jax','virtual_submeter'}:raise GuardError('training/model import denied: '+fullname)
  return None
sys.meta_path.insert(0,BlockImports());sys.addaudithook(hook);os.stat=checked_stat
if action=='seal':audit.seal()
elif action=='compute':audit.main()
elif action=='independent':independent_check.verify_all()
elif action=='test':guard_tests.run(GuardError)
elif action=='report_build':report_builder.build()
elif action=='deliver':verify.deliver()
elif action=='verify':verify.verify()
elif action=='report':verify.report()
else:raise SystemExit('unknown action')
