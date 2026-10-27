import json,hashlib,pathlib
ROOT=pathlib.Path.cwd().resolve();RUN=ROOT/'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def deliver():
 seal=json.loads((RUN/'input_seal.json').read_text(encoding='utf-8')); names=['input_seal.json','window_ids.csv','partitions.csv','daily.csv','fragments.csv','numerical_checks.json','报告.md','guard-tests.log','seal.log','compute.log','independent.log','fixture_weighted.csv','fixture_gap.csv']; paths={ROOT/x['path'] for x in seal['inputs']};paths.update(RUN/n for n in names); rows=[]
 for p in sorted(paths,key=str):
  s=p.stat();rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)})
 with (RUN/'delivery_seal.json').open('x',encoding='utf-8') as f:json.dump({'files':rows,'note':'This seal cannot contain its own digest.'},f,ensure_ascii=False,indent=2)
def verify():
 d=json.loads((RUN/'delivery_seal.json').read_text(encoding='utf-8'))
 for r in d['files']:
  p=ROOT/r['path'];s=p.stat();assert s.st_size==r['bytes'] and s.st_mtime_ns==r['mtime_ns'] and sha(p)==r['sha256'],r['path']
 print('OK unchanged inputs and deliveries:',len(d['files']))
def report(): print((RUN/'报告.md').read_text(encoding='utf-8'))
