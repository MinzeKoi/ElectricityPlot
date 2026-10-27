import json, csv, hashlib, sys
from pathlib import Path
RUN=Path(__file__).resolve().parent; ROOT=Path.cwd().resolve()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def verify():
 data=json.loads((RUN/'delivery_seal.json').read_text(encoding='utf-8'))
 for item in data['files']:
  p=ROOT/item['path']; s=p.stat(); assert s.st_size==item['bytes'] and s.st_mtime_ns==item['mtime_ns'] and sha(p)==item['sha256'],item['path']
 print('OK: input and delivery files unchanged:',len(data['files']))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='verify':verify()
 elif len(sys.argv)>1 and sys.argv[1]=='report': print((RUN/'报告.md').read_text(encoding='utf-8'))
 else: raise SystemExit('usage: verify | report')
