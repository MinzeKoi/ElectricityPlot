import pathlib,json,hashlib,sys
root=pathlib.Path.cwd();run=root/'outputs/accuracy/microwave-task23-guarded-audit-20261027-v4';d=json.loads((run/'delivery_seal.json').read_text(encoding='utf-8'));paths=[root/x['path'] for x in d['files']]+[run/'delivery_seal.json'];out=[]
for p in paths:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 s=p.stat();out.append({'path':str(p),'sha256':h.hexdigest(),'mtime_ns':s.st_mtime_ns,'bytes':s.st_size})
print(json.dumps(out,ensure_ascii=False,indent=2))

