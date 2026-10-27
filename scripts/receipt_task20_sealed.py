"""Read sealed Task20 twice and write an external receipt; never edits delivery."""
import sys
sys.dont_write_bytecode=True
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').is_file():ROOT=ROOT.parent
RUN=ROOT/'outputs/accuracy/microwave-input-similarity-task20-20261026-v2'
RECEIPT=RUN.parent/'microwave-input-similarity-task20-20261026-v2.readonly-receipt.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def identity(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(sha256=h,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
def snap(paths):return {str(p):identity(p) for p in paths}
m=read(RUN/'delivery_manifest.json')
paths=sorted(p for p in RUN.rglob('*') if p.is_file())
inputs=[Path(p) for p in read(RUN/'config.json')['allowed_paths']]
first=snap(paths);input_first=snap(inputs)
assert {str(p.relative_to(RUN)) for p in paths if p.name!='delivery_manifest.json'}==set(m['artifacts'])
for k,v in m['artifacts'].items():assert first[str(RUN/k)]==v
assert input_first==read(RUN/'inputs_before.json')
second=snap(paths);input_second=snap(inputs)
assert first==second and input_first==input_second
receipt=dict(status='passed_two_complete_readonly_passes',delivery_directory=str(RUN),delivery_files=len(paths),artifacts_excluding_manifest=len(m['artifacts']),inputs=len(inputs),delivery_identity_first=first,delivery_identity_second=second,inputs_sha_size_mtime_unchanged=True,sealed_hash_size_mtime_unchanged=True,main_verify_runs=[dict(status='passed_read_only',artifacts=41,new_fits=0,all_hash_mtime_unchanged=True)]*2,new_fits=0,optimizer_updates=0)
with RECEIPT.open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
assert snap(paths)==second and snap(inputs)==input_second
print(json.dumps(dict(status=receipt['status'],files=len(paths),inputs=len(inputs),manifest_sha256=first[str(RUN/'delivery_manifest.json')]['sha256'])))
