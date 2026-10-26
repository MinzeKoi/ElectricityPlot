"""Read-only entry contract check. Prints evidence; does not save any files."""
import sys
sys.dont_write_bytecode=True
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/accuracy/microwave-gate-recall-audit-20261025-v1'

def snapshot():
    return {str(p):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns)
            for p in OUT.rglob('*') if p.is_file()}

def main():
    before=snapshot()
    base=[sys.executable,str(ROOT/'scripts/audit_microwave_gate_recall.py')]
    r=subprocess.run(base+['--assert-acceptance'],capture_output=True,text=True,cwd=ROOT)
    result=json.loads(r.stdout)
    assert r.returncode==2 and result['eachseedR85'] is False
    v=subprocess.run([sys.executable,str(ROOT/'scripts/verify_microwave_gate_recall_audit.py')],capture_output=True,text=True,cwd=ROOT)
    assert v.returncode==0 and json.loads(v.stdout)['passed']
    rejected=[]
    for args,reason in [(['--build-source'],'requires --output'),(['--build-source','--details','--verify-ordinary','--output',str(OUT)],'already exists')]:
        x=subprocess.run(base+args,capture_output=True,text=True,cwd=ROOT)
        assert x.returncode==2 and reason in x.stderr
        rejected.append(dict(exit_code=x.returncode,reason=reason))
    assert before==snapshot(), 'Read-only entry changed audit content or modification time'
    print(json.dumps(dict(passed=True,read_only_replay_exit=r.returncode,replay_seconds=result['elapsed_seconds'],independent_verification=True,all_audit_file_hashes_and_mtimes_unchanged=True,files_checked=len(before),rejected_overwrite_or_missing_output=rejected),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
