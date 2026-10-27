"""Rerun static recovery tests and preserve both output streams and provenance."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
V1=ROOT/'outputs/accuracy/microwave-task22-recovery-preparation-20261027-v1'
V3=ROOT/'outputs/accuracy/microwave-task22-recovery-preparation-20261027-v3'
OLD=ROOT/'outputs/accuracy/microwave-task22-preparation-20261026-v4'
sys.path.insert(0,str(V3/'source'))
from virtual_submeter.accuracy import microwave_task22_recovery as m


def write_exclusive(path,text):
 with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)


def main():
 m.require(not (HERE/'evidence_manifest.json').exists(),'Evidence directory already completed')
 m.require((V3/'preparation_manifest.json').is_file() and V3.with_name(V3.name+'.readonly-receipt.json').is_file(),'Final v3 sealed preparation required')
 m.gate.check(V3,m.read_json(V3/'preparation_manifest.json')['artifacts'])
 m.gate.check(V1,m.read_json(V1/'preparation_manifest.json')['artifacts'])
 paths={Path(p).resolve() for p in m.read_json(V3/'inputs_before.json')}
 for folder in (V1,V3,OLD):paths.update(p.resolve() for p in folder.rglob('*') if p.is_file())
 for folder in (V1,V3,OLD):
  extra=folder.with_name(folder.name+'.readonly-receipt.json')
  if extra.exists():paths.add(extra.resolve())
 paths.add(Path(__file__).resolve())
 before=m.snapshot(paths)
 env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 checks=[]
 for label,folder in (('current_live',ROOT/'tests'),('v3_snapshot',V3),('v1_snapshot',V1)):
  print('Static tests: '+label,flush=True)
  command=[str(ROOT/'.venv-m2/Scripts/python.exe'),'-B','-m','unittest','discover','-s',str(folder),'-p','test_task22_recovery.py','-v']
  process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,encoding='utf-8',errors='strict')
  stdout=process.stdout or '';stderr=process.stderr or '';combined=stdout+stderr
  write_exclusive(HERE/f'{label}.stdout.log',stdout);write_exclusive(HERE/f'{label}.stderr.log',stderr);write_exclusive(HERE/f'{label}.combined.log',combined)
  count=re.search(r'Ran\s+(\d+)\s+tests?\s+in\s+',combined);passed=process.returncode==0 and count is not None and int(count.group(1))>0 and re.search(r'(?m)^OK\s*$',combined) is not None
  item=dict(label=label,returncode=process.returncode,tests=int(count.group(1)) if count else 0,ok=bool(passed),
      stdout_bytes=len(stdout.encode('utf-8')),stderr_bytes=len(stderr.encode('utf-8')),combined_bytes=len(combined.encode('utf-8')),
      stdout_sha256=hashlib.sha256(stdout.encode('utf-8')).hexdigest(),stderr_sha256=hashlib.sha256(stderr.encode('utf-8')).hexdigest())
  checks.append(item)
  if not passed:
   m.fresh(HERE/'FAILED.json',dict(failed=label,checks=checks,new_fits=0,optimizer_updates=0));raise RuntimeError('Static tests failed; evidence retained')
  m.require(m.snapshot(before)==before,'Test run modified a registered input or prior delivery')
 m.fresh(HERE/'test_results.json',dict(passed=True,checks=checks,v1_historical_empty_test_logs_retained=True,
   stdout_and_stderr_saved_separately=True,successful_unittest_output_was_stderr=True,new_fits=0,optimizers=0,optimizer_updates=0,backward_calls=0,development_reads=0,final_reads=0))
 write_exclusive(HERE/'TEST_EVIDENCE.zh.md','''# 恢复准备静态测试补充证据

原v1封存测试日志为空：runner成功时只保存stdout，而unittest成功报告位于stderr。原v1目录和清单保持原样，不将空日志当成通过证据。

本证据以UTF-8、-B、OMP/MKL线程4重新执行当前工作区、最终v3快照和原v1快照的静态测试，逐项保存stdout、stderr及合并日志、退出码和非零测试数；均实际报告OK。测试只用fakefit和静态合成数组，禁止优化器、反传及真实fit。

按逐文件白名单核对原Task22 v4、v1、v3以及v3注册输入的hash和mtime，测试前后不变；测试不会打开真实House17或最终8/9/20文件。此证据独立于已封存旧工件。
''')
 m.fresh(HERE/'evidence_manifest.json',dict(status='passed_static_tests_no_training',v3_registration_sha256=m.sha256(V3/'registration.json'),
   v1_registration_sha256=m.sha256(V1/'registration.json'),original_v4_registration_sha256=m.sha256(OLD/'registration.json'),
   checks=checks,artifacts=m.inventory(HERE),new_fits=0,optimizer_updates=0))
 delivery=m.snapshot([*paths,*[HERE/k for k in m.inventory(HERE)]])
 for _ in (1,2):m.require(m.snapshot(delivery)==delivery,'Readonly evidence hash/mtime changed')
 receipt=HERE.with_name(HERE.name+'.readonly-receipt.json')
 m.fresh(receipt,dict(passed=True,passes=2,all_input_delivery_hash_mtime_unchanged=True,evidence_manifest_sha256=m.sha256(HERE/'evidence_manifest.json'),
  v3_registration_sha256=m.sha256(V3/'registration.json'),checks=checks,snapshots=delivery,new_fits=0,optimizer_updates=0))
 print(json.dumps(dict(passed=True,checks=checks,v3_registration_sha256=m.sha256(V3/'registration.json'),receipt=str(receipt),new_fits=0,optimizer_updates=0),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
