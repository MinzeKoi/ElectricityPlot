"""Final supplemental execution seal after an immutable reviewer defect."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]


def audit(run):
    p=subprocess.run([str(ROOT/'.venv-m2/Scripts/python.exe'),'-B',str(ROOT/'scripts/verify_microwave_task21_execution.py'),'--run',str(run)],
        cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True,encoding='utf-8',errors='strict')
    if p.returncode:raise RuntimeError((p.stdout or '')+(p.stderr or ''))
    return json.loads(p.stdout or '{}')


def main(run,authorization):
    run=run.resolve();authorization=authorization.resolve();sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_task21 as t
    t.require(not (run/'execution_final_manifest.json').exists(),'Final execution seal cannot be overwritten')
    t.authorize(run,authorization);t.registered(run)
    result=audit(run)
    t.require(result['passed'] and result['source_qualified'] is False
        and result['original_date_core_boundaries_exact'] and result['original_min_positive_and_event_support_exact'], 'Incomplete final independent audit')
    t.require(not any((run/k).exists() for k in ('freeze.json','predictions','evaluation','evaluation_complete.json')),'Source rejected: 0devread required')
    t.fresh(run/'independent_execution_numeric_final.json',result)
    notes=dict(sealed_verify_report_succeeded=False,sealed_review_error="KeyError: 'event'",
        immutable_source_modified=False,new_training_or_retry=False,
        final_auditor_checks=['initialization','10584 actual batches and exposure','budget','all36 checkpoints',
            'original event/minimum positive support','original UTC/core boundaries','independent thresholds and ranks',
            'L95/H95/r15','source confusion and qualification','source rejection 0devread'],
        superseded_intermediate_manifest='execution_delivery_manifest.json',
        intermediate_readonly_attempt='External audit script was extended with explicit UTC/core-boundary checks while the intermediate read-only seal was in progress. Its hash/mtime changed, correctly preventing that receipt. No model, sealed source, recipe, index or original input changed. The final audit and two read-only passes below use stable final external scripts.')
    t.fresh(run/'execution_finalization_notes.json',notes)
    prior=(run/'EXECUTION_REPORT.zh.md').read_text(encoding='utf-8')
    extra='\n补充终审：原来源block4/5的UTC日期、中心core边界、最小正窗数与event支持条件均已显式独立核验。最终独立数值审计见independent_execution_numeric_final.json。封存verify/report执行分支仍存在event字段遗漏缺陷，外部独立审计通过不表示原分支通过。\n\n初次执行只读封存期间，外部审计程序补入了明确日期边界检查，触发脚本hash/mtime改变，因而未生成该次只读回执；未修改任何模型或封存配方。保留中间清单与失败过程，由execution_final_manifest.json作为最终清单。最终使用固定脚本重做两遍只读检查。\n'
    with (run/'EXECUTION_FINAL_REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as f:f.write(prior+extra)
    before=t.read_json(run/'inputs_before.json');t.require(t.snapshot(before)==before,'Original input changed')
    scripts=[ROOT/'scripts'/n for n in ('execute_microwave_task21_budget.py','verify_microwave_task21_execution.py',
        'seal_microwave_task21_execution.py','finalize_microwave_task21_execution_receipt.py')]
    t.fresh(run/'execution_final_manifest.json',dict(status='source_rejected_complete_independent_audit',new_fits=3,optimizer_updates=10584,
        registration_sha256=t.sha256(run/'registration.json'),preparation_manifest_sha256=t.sha256(run/'preparation_manifest.json'),
        authorization_sha256=t.sha256(authorization),source_eligible=False,development_reads=0,final_reads=0,
        sealed_verify_report_succeeded=False,independent_execution_audit_passed=True,
        supersedes_manifest_sha256=t.sha256(run/'execution_delivery_manifest.json'),
        artifacts=t.inventory(run),execution_tools=t.snapshot(scripts)))
    paths=[*before,*scripts,authorization,run.with_name(run.name+'.readonly-receipt.json'),*[run/k for k in t.inventory(run)]]
    sealed=t.snapshot(paths);calls=[]
    for iteration in (1,2):
        print(f'Final stable read-only pass {iteration}',flush=True)
        actual=audit(run);t.require(actual==result,'Repeated independent audit differs')
        report=(run/'EXECUTION_FINAL_REPORT.zh.md').read_text(encoding='utf-8')
        t.require(t.snapshot(sealed)==sealed,'Final read-only pass changed input/delivery hash/mtime')
        calls.append(dict(iteration=iteration,independent_verify_passed=True,
            result_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest(),
            report_sha256=hashlib.sha256(report.encode()).hexdigest()))
    receipt=run.with_name(run.name+'.execution-readonly-receipt.json')
    t.fresh(receipt,dict(passed=True,passes=2,calls=calls,all_input_delivery_hash_mtime_unchanged=True,
        execution_final_manifest_sha256=t.sha256(run/'execution_final_manifest.json'),snapshots=sealed))
    print(json.dumps(dict(status='complete_source_rejected',new_fits=3,updates=10584,development_reads=0,final_reads=0,
        manifest_sha256=t.sha256(run/'execution_final_manifest.json'),readonly_receipt=str(receipt)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    args=p.parse_args();main(args.run,args.authorization)
