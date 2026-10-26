"""Seal Task18 real execution evidence without changing any frozen preparation file."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
import numpy as np


def paired_scalar(run,e):
    """Independent per-point paired counters and segment walker; local truth only."""
    truth=e.npz(run/'evaluation/truth_once.npz',['t','y'])
    recorded=e.read_json(run/'evaluation/metrics.json');pairs=[];segments=[]
    for seed in e.SEEDS:
        w=e.npz(run/f'predictions/W50_{seed}.npz',['t','raw'])
        c=e.npz(run/f'predictions/control_guard95_{seed}.npz',['t','state'])
        new=e.npz(run/f'predictions/expanded_guard95_{seed}.npz',['t','state'])
        for z in (w,c,new):np.testing.assert_array_equal(z['t'],truth['t'])
        valid=np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1]);assert int(valid.sum())==137523
        t=truth['t'][valid];y=truth['y'][valid]>=200;after=new['state'][valid]
        for arm,before in [('W50',w['raw'][valid,1]>=200),('control_guard95',c['state'][valid])]:
            count=dict(removed_fp=0,added_fp=0,restored_tp=0,new_fn=0)
            flags=dict(removed_fp=[],added_fp=[])
            for old,pred,target in zip(before,after,y):
                assert old in (0,1) and pred in (0,1)
                if old and not pred:
                    count['new_fn' if target else 'removed_fp']+=1
                if not old and pred:
                    count['restored_tp' if target else 'added_fp']+=1
                flags['removed_fp'].append(bool(old and not pred and not target))
                flags['added_fp'].append(bool(not old and pred and not target))
            row=dict(seed=seed,reference=arm,**count)
            assert row==next(r for r in recorded['paired'] if r['seed']==seed and r['reference']==arm)
            pairs.append(row)
            for kind,flag in flags.items():
                start=None;previous=None;points=0
                for stamp,on in zip(t,flag):
                    stamp=int(stamp)
                    if start is not None and (not on or stamp!=previous+8):
                        segments.append(dict(seed=seed,reference=arm,kind=kind,start_utc=e.iso(start),end_utc=e.iso(previous+8),points=points))
                        start=None;points=0
                    if on:
                        if start is None:start=stamp
                        previous=stamp;points+=1
                if start is not None:segments.append(dict(seed=seed,reference=arm,kind=kind,start_utc=e.iso(start),end_utc=e.iso(previous+8),points=points))
    assert pairs==recorded['paired'] and segments==recorded['paired_fp_segments']
    selection=e.read_json(run/'source_selection.json')
    gates=e.independent_scalar(recorded,selection)
    assert gates==e.read_json(run/'evaluation/summary.json')['acceptance_gates']
    return dict(status='passed',method='Independent scalar paired counter and timestamp segment walker; independent eight-gate count arithmetic',
        local_truth_only=True,parent_truth_reopens=0,pairs=pairs,paired_fp_segments=len(segments),gates=gates)


def main(run):
    run=run.resolve();sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_expanded_guard95 as e
    e.authorize(run,run/'human_authorization.json')
    e.require(not any((run/n).exists() for n in ('EXECUTION_REPORT.zh.md','execution_delivery_manifest.json','execution_summary.json')),'New execution delivery outputs required')
    independent=e.read_json(run/'independent_source_verification.json')
    assert independent['status']=='independent_source_and_training_passed' and independent['actual_updates']==19404
    assert independent['source_selection_sha256']==e.sha256(run/'source_selection.json') and independent['training_complete_sha256']==e.sha256(run/'training_complete.json')
    with e.isolation('report',e.allowed_files(run,'report')) as access:
        _,identity,source=e.source_guard(run)
        result=e.verify(run)
        if source['expanded_eligible']:
            assert result['status']=='execution_verified_readonly'
            scalar=paired_scalar(run,e)
            e.fresh(run/'independent_paired_execution_verification.json',scalar)
            summary=e.read_json(run/'evaluation/summary.json')
            entered=True
        else:
            assert result['status']=='source_only_rejected' and not any((run/n).exists() for n in ('predictions','ordinary','freeze.json','evaluation'))
            summary=dict(selected_before_development=source['selected'],source_rejection=source['source_rejection'],development_acceptance_passed=False,prediction_improvement_demonstrated=False)
            entered=False
        old=source['rows']['control_guard95'];new=source['rows']['expanded_guard95']
        means={arm:{key:sum(r[key] for r in rows)/3 for key in ('precision','recall','f1')} for arm,rows in [('control_guard95',old),('expanded_guard95',new)]} if len(new)==3 else {}
        protection=dict(all_seed_blocks_R95_and_nondegenerate=len(new)==3 and all(r['source_gate_passed'] and not r['degenerate'] for r in new),
            mean_F1_not_below_control=len(new)==3 and sum(r['f1'] for r in new)>=sum(r['f1'] for r in old),
            each_seed_mean_R_drop_at_most_005=len(new)==3 and all(b['recall']-a['recall']>=-.05 for a,b in zip(old,new)),
            at_least_two_seed_P_up=len(new)==3 and sum(b['precision']>a['precision'] for a,b in zip(old,new))>=2)
        text=e.report(run)['report_zh']
        text+='\n实际执行：3次新增拟合、19404次更新（每种子6468），控制模型未重训，无自动重试或额外模型。\n'
        if means:
            text+=f"\n来源三种子平均P：{means['control_guard95']['precision']:.6f}→{means['expanded_guard95']['precision']:.6f}；平均F1：{means['control_guard95']['f1']:.6f}→{means['expanded_guard95']['f1']:.6f}。\n"
        text+='\n来源保护项：\n'+''.join(f'- {key}：{value}\n' for key,value in protection.items())
        if not entered:text+='\n来源资格拒绝，未运行12格真实开发冻结或开发八门槛验收，House17及最终8/9/20均未读取。来源结果不能当作跨户预测改善；保留冻结control_guard95，UI默认不变。数据和更新量同时改变，不能独立归因于扩数据。\n'
        with (run/'EXECUTION_REPORT.zh.md').open('x',encoding='utf-8') as f:f.write(text)
        summary.update(source_eligible=source['expanded_eligible'],new_fits=3,optimizer_updates=19404,control_retrains=0,
            source_means=means,source_protection=protection,
            development_evaluated=entered,default_changed=False,power_improvement_claim=False,final_house_reads=0,
            registration_sha256=e.sha256(run/'registration.json'),authorization_sha256=e.sha256(run/'human_authorization.json'),
            preparation_manifest_sha256=e.sha256(run/'final_delivery_manifest.json'),independent_source_verification_sha256=e.sha256(run/'independent_source_verification.json'))
        e.fresh(run/'execution_summary.json',summary)
    e.fresh(run/'execution_delivery_access.json',dict(reads=sorted(access['reads']),forbidden=access['forbidden'],development_evaluated=entered))
    e.require(not access['forbidden'],'Delivery read isolation failed')
    if not entered:
        numerical=[p for p in access['reads'] if not p.endswith('.py')]
        assert not any(any(token in p for token in ('truth_once','frozen_test','house_17','refit_17','evaluation/','evaluation\\')) for p in numerical)
    root=e.ROOT
    for src,name in [(root/'scripts/verify_microwave_expanded_guard95_execution.py','verify_microwave_expanded_guard95_execution.py'),(Path(__file__),'finalize_microwave_expanded_guard95_execution.py'),
        (root/'tmp/task18-expanded-guard95-real-train-20261026-v3.log','real_training.log'),
        (root/'tmp/task18-expanded-guard95-real-source-20261026-v3.log','real_source.log'),
        (root/'tmp/task18-expanded-guard95-independent-source-20261026-v3.log','independent_source.log')]:
        shutil.copyfile(src,run/name)
    if entered:
        for name in ('freeze','evaluate'):
            shutil.copyfile(root/f'tmp/task18-expanded-guard95-real-{name}-20261026-v3.log',run/f'real_{name}.log')
    def snapshot():return {str(p.relative_to(run)):(e.sha256(p),p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
    before=snapshot();checks=[]
    for _ in range(2):checks.append(dict(verify=e.verify(run),report_verification=e.report(run)['verification']))
    assert before==snapshot()
    e.fresh(run/'execution_readonly_verification.json',dict(status='passed',repetitions=2,all_hashes_and_mtimes_exact=True,checks=checks,preparation_manifest_sha256=e.sha256(run/'final_delivery_manifest.json')))
    e.fresh(run/'execution_delivery_manifest.json',dict(status='execution_complete',registration_sha256=e.sha256(run/'registration.json'),
        preparation_manifest_sha256=e.sha256(run/'final_delivery_manifest.json'),authorization_sha256=e.sha256(run/'human_authorization.json'),
        new_fits=3,optimizer_updates=19404,control_retrains=0,development_evaluated=entered,artifacts=e.inventory(run)))
    e.gate.check(run,e.read_json(run/'execution_delivery_manifest.json')['artifacts'])
    return dict(status='execution_delivered',run=str(run),source_eligible=source['expanded_eligible'],selected=source['selected'],development_evaluated=entered,
        development_acceptance_passed=summary['development_acceptance_passed'],new_fits=3,optimizer_updates=19404,
        execution_manifest_sha256=e.sha256(run/'execution_delivery_manifest.json'),preparation_manifest_sha256=e.sha256(run/'final_delivery_manifest.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    print(json.dumps(main(a.run),ensure_ascii=False,indent=2))
