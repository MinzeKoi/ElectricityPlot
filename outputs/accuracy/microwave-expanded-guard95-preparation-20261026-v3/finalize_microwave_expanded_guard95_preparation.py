"""One-shot Task18 preparation delivery seal. Never trains or opens real dev/final data."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True


def main(run,test_log,synthetic):
    run=run.resolve();synthetic=synthetic.resolve()
    sys.path.insert(0,str(run/'source'))
    from virtual_submeter.accuracy import microwave_expanded_guard95 as e
    e.require(not any((run/n).exists() for n in ('training_started.json','models','source_selection.json','predictions','evaluation','final_delivery_manifest.json','PREPARATION_REPORT.zh.md')),'Preparation only, new delivery seal required')
    independent=e.read_json(run/'independent_preparation_verification.json')
    e.require(independent['status']=='independent_preparation_passed' and independent['registration_sha256']==e.sha256(run/'registration.json'),'Independent preparation binding missing')
    task17_inputs={str(Path(r['path']).resolve()):r['sha256'] for r in e.read_json(e.AUDIT/'access_evidence.json')['reads']}
    identity=e.read_json(run/'identity.json');source_receipts=[]
    for house in ('2','19'):
        raw=e.ROOT/f'data/raw/refit_complete_20261022/households/CLEAN_House{house}.csv';receipt=raw.with_suffix('.receipt.json')
        for path in (raw,receipt):
            digest=task17_inputs[str(path.resolve())]
            e.require(identity['source_inputs'][str(path)]==digest==e.sha256(path),'Task17 raw/receipt identity mismatch')
            source_receipts.append(dict(path=str(path),sha256=digest,task17_exact=True))
        e.require(e.read_json(receipt)['sha256']==task17_inputs[str(raw.resolve())],'Original source receipt does not bind raw')
    verification=e.verify(run)
    raw=test_log.read_bytes();log=raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig',errors='replace')
    e.require('Ran 10 tests' in log and '\nOK' in log and 'FAILED' not in log,'All ten final synthetic tests must pass')
    sv=e.read_json(synthetic/'run/synthetic_verification.json')
    e.require(sv['status']=='passed' and sv['matrix_size']==12 and sv['support']==137523 and sv['real_fits']==sv['real_optimizer_updates']==0,'Complete retained synthetic verification missing')
    # All these files are explicitly synthetic; no real development cache is read.
    sm=dict(synthetic_only=True,real_fits=0,real_optimizer_updates=0,artifacts=e.inventory(synthetic))
    e.fresh(synthetic/'delivery_manifest.json',sm)
    with (synthetic/'README.zh.md').open('x',encoding='utf-8') as f:
        f.write('Task18保留合成链路。所有功率/标签/模型均为合成；没有真实训练或优化器更新。完成137523点评价支持、12格实际TCN/常量模型公共CLI推理与冻结回读、单次评价、两条算术复核、缺矩阵/重hash状态篡改/重复评价拒绝、两遍只读verify/report。训练完成来源元数据在测试驱动器中替代，不能作为真实训练预算或真实改善证据。父合成缓存已改名，verify只读本地一次副本。\n')
    # Bind the explanatory file too, without rewriting the first completed seal.
    e.fresh(synthetic/'delivery_complete.json',dict(manifest_sha256=e.sha256(synthetic/'delivery_manifest.json'),readme_sha256=e.sha256(synthetic/'README.zh.md')))
    shutil.copyfile(test_log,run/'synthetic_tests.log')
    shutil.copyfile(Path(__file__),run/'finalize_microwave_expanded_guard95_preparation.py')
    e.fresh(run/'synthetic_validation.json',dict(**sv,retained_artifacts=str(synthetic),delivery_manifest_sha256=e.sha256(synthetic/'delivery_manifest.json'),delivery_complete_sha256=e.sha256(synthetic/'delivery_complete.json')))
    registration=e.sha256(run/'registration.json')
    proposal=e.read_json(run/'authorization_template.disabled.json')|dict(registration_sha256=registration)
    e.fresh(run/'budget_proposal.disabled.json',proposal)
    proposal_text=f'''Task18待批准执行提案

准备已完成，当前尚未训练。若另行批准，仅执行expanded_guard95三个种子42/1019/2026，总新增3fit、19404次更新（每种子12×539=6468），不重训冻结control_guard95。

固定来源：保留原336704中心/2275正中心；追加House2 UTC [2014-10-14,2014-10-28)、House19 [2014-10-19,2014-11-02)，最终615922/2994/612928。归一化、原初始化、TCN127/5329参数、Adam .001 CPU4、BCE权重1和16正/240负采样保持固定。

先来源资格，再唯一预选与完整12格公共CLI冻结，然后只评价一次旧Task12缓存；来源拒绝不读开发。正式验收沿用原八门槛。数据和更新量同时改变，不能单独证明扩数据因果；合成通过不表示真实预测改善。功率/电量无改善声明，UI不切换。

注册SHA256：{registration}

待批准模板：budget_proposal.disabled.json（authorized=false；人类原话/出处仍为空）。当前准备授权和旧回执都不能启动训练。本交付不是新增训练批准。
'''
    with (run/'PROPOSAL.zh.md').open('x',encoding='utf-8') as f:f.write(proposal_text)
    e.report(run,output=run/'PREPARATION_REPORT.zh.md')
    e.fresh(run/'preparation_receipt.json',dict(status='preparation_complete_training_not_authorized',registration_sha256=registration,
        authorization_reference=e.read_json(run/'config.json')['preparation_authorization_reference'],new_fits=0,optimizer_updates=0,new_real_candidate_inference=0,new_real_candidate_calibration=0,development_reads=0,final_house_reads=0,
        future_budget=dict(additional_new_fits=3,maximum_updates=19404),verification=verification,raw_receipts_exact_task17=source_receipts))
    def snapshot(): return {str(p.relative_to(run)):(e.sha256(p),p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
    before=snapshot();checks=[]
    for _ in range(2):
        checks.append(dict(verify=e.verify(run),report_verification=e.report(run)['verification']))
    e.require(before==snapshot(),'Repeated readonly verify/report changed a file/hash/mtime')
    e.fresh(run/'readonly_verification.json',dict(status='passed',repetitions=2,file_count=len(before),all_hashes_and_mtimes_exact=True,checks=checks))
    e.fresh(run/'final_delivery_manifest.json',dict(status='prepared_not_trained',registration_sha256=registration,artifacts=e.inventory(run),
        new_fits=0,optimizer_updates=0,source17_original_20_hash_and_mtime_preserved=True,synthetic_delivery=str(synthetic)))
    # Final sealed default readback, no output files and no parent truth read.
    final_before=snapshot();last=e.verify(run);e.report(run)
    e.require(final_before==snapshot(),'Final sealed readonly replay changed a file')
    return dict(status='preparation_delivered_waiting_for_new_human_budget',run=str(run),registration_sha256=registration,
        delivery_manifest_sha256=e.sha256(run/'final_delivery_manifest.json'),artifact_count=len(e.read_json(run/'final_delivery_manifest.json')['artifacts']),
        verification=last,new_fits=0,optimizer_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--test-log',type=Path,required=True);p.add_argument('--synthetic',type=Path,required=True)
    a=p.parse_args();print(json.dumps(main(a.run,a.test_log.resolve(),a.synthetic),ensure_ascii=False,indent=2))
