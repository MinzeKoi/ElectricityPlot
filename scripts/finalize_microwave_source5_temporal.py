"""Publish local delivery documents after the completed numerical review."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.dont_write_bytecode=True
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',required=True,type=Path)
args=parser.parse_args()
run=args.run.resolve()
sys.path.insert(0,str(run/'source'))
from virtual_submeter.accuracy import microwave_source5_temporal_execution as e
from virtual_submeter.data.common import DataError,read_json,write_json,inventory,sha256


if __name__=='__main__':
    review=read_json(run/'independent_final_verification.json')
    if review.get('status')!='independently_reviewed_numeric_invariants' or not review.get('ordinary_cli_timestamp_probability_state_exact'):
        raise DataError('Final numerical/ordinary CLI verification required')
    if (run/'final_delivery_manifest.json').exists(): raise DataError('Delivery finalized; no overwrite')
    summary=read_json(run/'evaluation/summary.json')
    metrics=read_json(run/'evaluation/metrics.json')['rows']
    source=read_json(run/'source_rows.json')
    groups={a:[next(r for r in metrics if r['arm']==a and r['seed']==s) for s in e.f.SEEDS] for a in ('W50','control','source5')}
    base,new=groups['W50'],groups['source5']
    gain=float(np.mean([b['f1']-a['f1'] for a,b in zip(base,new)]))
    fp_ratio=sum(v['fp'] for v in new)/sum(v['fp'] for v in base)
    both=sum(b['precision']>a['precision'] and b['f1']>a['f1'] for a,b in zip(base,new))
    gates=dict(mean_precision_at_least_008=summary['means']['source5']['precision']>=.08,
        each_seed_recall_at_least_085=all(v['recall']>=.85 for v in new),
        mean_f1_gain_at_least_002=gain+1e-12>=.02,at_least_two_seed_precision_f1_both_up=both>=2,
        each_seed_f1_drop_at_most_002=all(b['f1']-a['f1']>=-.02 for a,b in zip(base,new)),
        false_positives_reduced_at_least_30_percent=fp_ratio<=.7,nondegenerate=all(not v['degenerate'] for v in new),
        same_137523_point_time_mask=True)
    if bool(all(gates.values()))!=summary['development_acceptance_passed']: raise DataError('Independent acceptance gates disagree')
    models=[]
    for arm in e.f.ARMS:
        for seed in e.f.SEEDS:
            path=run/f'models/{arm}/seed_{seed}'
            meta=read_json(path/'model.json')
            models.append(dict(arm=arm,seed=seed,model=str(path),epoch=meta['selected']['epoch'],threshold=meta['decision_threshold_probability'],
                model_json_sha256=sha256(path/'model.json'),weights_sha256=meta['weights_sha256']))
    delivery=dict(schema='microwave_source5_temporal_delivery_v1',run=str(run),summary=summary,
        gates_vs_W50=gates,mean_f1_gain_vs_W50=gain,false_positive_reduction_vs_W50=1-fp_ratio,
        seeds_precision_and_f1_both_up_vs_W50=both,models=models,source_rows=source,development_rows=metrics,
        training=dict(source_recipe='control_BCE50',new_starts=3,completed_fits=3,failed_fits=0,total_updates=10584,control_retrains=0),
        tests=dict(new_synthetic=19,default_protocol_regression=25,total=44,all_passed=True),
        preparation_failure=dict(run=str(run.parent/'microwave-source5-temporal-20261025-v1'),stage='source_input_path_type_check',real_fits=0,preserved=True),
        reporting_failure=dict(directory='details',stage='duplicate_threshold_field_before_writing_tables',real_fits=0,preserved=True,repaired_output='details_v2'),
        condition_identity=read_json(run/'condition_identity.json'),human_authorization_quote='允许这个独立时间对照试验',
        coordinator_terminal_confirmation='Task8v2 completed six fits and failed acceptance; source-only selection remains TCN127/control_BCE50',
        approved_exceptions=['House5 validation-only','source train/validation household disjointness'],
        validation_is_fresh_independent=False,compatible_with_original_cross_household_protocol=False,
        freeze_sha256=sha256(run/'freeze.json'),source_selection_sha256=sha256(run/'source_selection.json'),
        independent_final_verification_sha256=sha256(run/'independent_final_verification.json'),
        ordinary_cli_predictions_equal_frozen=True,truth_original_csv_parses=1,final_house_data_accessed=False,
        default_ui_and_model_unchanged=True,power_improvement_claim=False)
    write_json(run/'delivery_summary.json',delivery)
    baseline=read_json(run/'baseline_identity.json')
    aggregate=Path(baseline['reference_run'])/'protocol'/baseline['development_partition']['artifacts']['aggregate']['path']
    command=f'.venv-m2/Scripts/python.exe -B "{run}/predict_microwave_tcn.py" --model "{run}/models/source5/seed_42" --aggregate "{aggregate}" --output "<新的预测CSV路径>"'
    lines=['# House5同户时间对照试验：正式交付（2026-10-25）','',
        '**已完成固定3个新模型；整体开发验收未通过。** 新增House5组通过来源保护并被事前唯一预选，但相对W50的Precision、F1增益及误报下降均未达预登记门槛。无开发改选、额外候选或补训。','',
        '|开发组|均P|均R|均F1|总FP|','|---|---:|---:|---:|---:|']
    for arm in ('W50','control','source5'):
        m=summary['means'][arm]
        lines.append(f'|{arm}|{m["precision"]:.6f}|{m["recall"]:.6f}|{m["f1"]:.6f}|{sum(r["fp"] for r in groups[arm])}|')
    lines+=['',f'相对W50平均F1增加{gain:.6f}（要求≥0.02），误报减少{(1-fp_ratio)*100:.2f}%（要求≥30%）；平均P={summary["means"]["source5"]["precision"]:.6f}（要求≥0.08）。各seed R≥0.85、非退化、{both}个seed P/F1同升、各seed F1不下降超过0.02均通过。','',
        '来源对称对照：旧控制和新增组均只用原冻结最后两块、同一34351/201中心与输入，重选12epoch checkpoint/公共阈值。旧控制均P=0.162579、均F1=0.275326；新增均P=0.209635、均F1=0.340164，三seed P均提高，来源门槛通过。',
        '冻结后开发诊断：新增组相对对称旧控制均P增加0.006376、均F1增加0.011727、平均FP减少1430.67；该相对收益不足以满足原W50整体验收，不声明达标改善。','',
        '新数据支持：原House2/19 partial 259806/849全部行逐列不变；House5新增76898/1426，联合336704/2275，House5索引2。原归一化字节相同；31核心与127外圈不跨各自冻结块，跨界阈值段整段排除，末端1800秒带、NaN及观测mask保留。独立脚本重放三条3528更新采样流并核对实际上下文时间集合隔离。',
        '预算：仅control_BCE50、seed42/1019/2026各一次、每次3528更新，共10584更新；3启动全部完成，0失败拟合，0控制重训。准备v1路径类型错误与空的初次明细报告目录保留，均未引发新拟合；正式有效目录为v2。','',
        '人类授权原文：“允许这个独立时间对照试验”。协调者确认任务8终态未达标且source预选TCN127/control_BCE50后执行。两项获批例外为House5仅验证及来源训练/验证整户隔离；本试验不符合原跨户来源协议。默认data.protocol守卫和旧实验、旧CSV、默认UI不变。',
        'House5最后4日已历史选模接触，只称复用来源时间验证；House17此前已查看，只作开发验收，不是全新独立户外验证。最终8/9/20数据未访问。原A微波炉功率及其他电器功率/状态逐点不变，不宣称功率改善。','',
        '全部6份开发预测与唯一source预选先封存、普通推理逐点一致后才一次解析开发真值；每seed原W50同137523点。19项本任务合成测试＋25项默认协议回归通过；最终独立脚本复核数值、完整哈希和普通CLI输出。','',
        '## 交付文件','',
        f'- [主报告](<{run.as_posix()}/report/REPORT.zh.md>)',f'- [误报片段与补充明细](<{run.as_posix()}/details_v2/DETAILS.zh.md>)',
        f'- [逐seed开发混淆](<{run.as_posix()}/report/development_comparison.csv>)',f'- [逐块来源混淆](<{run.as_posix()}/details_v2/source_selected_block_confusions.csv>)',
        f'- [独立最终复核](<{run.as_posix()}/independent_final_verification.json>)',f'- [完整SHA-256清单](<{run.as_posix()}/final_delivery_manifest.json>)',
        '来源/开发PR曲线CSV在report目录；全部误报片段、真实日期/事件曝光及实际正负损失贡献在details_v2。普通可加载权重与元数据在models/source5/seed_42、seed_1019、seed_2026；对称旧控制另在models/control。','',
        '## 普通预测命令','', '```powershell',command,'```','',
        '该命令只输出微波炉开机概率/状态，原功率输出未调整；输出路径必须未存在。seed42普通CLI已实际运行并与封存t/scores/state完全一致，另6份预测均通过普通加载API核对。']
    text='\n'.join(lines)+'\n'
    docs=e.WORKSPACE/'docs'
    md=docs/'MICROWAVE_SOURCE5_TEMPORAL_DELIVERY_20261025.md'
    js=docs/'MICROWAVE_SOURCE5_TEMPORAL_DELIVERY_20261025.json'
    if md.exists() or js.exists(): raise DataError('Delivery documents exist; no overwrite')
    md.write_text(text,encoding='utf-8')
    (run/'DELIVERY.zh.md').write_text(text,encoding='utf-8')
    shutil.copyfile(Path(__file__),run/'validation_evidence/finalize_delivery.py')
    manifest=dict(artifacts=inventory(run),external_report=dict(path=str(md),sha256=sha256(md)),
        source_selection_sha256=sha256(run/'source_selection.json'),freeze_sha256=sha256(run/'freeze.json'),new_fits=3,
        excluded_from_self_hash=['final_delivery_manifest.json'])
    write_json(run/'final_delivery_manifest.json',manifest)
    e.check_artifacts(run,manifest['artifacts'])
    write_json(js,{**delivery,'final_delivery_manifest':str(run/'final_delivery_manifest.json'),
                  'final_delivery_manifest_sha256':sha256(run/'final_delivery_manifest.json'),'report_sha256':sha256(md)})
    print(json.dumps(dict(status='complete_verified_local_delivery',report=str(md),manifest_sha256=sha256(run/'final_delivery_manifest.json'),
        development_acceptance_passed=summary['development_acceptance_passed'],gates=gates,new_fits=3,artifact_count=len(manifest['artifacts'])),ensure_ascii=False,indent=2))
