"""Write new local delivery artifacts after review; do not modify a completed run."""
import argparse
from pathlib import Path
import shutil
import sys
import json

sys.dont_write_bytecode = True
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
a = p.parse_args()
run = a.run.resolve()
sys.path.insert(0,str(run/'source'))
from virtual_submeter.accuracy import microwave_event_gate as e


if __name__=='__main__':
    e.require(not (run/'delivery_manifest.json').exists(),'Completed run cannot be modified')
    review = e.read_json(run/'independent_delivery_review.json')
    e.require(review['ordinary_cli_all_150750_rows_exact'] and review['independent_acceptance_gates_exact']
        and review['final_verification']['status']=='verified','Final review incomplete')
    e.require(e.sha256(Path(__file__).with_name('verify_microwave_event_gate.py'))==review['verification_script_sha256'],'Review script changed')
    e.verify(run)
    source = e.read_json(run/'source_selection.json')
    summary = e.read_json(run/'evaluation/summary.json')
    metrics = e.read_json(run/'evaluation/metrics.json')
    identity = e.read_json(run/'identity.json')
    parent = Path(identity['parent'])
    default_protocol = parent/'source/virtual_submeter/data/protocol.py'
    workspace = Path(__file__).resolve().parents[1]
    e.require(e.sha256(workspace/'src/virtual_submeter/data/protocol.py')==e.sha256(default_protocol),'Default protocol changed')
    test_logs = {}
    for name,expected in [('gate_tests.log',21),('protocol_tests.log',25)]:
        path = run/'validation_evidence'/name
        text = path.read_text(encoding='utf-8')
        e.require(f'Ran {expected} tests' in text and '\nOK' in text,'Tests not successful')
        test_logs[name] = dict(count=expected,passed=True,sha256=e.sha256(path))
    # Failed attempts remain intact. v3 stopped only at JSON serialization;
    # all scores, thresholds and decisions must exactly match the successful run.
    previous = run.parent/'microwave-event-gate-20261025-v3'
    same = []
    for name in ['block_metrics.csv']+[f'{kind}_curve_{seed}.csv' for kind in ('L','H') for seed in e.SEEDS]:
        e.require(e.sha256(previous/'source_results'/name)==e.sha256(run/'source_results'/name),'Rerun altered source result')
        same.append(name)
    for seed in e.SEEDS:
        old = e.read_json(previous/f'source_results/recipe_{seed}.json')
        current = next(r for r in source['recipes'] if r['seed']==seed)
        e.require(all(old[k]==current[k] for k in ('L','H','eligible','parent_epoch','parent_point_threshold')),'Rerun altered recipe')
    gate = summary['means']['gate']
    base = summary['means']['W50']
    point = summary['means']['parent']
    reductions = dict(vs_W50=1-gate['fp']/base['fp'],vs_parent=1-gate['fp']/point['fp'])
    source_counts = review['blocks']
    delivery = dict(schema='microwave_event_gate_delivery_v1',run=str(run),source_selected=source['selected'],
        gate_source_qualified=source['gate_qualified'],source_protection=source['protection'],summary=summary,
        source_seed_rows=dict(parent=source['parent_rows'],gate=source['gate_rows']),development_rows=metrics['rows'],
        source_recipes=source['recipes'],false_positive_reduction=reductions,
        false_positive_change_counts=review['false_positive_change_counts'],tests=test_logs,
        new_fits=0,new_training_model_initializations=0,optimizer_updates=0,task10_new_models_retrained=0,
        parent_weights_and_metadata_rechecked=True,original_development_support=metrics['supports'],
        ordinary_prediction_cli_rows=150750,ordinary_prediction_cli_exact=True,
        source_extra_non_evaluation_centers_per_seed=1222,source_truth_not_used_for_feature_segments=True,
        original_protocol_sha256=e.sha256(default_protocol),freeze_sha256=e.sha256(run/'freeze.json'),
        source_selection_sha256=e.sha256(run/'source_selection.json'),final_review_sha256=e.sha256(run/'independent_delivery_review.json'),
        failed_attempts=[dict(run='microwave-event-gate-20261025-v1',stage='prepare code-file audit guard too broad',source_thresholds_computed=False),
            dict(run='microwave-event-gate-20261025-v2',stage='strict historical score equality caught CPU tail batch roundoff',source_thresholds_computed=False),
            dict(run='microwave-event-gate-20261025-v3',stage='numpy bool JSON serialization after deterministic source computation',source_thresholds_computed=True)],
        failed_attempts_new_fits=0,failed_attempts_development_truth_reads=0,v3_v4_identical_source_curves=same,
        roundoff_note='Full aggregate batches include unlabeled centers; max historical score difference 3.183231456205249e-12; original point decisions exactly unchanged. Comparison rtol=1e-5, atol=1e-7; all current ordinary replay scores/states exact.',
        final_houses_accessed=False,default_changed=False,power_improvement_claim=False,independent_new_house_validation=False)
    docs = workspace/'docs'
    md = docs/'MICROWAVE_EVENT_GATE_DELIVERY_20261025.md'
    js = docs/'MICROWAVE_EVENT_GATE_DELIVERY_20261025.json'
    e.require(not md.exists() and not js.exists(),'Delivery documents already exist')
    lines = ['# 固定冻结模型事件高分门控：任务11交付','',
        '**来源门控通过，开发整体验收未达标。全程0次新训练、0次优化更新，任务10的3个新增House5模型未重训。**','',
        '|开发组|平均P|平均R|平均F1|总FP|','|---|---:|---:|---:|---:|']
    for arm in ('W50','parent','gate'):
        m = summary['means'][arm]
        total = sum(r['fp'] for r in metrics['rows'] if r['arm']==arm)
        lines.append(f'|{arm}|{m["precision"]:.6f}|{m["recall"]:.6f}|{m["f1"]:.6f}|{total}|')
    lines += ['',f'门控相对W50：平均F1增加{gate["f1"]-base["f1"]:.6f}，误报减少{100*reductions["vs_W50"]:.2f}%；相对父point规则：平均F1增加{gate["f1"]-point["f1"]:.6f}，误报减少{100*reductions["vs_parent"]:.2f}%，平均R变化{gate["recall"]-point["recall"]:+.6f}。',
        '验收要求平均P≥0.08、F1提升≥0.02及误报下降≥30%，这三项未通过。各seed R≥0.85、至少2seed P/F1同升、任seed F1不下降超过0.02及非退化均通过。来源预选不因开发结果改变；未切默认、未追加候选。','',
        '|seed|固定L|固定H|父epoch|父point阈值|','|---|---:|---:|---:|---:|']
    for r in source['recipes']:
        lines.append(f'|{r["seed"]}|{r["L"]:.12f}|{r["H"]:.12f}|{r["parent_epoch"]}|{r["parent_point_threshold"]:.12f}|')
    lines += ['','来源只有固定block4/5（Nov23–27 UTC），沿用原34351/201个完整C31评价中心。各seed先L保护每块R≥.95，再固定半径15（120秒）的一次中心max，以H保护每块R≥.85。三个seed全部可行且通过来源保护，来源均P 0.209635→0.220452，均F1 0.340164→0.354818。',
        '每块完整aggregate推理得到19987/15586个有限分数，其中860/362个不属于原评价中心，仍只作为无标签特征；标签缺失不当关闭，也不决定max连续段。连续段只由原始有限score、8秒连续与块边界决定。低于L的原评价点保留off/FN；独立复核逐块核对相应数量。','',
        '|seed|开发P|开发R|开发F1|TP|FP|FN|TN|FPR|预测开机段|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics['rows']:
        if r['arm']=='gate':
            lines.append(f'|{r["seed"]}|{r["precision"]:.6f}|{r["recall"]:.6f}|{r["f1"]:.6f}|{r["tp"]}|{r["fp"]}|{r["fn"]}|{r["tn"]}|{r["fpr"]:.6f}|{r["predicted_on_segments"]}|')
    lines += ['','|seed|对照|新增FP|移除FP|FP净变化|','|---|---|---:|---:|---:|']
    for r in review['false_positive_change_counts']:
        lines.append(f'|{r["seed"]}|{r["reference"]}|{r["FP_added"]}|{r["FP_removed"]}|{r["FP_delta"]}|')
    lines += ['','## 复核和边界','',
        '21项门控合成测试与25项原协议测试通过。独立直方图阈值枚举、填充窗口max、标量TP/FP/FN/TN与来源保护/开发验收均一致。普通CLI的150750行t/score/h/state与封存完全一致，三个seed的普通加载API也逐点重放；开发每seed严格保持原137523点与时间mask。',
        '任务10已完成终态数值复核后才正式执行。prepare/source访问日志不含House17数值输入；6份开发预测、source预选及普通推理重放先封存，才一次读取已核验旧真值缓存。复核只重读本任务封存缓存。确认终态时工具意外返回任务10已有汇总开发结果的情况已在预登记protocol.md记录；未读逐点误报来设计规则。',
        '权重、epoch和归一化保持；加载网络结构以读回冻结权重，不产生新初始化训练。运行期间禁止建立优化器和backward，0新训练/0更新可从入口、访问记录和父权重哈希复核。',
        '源完整aggregate含额外无标签中心，CPU最后一批大小变化使历史分数最多相差3.183×10⁻¹²；使用rtol=1e-5、atol=1e-7核对历史score，同时严格要求父point状态完全相同。新普通入口与本任务封存分数仍严格逐点一致。',
        '准备v1因审计守卫误拦代码文件而停止，v2因上述极小浮点严格比较停止，v3在来源结果JSON序列化时停止；均原样保留、0拟合、未读开发真值。只修复执行/记录检查，算法和参数未变；v3与正式v4的全部L/H曲线、来源混淆及配方阈值已核对完全一致。',
        '最大有效上下文157样本约21分钟，仍含离线未来上下文。House5末4日是历史复用来源验证，House17只是已有开发工程试验，不称独立新户验证；最终8/9/20任何数据未访问。原A功率、其他电器、默认UI与默认协议保持，无功率改善声明。','',
        '## 文件与复核命令','',
        f'- [完整试验报告](<{run.as_posix()}/REPORT.zh.md>)',
        f'- [来源逐块指标、L/H曲线与普通recipe](<{run.as_posix()}/source_results/block_metrics.csv>)',
        f'- [来源独立复核（含无标签特征中心）](<{run.as_posix()}/independent_source_review.json>)',
        f'- [逐seed开发混淆](<{run.as_posix()}/evaluation/development_confusions.csv>)',
        f'- [开机预测段](<{run.as_posix()}/evaluation/predicted_on_segments.csv>)',
        f'- [误报增加/移除逐点对照](<{run.as_posix()}/evaluation/false_positive_changes.csv>)',
        f'- [最终独立复核](<{run.as_posix()}/independent_delivery_review.json>)',
        f'- [整包SHA-256清单](<{run.as_posix()}/delivery_manifest.json>)','',
        '```powershell',f'.venv-m2/Scripts/python.exe -B "{run}/run_microwave_event_gate.py" verify --run "{run}"','```','',
        '公共predict入口只接受合格recipe；只需aggregate输入。已实际运行seed42，其他seed替换recipe文件即可，输出不可覆盖：','',
        '```powershell',f'.venv-m2/Scripts/python.exe -B "{run}/run_microwave_event_gate.py" predict --run "{run}" --recipe "{run}/source_results/recipe_42.json" --aggregate "<aggregate.csv>" --output "<新输出.csv>"','```']
    md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copyfile(md,run/'DELIVERY.zh.md')
    e.fresh_json(run/'delivery_summary.json',delivery)
    shutil.copyfile(Path(__file__),run/'validation_evidence/finalize_delivery.py')
    shutil.copyfile(Path(__file__).with_name('verify_microwave_event_gate.py'),run/'validation_evidence/verify_delivery.py')
    manifest = dict(artifacts=e.inventory(run),external_report=dict(path=str(md),sha256=e.sha256(md)),
        new_fits=0,optimizer_updates=0,source_selection_sha256=e.sha256(run/'source_selection.json'),
        freeze_sha256=e.sha256(run/'freeze.json'),excluded_from_self_hash=['delivery_manifest.json'])
    e.fresh_json(run/'delivery_manifest.json',manifest)
    e.check(run,manifest['artifacts'])
    e.fresh_json(js,delivery | dict(report=str(md),delivery_manifest=str(run/'delivery_manifest.json'),delivery_manifest_sha256=e.sha256(run/'delivery_manifest.json')))
    print(json.dumps(dict(status='complete_verified',report=str(md),new_fits=0,optimizer_updates=0,tests_passed=46,
        development_acceptance_passed=summary['development_acceptance_passed'],artifact_count=len(manifest['artifacts']),
        false_positive_reduction=reductions),ensure_ascii=False,indent=2))
