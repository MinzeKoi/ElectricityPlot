"""Produce task-6 delivery documents from sealed results; no inference or fitting."""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from virtual_submeter.accuracy import microwave_regression_calibration as c


def main():
    run = ROOT/'outputs/accuracy/microwave-regression-calibration-20261025-v2'
    c.validate_freeze(run)
    c.stage_check(run, 'evaluation_complete')
    results = c.read_json(run/'results.json')
    source = c.source_rows(run)
    selection = c.read_json(run/'source_selection.json')
    validation = c.read_json(run/'delivery_validation.json')
    groups = ['W50', 'W50_CAL', 'W10_CAL', 'W1_CAL', 'W50_HN_CAL']
    averages = {}
    for g in groups:
        rows = [r for r in results['rows'] if r['candidate'] == g]
        averages[g] = {k: sum(r[k] for r in rows)/3 for k in
                       ('precision', 'recall', 'f1', 'false_positive', 'false_negative')}
    averages['W50_CAL']['mean_fp_change_fraction'] = averages['W50_CAL']['false_positive']/averages['W50']['false_positive']-1
    files = ['registration.json', 'source_rows.json', 'source_selection.json', 'source_support.json',
             'source_complete.json', 'freeze.json', 'recipe.json', 'evaluation_started.json', 'evaluation_complete.json',
             'verification.json', 'delivery_validation.json', 'execution_repair/receipt.json']
    delivered = {**results, 'source_rows': source, 'source_selection': selection, 'averages': averages,
                 'protocol_deviation': validation['protocol_deviation'], 'overall_acceptance_passed': False,
                 'run': run.relative_to(ROOT).as_posix(), 'fits': 0,
                 'artifact_sha256': {n: c.sha256(run/n) for n in files},
                 'pipeline_prelabel_flags_scope': 'automated stages only; earlier manual schema inspection disclosed separately',
                 'source_and_development_PR': {'source': 'source_pr/*.npz', 'development': 'development_pr/*.npz'},
                 'immutable_model_and_input_hashes': c.read_json(run/'registration.json')['inputs'],
                 'snapshot_source_hashes': c.read_json(run/'registration.json')['execution'],
                 'current_workspace_module_sha256': c.sha256(ROOT/'src/virtual_submeter/accuracy/microwave_regression_calibration.py')}
    prefix = ROOT/'docs'/c.PREFIX
    c.save(Path(str(prefix)+'_RESULTS_20261025.json'), delivered)
    tests = ROOT/'tmp/microwave-regression-calibration-tests-20261025.log'
    validated = {**validation, 'automated_stage_verification': c.read_json(run/'verification.json'),
                 'unit_tests': {'passed': 9, 'failed': 0, 'fits': 0, 'log': tests.relative_to(ROOT).as_posix(),
                                'log_sha256': c.sha256(tests)},
                 'repairs': ['v1 stopped at source support guard before calibration; v2 uses frozen block timestamps',
                             'v2 object-only JSON reader could not reopen source list; sealed wrapper repairs list parsing only',
                             'truncated recipe probe uses original numerical tolerance for tiny last-batch float32 differences; state/missing exact'],
                 'historical_direct_snapshot_entry_requires_compatibility_wrapper': True}
    c.save(Path(str(prefix)+'_VALIDATION_20261025.json'), validated)
    lines = ['# 冻结回归分数的来源状态阈值校准', '',
             '结论：未实现预测改进，保留原模型与默认界面。来源唯一预选为校准W50；其开发精确率、F1下降，误报增加。0次新拟合，未追加候选或检查点。', '',
             '|House17，三seed均值|Precision|Recall|F1|FP|FN|', '|---|---:|---:|---:|---:|---:|']
    for g in groups:
        r = averages[g]
        lines.append(f'|{g}|{r["precision"]:.6f}|{r["recall"]:.6f}|{r["f1"]:.6f}|{r["false_positive"]:.3f}|{r["false_negative"]:.3f}|')
    lines += ['', 'W50表示原固定200W基准；CAL表示独立状态决策头。只有W50_CAL是来源预选者，其余行仅评价后诊断，不据此换候选。', '',
              '预选者对原基准：平均P 4.0128%→3.5758%，R 92.4282%→96.2576%，F1 7.6913%→6.8945%；F1变化−0.007968，FP平均增加17.8218%。每seed召回保护通过，但平均P、F1增益、至少2seed同时提高P/F1、误报降低30%四项均失败。', '',
              '## 来源选择证据', '',
              '只使用House5六个已冻结UTC双日块，按House5协议身份与时间戳匹配原回归raw分数及>=200W标签；块中缺失不当关闭。任务4的边界剔除原样保留，不使用其raw_index作为回归索引。每块有效点数/正例数为18063/676、20518/205、20641/303、16213/183、19127/136、15224/65，合计109786/1568。', '',
              '每模型枚举精确PR阈值，单阈值须六块均R>=85%且每块状态混合，按最差P、块均P、块均F1、较高阈值确定性选。12套模型均有合格阈值。来源合并均P/F1：校准W50为0.116622/0.207233，W10为0.095611/0.173938，W1为0.013389/0.026401，HN为0.102942/0.186243；三个非W50家族均因F1低于校准控制而不合格。', '',
              '|家族|seed|来源阈值W|来源P|来源R|来源F1|', '|---|---:|---:|---:|---:|---:|']
    for r in source:
        m = r['metrics']
        lines.append(f'|{r["candidate"]}|{r["seed"]}|{r["threshold"]:.12g}|{m["precision"]:.6f}|{m["recall"]:.6f}|{m["f1"]:.6f}|')
    lines += ['', 'W1在固定200W下高分端较好的历史观察不能证明高召回端有效。本轮六块召回保护将其阈值压到接近零，开发误报达到每seed约10.7万～11.6万；这仅支持本次四族及固定来源校准协议的负结论。', '',
              '## 逐seed开发混淆', '', '|候选|seed|TP|FP|FN|P|R|F1|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results['rows']:
        lines.append(f'|{r["candidate"]}|{r["seed"]}|{r["true_positive"]}|{r["false_positive"]}|{r["false_negative"]}|{r["precision"]:.6f}|{r["recall"]:.6f}|{r["f1"]:.6f}|')
    lines += ['', '每seed137523个有效点、383个真实开机点；各候选时间掩码与原固定200W基准完全相同。固定基准逐seed混淆、P/R/F1及支持哈希与旧precision结果精确重现。', '',
              '## 功率与归因', '',
              '仅校准微波炉状态。组合输出微波炉仍沿用原A功率；水壶/洗衣机原始功率与状态逐点相同。旧W10/W1/HN回归功率误差单列power_diagnostics.csv，不能称为本任务功率改善。来源MAE选中权重、selected_epoch、归一化、训练/验证家庭和原A三seed参数相等证据见VALIDATION JSON的ancestry。', '',
              '## 核验范围与偏差', '',
              '24份完整普通推理重放覆盖12模型×来源/开发，每份151200/150750行，raw、缺失及阈值状态逐点完全一致。另有导出配方普通入口2033行核验，截断末批raw最大差1.90735e−6 W，沿用rtol=1e−6/atol=2e−4 W容差，状态/缺失完全一致。来源精确PR另作864个直接混淆核对；9项专项测试通过，均无拟合。以上为本机自动核验，不是独立成员或第二机器复现。', '',
              '**隔离偏差：**准备阶段为检查旧真值缓存字段形状，NumPy曾解压House17的y数组；只查看t/y形状和dtype，未查看标签值或统计、未用于阈值选择。但这不满足严格的“预测封存前零标签读取”。自动阶段中的development_labels_parsed=false仅描述阶段代码，不覆盖此人工检查。数值出口本身失败，且本轮不具备完全盲化验收资格。后续只作冻结结果诊断，不重复搜索或将其称作新户泛化证明。', '',
              '工程记录：v1在来源块支持检查中止，未产出阈值；v2按冻结块t字段对齐。v2来源完成后，公共JSON读取器拒绝列表格式，新增封存execution_repair包装器，仅修复source_rows列表读取；旧快照及来源结果未改，已完成推理未重跑。当前工作区模块已修复该接口，未来新运行可直接使用。', '',
              'House17此前已反复查看；最终House8/9/20任何数据未访问。任务4、任务5及其他旧目录未修改。本轮是新状态协议，不能称旧固定200W方案已经通过。', '',
              '## 使用与复核', '',
              '配方可加载作诊断，效果未通过，不建议替换原模型。预测只读取总表CSV，输出raw分数、来源阈值和状态（空值为不可用），不修改功率、不训练、不切换UI。', '', '```powershell',
              '$run = "outputs/accuracy/microwave-regression-calibration-20261025-v2"',
              '.venv-m2/Scripts/python.exe "$run/execution_repair/run_microwave_regression_calibration_snapshot.py" verify --run $run',
              '.venv-m2/Scripts/python.exe scripts/verify_microwave_regression_calibration_delivery.py --run $run',
              '.venv-m2/Scripts/python.exe scripts/run_microwave_regression_calibration.py predict --recipe "$run/recipe.json" --candidate W50 --seed 42 --input "your_aggregate.csv" --output "new_state_output.csv"',
              '$env:PYTHONPATH = "src"',
              '.venv-m2/Scripts/python.exe -m unittest discover -s tests -p test_microwave_regression_calibration.py -v', '```', '',
              '完整新运行用工作区入口依次prepare/source_select/freeze/evaluate/verify/report，并为每步指定同一全新--run路径。旧目录禁止覆盖。对已完成阶段调用会核对封存结果；普通推理收据允许中断后复用，不新增模型。', '',
              '交付：RESULTS JSON含全部来源块/阈值、逐seed开发混淆、误报增减、功率诊断与输入/源码哈希；VALIDATION JSON含测试、重放、溯源和隔离偏差。运行目录保留source_pr/、development_pr/精确PR数组，comparison.csv、false_positives.csv、recipe.json及完整阶段收据。']
    text = '\n'.join(lines)+'\n'
    path = Path(str(prefix)+'_REPORT_20261025.md')
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise RuntimeError('Delivery report differs')
    path.write_text(text, encoding='utf-8')
    delivered_files = {k: v for k, v in c.inventory(run).items()
                       if k not in ('delivery_manifest.json', 'final_delivery_manifest.json', 'combined_export.json')
                       and not k.startswith('combined_predictions/')}
    c.save(run/'delivery_manifest.json', {'run_artifacts': delivered_files,
           'documents': {p.relative_to(ROOT).as_posix(): c.sha256(p) for p in (ROOT/'docs').glob(c.PREFIX+'*20261025.*')},
           'authoritative_report': path.relative_to(ROOT).as_posix(), 'overall_acceptance_passed': False,
           'fits': 0, 'final_households_accessed': False})
    print(json.dumps({'report': str(path), 'acceptance_passed': False, 'fits': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
