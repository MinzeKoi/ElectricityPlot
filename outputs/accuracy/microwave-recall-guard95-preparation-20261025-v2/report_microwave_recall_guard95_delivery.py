"""Render already-frozen fixed results; performs no inference or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main(run):
    def read(name):return json.loads((run/name).read_text(encoding='utf-8'))
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    s=read('source_selection.json');v=read('independent_execution_verification.json');summary=read('evaluation/summary.json');identity=read('identity.json');auth=read('human_rule_authorization.json')
    assert v['formal_passed']==summary['development_acceptance_passed'] is False
    assert s['selected']=='guard95' and s['guard95_eligible']
    parent=Path(identity['parent'])
    assert sha(parent/'final_delivery_manifest.json')==identity['parent_manifest_sha256']
    old=json.loads((parent/'evaluation/summary.json').read_text(encoding='utf-8'))
    assert old['development_acceptance_passed'] is False
    lines=['# Recall95 独立零训练对照：正式开发验收失败','',
        '固定来源预选guard95通过；一次正式开发评价未通过平均精确率8%和平均误报减少30%两项门槛。各seed召回率85%门槛通过。结果保留为失败，不部署、不切换UI、不继续调阈值或追加候选。','',
        '## 授权与执行边界','',
        '根对话01a16d7b-478e-7780-a999-88abe71e8c94转交真实人类答复：**'+auth['human_authorization_quote']+'**。问题明确为保留冻结模型及原开发验收，将来源峰值保护85%改为95%，按原W50预筛并封存评价一次。回执绑定注册SHA256：`'+auth['registration_sha256']+'`。配置保持authorized=false，执行依赖单独真实授权。','',
        '0新fit / 0 optimizer update。固定三模型、原epoch、归一化、原L、radius15和8秒规则不变；没有半径/温度/min-run/H90等搜索。source_select一次（三seed各一次H95选择），freeze一次，evaluate一次。原父任务H85失败结论及所有完成文件保持不变。','',
        '## 来源预选','',
        'House5固定block4/5、同34351个C31有效中心/201阳性；原W50 normalized_MAE选定epoch与固定200W来源身份精确对齐，显式只读stamps/raw。旧L完全重现；新H仅按预登记H95精确候选排序。','',
        '|seed|原L|旧H85|本轮H95|block4 R|block5 R|本轮源均P/R/F1|W50源均P/R/F1|','|---|---:|---:|---:|---:|---:|---|---|']
    proof=read('source_identity_replay.json')
    for recipe in s['recipes']:
        seed=recipe['seed'];b=recipe['blocks'];c=next(r for r in s['rows']['guard95'] if r['seed']==seed);w=next(r for r in s['rows']['W50'] if r['seed']==seed)
        fmt=lambda r:'/'.join(f'{r[k]*100:.4f}%' for k in ('precision','recall','f1'))
        lines.append(f"|{seed}|{recipe['L']}|{proof['seeds'][str(seed)]['H85']}|{recipe['H']}|{b[0]['recall']*100:.4f}%|{b[1]['recall']*100:.4f}%|{fmt(c)}|{fmt(w)}|")
    lines+=['','三个seed均source95/nondegenerate；均F1≥原W50，各seed均R未超过5pp下降限制，三个seed均P提高。按meanP/meanF1预选guard95，未使用开发结果改选。Task12 H85只作诊断，不参加本轮来源竞争。','',
        '## 三seed正式开发混淆矩阵','',
        '每seed同137523个有效点、383阳性/137140阴性，t/mask与原A和原W50精确一致。','',
        '|seed|方法|TP|FP|FN|TN|P|R|F1|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in v['rows']:
        if r['arm']=='A':continue
        lines.append(f"|{r['seed']}|{r['arm']}|{r['tp']}|{r['fp']}|{r['fn']}|{r['tn']}|{r['precision']*100:.4f}%|{r['recall']*100:.4f}%|{r['f1']*100:.4f}%|")
    lines+=['','原A与W50在本次固定mask上的混淆数相同；A完整记录在evaluation/confusions.csv。','',
        '|三seed均值|P|R|F1|FP|','|---|---:|---:|---:|---:|']
    for arm in ('W50','Task12_H85','guard95'):
        m=v['means'][arm];lines.append(f"|{arm}|{m['precision']*100:.4f}%|{m['recall']*100:.4f}%|{m['f1']*100:.4f}%|{m['fp']:.4f}|")
    labels=dict(mean_P_at_least_008='meanP≥8%',each_R_at_least_085='各seedR≥85%',mean_F1_gain_at_least_002='meanF1相对W50增益≥2pp（1e-12容差）',at_least_two_seed_P_F1_up='至少2seed P/F1同时提高',each_F1_drop_at_most_002='各seedF1下降≤2pp',mean_FP_reduction_at_least_30_percent='meanFP减少≥30%',no_new_all_on_or_off='无新增全开全关',support_137523_exact='每seed同137523有效点')
    values=dict(mean_P_at_least_008=f"{v['means']['guard95']['precision']*100:.4f}%",each_R_at_least_085='/'.join(f"{r['recall']*100:.4f}%" for r in v['rows'] if r['arm']=='guard95'),mean_F1_gain_at_least_002=f"+{v['mean_F1_gain']*100:.4f}pp",at_least_two_seed_P_F1_up='3seed',each_F1_drop_at_most_002='3seed均提高',mean_FP_reduction_at_least_30_percent=f"{v['mean_FP_reduction']*100:.4f}%",no_new_all_on_or_off='3seed均非退化',support_137523_exact='时间及mask exact')
    lines+=['','## 原样正式门槛','', '|门槛|实测|通过|','|---|---|---|']
    for key,passed in v['formal_gates'].items():lines.append(f"|{labels[key]}|{values[key]}|{'是' if passed else '否'}|")
    lines+=['','## 成对差异（本轮减对照）','',
        '|seed|对照|恢复TP|新增FN|移除FP|新增FP|ΔTP|ΔFP|ΔP(pp)|ΔR(pp)|ΔF1(pp)|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in v['paired']:lines.append(f"|{r['seed']}|{r['reference']}|{r['restored_tp']}|{r['new_fn']}|{r['removed_fp']}|{r['added_fp']}|{r['delta_tp']:+d}|{r['delta_fp']:+d}|{r['delta_P']*100:+.4f}|{r['delta_R']*100:+.4f}|{r['delta_F1']*100:+.4f}|")
    lines+=['','1019相对旧H85恢复61个TP、增加2358个FP，没有新增FN。新保护修复了旧召回不足，但精确率和F1均低于旧H85；旧H85自身仍因1019召回不足而失败，不能据此换回旧候选或重命名挽救。','',
        '## 冻结与独立复核证据','',
        '9份A/W50/guard95预测及普通aggregate-only API读回先冻结；随后三seed公开predict入口各导出150750行CSV，与冻结t/score/state逐项完全相同，并在读取真值前封存ordinary_pretruth_seal.json。','',
        '正式评价只读取Task12已核验truth_once.npz的t/y；没有重新打开House17标签CSV。独立复核用逐点标量循环重算12组混淆数（含旧H85），全部与正式结果相符；独立重算八项门槛和来源相对W50保护均一致。','',
        '候选所有raw功率列与原A exact，corrected功率亦保持；其他电器状态列与保存的原A状态exact。训练/更新硬禁，source/freeze/evaluate审计无禁止读取；最终8/9/20数据未访问。源码、权重、归一化、原L均由注册/输入hash保持。普通CSV为本轮对照产物，不构成上线。','',
        'House17为反复使用工程开发集；不声称独立新户泛化，也不把source95解释为跨户召回保证。','',
        '|封存对象|SHA256|','|---|---|']
    for name in ('registration.json','human_rule_authorization.json','source_selection.json','source_complete.json','freeze.json','ordinary_pretruth_seal.json','evaluation_complete.json','independent_execution_verification.json'):
        lines.append(f'|{name}|`{sha(run/name)}`|')
    target=run/'DETAILED_EXECUTION_REPORT.zh.md'
    with target.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
