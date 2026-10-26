"""Render the frozen audit tables and disabled next-step proposal in Chinese."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/accuracy/microwave-source5-loss-preparation-20261025-v4'
OUT=ROOT/'outputs/accuracy/microwave-gate-recall-audit-20261025-v1'
DOC=ROOT/'docs/MICROWAVE_GATE_RECALL_AUDIT_20261025.md'

def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def table(headers,values):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in values])

def main():
    summary=js(OUT/'summary.json'); d=summary['development']; source=summary['source']
    selection=js(RUN/'source_selection.json'); detail=js(OUT/'detail_summary.json')
    contrasts=js(OUT/'lost_fragment_contrasts.json'); peak=js(OUT/'peak_distance_summary.json')
    margins=rows(OUT/'margins.csv'); config=js(RUN/'config.json')
    formal=js(RUN/'evaluation/summary.json'); independent=js(OUT/'independent_verification.json')
    assert independent['passed'] and not summary['eachseedR85']
    fixed=dict(parent_run=str(RUN),selected='new_gate',seeds=config['seeds'],recipes=selection['recipes'],truth_threshold_w=200,radius_samples=15,interval_seconds=8,source_blocks=[4,5],evaluation_support=137523,positive_support=383,new_fits=0,optimizer_updates=0,threshold_searches=0,extra_development_candidates=0,final_houses_read=False,original_development_label_csv_read=False,source_only_descriptive_groups=js(OUT/'group_definitions.json'),replay_command='.venv-m2/Scripts/python.exe scripts/audit_microwave_gate_recall.py --assert-acceptance',expected_acceptance_exit_code=2)
    fixed.update(read_only_default_replay=True,read_only_independent_verifier=True,full_reaudit_requires_explicit_new_directory=True,existing_output_rejected=True)
    write(OUT/'fixed_config.json',fixed)
    proposal=dict(status='disabled',executed=False,title='来源逐seed逐块召回退化拒绝条件',scope='仅作为下一轮预注册提案；任何seed×来源块中固定gate召回低于其对应冻结point召回，则gate不具资格进入开发。不得以均值掩盖该条件。',changes_previous_fixed_rules=True,changes='新增来源资格规则，不改变本轮L/H/radius、来源选择或正式开发失败。',new_fit_budget_required_for_this_guard=False,new_fits_proposed=0,new_training_requires_separate_human_authorization=True,development_evaluation_requires_separate_authorization=True,not_proposed='本审计不生成段级gate，不扫半径/H，不后选本轮point，不计算新规则开发成绩。',limitation='来源保护可以减少已有来源召回退化，不保证跨户召回≥.85，也不能自动选择更优模型。')
    write(OUT/'disabled_next_step.json',proposal)
    main_table=table(['seed','TP','FP','FN','TN','R','非有限','p<L','p≥L且h<H','point TP','新增FN','恢复TP'],[[r['seed'],r['tp'],r['fp'],r['fn'],r['tn'],f"{r['recall']:.6f}",r['nonfinite'],r['low_fail'],r['high_fail'],r['point_tp'],r['added_fn'],r['recovered_tp']] for r in d])
    src_table=table(['seed','源块','TP/正例','FP','FN','TN','R','p<L','高峰失败','新增FN/恢复TP'],[[r['seed'],r['part'][-1],f"{r['tp']}/{r['tp']+r['fn']}",r['fp'],r['fn'],r['tn'],f"{r['recall']:.6f}",r['low_fail'],r['high_fail'],f"{r['added_fn']}/{r['recovered_tp']}"] for r in source])
    def m(seed,part,group,field,q): return float(next(r[q] for r in margins if int(r['seed'])==seed and r['part']==part and r['group']==group and r['field']==field))
    margin_table=table(['seed','分区','真开机p−L q05 / q50','真开机h−H q05 / q50','FP p−L q50','FP h−H q50'],[[seed,part,f"{m(seed,part,'true_on','p_minus_L','q05'):.4f} / {m(seed,part,'true_on','p_minus_L','q50'):.4f}",f"{m(seed,part,'true_on','h_minus_H','q05'):.4f} / {m(seed,part,'true_on','h_minus_H','q50'):.4f}",f"{m(seed,part,'false_positive','p_minus_L','q50'):.4f}",f"{m(seed,part,'false_positive','h_minus_H','q50'):.4f}"] for seed in [42,1019,2026] for part in ['source_4','source_5','development']])
    frag_table=table(['UTC开始','n','新增FN','1019峰h最大','42 gate TP','2026 gate TP','同源分组片段数'],[[r['start_utc'],r['n'],r['added_fn'],f"{r['h_max']:.4f}",r['42_gate_tp'],r['2026_gate_tp'],r['source_similar_fragments']] for r in contrasts['lost_fragments']])
    peak_table=table(['seed','分区','高峰失败点','低L段数','同低L段有H峰','无同段H峰','完整有限段也无H峰'],[[r['seed'],r['part'],r['high_fail_points'],r['unique_low_segments'],r['same_low_segment_has_high'],r['no_same_low_segment_high'],r['no_high_in_full_finite_segment']] for r in peak])
    pp=next(r for r in peak if r['seed']==1019 and r['part']=='development')
    text=f'''# 固定微波炉门控召回只读审计（2026-10-25）

权威父轮：`outputs/accuracy/microwave-source5-loss-preparation-20261025-v4`。本审计完成回放与独立复核，未发现本路径的实现不一致；正式失败保留。唯一已选来源候选仍为 new_gate，不能后选 new_point。父轮3个fit、10584次更新已用尽；本审计新增fit=0、optimizer更新=0、阈值搜索=0、新开发候选=0。默认UI、功率输出、模型和父轮产物未修改。

## 确定性反馈与访问边界

秒级只读入口：`.venv-m2/Scripts/python.exe scripts/audit_microwave_gate_recall.py --assert-acceptance`。仅stdout返回计数和退出码，不写入审计目录；已运行，内部耗时约 {summary['elapsed_seconds']:.3f} 秒；`eachseedR85=false`，预期退出码2。普通回放模式退出0仅表示回放通过，不表示正式达标。完整重新取证必须显式指定不存在的新目录：`.venv-m2/Scripts/python.exe scripts/audit_microwave_gate_recall.py --build-source --details --verify-ordinary --output outputs/accuracy/NEW_AUDIT_DIRECTORY`。仅允许outputs/accuracy下全新同级目录，已存在即拒绝，缺少--output也拒绝；只运行冻结模型前向推理。独立只读入口：`.venv-m2/Scripts/python.exe scripts/verify_microwave_gate_recall_audit.py`，仅stdout返回复核结果，独立计数、deque窗口、FN清单与峰距离复核9个seed×分区，通过。只读入口已禁止字节码写入，回放还以运行时文件写入拦截器约束。

开发仅用本轮已封存 `evaluation/truth_once.npz` 的显式t/y键；未重开原House17标签CSV，未读最终House8/9/20数据。源只读identity明确白名单，块4/5使用父validation t/y；源aggregate整块推理后保存完整p。训练覆盖只读父train.npz的t/y/event/house/outer_fraction五个字段。NPZ不遍历items/values。数值读取白名单和显式键见 `full_read_boundary.json`；缓存来历、源码和模型哈希见 `source_cache_provenance.json`、`full_input_hashes.json`、`audit_manifest.json`。

根目录及相关src/scripts/docs未找到AGENTS.md、GLOSSARY.md或相关ADR；已读DATA_CONTRACT，遵守UTC、8秒、缺失不填零、标签仅评价的约束。使用diagnosing-bugs的真实轨迹回放，不预设代码bug。未将支持缩为仅FN：h依赖完整有限p段，去掉非评价有限中心会改变真实路径。确定性回放只能核验冻结结果，不能证明泛化。

交付入口的只读契约检查：`.venv-m2/Scripts/python.exe scripts/verify_microwave_gate_recall_entry.py`，本身仅stdout输出；逐文件SHA256和修改时间前后核对，确认秒级回放及独立复核不改审计文件，并验证缺少新输出目录和使用已有目录都会退出2拒绝。见 `readonly_entry_verification.json`。初版回放会重写审计文件的交付问题已纠正，不涉及模型、门控规则或父轮数据修改。

## 正式计数与漏报机制

每seed评价点137523，真开机383；mask为cached truth有限且原W50微波炉raw有限。t逐点一致、UTC为8秒整数倍，p/h非有限支持一致。h在完整aggregate有限p、相邻8秒段上对原p做一次中心max，半径15；包含开发1914、源块4为860、源块5为362个非评价但有限分数点，未从标签有效中心重建。

{main_table}

1019的point→gate TP净少52，实际为新增59、恢复7。三个seed的低门槛L分别为0.09719955、0.11615904、0.06281075；H分别为0.52479666、0.82535273、0.71694928。point门槛分别为0.16646883、0.15353359、0.06281075。故L不高于point阈值，所有新增FN均来自高峰条件；不能把31个低L失败都解释为门控新增。

{src_table}

正式meanP={formal['formal_means']['new_gate']['precision']:.7f}、R={formal['formal_means']['new_gate']['recall']:.7f}、F1={formal['formal_means']['new_gate']['f1']:.7f}、FP={formal['formal_means']['new_gate']['fp']:.2f}，原W50 FP=8482；正式门槛中仅eachseedR85失败。new_point三个seed召回虽均≥.85，平均P约.05762，未达P=.08，且它并非本轮可后选正式候选。

## 预列五个可证伪原因及检验结果

A：若p低于L是门控额外丢失主因，则新增FN应落在p<L。结果反驳：新增59个1019 FN均已过point阈值且≥L；其全部92个FN中仍有31个低L失败，是基础分数不足的重要组成。

B：若H峰值要求拒绝了已过L的真实点，则高峰失败应占主导。支持：1019有61/92个FN属于p≥L且h<H，新增59个在其中。源块4此类为0，块5为5；开发显著恶化。这里只归因固定规则触发条件，不据此选新H或宣称降低H会达标。

C：若跨户输入/功率/背景/缺失差异相关，错误应在预定分组或同时间片段中呈差异。观察支持分布差异，但不足以建立因果。source-only真开机功率四分位切点1153.5/1416.5/1614.375 W，背景（aggregate−truth，可能受噪声影响）切点478.875/565/693.375 W。1019新增FN在背景Q1/Q2/Q3/Q4分别28/4/0/27，既有低背景也有高背景，不能只归咎高背景。完整外圈360个开发真点中检出281、部分外圈23个中检出10；59个新增FN中47个外圈完整。源块5全部65个真点外圈不完整仍检出57，故缺外圈不能单独解释1019问题。

D：若不同seed/选中epoch的分数峰值校准不同，则同点错误和固定H margin会分离。支持：epoch为42→5、1019→12、2026→11；1019 H最高。同一批59个1019新增漏报点，42既有gate检出59，2026检出48。初始化和洗牌同时变化、epoch也不同，不可单独归因初始化或模型容量。

E：若对齐/缺失/全aggregate门控有实现错误，则独立窗口、计数、普通推理或封存state应不一致。当前证据反驳本路径此假设：shift-max、sliding-window实现以及独立deque实现一致；源全seed普通前向分数与所选epoch缓存一致（浮点allclose且point判断完全一致），开发1019普通前向与冻结全p完全一致；9组TP/FP/FN/TN与FN分解一致。283个父artifact哈希在回放和独立复核后全部一致。此结论限定于读取路径，不保证所有系统代码无bug。

## source→development margin

{margin_table}

分布表给出固定L/H的有符号margin，未改变阈值。完整q05/q25/q50/q75/q95与n、真开机/误报/真关闭/高峰拒绝真开机的功率、aggregate及背景分布见 `margins.csv`。1019开发真开机h−H中位数仍为正，却存在61个高峰失败；一次组均值或中位数好不能推出尾部召回受保护。

## 1019损失片段与覆盖

下表是评价支持内、y≥200W且相邻8秒的可观察阈值片段，遇缺标签、缺支持或时间断裂分开；不是物理运行cycle。59个新增FN集中于9片段，前三大16+12+11=39点。逐点p/h/margin与其他seed判断见 `false_negatives.csv`、`seed1019_added_fn_cross_seed.csv`，全部片段见 `fragments.csv`。

{frag_table}

源块4/5合计只有201个真点、20个可观察阈值片段。源片段自身的长度/均值功率/均值背景四分位定义描述性三维组，9个损失片段中7个组在源验证无同组片段，其余各仅1个。切点和组定义见 `lost_fragment_contrasts.json`，这只是稀疏支持证据；20片段的高维空格本身不能证明域偏移。父训练正中心2275：house代码0（House2）466点/74片段、1（House19）383点/80片段、2（House5）1426点/136片段；训练片段和event ID均不当作独立物理cycle。

## 既有高峰最近距离（只诊断当前FN）

{peak_table}

1019开发61个高峰失败点分属{pp['unique_low_segments']}个低L连通段，**同低L段达到H的峰值为0**。其完整有限8秒段内都有远处达到H的峰，但到最近峰距离q05/q50/q95为{pp['nearest_full_high_distances_seconds']['q05']:.0f}/{pp['nearest_full_high_distances_seconds']['q50']:.0f}/{pp['nearest_full_high_distances_seconds']['q95']:.0f}秒；这些峰与当前点之间已被p<L截断。低L段观察时长q05/q50/q95为{pp['low_segment_seconds']['q05']:.0f}/{pp['low_segment_seconds']['q50']:.0f}/{pp['low_segment_seconds']['q95']:.0f}秒，按n×8计，首末时间差另列，不跨缺口。

例如2014-10-15T15:32:32Z的1019 FN处于15:32:32–15:34:32低L段（16点、128秒），段最大p=.5004515<H=.8253527；完整有限段最近H峰在15:42:08，距576秒且不属于同低L段。2026有8/11个高峰失败点在同低L段有远于120秒的既有H峰，42为5/26；这是seed特定结构现象，不能用于证明1019可由段级峰门控解决。逐点最近峰、低L段边界/持续长度、其他seed同点及同段已有point/gate计数见 `high_fail_peak_distances.csv`。未生成整个component gate状态、未扫半径或H、未计算新结构开发成绩。

## 未解问题与唯一disabled提案

本路径证据支持“1019跨户分数与H峰值校准未保住尾部召回”，尚不能分离家庭电器形态、背景混合、优化轨迹、epoch选择各自因果贡献；重复开发House17也不是新户独立验证。冻结point在开发的较高召回不意味着精确率达标。没有证据支持为1019直接改为低L段峰门控作为最小修复。

唯一下一轮提案已写入 `disabled_next_step.json`，status=disabled：在来源资格判定加入逐seed×块的gate召回不得低于对应冻结point的拒绝条件，禁止只看均值。它改变此前来源资格规则；仅实施资格检查本身不需新fit预算，但本审计未执行该新选择、未形成候选、未评价新规则开发成绩。该保护不保证跨户R≥.85。后续任何新开发实验或真实训练须由根协调者另获人类授权；当前3fit预算已耗尽。

合成验证覆盖NaN、时间断裂、一次中心max、非评价有限峰、low-fail FN、新增FN/恢复TP分别计数、缺标签/时间断裂片段分离。没有静默修补父轮，没有将统计失败包装成代码修复完成。审计复核成功，正式验收仍失败。
'''
    DOC.write_text(text,encoding='utf-8')
    (OUT/'REPORT.zh.md').write_text(text,encoding='utf-8')
    inputs=[ROOT/'scripts/audit_microwave_gate_recall.py',ROOT/'scripts/verify_microwave_gate_recall_audit.py',ROOT/'scripts/verify_microwave_gate_recall_entry.py',Path(__file__),ROOT/'src/virtual_submeter/accuracy/microwave_event_gate.py',ROOT/'src/virtual_submeter/models/center_mask_tcn.py',RUN/'final_delivery_manifest.json',DOC]
    write(OUT/'audit_manifest.json',dict(parent_manifest_sha256=sha(RUN/'final_delivery_manifest.json'),source_and_report_hashes={str(p):sha(p) for p in inputs},artifacts={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='audit_manifest.json'},status='read_only_audit_complete_formal_acceptance_failed'))
    print(str(DOC))

if __name__=='__main__': main()
