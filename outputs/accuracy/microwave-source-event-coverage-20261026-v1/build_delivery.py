"""Write immutable Chinese report, disabled proposal and delivery hashes."""
import csv,hashlib,json,math
from pathlib import Path
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2]
def read(name):return json.loads((RUN/name).read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    with (RUN/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')

s=read('summary.json');pool=read('sealed_pool_metadata.json');cfg=read('config.json')
assert read('independent_verification.json')['status']=='passed'
candidate=[x for x in s['houses'] if x['role']=='candidate_audit_only_not_training']
old=[x for x in s['houses'] if x['role']=='frozen_train_original14d']
added=sum(x['eligible_centers'] for x in candidate);pos=sum(x['eligible_positive_centers'] for x in candidate)
newn=336704+added;newpos=2275+pos;oldneg=336704-2275;newneg=newn-newpos
steps=math.ceil(294*newneg/oldneg);updates=12*steps

# Source-only earlier audit establishes why frozen labels are missing; no filling or new old-date parsing.
prior=ROOT/'outputs/accuracy/microwave-data-audit-20261025-v2/daily_support.json'
daily=json.loads(prior.read_text(encoding='utf-8'))['rows']
attribution={h:dict(aggregate_uncovered_seconds=sum(sum(r['uncovered_seconds_by_raw_cause'].values()) for r in daily if r['household']=='refit_'+h),
                   label_bins_lost_to_issues=sum(r['label_bins_lost_to_issues_after_resampling'] for r in daily if r['household']=='refit_'+h)) for h in ('2','19')}
save('frozen_missingness_attribution.json',dict(metadata_path=str(prior),sha256=sha(prior),houses=attribution,
    interpretation='Observed frozen labels are exactly90% finite; source metadata attributes losses to strict8 gap bins plus Issues label invalidation. No explicit artificial-mask flag found; do not infer acquisition missingness or artificial dropout solely from this ratio. No values repaired.'))

proposal=dict(status='DISABLED_AUDIT_PROPOSAL_ONLY',authorized=False,training_enabled=False,fit_calls_permitted_now=0,
    requires_separate_root_human_authorization=['extend original2/19 each14d source identity','new paired-control and expansion training budgets','any later source inference/calibration and separately gated development evaluation'],
    exact_additions=[{k:r[k] for k in ['house','start_utc','end_utc','eligible_centers','eligible_positive_centers','supported_segments']} for r in candidate],
    expected_combined=dict(windows=newn,positive_centers=newpos,negative_centers=newneg),
    data_rule='preselected consecutive complete UTC dates; retain low coverage and zero-positive days; core31>=24 observed/no adjacent missing, context127 original mask/boundary truncation; reject whole cross-role positive segments; audit counts must be independently reproduced before any train materialization',
    source_normalization=dict(mode='exact frozen source2/19 constants, no refit',input_log_mean=5.370561599731445,input_log_std=0.8337093591690063),
    house5='same existing prefix, no new dates, block4/5 same frozen validation centers; suffix never new training',
    seeds=[42,1019,2026],recipe='unit_BCE',pos_weight=1,model='unchanged CenterMaskTCN16 six dilation layers/context127/5329 parameters',
    optimizer='Adam lr0.001, batch256, positive16/negative240, CPU4',
    budget=dict(original_reference_updates=3528,old_negatives=oldneg,expanded_negatives=newneg,
                reason='expanded negative uniform-draw expectation must not fall below old3528 baseline; round up per checkpoint',
                formula='12*ceil(294*Nneg_expanded/Nneg_original)',steps_per_checkpoint=steps,checkpoints=12,updates_per_model=updates,
                old_reference_expected_draws_per_negative=3528*240/oldneg,expanded_expected_draws_per_negative=updates*240/newneg,
                paired_controls_new_fits=3,expanded_new_fits=3,total_new_fits_if_separately_approved=6,total_optimizer_updates_if_approved=updates*6,
                per_model_total_draws=updates*256,per_model_positive_draws=updates*16,per_model_negative_draws=updates*240),
    exposure_fairness=dict(primary_comparison='three paired same-seed refreshed controls and three expansion models; both receive exactly6468 updates, 12 equal checkpoints and same initialization/sampler/quota; frozen3528 controls are descriptive references only',
        caveat='Equal compute is not equal per-center or per-event exposure. Additional dates change hierarchical positive denominators; the single negative-based multiplier cannot match every event exposure.',
        positive_probability='P(center in positive slot)=1/(G_house_date*E_in_group*C_in_event); expected positive draws=16*U/(G*E*C)',
        negative_probability='P(negative center per draw)=1/Nneg; expected draws=240*U/Nneg',
        required_before_start='audit-only deterministic sampling ledger for both arms and all three seeds: each house/date/event/center exposures, unique centers/segments, positive mass per group, expected and realized counts; reject if original source support/16:240 quota or frozen identities change; no selecting dates/budgets by source or development performance',
        exact_exposure_claim_permitted=False),
    checkpoint_rule='12 source checkpoints; original source85 worst-P/mean-P/F1/earlier selection and original source relative qualification unchanged; stride changes294 to539 solely by preregistered budget and requires approval',
    terminal='fixed L95/H95/radius15, source-only calibration; no search',
    source_control=dict(each_seed_each_block_recall_min=.95,nondegenerate=True,mean_f1_not_below_control=True,max_seed_average_recall_drop=.05,min_seeds_precision_improved=2,rank=['mean_precision','mean_f1','control_on_tie']),
    development_gates=dict(mean_precision_min=.08,each_seed_recall_min=.85,mean_f1_gain_vs_W50_min=.02,min_seeds_precision_and_f1_improved=2,
        max_seed_f1_drop=.02,mean_fp_ratio_max=.7,nondegenerate=True,original137523_support_mask_raw_power_other_appliance_states_exact=True,tolerance=1e-12),
    final_houses='8/9/20 prohibited; House17 prohibited in audit; no development read or evaluation until separately authorized and unique source selection frozen',
    fallback_if_only_three_fits_approved='do not run this six-fit paired proposal; require a newly approved, explicit alternative. Never silently substitute frozen3528 controls or claim equal exposure.')
save('followup_proposal.disabled.json',proposal)
save('test_verification.json',dict(status='passed',tests=5,command='.venv-m2/Scripts/python.exe -B -m unittest discover -s outputs/accuracy/microwave-source-event-coverage-20261026-v1 -p test_boundaries.py -v',
    cases=['house/date allowlist','NaN/time-gap segmentation','9s gap retains unsupported coverage','two consecutive core missing rejected','label-independent chronological date calculation']))

lines=['# Task17 微波炉来源真实事件覆盖与扩展可行性审计','',
'本地存在可用的额外 House 2/19 真实总表与微波炉标签。按预先固定的连续日期，可增加审计合格中心 279,218 个、正中心 719 个，支持 160 个可观察正片段。本轮只核算计数，没有生成新增训练数组、拟合、optimizer 更新、模型推理、阈值校准或开发评价。',
'','## 授权与隔离','',
'用户授权为“允许新开并执行审计”，来源根聊天 01a16d7b-478e-7780-a999-88abe71e8c94 的 request_user_input_async call_ca4b91662f3b4bf2a2f87949e0ebd797。本轮仅允许 House 2/19 数值审计；House 5 仅复用既有冻结训练聚合元信息。House 17、最终 8/9/20 文件未打开、未 hash、未查看 shape；未读取任何开发指标、预测或真值。',
'',
'检查了工作区祖先、scripts/src 与所访问数据/输出祖先目录，未发现适用 AGENTS.md。读取现有 source 身份、原时间协议、原重采样实现与 source-only 池元信息；没有调用训练模块。最初一次文件名发现范围过宽，仅枚举了 outputs/accuracy 路径名，未打开其数据；随后全部改为显式路径。一处既有 28 日缓存目录非递归列举被拒绝，未重试。访问局限如实记录于 access_evidence.json。',
'','## 冻结来源覆盖','',
'全部时间左闭右开、UTC。原“各14d”是各 1,209,600 秒，跨 15 个 UTC 日期，其中仅 13 个为完整 UTC 日。每户 151,200 行只是规则网格，不能表示 151,200 次真实完整观测。',
'',
'|户|固定训练范围|有限总表/标签|可观察正标签点/段|训练窗口/正中心|训练支持段/完整支持段|',
'|---|---|---:|---:|---:|---:|']
for r in old:
    lines.append(f"|{r['house']}|{r['start_utc']} → {r['end_utc']}|{r['aggregate_known']:,}/{r['label_known']:,}|{r['positive_label_bins']}/{r['observable_segments']}|{r['eligible_centers']:,}/{r['eligible_positive_centers']}|{r['supported_segments']}/{r['fully_supported_complete_segments']}|")
lines += ['',
'House 5 既有前缀聚合对照：76,898 个窗口、1,426 个正中心；原 2/19 为 259,806/849，合计冻结 train.npz 336,704/2,275，SHA256 与 Task15 source 身份完全一致。只读取 train 的 house/t/y/event，不读 x，不重新解析 House 5 CSV。验证 block4/5 固定为后缀，未纳入新增训练。',
'',
'原两户标签各恰好 90% 有限，不代表随机预设丢弃10%。已有只读来源审计将总表不支持时间核对为 House2 45,258秒、House19 16,911秒；Issues=1 另外导致 904/2,160 个标签网格失效。固定日期来源于既有按时间先后达到覆盖门槛的选择协议；本轮未重选、填补或更改原缺失。上游 CLEANED 数据已做过补值/尖峰处理，原始未清洗缺失不可恢复，所以“真实标签”只指本地现有官方清洗来源测量标签，并非未经处理的物理真值。',
'',
'31 点核心跨度为 248秒（首末中心差240秒），127点上下文为1,016秒（首末差1,008秒），相邻中心仅差8秒；窗口大量重叠。>=200W 的连续可观察标签段遇 NaN、关闭或非8秒时断即结束；不得跨缺失拼接。9秒采样间隔仅允许保持前值8秒，剩余1秒仍不支持；非满覆盖网格值保持NaN。片段不是独立物理加热 cycle，长物理运行也可能被缺失切成多个短片段。',
'',
'原House2/19观测段时长中位数24/32秒、范围8–280/8–104秒；正标签功率中位数约1,097.94/1,258.69W。关闭背景残差中位数156.0/225.5W，q90为444.375/459.5W。训练仅支持74/80段，而可观察总段为81/83，进一步说明不能用466/383正中心充当事件数。',
'','## 已封存模板池的局限','',
'214个完整127标签模板只来自16个可观察片段：House2为39模板/4段、House19为44/6、House5为131/6。2/19合计83模板只覆盖10段，分别仅覆盖其训练已支持段的4/74和6/80。House5贡献61.2%模板；本轮只引用其既有聚合元信息，未解析新日期。',
'',
'2/19分别有427/339个训练正中心因为127标签不全而不能作为完整模板；仅39/466与44/383保留。完整模板规则会偏向更连续的标签区间，16段并不代表全部真实使用多样性。模板间仍有高度重叠，背景组合也不会创造新真实事件、改变实际日期覆盖或证明泛化。',
'','## 本地额外日期与真实覆盖','',
'先查导入receipt、来源28日manifest和通道映射，再固定日期，最后读取允许户的功率列。两份完整原文件SHA256均与导入receipt一致。House2用Appliance5，House19用Appliance4；缓存28日范围覆盖大部分候选，最后一日超出缓存边界的部分由同一完整原文件读取。没有下载。',
'',
'候选固定为原14d末端之后的第一个完整UTC日开始，连续14日：House2 [2014-10-14,2014-10-28)，House19 [2014-10-19,2014-11-02)。原末日剩余部分排除；不按正例丰度跳日。原始CSV只解释该范围的总表/微波炉/Issues数值，另用一个前驱与首个闭合记录保持hold年龄。House2闭合记录因长缺口实际在10-28T13:46:49Z，其值不贡献候选时间内观测。',
'',
'|户|总表完整网格|标签完整网格|可观察正点/段|可合格中心/正中心|支持段/完整支持段|',
'|---|---:|---:|---:|---:|---:|']
for r in candidate:
    lines.append(f"|{r['house']}|{r['aggregate_known']:,} ({r['aggregate_known']/151200:.2%})|{r['label_known']:,} ({r['label_known']/151200:.2%})|{r['positive_label_bins']}/{r['observable_segments']}|{r['eligible_centers']:,}/{r['eligible_positive_centers']}|{r['supported_segments']}/{r['fully_supported_complete_segments']}|")
lines += ['',
'合计748个已观测正标签网格、169个可观察段；应用原核心规则后保留719正中心、160个支持段。两户完整支持段57+19=76；没有把169或160解释为新增独立物理运行。新增中心仅为可复核审计计数，未生成127训练模板或实际训练集；后续准备仍须以原外圈截断、跨角色整段排除和原归一化独立核对。',
'',
'House2候选总表完整网格率97.36%，高于原90.60%；House19为91.64%，接近原91.43%，不能笼统宣称新增日期都更优。House2候选总表不支持秒为29,005秒、House19为99,522秒；新House19长缺口跨10-27/28，原始相邻间隔最大88,912秒；House2最后候选日受74,510秒长间隔影响。不支持网格数分别3,992/12,640，不能用10800×日期覆盖真实观测缺口。',
'',
'House2的10-18、10-19为零正例日，仍保留。10-27仅7,595个已知标签网格；House19的10-27/28分别5,259/4,519个已知标签网格。候选House2/19正功率中位数1,093.125/1,284W，段时长中位数16/40秒，关闭背景残差中位数190/233.125W。日期变化同时改变缺失、段长和背景，新增段数不能直接推断模型将改善。',
'','## 逐UTC日期计数','',
'完整逐日表同时提供已知点、覆盖秒、段跨日触及数与按开始日期计数；触及数不可直接跨日相加为独立段。以下“段”按该日开始计数。原15日的头尾按实际范围统计，不能按满日10800补齐。',
'',
'|户|角色|UTC日期|完整日|总表已知|标签已知|正标签|开始段|合格中心|正中心|',
'|---|---|---|---|---:|---:|---:|---:|---:|---:|']
with (RUN/'daily_counts.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        role='原训练' if r['role']=='frozen_train_original14d' else '仅审计'
        lines.append('|'+ '|'.join([r['house'],role,r['utc_date'],'是' if r['full_utc_date']=='True' else '否',r['aggregate_known'],r['label_known'],r['positive_label_bins'],r['observable_segments_starting_date'],r['eligible_centers'],r['eligible_positive_centers']])+'|')
lines += ['',
'## 禁用的后续对照提案','',
f'唯一提案已写入 followup_proposal.disabled.json，authorized=false、training_enabled=false。本授权不允许执行。若另行批准，新增279,218/719后预计总池615,922/2,994，负中心由334,429增加至612,928。House5前缀、原2/19归一化常数、三seed42/1019/2026、BCE1、TCN127、Adam .001、batch256的16正/240负规则保持。',
'',
f'预算不能照旧3528步就声称曝光公平。均匀负采样下，单负中心期望曝光=240U/Nneg。固定预算为12×ceil(294×612928/334429)=12×539={updates}步/模型，使扩展池负中心期望曝光{updates*240/newneg:.6f}次，不低于旧3528步的{3528*240/oldneg:.6f}次。每模型1,655,808次样本抽取，103,488正槽、1,552,320负槽。',
'',
'为避免新增数据与训练步数同时改变却只对比旧3528模型，提案要求同seed成对刷新原三户控制与扩展臂，各3模型、均6468步，共6次拟合/38,808更新；原三份unit_BCE模型只作冻结历史参照。该新增控制预算尚未授权；如果只获批3次拟合，这一提案仍不得启动，不能悄悄用3528控制替代。12个checkpoint与原排序规则保持，间隔由294变539属于显式预算修改，同样需批准。',
'',
'等总更新并不等每中心/每段曝光：正槽按house/date组→事件→中心分层，单中心期望16U/(G×E×C)，新增日期会改变G/E/C；一个负样本倍率不可能同时精确匹配全部正事件。后续启动前必须先封存两臂三个seed的纯采样曝光台账，逐户/日/片段/中心报告理论与实际次数、独特支持数、旧数据与新数据质量；不以效果反向改日期或预算。本提案只保证预设相同计算预算与扩展负中心旧基线期望下界，明确不宣称逐事件曝光完全相同。',
'',
'仍用原checkpoint来源选择规则、固定L95/H95/radius15、相同block4/5 source control。各seed各块R≥.95、非退化，扩展均F1不低于控制、每seed平均R下降≤.05、至少2seed P提升，按均P/F1、平手控制。只在唯一source选择完整封存后，且另获授权时才允许一次开发验收；本轮不执行。',
'',
'原8门槛原样保留：(1)均P≥.08；(2)各seed R≥.85；(3)相对W50均F1增≥.02；(4)至少2seed P/F1同升；(5)任seed F1降≤.02；(6)均FP≤旧70%；(7)非退化；(8)原137523支持点/mask、原微波炉功率及其他电器状态逐点不变。容差1e-12。最终8/9/20保持禁止访问。',
'','## 复核与交付','',
'独立verify.py不用audit.py的重采样、分段或核心算法，分别用原始<=8秒区间交集、段起点累计和、卷积重新核算两户原数据及额外日期；已复现已知点、正点、总段、合格中心、正中心、支持段全部数值，并重新验证允许输入hash。5项必要测试通过，覆盖户/日隔离、NaN/时断不串段、9秒gap不填补、两连缺核心拒绝与日期选择不接触标签。没有任何真实或合成模型训练。',
'',
'audit.py只支持audit且拒绝已启动/完成目录；verify.py仅只读源并以独占方式写新复核文件，重复执行不会覆盖已完成复核；报告生成同样拒绝覆盖。delivery_manifest.json封存本目录源码、配置、报告和全部计数文件hash。旧实验与默认协议没有修改。逐段明细见segment_details.csv、逐日期计数见daily_counts.csv、日期角色见date_roles.json、实际输入hash与访问范围见access_evidence.json。']
with (RUN/'REPORT.zh.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
save('delivery_manifest.json',dict(status='completed',files={p.name:sha(p) for p in sorted(RUN.iterdir()) if p.is_file() and p.name!='delivery_manifest.json'},
    configuration_sha256=sha(RUN/'config.json'),audit_code_sha256=sha(RUN/'audit.py'),verification_code_sha256=sha(RUN/'verify.py'),
    source_dependency_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/virtual_submeter/data/prepare.py',ROOT/'src/virtual_submeter/data/common.py']},
    current_authorization='audit only; zero fits/updates/inference/calibration/development evaluation',new_training_data_materialized=False))
print('Delivered',RUN/'REPORT.zh.md')
