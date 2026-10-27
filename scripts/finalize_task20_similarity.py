"""Build Chinese Task20 report and exclusive manifest after independent verification."""
import sys
sys.dont_write_bytecode=True
import csv
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').is_file():ROOT=ROOT.parent
RUN=ROOT/'outputs/accuracy/microwave-input-similarity-task20-20261026-v2'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def snap(paths):return {str(p):dict(sha256=sha(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
def fresh(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def med(v):return f'{np.median(v):.4f}' if v else 'null'
def table(header,rows):return ['|'+'|'.join(header)+'|','|'+'|'.join(['---']*len(header))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows]+['']

def main():
    if (RUN/'delivery_manifest.json').exists():raise FileExistsError('Sealed directory cannot be finalized again')
    verification=read(RUN/'independent_verification.json')
    assert verification['status']=='passed'
    cfg=read(RUN/'config.json');paths=[Path(p) for p in cfg['allowed_paths']]
    allowed=set(paths)|{Path(__file__).resolve()}
    def audit(event,a):
        if event=='open' and not isinstance(a[0],int):
            p=Path(os.fsdecode(a[0])).resolve()
            writing=(isinstance(a[1],str) and any(c in a[1] for c in 'wax+')) or (isinstance(a[2],int) and a[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and not p.is_relative_to(RUN):raise PermissionError('Writes only inside unsealed delivery')
            if not writing and p.is_relative_to(ROOT) and not p.is_relative_to(RUN) and not p.is_relative_to(Path(sys.prefix)) and p not in allowed:raise PermissionError('Outside whitelist')
        elif event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):raise PermissionError('No external execution')
    sys.addaudithook(audit)
    before=snap(paths);assert before==read(RUN/'inputs_before.json')
    with (RUN/'support_and_coverage.csv').open(encoding='utf-8-sig',newline='') as f:support=list(csv.DictReader(f))
    a=read(RUN/'analysis_complete.json');dup=read(RUN/'exact_duplicates.json');frozen=read(RUN/'samples_frozen.json')
    aggregate=defaultdict(lambda:defaultdict(list));role=defaultdict(lambda:defaultdict(list));counts=defaultdict(int)
    with (RUN/'pairs.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['family'],r['category'],r['mode']);label=r['reference_label']
            if r['distance']:aggregate[key]['d'+label].append(float(r['distance']))
            if label=='1':
                counts[key]+=1
                if r['positive_over_negative_ratio']:aggregate[key]['ratio'].append(float(r['positive_over_negative_ratio']))
                rk=(r['family'],r['query_house'],r['query_role'],r['category'],r['mode'])
                role[rk]['n'].append(1)
                if r['positive_over_negative_ratio']:role[rk]['ratio'].append(float(r['positive_over_negative_ratio']))
    curves=read(RUN/'curve_index.json')
    for curve in curves:
        svg=ET.parse(RUN/curve['file']).getroot()
        assert svg.tag.endswith('svg') and len(list(svg))>5
    repeats=read(RUN/'exact_repeat_groups.json')['groups']
    assert len(repeats)==dup['repeated_groups']
    common_queries=sum(int(x['sample_full_n']) for x in support if x['kind']=='query')
    strict_paired=sum(len(aggregate[(f,c,'strict31')]['d1']) for f in ('control','expanded') for c in ('FP','FN','TP','TN'))
    common_paired=sum(len(aggregate[(f,c,'common127')]['d1']) for f in ('control','expanded') for c in ('FP','FN','TP','TN'))
    lines=['# Task20：冻结模型输入相似性零训练诊断','',
        '本轮已完成真实数值分析、独立实现的逐配对复算、固定顺序曲线和封存。没有新增拟合、优化器更新、监督探针、阈值或checkpoint搜索，也没有修改模型输出或默认模型。', '',
        '## 结论与下一步', '',
        f'在{a["strict_windows"]:,}个核心31点完整窗口中，完整127点标准化功率与掩码共形成{dup["unique_effective_inputs"]:,}种有效输入。只有{dup["repeated_groups"]}组完全重复（236窗，235次逐元素确认）；该组均为关闭标签、完整127点无缺失且未跨UTC日，未发现中心200W开关标签相反的精确重复。这不能排除未重复输入的标签或时间对齐问题。', '',
        '在固定跨日期参考库内，FP与FN的最近正/负距离比中位数均大于1；它们的已观测输入通常仍有更近的同户关闭参考。FP与TP的距离分布也存在接近范围。它们是模型错误来源的线索，不能把一个中位数或近邻相似解释为物理不可辨识、标签错误或可直接提升模型的规则。', '',
        '在相同完整127点查询与参考的对照中，127点没有表现出对全部错误类别一致的距离比变化；正确TP的距离比中位数还从31点略小于1变成127点大于1。因而本轮不支持“直接扩大上下文就能解决”的结论。距离比只是描述量，未用来构造分类器或计算新验收成绩。', '',
        '正负参考候选数明显不均衡：负参考每层上限4096，正参考上限256，原支持也不同。更多负候选会影响最近距离；R>1不能直接判定输入可分/不可分，也不是新分类准确率。127点RMS增加背景后改变距离几何，未一致改善该描述量，不能推出网络无法利用长上下文。', '',
        '**下一轮优先建议：先提出受控的训练/表示诊断，而不是据此大范围改标签。** 可证伪假设为：固定这些错误片段和既有输入后，错误主要来自当前学习/表示不足；先用本轮固定曲线片段复核输入与中心200W标签的对应关系，若发现系统时间错位或标签异常，则推翻训练优先顺序，转为输入/标签修复。任何新增训练或扩大原始标签读取范围须另行授权具体方案与预算；本轮没有执行。', '',
        '## 固定口径与可复现身份', '',
        '输入仅限config.json的逐文件白名单：Task19六模型18份冻结概率、阈值/身份/manifest，以及Task18 train、两块validation、normalization和原输入转换源码。未打开模型权重、其他epoch或其他实验成绩。Task19 manifest只核验白名单成员及56项头部，不把“未读取的56项全部重新哈希”当作已完成。报告身份固定为Task19 SHA256 fe5fd289e3129ef7eeb3f99c6e717d93d018aac7335483085edfd3e46a03d58c。', '',
        'House17、最终House8/9/20未读数值、形状或hash；不增加原始数据读取。control参考仅原336704窗训练前缀（历史3528更新）；expanded参考来自完整615922窗训练角色（历史6468更新）。expanded是此前来源被拒模型，未晋升。追加日期只对control未训练；House5后缀曾参与选模，不能称盲测或独立泛化。', '',
        'strict31 = 原mask[:,48:79].all；模型输入始终保留原127点功率和127点掩码。功率严格复用float32 (log1p(x)-5.370561599731445)/0.8337093591690063；缺失为功率通道0与掩码0。距离用已冻结功率通道的直接float64逐点RMS，不重估尺度、不减均值、不填缺失。独立合成验证与原tensor_input源码的双通道输出完全相同。', '',
        '稳定ID为H{物理户号}|{原角色}|{UTC秒}|raw{原行号}，按UTF-8 SHA256升序抽样。同户/真实训练角色/标签参考每层最多256正、4096负；查询按family/seed/house/role/UTC日期/TP、FP、FN、TN每层最多32，错误类别仅用于描述分层。参考与查询同日禁配、中心间隔小于1016秒禁配；无参考为null，不跨户补齐。距离相同按参考stableID字典序。每次最多8个查询×256个参考×127点直接分块，不保存全量平方矩阵。', '',
        '注册配置和源码先冻结，之后支持清单、参考/查询样本再冻结，最后才计算距离。common31与common127均从已经冻结的strict31列表中筛出完整127点查询和参考，不重新抽样或补样；两个模式的候选完全相同。明细另保存common31选中参考在127点的距离，支持相同配对检查。', '',
        '## 支持、覆盖与缺失', '',
        f'共{a["total_windows"]:,}个原窗口，strict31支持{a["strict_windows"]:,}（{a["strict_windows"]/a["total_windows"]:.2%}），完整127点支持{a["full127_windows"]:,}（{a["full127_windows"]/a["total_windows"]:.2%}）。抽取{frozen["queries"]:,}条查询记录和{frozen["references"]:,}条参考记录；记录跨seed/family重复，不是独立物理事件数。', '',
        f'固定查询中完整127点有{common_queries:,}/{frozen["queries"]:,}（{common_queries/frozen["queries"]:.2%}），不完整的{frozen["queries"]-common_queries:,}条保持null，未补样。strict31最近正参考可配{strict_paired:,}条；common127可配{common_paired:,}条。共交付{a["pair_rows"]:,}条“查询×模式×参考标签”明细，其中{a["null_pairs"]:,}条为明确null。common31与common127都保留不完整查询的空行，以避免隐去覆盖损失。', '',
        '下表按原house/role聚合，不跨seed或模型重复计数。完整UTC日期、各错误类别原支持、抽样量和完整127覆盖见support_and_coverage.csv；计数不把窗口或可观察片段当独立开机事件。', '']
    coverage=defaultdict(lambda:np.zeros(3,dtype=np.int64))
    for r in support:
        if r['kind']=='query' and r['family']=='expanded' and r['seed']=='42':
            coverage[(r['house'],r['role'])]+=np.array([int(r[k]) for k in ('original_n','strict_n','full_n')])
    lines+=table(['house','role','原窗','strict31','完整127'],[(h,r,*map(int,v)) for (h,r),v in sorted(coverage.items())])
    lines+=['## 固定查询的距离分布','', '下表跨seed描述性汇总，查询按日期/类别等分层抽样，不能外推为总体发生率；相同窗口可出现在不同seed中。d+、d−为参考库内最近正/负距离的各自中位数；R是逐查询d+/d−的中位数，不能用两个中位数相除代替。', '']
    lines+=table(['family','类别','查询n','RMS31 d+','RMS31 d−','R31'],[(f,c,counts[(f,c,'strict31')],med(aggregate[(f,c,'strict31')]['d1']),med(aggregate[(f,c,'strict31')]['d0']),med(aggregate[(f,c,'strict31')]['ratio'])) for f in ('control','expanded') for c in ('FP','FN','TP','TN')])
    lines+=['共同完整127子集对照（同查询、同候选参考；每种长度仍各自找最近参考）：','']
    lines+=table(['family','类别','共同子集有效n','R common31','R common127'],[(f,c,len(aggregate[(f,c,'common31')]['ratio']),med(aggregate[(f,c,'common31')]['ratio']),med(aggregate[(f,c,'common127')]['ratio'])) for f in ('control','expanded') for c in ('FP','FN','TP','TN')])
    lines+=['R在127点的变化不是上下文增益或新分类准确率证据。RMS长度改变会重分配原31点与外围点的权重；参考库是有限抽样而非所有可能输入。任一层完整127支持不足或没有合格参考时只报告null，不改口径。','', '## 跨日期与角色的错误来源线索','', '所有参考均来自不同UTC日。下表进一步保留house/role区分；原日期、追加日期和House5前缀/后缀不能混成相同训练角色。表中n为strict31抽样错误窗记录，非事件数；细至每个UTC日、seed的分位数见group_distributions.csv。','']
    rr=[]
    for (f,h,r,c,mode),values in sorted(role.items()):
        if mode=='strict31' and c in ('FP','FN'):
            rr.append((f,h,r,c,len(values['n']),med(values['ratio']),med(role[(f,h,r,c,'common31')]['ratio']),med(role[(f,h,r,c,'common127')]['ratio'])))
    lines+=table(['family','house','role','类别','n','R31','R common31','R common127'],rr)
    lines+=['参考库会随family真实训练角色改变；control和expanded的距离差异同时包含参考覆盖差异，不能归因于训练收益。日期与缺失分布也不是随机分配，分组差异仅是工程线索。','', '## 固定配对曲线','', '按预先冻结的query SHA顺序，每family/类别最多2条；FP选最近正参考、FN选最近负参考、TP/TN选同标签参考。曲线显示原功率W与完整127点、缺失断线及核心31区间。正例按原event ID，关闭窗按可观察连续片段去重展示；统计计数保持原窗口口径。图示只是“冻结参考库内最近”，不代表全局最近或完整物理事件。','']
    lines+=table(['family','类别','seed','查询中心UTC秒','曲线'],[(c['family'],c['category'],c['seed'],c['query_utc'],f'[{Path(c["file"]).name}]({c["file"].replace(os.sep,"/")})') for c in curves])
    lines+=['## 独立复核与封存','',
        f'独立复核器没有复用主分析的抽样/近邻/身份实现：重新构造全部支持与SHA前K清单，重新遍历所有合格参考核对最近身份和eligible_n，再以math.fsum标量逐点重算全部{verification["pair_result"]["nonnull_pair_rows"]:,}条非空交付距离及common31选中配对的127点距离；最大距离绝对误差{verification["pair_result"]["max_scalar_distance_abs_error"]:.3g}。核对原行号、UTC、家庭、真实训练角色、原200W标签、冻结概率和阈值、错误类别、无同日/重叠、null和共同候选，全部通过。', '',
        '7组针对性合成检查通过：分块对标量、跨块全局稳定并列与全不合格null、同日/重叠/跨户/不完整阻断、完整功率与掩码重复语义、与原转换源码完全一致、模型矩阵缺项拒绝、文件与目录不可覆盖。测试中导入主代码仅用于构造被测对象；真实数值复核使用独立实现。', '',
        '过程说明：v1在读取validation不存在的source_role字段时停止，未产生抽样/距离；v2仅按Task19固定角色赋值修复，v1未覆盖。最初启动过独立review子代理，收到根对话约束后已停止；主对话接管并审阅、修正、运行分离的复核实现。独立性指实现与数值重算，不宣称盲态外部审计。没有扩大路径扫描。重复补充清单首次已成功写出，但复制自身源码被白名单阻断，留下0字节duplicate_inventory.py；保留该失败工件，修复为在审计前读取自身源码，重算确认清单完全相同后另存duplicate_inventory_source_v2.py，未覆盖原输出。', '',
        '全部47个白名单输入在注册、主分析前后、独立复核前后与封存前后逐文件核对SHA256/size/mtime_ns。最终manifest绑定所有交付配置、脚本、数据身份、抽样、明细、图、报告和独立复核结果。verify为只读，复验完整文件清单以及全部hash/mtime；完成目录以exclusive创建，禁止重跑覆盖。完成后的两遍只读复验记录另存同级receipt，避免修改已封存目录。', '',
        '## 交付索引','',
        '- `config.json` / `registration.json`：规则、授权来源与源码hash。',
        '- `inputs_before.json` / `inputs_after_analysis.json` / `independent_inputs_after.json` / `inputs_after_finalization.json`：原输入只读身份。',
        '- `support_and_coverage.csv` / `reference_sample.json` / `query_sample.json` / `samples_frozen.json`：原支持及固定抽样。',
        '- `pairs.csv`：完整191250条配对、时间/行号/标签/概率/掩码/距离及null。',
        '- `group_distributions.csv`：8760条预定义分组分布。',
        '- `exact_duplicates.json` / `exact_conflict_members.json` / `exact_repeat_groups.json`：完整有效输入重复、日期/掩码明细与冲突。',
        '- `curve_index.json` / `curves/*.svg`：固定顺序、片段去重的16条曲线。',
        '- `independent_verifier.py` / `independent_verification.json` / `independent_synthetic_tests.json`：独立实现、逐对复算与测试。',
        '- `diagnostic.py` / `finalizer.py` / `delivery_manifest.json`：冻结源码与封存清单。', '',
        '本轮没有提高预测效果的交付结论。精确冲突未发现、跨日期近邻接近或127距离改变，都不能证明标签正确/错误、物理不可分或已实现泛化收益。','']
    report='\n'.join(lines).replace('全部47个白名单输入',f'全部{len(paths)}个白名单输入')
    with (RUN/'DELIVERY_REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as f:f.write(report)
    with (RUN/'finalizer.py').open('x',encoding='utf-8',newline='\n') as f:f.write(Path(__file__).read_text(encoding='utf-8'))
    after=snap(paths);assert before==after
    fresh(RUN/'inputs_after_finalization.json',after)
    artifact_paths=sorted(p for p in RUN.rglob('*') if p.is_file())
    artifacts={str(p.relative_to(RUN)):v for p,v in ((p,snap([p])[str(p)]) for p in artifact_paths)}
    fresh(RUN/'delivery_manifest.json',dict(status='complete_zero_fit_input_similarity',artifacts=artifacts,new_fits=0,optimizer_updates=0,supervised_probes=0,threshold_searches=0,checkpoint_searches=0,model_promotions=0,report_sha256=sha(RUN/'DELIVERY_REPORT.zh.md'),independent_verification='passed'))
    print(json.dumps(dict(status='sealed',artifacts=len(artifacts),report_sha256=sha(RUN/'DELIVERY_REPORT.zh.md')),ensure_ascii=False))

if __name__=='__main__':main()
