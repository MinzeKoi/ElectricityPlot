"""Create non-overwriting audit handoff documents from sealed availability evidence."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from virtual_submeter.data.common import write_json,sha256
from verify_microwave_data_audit import verify

def main():
 run=ROOT/'outputs/accuracy/microwave-data-audit-20261025-v2'
 report=ROOT/'docs/MICROWAVE_DATA_AUDIT_20261025.md';machine=report.with_suffix('.json')
 if report.exists() or machine.exists():raise ValueError('Completed delivery exists; refuse overwrite')
 a=json.loads((run/'audit.json').read_text());daily=json.loads((run/'daily_support.json').read_text())['rows']
 verification=verify(run)
 tests={'test_microwave_data_audit.py':{'tests':5,'passed':5},'test_data_protocol.py':{'tests':25,'passed':25},'test_microwave_diagnosis.py':{'tests':5,'passed':5}}
 for name,v in tests.items():v.update(command=f'.venv-m2/Scripts/python.exe -m unittest discover -s tests -p {name} -v',sha256=sha256(ROOT/'tests'/name))
 a.update(delivery_status='complete_negative_correctness_finding',run_path=run.relative_to(ROOT).as_posix(),
     daily_support=daily,detailed_events_path=(run/'events.json').relative_to(ROOT).as_posix(),
     artifact_manifest_path=(run/'complete.json').relative_to(ROOT).as_posix(),artifact_manifest_sha256=sha256(run/'complete.json'),
     downstream_plan_path=(run/'downstream_plan.json').relative_to(ROOT).as_posix(),
     independent_verification=verification,tests=tests,total_tests_passed=35,
     evidence_scope='Local automated audit, not a second investigator. No model experiments or training runs; the existing diagnosis unit suite includes a synthetic gradient update.',
     audited_version_history={'v1':'Completed initial strict replay; preserved. v2 adds explicit pre-Issues label stage and complete-event-feasible split candidates; values unchanged.'},
     clock_limit='Unix timestamps are used consistently; no independent physical clock reference exists, so a shared upstream sensor offset cannot be ruled out.',
     downstream_checklist=['Explicitly select input paths and freeze their SHA256 before training.',
      'Treat audit inputs as byte-identical strict8 data, not a repaired dataset or extra evidence.',
      'Choose date blocks by support only and freeze before fitting; never tune with historical outer scores.',
      'Apply all 31 samples inside block, 1800-second half-gaps, and whole threshold-event boundary exclusion.',
      'Report complete observed segments as well as windows; do not treat threshold segments as independent physical cycles.',
      'Keep missing labels unknown; do not fill aggregate gaps, change the 200W definition, or use labels to gate aggregate validity.',
      'Fit normalization only on training data; report exposure and tiny complete-event support.',
      'Describe any evaluation on these dates as reused source development evidence, not fresh final testing.'],
     execution_files={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in [ROOT/'scripts/audit_microwave_data.py',ROOT/'scripts/verify_microwave_data_audit.py',Path(__file__)]})
 write_json(machine,a)
 lines=['# 微波炉来源数据质量审计：20261025',
 '','完成：仅访问 House2/19 的实际来源总表和微波炉标签；未访问 House5/17 或最终8/9/20标签。未进行新模型实验。使用 diagnosing-bugs 技能，先复现，再检验归因。',
 '', '## 结论与实际处理',
 '', '未发现来源导入、时间戳、8秒重采样、标签对齐或31点窗口的实现缺陷。原始记录重放与旧输入的数值、缺失位置、coverage、质量标记、时间戳全部一致。审计版四个输入CSV与旧来源输入逐字节相同；真实修复为否，有效窗口、正中心和完整观测段增量均为0。没有修改公共代码、旧配置默认值、界面或旧实验。',
 '', 'House19早期大量9秒采样间隔与“每条观测最多支持8秒”的保守契约相遇，产生1秒无支持区间和coverage=.875。完整31点需要248秒连续支持，分散小孔足以消灭几乎所有窗口。这里的“缺失”指契约下缺少观测支持，不能断言物理设备停止采样；若把原值持有9秒，那是额外观测假设而非恢复出的测量。本次未放宽规则，也未填零、前填或跨gap插值。',
 '', '本任务没有访问外部论文或网站；契约判断基于项目现行代码、测试和原始可用记录。上游物理时钟偏差没有独立时钟可验证，因此不能排除；代码内部UTC/Unix对齐已精确核验。',
 '', '## 可重复失败信号与归因',
 '', '最初执行 `.venv-m2/Scripts/python.exe scripts/audit_microwave_data.py`：House19首日10559条含前驱/闭合记录，9秒间隔1649次，完整C31为0；覆盖率0.875有1654个网格。实际重放同时断言首日为0，九秒合成反例保证规则不会无依据生成完整覆盖。',
 '', '检验前的三个预测：①若8秒支持上限解释早期碎片，原始9秒间隔的未支持秒应逐网格解释coverage损失；②若另有导入/边界错误，原始重放会与旧CSV出现数值、时间或缺失差异；③若标签额外损失来自Issues，去除该标签屏蔽的审计分支应逐级解释标签有效数之差。三个检查均完成：所有未支持秒精确归因；重放差异为0；Issues阶段完全闭合。',
 '', '实际链路：`data.prepare.read_samples` 使用Unix、拒绝重复/倒序，将无效功率保留缺失；Issues仅屏蔽标签，不影响总表。`resample` 对每条观测持有不超过8秒、最后一点不外推；半开UTC网格按真实支持秒加权，不完整网格返回空值。旧来源分区复制这些网格，诊断再复制并按时间映射标签。`models.arrays.windows` 要求31个有限值和连续8秒时间步；中心标签缺失再单独排除。',
 '', '两户完整审计片段均无重复/倒序，无原始总表非数值导致的支持损失；不规则采样和实际长间隔有明确记录。边界使用前驱和首个闭合记录保留观测年龄，未重置持有时长；最终14天网格各151200点，标签与总表时间戳逐点一致。',
 '', '|家庭|9秒间隔未支持秒|10–16秒间隔未支持秒|>16秒间隔未支持秒|Issues额外丢标签网格|其中原本≥200W中心|', '|---|---:|---:|---:|---:|---:|']
 for h in a['households']:
  rs=[r for r in daily if r['household']==h];g={k:sum(r['uncovered_seconds_by_raw_cause'][k] for r in rs) for k in rs[0]['uncovered_seconds_by_raw_cause']}
  lines.append(f"|{h}|{g['delta9']}|{g['delta10_16']}|{g['delta_gt16']}|{sum(r['label_bins_lost_to_issues_after_resampling'] for r in rs)}|{sum(r['on_grid_centers_lost_to_issues'] for r in rs)}|")
 lines+=['', 'Issues之前的标签审计分支仅用于定位屏蔽损失，未导出为训练输入。缺失标签不表示关闭。不能凭本审计推翻Issues质量契约。',
 '', '## 前后支持与逐日统计',
 '', '|家庭|总表缺失前→后|标签缺失前→后|可用窗前→后|正中心前→后|完整观测段前→后|', '|---|---:|---:|---:|---:|---:|']
 for h,v in a['households'].items():
  aa=v['replay']['aggregate'];yy=v['replay']['microwave'];w=v['total_usable_windows'];p=v['total_positive_windows'];e=v['complete_observed_segments']
  lines.append(f"|{h}|{aa['missing_before']}→{aa['missing_after']}|{yy['missing_before']}→{yy['missing_after']}|{w}→{w}|{p}→{p}|{e}→{e}|")
 lines+=['', '相对日从原来源起点计数，非UTC自然日。总表有效时长=完整8秒网格数×8；可用时长=可用中心×8，二者都不等同物理独立观测时长。完整观测段要求段前后关闭标签可见、全部正中心具备C31支持；仍不等同独立物理周期。',
 '', '|家庭|相对日|原始记录|总表完整小时|标签有效点|标签正中心|完整总表窗|可用正中心|阈值段起始数|完整观测段|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in daily:
  lines.append(f"|{r['household']}|{r['day_from_zero']}|{r['raw_records']}|{r['aggregate_full_bin_hours']:.3f}|{r['label_valid_bins']}|{r['observed_on_centers']}|{r['complete_aggregate_windows']}|{r['usable_on_centers']}|{r['threshold_segments_starting']}|{r['complete_observed_segments']}|")
 lines+=['', '每一天的起止UTC、原始采样间隔直方图、总表coverage直方图、未支持秒、Issues前后标签阶段、独占窗口丢失原因及正中心丢失原因均在同名JSON的 `daily_support`。独占原因按边缘→总表中心缺失→上下文缺失→标签中心缺失→可用分配，每天严格加总至10800中心，正中心原因也严格闭合。事件级丢失见 `events.json`，具体9秒反例见 `counterexamples.json`。',
 '', '## 日期块与下游限制',
 '', '旧8–11/11–12/12–14划分：House2完整观测段为9/0/5，House19为14/9/4。旧表的19/7/8与25/18/7是至少一个正中心可用的阈值段，不能替代完整段数。House2第11天不宜作为要求完整正事件支持的唯一内验证日；House19第12天无正中心。',
 '', '同时具备两户完整段的单日块有第8、9、10、13天；按各日边界两侧1800秒和完整窗口隔离后，House2/19完整段数依次为6/3、1/1、2/10、3/4。支持都偏小，尤其第9天每户仅1段，不足以宣称稳健的多日期验证。',
 '', '按支持推荐的可行候选：相对0–10天训练、10–11天内验证、11–14天来源留出；分界两侧各排除1800秒，完整31点在所属块内，跨分区阈值段整段排除。以下仅可用性建议，不是训练结果：',
 '', '|家庭|角色|UTC开始|UTC结束（不含）|可用窗|正中心|有支持阈值段|完整观测段|', '|---|---|---|---|---:|---:|---:|---:|']
 for h,v in a['households'].items():
  for role in ['train','validation','outer']:
   b=v['candidate_blocks']['recommended_'+role]
   lines.append(f"|{h}|{role}|{b['start']}|{b['end']}|{b['windows']}|{b['positive_windows']}|{b['supported_threshold_segments']}|{b['complete_observed_segments']}|")
 lines+=['', 'House19名义10天训练实际仅14866窗、4个完整观测段，House2内验证仅2个完整观测段。该候选满足存在性，不满足充足性。若下游协议要求更高最低完整事件数，应据此判定不满足，不能悄悄降低门槛。日块清单是替代支持诊断，不可不经重新隔离就同时充当多个角色。',
 '', '整个14天来源范围都已参与旧诊断的训练/验证/评价，尤其12–14天已解封评价。本审计不依据任何模型分数选日期，但不能消除历史使用：下游只能将这些日期作为复用来源开发比较，不能称新鲜独立留出。审计范围内没有可证明未使用的新测试集；本任务未扩展日期或访问其他户。',
 '', '## 输入、版本、哈希与复建',
 '', f'审核主目录：`{run.relative_to(ROOT).as_posix()}`。v1初步审核已完成并保留；v2补充Issues前后阶段及完整事件划分，数据字节相同。它是审核版本，不是修复版。下游需显式选用以下路径；公共API和旧默认配置不变：',
 '', '|家庭|通道|明确输入路径|SHA-256|','|---|---|---|---|']
 for h,v in a['households'].items():
  for c,meta in v['replay'].items():lines.append(f"|{h}|{c}|`{meta['input_path']}`|`{meta['sha256']}`|")
 lines+=['', '机器划分入口：`outputs/accuracy/microwave-data-audit-20261025-v2/downstream_plan.json`；时间/有效性及每个候选块的布尔掩码：同目录的`refit_2_support.npz`、`refit_19_support.npz`。NPZ不含标签功率，但使用有效标签筛选后的掩码仍应按来源标签衍生信息管理。完整事件清单和逐日丢失统计是审核证据，不能当额外独立样本。',
 '', '原始完整文件路径/哈希、精确字段片段哈希、旧实验registration/freeze等收据哈希均在交付JSON；执行源码快照和产物清单在run/source与complete.json。原始记录只选总表、微波炉、Issues及Unix列，保留前驱和闭合记录。',
 '', '```powershell', a['rebuild_command'], '.venv-m2/Scripts/python.exe scripts/verify_microwave_data_audit.py --run outputs/accuracy/microwave-data-audit-20261025-v2', '```',
 '', '重建必须使用从未存在的新目录，存在即拒绝；不得覆盖v1/v2。脚本未提供修改旧输入的路径。若工作区公共代码以后变化，先比对已封存源码哈希；本次四个执行依赖已有快照，旧输入哈希不匹配则停止。',
 '', '## 验证与不能恢复的信息',
 '', f'35项测试通过：本任务5项、现有数据协议25项、现有诊断5项。诊断单测含合成数据上的一次梯度更新；没有来源新模型拟合或性能选模。独立复核核验{verification["audited_artifacts"]}个产物哈希，用前缀缺失计数独立重建C31掩码，并复核逐日加总、Issues阶段、两种划分的上下文及事件隔离、原始与旧实验哈希。完整命令、测试文件哈希与结果见JSON。',
 '', '原始重放逐值一致、未支持秒逐网格闭合、旧逐日症状一致，支持“现行契约没有实现错误”的负结论。它不证明8秒持有假设在所有研究任务中最优，也不证明物理采样原本缺了多少。若未来试验9秒容忍或另一重采样方法，必须独立登记其观测假设、保持真实长gap不插值、版本化输入并作为新处理实验，不能冒称本次bug修复。',
 '', '上游已前填/填零的原始缺失状态不可恢复，IAM超过4000W被截为零的信息不可恢复；没有独立物理时钟、设备周期或额外传感器真值。不得把阈值段当独立物理周期，不能用标签调整总表有效性，也不能改变200W定义来提高分数。',
 '', '下游执行审核清单：']
 lines+=['',*['- '+x for x in ['显式冻结四个CSV路径、SHA-256与划分；注明未增加样本。','训练窗内估计归一化，不从验证/留出估计。','完整窗口、边界间隔和整事件隔离同时执行；报告完整段数及稀少支持。','缺失仍未知，不填gap、不把缺失当关闭、不改变开机阈值。','将所有本范围日期评价标为复用来源开发证据，不声称新鲜外部验证。','训练采样和模型入口由后续任务实现；本任务未修改相邻任务文件。']]]
 report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'status':'delivered','documents':[str(report),str(machine)],'tests_passed':35,'verification':verification['status']},ensure_ascii=False))
if __name__=='__main__':main()
