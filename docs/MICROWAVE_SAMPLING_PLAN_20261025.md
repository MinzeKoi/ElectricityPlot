# 微波炉事件曝光与多日期验证：训练前方案

2026-10-25，任务2准备版本。训练前登记；任务1审计现已完成，协调者已授权实际执行，真实训练尚未开始。本文件、配置和实现须在实际训练前封存。审计输入接口若需适配，只调整清单转换，不按模型成绩修改配方。保留已有修改、历史失败实验、旧水壶/洗衣机模型和默认界面。

已核对MICROWAVE_DIAGNOSIS的REPORT、REGISTRATION、RESULTS、VALIDATION及其配置/实现，以及state_classifier/state_selection和旧STATE/PRECISION登记、结果、验证。旧C_EVENT保留正类质量；旧小集合成功不能证明全量容量足够。先只改变采样，禁止同时改网络、加长训练、追加权重或采样率搜索。House17已查看，不能称独立测试；本轮第一批不读取House17及最终House8/9/20标签。

## 拟合数量与固定对照

House2/19来源训练块联合训练一个C31模型，两种采样各种子42/1019/2026，**总共6次拟合**，不分户/折额外训练。联合模型是本轮任务，不与旧分户模型作等预算因果对比。每对种子网络初始参数完全相同；来源数据、划分、训练log1p均值/标准差、完整256批次、12×294=3528次更新、Adam .001、BCE pos_weight50相同。保存每轮权重与内验证分数。仅保留现有31点、8秒、单总有功、两层8通道CNN、32隐单元，200W状态定义。

自然组：训练窗打乱无放回循环，每批固定256，跨循环补齐，不缩短末批。事件组：每批16正中心（6.25%目标且严格下限16），剩余240个关闭中心按原始训练负窗口分布均匀打乱、无放回轮换。高总功率负样本自然出现，不显式偏采；不改变负类条件分布，不额外加入困难挖掘。此项按协调者训练前跟进修正，未增加候选、未进行真实拟合。

正中心依次对“家庭＋UTC日期”、该日期的阈值段、段内中心作打乱无放回轮换；短段和长段获得相同段级机会，中心轮换避免总抽同一邻近窗。跨午夜事件可贡献两个日期组，明确不称新增独立事件。事件定义为同户相邻8秒、观测功率≥200W的连续段；缺失和时间间断断开，不能称独立物理周期。记录完整/不完整段标志。采样增加的是重复曝光，不制造独立标签。

BCE50不变**不意味着有效损失相等**。事件组名义正权重质量为800/(800+240)=76.923%，自然组取决于自然正比例；记录实际正/负窗次数、逐窗/日期/事件曝光、每批正负损失和正损失占比、计算秒数与总更新数。每批独立事件数量也记录；稀少日期/段不能伪装为更多独立支持。

## 时间协议与可行性门禁

各户使用原14天起点，固定相对[0,10)天训练，[10,11)、[11,12)两个日期块内验证，[12,14)外层；这是训练前按时间预留，不依据旧外层分数挑切点。第10/11/12天边界各侧1800秒隔离。31点全部上下文必须位于各自块内，任何跨块阈值段的正中心整段排除。缺失不插补、不当关闭；总表缺失使上下文整窗无效，标签只取同UTC中心。这里“日期块”是相对起点24小时块，非必然自然日；报告另列UTC日期和事件。

每户训练和外层必须都有正负类；每个内验证块至少5正中心、2个不同阈值段且有负类才称满足预登记最低支持门槛。至少两个满足最低门槛的块才允许运行；仅2/3块可预选诊断阈值，但标为有限证据，无法通过整体部署门槛。四块均充分才有完整多日期/双户资格。缺乏支持不重新搜索切点；保留准备失败目录、0次拟合，交协调者判定下一步。审计若确认不可恢复缺失如实记录；此版本不假设修复可行。

训练归一化只由两户训练窗计算，保持log1p标准化形式。每个窗保留户、UTC、原行、窗口ID；NPZ中中心与原API标签/直接时间映射独立断言一致。外层训练可读输入只含x/t，标签与事件身份单独封存。prepare按明确清单处理来源数据以构建隔离；train和来源选择不得打开外层标签，freeze只打开外层x/t。

## 内验证预选与改善门槛

每轮仅用四个来源内验证块的分数并集确定一个公共概率阈值。每个满足最低门槛的块都须R≥.85，并拒绝全开/全关。零正样本块和不满足最低门槛的块只报告指标，不承担召回/全关门槛；零正样本块全关是正确预测。来源选择不另设P、F1或off-FPR绝对门槛，低P/高FPR完整报告。阈值确定性排序：满足最低门槛的块最差P、平均P、平均F1、更高阈值；检查点按前三项、较早轮次排序。不支持的块明示，不把它算独立召回保护证据。此规则按协调者训练前复核修订，8%绝对效果门槛仅留在外层诊断。

无合格检查点时保存末轮/.5仅诊断，并标source_gate_passed=false，不伪造通过。每个候选三个种子全部通过且四块满足预登记最低支持门槛才有整体资格；事件组源内平均F1不得低于自然组（相对差下限0），然后按来源平均P、平均F1、自然组优先的固定顺序选候选；在外层评价前写入source_selection.json。可以无合格候选。

主要改善门槛预登记：每个外层目标户，事件组三种子均R≥.85，平均P≥.08，配对平均F1至少提高.02，FP总数≤自然组70%，至少两种子P/F1同时提高，任一种子F1下降≤.02。两个户都通过、事件组在源内被预选且具有完整支持，才报告本轮改善得到支持。否则报告失败/有限证据，不追加搜索。外层R85处Precision只作事后诊断，不反过来部署或选择阈值。外层曾被查看，应称来源时间开发诊断，不称新独立泛化证明。

逐种子/分区报告P/R/F1、TP/FP/FN/TN、off-FPR、AP、完整精确PR、固定.5、冻结阈值、事件支持/命中、有效秒数、缺失/排除点；命中一中心不等于完整事件召回。报告训练及内验证重放帮助识别过拟合，但主要判据只用冻结时间留出。House17比较须由协调者在源结果之后另行安排。

## 审计清单输入接口

任务1审计ID为microwave-data-audit-20261025-v2，status=complete，35项测试及独立复核通过；严格重放数据与旧版逐字节一致、样本增量0。协调者明确授权生成适配清单、绑定执行凭据并执行6拟合完整流程，保持本协议日期。不要改写任务1的审计实现。本入口接受单独的适配清单，字段示例（路径相对清单所在目录；使用真实SHA256，不能用占位符执行）：

```json
{
  "schema": "microwave_sampling_audit_v1",
  "status": "ready",
  "audit": {"path": "MICROWAVE_DATA_AUDIT_20261025.json", "sha256": "..."},
  "sources": [
    {"household_id": "refit_2", "start": "原14天UTC起点", "end": "原14天UTC终点",
     "artifacts": {"aggregate": {"path": "...", "sha256": "..."},
                   "microwave": {"path": "...", "sha256": "..."}}},
    {"household_id": "refit_19", "start": "原14天UTC起点", "end": "原14天UTC终点",
     "artifacts": {"aggregate": {"path": "...", "sha256": "..."},
                   "microwave": {"path": "...", "sha256": "..."}}}
  ]
}
```

CSV采用已有规范：aggregate为timestamp/household_id/aggregate_power_w/coverage/quality_flags；microwave为timestamp/household_id/power_w。标签和总表唯一递增UTC、8秒网格、非负或空缺；覆盖不足不可有非空功率。清单只接受两个来源户且文件内容户ID必须一致。封存审计回执、清单、源码、环境版本、输入文件、窗口/事件/划分、统计、配置和本协议的哈希。

## 运行及中断策略

准备后协调者发执行消息，再提供执行凭据JSON：execute=true、maximum_fits=6、registration_sha256、audit_sha256及非空coordinator_execution_message。该凭据是依赖门禁，不要求用户再次常规确认；本轮已收到协调者明确执行消息，获授权自行生成绑定哈希的执行凭据。

```powershell
.venv-m2/Scripts/python.exe scripts/run_microwave_sampling.py prepare --run outputs/accuracy/microwave-sampling-20261025-v1 --data-manifest docs/MICROWAVE_SAMPLING_AUDIT_INPUT_20261025.json
.venv-m2/Scripts/python.exe scripts/run_microwave_sampling.py train --run outputs/accuracy/microwave-sampling-20261025-v1 --authorization tmp/microwave-sampling-execute.json
.venv-m2/Scripts/python.exe scripts/run_microwave_sampling.py freeze --run outputs/accuracy/microwave-sampling-20261025-v1
.venv-m2/Scripts/python.exe scripts/run_microwave_sampling.py evaluate --run outputs/accuracy/microwave-sampling-20261025-v1
.venv-m2/Scripts/python.exe scripts/run_microwave_sampling.py verify --run outputs/accuracy/microwave-sampling-20261025-v1
.venv-m2/Scripts/python.exe scripts/run_microwave_sampling.py report --run outputs/accuracy/microwave-sampling-20261025-v1
```

已存在运行/已启动训练/预测/评价/报告禁止覆盖；本版本不自动断点重训。中断保留全部文件和已消耗拟合，需协调者明确登记恢复成本和新运行，不能静默超过6。源码变化后用运行根下run_microwave_sampling.py加载封存source；不改写旧封存源码。普通load_state_model/predict_windows可读取每个模型，verify复核12份外层概率容差和逐点状态完全相同、初始化相同、归一化及源内选择与外层计数。所有12份预测完成并统一freeze后才允许evaluate；未freeze拒绝且不创建评价目录。测试使用合成数据的小预算流程，与真实6次拟合预算分别标识。


训练前审计限制补记：House2相对第11天有部分支持阈值段，但完整可观测段为0；House19训练仅4个完整段。最低门槛以至少部分有效中心的阈值段计数，不代表完整事件足够或相互独立。逐块同时输出supported_threshold_segments和complete_observed_segments，不能以最低门槛通过掩盖这些限制。
