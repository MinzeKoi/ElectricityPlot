# 微波炉背景组合：唯一候选的训练前登记

2026-10-25，任务3。只新增 event_background，对照为任务2封存的 event 与 natural 各三个种子。最多新增3次真实拟合，种子42/1019/2026；本准备阶段0次拟合，必须等待任务2完成及协调对话执行消息。不得读取House17、最终House8/9/20，或使用来源外层标签、分数、误报设计配方。背景依赖仅为待检验假设。

实现从任务2的封存生命周期派生独立模块，保持其BatchStream、selection、rank、模型、初始化、优化器和普通推理逻辑。训练数据与内验证数组逐字节复用，log1p统计复制自真实来源训练，绝不重估。来源两户训练[0,10)天、1800秒边界隔离；验证[10,11)、[11,12)，来源时间开发诊断[12,14)。31点、8秒、200W、每批16正加240自然负、Adam .001、BCE50、12×294=3528更新完全相同。模型网络、默认模型、其他电器、界面及公共数据不改。

## 唯一变换和可用性

每个正样本的模板必须是该被抽到中心的原始31点微波炉功率，31点总表和标签均需有限非负且全在来源训练区间，中心标签必须等于封存采样标签。缺失保持缺失，不插值、不换模板、不改事件身份。为排除明显物理错位，模板窗的总表减微波炉也须逐点非负，否则记录排除。

背景仅取原来源训练中心，31点微波炉均低于200W，31点总表/标签均有限非负，背景=原总表−完整原微波炉功率；残差任一点负数即排除，禁止裁零。低于阈值的待机目标功率同样完整减去。合成总表=背景+原模板，输出float32；模板中心按200W定义标签，无缩放、平移、打乱时间或双重目标。

独立SeedSequence([训练种子,310025])随机流，在原批16正槽中无放回选8槽尝试增强。原始BatchStream完全不消耗增强随机流。背景优先顺序预先固定：其他户且其他UTC日期、其他户、其他日期、同日；选首个非空支持层，在层内打乱无放回轮换。模板无完整支持或无背景时保留原正窗，记录回退，禁止重试其他模板或增选槽位。实际比例可能低于8/16，按每种子报告。

prepare只读取任务2训练/内验证数组与无标签外层输入，解析CSV功率时先按训练时间过滤。登记模板/背景中心数、排除原因、每户UTC日期和阈值段身份；3528步×3种子的无拟合干跑确认原批次可重放和实际增强可用性。若0可增强，保留失败登记并禁止拟合，不扩大数据/变换。重复增强次数不能当新事件。任务2完整训练段House2/19仅14/4，House2第二内验证块完整段0；最低支持通过不等于独立日期或完整事件充分。

## 选择、冻结及比较

每轮只用来源内验证选择公共阈值，完全复用任务2：至少2个满足最少5正窗和2阈值段且有负类的块，每个受支持块R≥85%且非全开全关；按最差P、平均P、平均F1、更高阈值排序，检查点再按较早轮次。未支持块仍报告，不能承担召回保护。无合格时末轮/.5仅诊断，source_gate_passed=false。

三种子全部通过且四块均满足最低支持才具整体资格；event沿用F1不低于natural门槛，event_background额外要求源内平均F1不低于event和natural。三候选按源内平均P、平均F1排序；完全相同依次优先natural、event、event_background。允许无合格候选。先完成18份概率/状态冻结及source_selection，再统一打开来源外层标签。不得用外层改阈值或增强强度。

来源时间开发诊断按预登记门槛分别比较背景候选与event、natural：每户三种子R≥.85，平均P≥.08，配对平均F1增量≥.02，FP总数≤对照70%，至少两个种子P/F1同时提高，任一种子F1下降≤.02。两户对两对照全部通过且候选源内被预选才称本轮改善有支持；已用过的来源外层不能称新独立泛化证据。House17不在本阶段验收范围。

## 留痕和执行

每次尝试保存种子、更新/槽位、原批索引哈希、模板中心/事件/户、背景中心/户、固定变换参数、模板/背景和输出哈希。输入CSV哈希、审计回执、池、逐窗原因、归一化、配置、协议、源码、环境均由registration封存。增强日志逐条可复算，保存唯一阈值段覆盖和实际比例。推理读取标准state_classifier metadata，不使用增强。

入口支持prepare/train/freeze/evaluate/verify/report，所有写入禁止覆盖。本地工作副本改变后用运行目录内封存入口。训练前执行凭据必须有execute=true、maximum_fits=3、registration_sha256、audit_sha256、非空coordinator_execution_message，以及任务2的control_training_complete_sha256、control_freeze_sha256、control_evaluation_complete_sha256。依赖必须已完成且模型哈希有效；不自动恢复、不重训对照。执行凭据不在本准备阶段生成。

```powershell
.venv-m2/Scripts/python.exe scripts/run_microwave_background.py prepare --run outputs/accuracy/microwave-background-20261025-v1 --control-run outputs/accuracy/microwave-sampling-20261025-v1
.venv-m2/Scripts/python.exe outputs/accuracy/microwave-background-20261025-v1/run_microwave_background.py train --run outputs/accuracy/microwave-background-20261025-v1 --authorization tmp/microwave-background-execute-20261025.json
```

后续freeze/evaluate/verify/report同一入口和--run。报告包含逐种子/分区P/R/F1、混淆、FPR、AP、PR与事件覆盖、配对误报片段、增强前后来源验证曲线；prepare阶段不生成任何模型成绩。测试仅用合成数据小预算验证生命周期，不计入真实拟合。
