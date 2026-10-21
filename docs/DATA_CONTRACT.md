# 数据与结果契约 v6：M1–M5 与 A0 开发诊断接口

状态：M1/M2、M3 选片/状态适配和 M4 可信输出工程接口已实现；正式多户协议与算法效果尚未冻结。时间字段持久化为 UTC ISO 8601（以 Z 结尾），区间均为左闭右开 `[start,end)`。

## 1. 规范化总表

| 字段 | 类型 | 含义 |
|---|---|---|
| household_id | string | 数据集内匿名家庭标识 |
| timestamp | datetime | 带时区时间戳 |
| aggregate_power_w | number / null | 总有功功率 W；缺失保留 null，不填 0 冒充关闭 |
| quality_flags | string[] | 缺失、重复、异常、插值或不支持输入等标记 |
| coverage | number | 当前 8 秒区间内有效观测所覆盖的时间比例，范围 0–1 |

不得把累计 kWh 读数直接当 W。重采样使用明确时间区间、均值/能量口径及覆盖率；不能简单抽取每 N 点替代区间平均。缺口不假定整段功率不变并生成未经说明的电量。

当前实现：以 Unix 时间的 8 秒整数倍为区间边界；一次观测最多沿用到下一观测或 8 秒后，取较早者。不外推最后一个点，不插值。完整覆盖区间输出时间加权均值；部分覆盖区间功率留空，并保留 coverage。负数和非有限功率变为缺失并标记；非数字、时间倒序/重复、无时区用户时间或字段损坏直接失败。CSV 内 quality_flags 是 JSON 字符串数组，空数值单元格表示 null。仅积分完整有效区间，报告名称为 observed_energy_kwh，不推算整日耗电。

REFIT 时间使用 Unix 列，避免对本地 Time 列的夏令时再做猜测。清洗版原有补值和清零无法撤销，所有记录保留 upstream_cleaned_missingness_unknown 提示。Issues=1 由分表和总表关系生成，因此只影响分表标签有效性，**不得用于筛选或标记模型可见的总表**。测试验证了修改分表答案及 Issues 不会改变导出的总表字节。

首版只接受无光伏、无储能的适用家庭。大量负功率、时间倒序、单位不明或严重缺失均需拒绝或显式降级。

## 2. 分表标签与划分

分表标签包含 `household_id`、`timestamp`、`appliance_id`、`appliance_type`、`power_w` 和质量标记。标签与总表按数据源元数据对齐，通道不存在和通道缺失不能视为设备关闭。

分表有独立访问边界：来源训练标签提供给训练模块；目标测试标签仅提供给评价模块。划分清单记录来源文件、哈希、通道映射、家庭、区间、间隔、隔离带和生成版本。

`prepare-refit` 输出 `aggregate/house_<id>.csv`、`labels/house_<id>/<type>.csv`、`manifest.json`、`catalog.json` 和 `quality.html`。分表字段为 household_id、timestamp、appliance_id、appliance_type、power_w、coverage、quality_flags；设备标识同时包含原始通道。House 4 的两台洗衣机为明确标记的类型级合计，不当作单台设备。不存在的通道不生成全零标签。

`freeze-protocol` 把输入放在 inputs/，来源标签放在 source_labels/，适配池分表放在 oracle_labels/，测试答案放在 evaluation_labels/。协议记录配置快照、包源码指纹、上游审计/计划/文件哈希、预算和随机种子。分区按完整区间复制，适配池与测试之间的隔离带至少为模型上下文和确认窗口的最大值。已有输出拒绝覆盖，失败时临时派生产物不发布为成功目录。

调用 `virtual_submeter.data.protocol.read_partition(manifest_path, partition_id, consumer, include_labels=False)`：

| consumer | 可访问分区 | 可读取分表 |
|---|---|---|
| training | source_train / source_validation | 是 |
| calibration | source_validation | 是；不得读取来源训练/目标家庭标签 |
| selection / adaptation | target_adaptation | 否 |
| inference | source_validation / frozen_test | 否 |
| evaluation | source_validation / frozen_test | 是 |

每次读取校验实际文件哈希。此接口防止合作模块误用答案，不是抵抗拥有本地文件权限者的操作系统安全隔离。M3 的 `data.oracle.answer_pending(session_dir)` 仅向已持久化、预算内的待确认问题返回窗口弱标签，不提供整池 oracle 读取接口。协议中 purpose=engineering_pilot_only 的样本仅用于工程验证；前缀或来源未校验的输入不能冻结为 final_evaluation。

### A0 开发诊断

A0 从已查看的工程协议复制出独立协议，在副本设置 `accuracy_usage=development_only`、`parent_protocol_sha256` 和明确的历史已查看说明；旧协议与旧产物不修改。基础权重不变，仅在模型副本重新绑定开发协议哈希，并记录原模型清单哈希。查询种子不是重新训练种子。

新增 consumer `development_diagnosis` 可读取来源和开发目标分表，但必须同时满足 `purpose=engineering_pilot_only` 与 `accuracy_usage=development_only`。旧协议、正式协议和未标记协议均拒绝；selection/adaptation 权限不变。该消费者仅供所有适配模型和未来预测冻结后的离线诊断，不得传入学习或选片路径。

A0 输出 `diagnosis.json`、`query_distribution.csv`、`queries.csv`、`state_metrics.csv`、`availability.json`、`report.html`，并保留逐案例参数、训练历史、预测、指标、源码快照及哈希证据。状态指标按相同有效时间支持比较；无支持为 null，不判为全关。完整事件指由有效关闭样本包围、连续且无缺失的开机段，只表示固定阈值事件，不代表完整物理运行周期。开发诊断不作为最终盲测或精度改进证据。

### A1 基础模型

A1 另建 `accuracy_usage=development_only` 协议，来源训练/来源验证/开发家庭按整户隔离，适配池与开发未来区间继续保留至少模型上下文及确认窗口的隔离带。原始文件、审计、截取天数、纳入/排除理由与下载回执绑定到新协议。来源户纳入只使用预先声明的数据质量门槛，不能按模型效果挑户。

模型可声明 `state_threshold_view=raw_power_w`：状态与 raw_state 都按原始非负功率和来源冻结阈值计算，功率修正仍独立保存。未声明时维持 M2 的 state 按修正功率、raw_state 按原始功率行为。A1 统一原始阈值，避免仅启用 INITIAL 状态头就改变决策；不将这个口径变化算作确认收益。历史模型须标记原始视图与历史默认视图。

基础候选、顺序、选择规则和种子在比较前保存；训练及逐轮 checkpoint 选择仍只读来源训练/验证标签。开发评价允许选择有限的基础配置，不能用于最终效果声明。选定配置冻结后训练其余种子，各确认策略复用相同种子下的同一文件哈希。阈值事件检出与逐点状态、功率/电量、共同有效支持同时报告。

## 3. 用户确认记录

### A3 动态会话契约

A3 新增独立版本 2 会话，不将动态历史写入 M3 版本 1。固定 A2 已选择的学习器、来源权重、来源状态头、回放和训练参数；每累计 5 次回答从来源初始化，用全部可见回答重新适配。各来源种子独立生成轨迹，0/5/10/20 是同一条最长轨迹的前缀。所有策略使用同一完整候选池与更新频率；随机、不确定性、拟议策略及静态／去多样性／去覆盖消融均保存。

每轮保存可见回答数、前缀哈希、模型清单哈希与当前状态窗口分数；会话绑定候选池、协议、基础模型及策略参数哈希。pending 在回答代理读取前落盘，重复请求返回同一问题；回答原子提交，跳过和不确定均计费。恢复时按各轮模型与当时可见前缀重放选片，不能用最终模型重放早期问题。静态消融始终使用零确认评分，仍按相同频率拟合和评价。候选形态来自总表，覆盖仅按设备身份和已回答次数，探索使用固定种子；未知正负标签不参与候选筛选或平衡。

数据权限沿用 A2；仅数据侧代理可读取已持久化问题的窗口标签，学习器无新增标签权限。全矩阵查询、模型与未来预测冻结后才读取开发未来分表。当前 A3 仅用于已查看开发家庭，不支持最终效果声明；功率输出冻结，状态分数未经校准。会话恢复要求原协议、基础模型与轮次模型仍可访问。

### A2 固定查询与状态学习器契约

A2 仅接受 `purpose=engineering_pilot_only` 且 `accuracy_usage=development_only` 的 A1 协议与已冻结来源权重，不新增消费者权限。来源状态头与特征标准化通过 `training` 读取来源训练/验证；目标拟合只接收总表特征和经 `validate_confirmation_history` 校验的回答前缀。未查询目标分表、目标逐点状态及未来分表不能传给学习器。查询代理沿用 M3 的已持久化 pending 机制，标签来源标记 `simulated_from_submeter`。

每个查询种子固定一条最长 20 次、三设备共享预算的随机序列，0/5/10/20 为嵌套前缀；同一序列跨来源种子和学习器复用。候选完整性只依赖总表上下文，不按未知答案平衡正负类别。`unsure/skipped` 计费。候选不足时运行失败，不能缩减预算后宣称相同预算。

L0 保留旧三系数 max MIL；L1 仅更新有界偏置；L2 使用冻结 CNN 的倒数第二层时序特征和原始功率对数比，来源逐点状态 BCE 与目标窗口 MIL 共用同一线性状态输出。L2 比较 max 与温度固定的 log-mean-exp 平滑聚合；负窗口的零活动语义由高分点约束，不展开正回答为逐点标签。单类回答的受保护规则预先固定为仅更新受限偏置；未查询设备和无有效回答严格返回该学习器自身来源决策。另保存无保护、仅限幅、限幅加回放、完整正则的逐步消融，不依据开发未来结果决定回退。

新增可选 `model.json.accuracy_state_head`，版本 1；记录特征归一化、每设备系数、来源先验与目标家庭。L0/L1 使用原始阈值对数特征，L2 使用来源标准化的冻结时序特征。功率权重不更新，`confidence` 保持 null，`state_score` 是未校准分数。通用预测接口支持这些独立模型；旧 `state_adapter` 和旧会话格式保留。

实验保存一份每来源种子的基础未来预测，逐案例 `.npz` 仅存状态分数与有效支持，恢复时与共享基础功率逐点合成；不把省略的重复功率当成缺失。来源特征缓存与模型/协议/聚合输入哈希绑定。冻结清单绑定配置、来源/模型/查询/状态数组和源码；评价阶段才读取开发未来分表，核验不重新打开分表。主比较保留逐种子、逐设备 Precision/Recall/F1、开机比例、支持、退化、功率及电量误差；分别报告结构零基线变化与确认增益。

字段草案：`query_id`、`household_id`、`window_start`、`window_end`、`appliance_type`、`response`、`label_source`、`selection_strategy`、`model_version`、`budget_before`、`budget_after`。

- `response` 为 `used`、`not_used`、`unsure` 或 `skipped`。
- `label_source` 明确为 `simulated_from_submeter` 或 `human_confirmed`。
- 一段时间某设备的一次回答算一次，不因前端合并提问而减免预算。
- 弱标签只表达窗口中是否用过，不能反推为窗口内每一点都开启，也不等于设备耗电量。
- 离线代理按冻结的训练协议从已查询片段生成弱标签，其余分表数据不向模型开放。
- 选择策略只能使用当时可见的总表、预测、已确认标签及训练/验证阶段冻结的信息。

`validate_confirmation` / `validate_confirmation_history` 校验上述固定字段、设备、窗口边界、适配池归属、标签来源、响应枚举及逐户连续预算。窗口必须恰为配置的 query_window_seconds，边界须与网格对齐；不确定和跳过同样占用一次预算。历史中 query_id 不得重复。M3 进一步把记录绑定到持久化待确认问题和模型版本，并回放策略核验历史；同设备不重复/重叠选片。

### M3 会话与适配

- `adaptation.session.create_session(model_dir, manifest_path, partition_id, output, settings_path, budget, strategy, seed)` 只接受未适配基线和 `target_adaptation` 分区；预算/种子必须在冻结协议中。输出 `session.json`、协议快照和只使用总表生成的 `prediction/`。
- `select_next(session_dir)` 先持久化唯一 `pending` 问题；重复调用返回同一问题，不消耗预算。`confirm(session_dir, query_id, response, label_source='human_confirmed')` 原子保存回答并加一，拒绝过期或重复回答。文件锁防止同时提交；失败保留原记录。重新加载会话可继续。
- `strategy` 为 `random`、`uncertainty`、`proposed`。候选必须有完整、连续的模型上下文与确认窗口；按设备采用不重叠网格。窗口状态取初始状态分数的最大值，不确定性为到 0.5 的距离；组合策略增加估计耗电影响和由总表事件数计算的易回答性代理。代理并非真实用户可回答性证据。
- `data.oracle.answer_pending` 只读取当前问题对应设备的适配池 oracle 文件，返回 `query_id/response/label_source`；“用过”为窗口内至少一个有效样本达到来源阶段冻结的开机阈值。任一缺失、覆盖不足或时间不全返回 `unsure`，不暴露逐点功率、开关时刻或未查询答案。模拟来源固定标为 `simulated_from_submeter`。
- `adaptation.training.adapt(session_dir, output)` 只使用已确认的 `used/not_used` 窗口，最大池化 MIL 损失监督“至少一刻用过”，不把正标签铺成全窗口开机。冻结功率模型，训练每设备三个系数的状态头；固定迭代次数，不使用测试集早停/调参。`unsure/skipped` 只计预算；无可用标签时保留原始决策规则。
- 产物为独立模型目录，含 `model.json`、原功率权重（如适用）、`queries.json`、`adaptation_history.json`、设置及协议快照。记录基线模型指纹、目标家庭、预算/回答数、真实/模拟来源、策略、种子、源码指纹和耗时。查询记录有 SHA-256 校验；适配模型只用于该目标家庭。

## 4. 预测结果

单次运行包含 `run_id`、数据/配置/代码/模型版本、输入区间与运行状态。逐时间输出每个目标设备的估计功率、状态或状态分数、未解释功率、质量标记和可选可信信息。

同时保留原始预测及约束修正后的预测。未解释部分包含未建模负荷与预测/计量误差，不自动命名设备。无真实评价支持的可信度返回缺失，不默认为 1。

电量使用功率与有效时间长度积分，单位 kWh；覆盖不足时报告覆盖率或不可用，不跨缺口补造完整日用电量。

`models.prediction.predict(model_dir, output, manifest_path=None, partition_id=None, input_path=None)` 接受协议分区或规范化单户总表 CSV，二者互斥。预测仅调用 aggregate-only 访问；适配池用 selection 消费者，验证/测试用 inference 消费者。模型不存在、状态非 trained、权重哈希错误、协议不匹配或没有完整模型窗口时失败，不发布成功目录。

M2 `predictions.json` 为 schema_version=1 的运行对象，包含模型/代码/协议/输入指纹、sample_interval_seconds、appliances 和 rows。每个 row 包含 household_id、timestamp、aggregate_power_w、appliances、unexplained_power_w、raw_excess_w、correction_method、quality_flags。每个设备子对象字段为：

| 字段 | 语义 |
|---|---|
| raw_power_w / power_w | 非负原始估计 / 一致性修正估计，单位 W |
| raw_state / state | 基线使用功率阈值；M3 有可用回答的设备使用独立状态头，详见下文 |
| correction_w | 修正后减原始功率，M2 不向上摊满电量 |
| confidence | 恒为 null；M4 功率经验区间不作为状态可信概率 |
| state_score | M3 适配设备可选的未校准状态分数，不能作为 confidence；缺失保持 null |

当整个中心窗口的总表无法完整取得时，该时点各设备功率与状态、剩余项均为 null，并标记 context_missing_or_boundary；不能标为 off。当前窗口 31 点 / 248 秒，使用未来上下文，只适用于离线分析。窗口不能跨分区、缺失或时间跳变，归一化和功率缩放统计只在来源训练窗口上计算。

设备总和仅在超出总表时按比例缩小，记录原始超限和 correction_method=scale_down_only；不足时保留剩余项。数值一致性允许浮点舍入误差。M2 的这一处理是可追溯工程约束，不等于 M4 可信输出实验或未知电器识别已完成。另有宽表 predictions.csv、元数据 run.json 和独立离线 report.html。

`models.training.train` 固定来源训练/验证家庭，验证归一化 MAE 选择 checkpoint；不会读取适配池或测试标签。模型导出包含 model.json、weights.pt（CNN）、配置/环境/训练历史/零查询记录，加载权重显式使用 weights_only=True 并核验 SHA-256。常量对照使用来源训练中心标签均值；零预测保留为单独对照。所有基线评价使用相同完整输入窗口，以便比较。

`evaluation.runner.evaluate(prediction_dir, manifest_path, output)` 独立读取 source_validation 或 frozen_test 分表，拒绝适配池答案和删减过的预测时间轴。逐设备同时输出 raw / corrected 的 MAE、运行片段 MAE、Precision/Recall/F1、支持样本数、共同有效区间 kWh 误差及按 UTC 日归并的积分和覆盖率。无有效支持或无定义的指标为 null，不默认为满分。不存在除以零真值的电量百分比。评价覆盖率的分母是整个预测时间轴，设备关闭样本不代替运行片段指标。

## 5. 运行状态

初始化、数据检查、训练、预测和实验评价为不同状态。模型或输入无效时返回具体原因，不能展示成功结论。模拟数据、真实数据、模拟确认和真实确认在运行记录与界面均要可区分。

状态入口只报告能力与配置，不推断本机是否已有成功运行。训练/预测/评价产物分别标记 trained、complete、evaluated；Streamlit 在输入或模型失效、切换数据或分析失败后清除旧结果。M3 变更家庭、模型、预算、策略或种子同样清除适配结果。批量运行先冻结全部策略/预算/种子模型，再开启测试评价；候选不足则记 failed，不以更少回答冒充相同预算。模型和数据均不提交 Git；M3/M4 真实数据效果、界面运行和正式实验仍需独立验证。

M3 在 `predictions.json` 中增加 `adaptation`、`base_model_id`、`state_definition` 和 `query_summary`，保持 schema_version=1 的兼容扩展。未适配模型的状态仍为功率阈值；有可用弱标签的设备使用独立 MIL 状态头（阈值 0.5），因此状态不再必须等于功率是否越过阈值。评价读取显式 `raw_state/state`，旧格式缺少状态时才回退到功率阈值。功率 MAE、电量和一致性与基线分别报告；M3 不宣称功率改善。未查询设备的决策、缺失状态及 `confidence=null` 保留。

## 6. M4 可信输出、校准与压力测试

`trust.calibration.calibrate(model_dir, manifest_path, settings_path, output)` 只接受未适配功率基线和隔离的来源验证家庭。新增 consumer=`calibration` 只允许 `source_validation`，不能访问训练、适配或测试分表。对各设备的原始/修正功率分别计算绝对残差，取排序后第 `ceil(n * interval_level)` 个值为半径；有效样本少于 `minimum_samples` 则失败。默认标称水平 0.9、最少 20 点；时间相关样本不等于独立校准样本。

校准目录包含 `calibration.json`、原字节 `settings.json`、`protocol.json`。记录设备、来源分区和样本数、经验覆盖、功率指纹、协议/设置/源码/模型指纹、总功率来源范围及 `target_labels_read=false`。来源验证重复参与模型选择，故 `coverage_guarantee=false`；不是 split-conformal，也没有目标家庭或同时覆盖保证。不得依照测试覆盖率反复调半径。

`predict(..., calibration_dir=...)` 通过功率指纹匹配校准，M3 仅更新状态头可复用；改变功率主干、归一化、设备、协议或采样不能复用。预测保持 schema_version=1，增加：

| 字段 | 语义 |
|---|---|
| confidence_method | `source_validation_absolute_residual_quantiles`，无校准为 null |
| calibration | 校准快照 SHA-256、标称水平、来源样本和限制；目录保留快照 |
| raw_lower_w / raw_upper_w | 原始功率的来源经验区间；下界截零 |
| lower_w / upper_w | 修正功率的来源经验区间；不按总表截断上界 |
| quality_summary | 各质量标记数量、覆盖率、超限率、修正幅度和有效区间电量 |
| data_kind / model_data_kind | 当前输入/训练数据性质；合成契约为 `synthetic_contract_fixture`，用户导入为 `user_import` |

区间是逐设备边际区间，不表示各设备同时取上界仍满足总表约束；`confidence` 始终 null，状态分数仍未校准。缺上下文时所有功率、状态、区间和未解释项均为 null。`power_uncalibrated`、`source_validation_intervals_no_target_guarantee`、`aggregate_outside_calibration_range`、`unexplained_load_present` 分别标记区间可用性、适用范围和剩余项；范围标记只是外推提示，不是未知电器检测。

评价新增 `intervals`（逐设备 raw/corrected 的经验覆盖率、平均宽度、支持数）、`attribution`（共同标签支持上的归因电量比例、平均设备 MAE、过归因及漏掉已知电量）。`all_unexplained_control` 以所有设备输出零作为对照，明确展示其归因比例为零但会漏掉真实已知设备电量。`consistency` 同时保存原始/修正超限率、移除幅度、平衡残差和未解释电量。缺少支持的指标为 null。

`evaluation.stress.stress_test(...)` 只对 `frozen_test` 运行固定矩形背景脉冲。默认注入 1500 W，每 40 点持续 10 点，偏移 10 点；按时间网格定义，无总表时不注入或补值。目标设备标签不变。保存基准/注入输入、两份完整预测及其哈希到 `frozen_predictions.json` **之后**才打开测试标签。按中心时刻有/无背景分组报告误差、错误激活、区间、归因和一致性；离线上下文影响可能超出脉冲时间。额外已知归因占注入电量比例只是敏感性指标，不是未知设备识别准确率。

压力测试输出 `stress.json`、`injection.json`、`injected_aggregate.csv`、设置/协议快照、`baseline/`、`injected/` 和汇总报告。注入预测不能作为原始协议总表交给普通 `evaluate`；只能由成对压力评价路径读取原标签。`verify_m4.py` 校验所有产物哈希与预测/评价身份，不重开标签；这属于本地产物完整性检查，不是对恶意修改者的密码学认证。

新预测的 `run.json.artifacts_sha256` 绑定预测 JSON、CSV、HTML 与校准快照；界面恢复前校验，拒绝损坏导出或不匹配元数据。旧 M2/M3 无哈希清单时保留兼容加载，但仍验证数值一致性。切换输入/模型/校准、发现总表哈希变化或运行失败均清除旧结果。合成数据与合成模型在报告和界面单独提示，不冒充 REFIT 测量。真实 REFIT 前缀/CNN/Streamlit 已完成单目标家庭工程运行，正式多户与独立成员复跑仍待验收。界面时间轴强制 UTC，原始时间戳作提示，缺失段断线。
## 7. M5 实验运行附加契约

M5 不修改 M1–M4 的输入、弱标签或预测字段。新增 `configs/m5_experiment.json` 声明实验模式、主干、预算、种子、背景压力开关和测试查看历史；实现及命令见 [M5_HANDOFF.md](M5_HANDOFF.md)。

`outputs/<run_id>/frozen/` 包含原配置字节、协议、源码/依赖锁、每种子模型、查询记录、来源校准和全矩阵预测。`freeze.json` 保存文件 SHA-256 清单，必须先于任何测试评价；`run.json` 的状态为 freezing → frozen → evaluating → evaluated，异常进入 failed_freeze/failed_evaluation，失败不得伪装完成或覆盖重跑。

`evaluation/` 保留每个实际预测的完整 raw/corrected 指标和报告；`stress/` 为 P1/P2 共用的成对背景证据；`summary.json` 保存每组/家庭/种子/预算/设备数值、逐户及宏平均、配对差值、耗时与限制。P2 是 P1 的修正视图，不是重复训练，也不增加确认数；B0–P1 主比较均取 raw，P2 取 corrected。所有视图保留共同有效支持。

未定义指标保持 null。汇总先种子后家庭等权平均，报告有效/缺失数和样本标准差。`evidence.json` 保存评价/汇总文件哈希；`verify` 不读取原始分表，须重新从逐次指标核对汇总。`compare` 的数值一致性不构成独立人员/机器证明。测试历史仅为操作者声明，不是访问审计或操作系统级隔离。
