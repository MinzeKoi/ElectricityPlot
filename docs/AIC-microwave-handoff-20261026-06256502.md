# AIC 微波炉实验交接（2026-10-26）

## 先读：暂停状态与用户当前意图

- 用户明确要求过“停止.先汇报一下当前进度”。总目标已通过 goal 工具标为 **paused**；本次 handoff 不表示恢复工作。
- 所有训练已结束，Task19 对话已确认停止，当前没有仍运行的执行会话。不要重启旧 session、自动训练或自动建立下一项实验。
- 用户随后要求汇总十几个任务；已经解释后续编号共有19项，其中10项包含新训练，完成48个模型，另有一次294步的工程失败启动。前置损失/状态识别/小样本实验另计。
- 最新用户请求是调用 handoff，未指定下一会话重点。下一会话应先读取本交接，等待用户明确恢复或指定工作。
- 业务目标仍未实现：尚无通过原全部验收条件的新模型；默认界面模型未切换。不要将“准备完成”“测试通过”“某一分组变好”说成目标达成。

## 环境与对话定位

- 工作区：`D:\system position\document\真正的文档\AIC`。下文工件路径均相对此目录。
- Windows / PowerShell；Python：`.venv-m2/Scripts/python.exe`（Python 3.13、NumPy、Torch，CPU训练）。Matplotlib 未安装，不必为交接安装依赖。
- 本地项目 ID：`669e8dd6-6a81-419d-81c1-b950a1837ca9`。
- 根聊天：`01a16d7b-478e-7780-a999-88abe71e8c94`，标题“分析F1低的原因与改进方向”。总目标原文：“请你实现改进，提高模型预测效果，每一个任务开一个新对话，模型使用GPT6.1 Sol Medium”。
- 用户要求的新任务模型配置是 `gpt-6.1-sol` / `thinking: medium`，本地同项目。已有任务的继续工作使用原对话，避免重复建任务。
- 创建对话前调用 list_projects；需要用户对具体新任务的直接授权。曾有自动审批拒绝仅凭宽泛目标创建Task17；后经用户明确授权才创建。不能绕过自动审批。
- 用户曾授权本轮任务间必要跟进。根向任务对话的必要修正消息多数可用；任务对话向根转发曾被自动审批拒绝。无需为进度转发扩权限，优先 read_thread / wait_threads。
- 没有请求提交、推送、PR、部署、默认模型切换或清理旧版本。先前未发现适用AGENTS，但重新编辑前仍按实际路径检查。

## 当前唯一未完成事项：Task19 最终诊断交付

对话：`01a17683-26c5-7b12-ba56-3b4c8d61f5e3`
标题：“微波炉完整来源拟合与同户时间泛化诊断”。

正式待完成目录：
`outputs/accuracy/microwave-fit-generalization-diagnostic-task19-20261026-v4`

停止时的权威状态：
- v4六个冻结模型全量推理已完成，18份预测缓存、7596条分组结果。
- v3已完成两遍独立数值复核；v4尚未运行最终独立复核和交付封存。
- 报告此前重复选到“dataset总计/partition总计”两行，已修正为明确选择partition；13项针对性检查已通过。
- v4的 `DELIVERY_REPORT.zh.md`、`independent_verification.json`、`readonly_verification.json`、`delivery_manifest.json` 尚未生成。不要称v4已经正式交付。
- 0新训练、0优化器更新、0阈值/检查点搜索；没有读取House17或最终8/9/20。旧目录未覆盖。

已存在的重要文件：
- `diagnostic.py`：完整独立入口，包含模型推理、指标、只读复核、报告及seal逻辑。
- `config.json`, `pre_registration.json`, `data_identity.json`, `model_identity.json`
- `started.json`, `inference_complete.json`, `inputs_before.json`, `inputs_after.json`
- `metrics.json`, `metrics.csv`, `predictions/`, `identities/`
- `execution_notes.json`：v1–v3的失败与修复记录，直接读它，不在交接重复其全部内容。
- `synthetic_tests.log`, `test_task19_fixed_diagnostic.py`

**仅当用户明确恢复后**，先核实目录/对话状态，然后从现有v4完成，不要无故重做六模型全量推理。已读取确认的CLI：

```powershell
.venv-m2/Scripts/python.exe -B outputs/accuracy/microwave-fit-generalization-diagnostic-task19-20261026-v4/diagnostic.py verify --output outputs/accuracy/microwave-fit-generalization-diagnostic-task19-20261026-v4
.venv-m2/Scripts/python.exe -B outputs/accuracy/microwave-fit-generalization-diagnostic-task19-20261026-v4/diagnostic.py report --output outputs/accuracy/microwave-fit-generalization-diagnostic-task19-20261026-v4
```

确认只读核验/报告正常后，`seal` 动作会两遍verify、写入新的独立复核证据和正式中文报告/manifest。执行前先检查目标文件是否已经存在，禁止盲目重复或覆盖。若seal部分失败，保留证据，先判断能否安全补齐，必要时新版本；不要删除旧文件求通过。

### Task19 的指标解释边界

- 模型仅两族各三个种子：`control`（Task12旧unit_BCE）与 `expanded`（Task18扩展模型，已来源拒绝）。每模型只用已冻结选中权重和model.json中的原点分类阈值。
- 这是**固定点分类诊断**，不是正式L95/H95事件门控验收。不得把两者数值混用。
- `metrics.json`为列表；主要字段：family、seed、dataset、level、group、view。view精确值为 `original_support` / `strict_core31`。
- level包括 dataset / aggregate / partition / role / date / event_date。报告选择需明确level，block_4/5同时有dataset和partition行。
- aggregate组：`original_train`、`added_dates`、`validation_pooled`。partition组：`original_H2`、`original_H19`、`original_H5`、`added_H2`、`added_H19`、`block_4`、`block_5`。
- control的dataset=train指完整扩展矩阵，包含它未训练过的added_dates；只有original_train是旧模型的实际训练范围。expanded已训练过added_dates，该组结果不能称泛化。
- original_train混有三户，added_dates只有两户；同户时间比较必须按对应partition，而非直接比这两个总计。
- 核心31点部分缺失窗口使用同一冻结网络无梯度前向；strict核心使用原predict_windows。保留原127上下文和mask，不能为比较改变输入。
- AP使用同分组平均精确率定义，不是阈值搜索；零正例AP/Recall应是None。Precision/AP受开机率影响，必须结合FPR、Recall、样本量解释。
- 最新初步证据：扩展模型对新增2/19日期拟合提高，同时House5原训练分组AP下降、FPR上升；House5验证AP略升但固定点F1下降。支持检验多来源学习取舍，不证明某个唯一原因。小集合能记忆不能证明大数据分布的容量/优化已充分。
- 家庭优先均衡采样是被讨论的**未批准假设**，不是已确定方案；没有新的训练额度，也没有Task20授权。

### 根已完成的外部交叉核对

- v2 control42全615922窗口与根独立前向，row_index顺序一致，最大概率差约1.14e-13。
- v2主表156行(dataset/aggregate/partition，两族三seed两视图)的冻结阈值、混淆计数、P/R/F1/FPR和同分AP均独立复算一致。
- 临时证据：`tmp/task19-root-control42-points.npz`、`tmp/task19-root-control42-summary.json`。仅诊断证据，不是已选中新模型，不要直接把旧cache当v4最终交付。
- 预测NPZ键名是 `probability`, `row_index`, `binding_json`，不要猜作p/scores。
- 普通接口回读必须保留原批次边界，尤其末尾不足256的batch。改变batch形状可产生极小浮点差异，不应通过改权重或阈值修复。

## 固定数据、模型与验收协议：读现有工件，不重建规格

优先读取这些权威入口：
1. `outputs/accuracy/microwave-expanded-guard95-preparation-20261026-v3/protocol.md`
2. 同目录 `config.json`, `registration.json`, `identity.json`, `support.json`
3. 同目录 `PREPARATION_REPORT.zh.md`, `PROPOSAL.zh.md`, `EXECUTION_REPORT.zh.md`
4. 同目录 `final_delivery_manifest.json`（133项准备工件）和 `execution_delivery_manifest.json`（289项执行工件）。根已全部核对。

必要定位摘要：
- Task18 train.npz：615922窗口、2994正中心；前336704行是精确保留的原训练数据，后续为已批准固定2/19新增日期。source_role和house/t/event/raw_index均已封存。
- 旧Task10数据：`outputs/accuracy/microwave-source5-temporal-20261025-v2`，原336704/2275；House5仅已批准前缀，验证block4/5为既有后缀，共34351/201。
- 固定旧模型：`outputs/accuracy/microwave-source5-loss-preparation-20261025-v4/models/unit_BCE/seed_{42,1019,2026}`。
- 固定扩展模型：Task18 v3的 `models/expanded/seed_*`。
- 普通网络/输入方法：`src/virtual_submeter/models/center_mask_tcn.py`；旧训练与采样：`src/virtual_submeter/accuracy/microwave_source5_temporal_execution.py`、`microwave_sampling.py`。
- 正式开发目标与全部八门槛以Task18 protocol/code为准：三种子平均Precision>=8%、每种子Recall>=85%，并有F1/FP/非退化及原137523支持一致性保护。不能改成平均Recall达标。
- 当前Task19范围禁止House17开发和最终8/9/20的数值、hash、shape读取。不要为补报告打开其真值缓存。
- House5已获批前缀训练，不能误用最初“House5仅验证”的旧条件撤销后来授权；也不能继续扩House5日期。
- Task18是数据与更新量同时增加的工程对照，不能独立归因于扩数据。原控制不重训，不得暗加三控制模型。

## 已完成的实验历史：按报告查详细数值

这里仅给工件索引，避免重复各报告和协议。标题带“准备”的Task12/14/15/16/18后来已实际执行相应已批准实验。

|后续编号|内容|权威报告/入口|
|---|---|---|
|1|数据缺失审计|`docs/MICROWAVE_DATA_AUDIT_20261025.md`|
|2|事件采样与时间泛化，6模型|`docs/MICROWAVE_SAMPLING_REPORT_20261025.md`|
|3|早期两户背景增强，3模型|`docs/MICROWAVE_BACKGROUND_RESULTS_20261025.md`|
|4|完整来源训练，9模型|`docs/MICROWAVE_FULLSOURCE_REPORT_20261025.md`|
|5|缺失上下文感知，6模型|`docs/MICROWAVE_MASK_RESULTS_20261025.md`|
|6|冻结回归阈值校准|`docs/MICROWAVE_REGRESSION_CALIBRATION_REPORT_20261025.md`（记录过提前解压开发缓存shape的偏差，不得称完全盲化）|
|7|TCN31/127，6有效模型及1次294步故障|`docs/MICROWAVE_TCN_RESULTS_20261025.md`|
|8|两户采样损失校正，6模型|`docs/MICROWAVE_LOSS_CORRECTION_RESULTS_20261025.md`|
|9|来源验证代表性审计|`docs/MICROWAVE_SOURCE_VALIDATION_AUDIT_20261025.md`|
|10|加入House5前缀，3模型|`docs/MICROWAVE_SOURCE5_TEMPORAL_DELIVERY_20261025.md`|
|11|冻结模型事件门控|`docs/MICROWAVE_EVENT_GATE_DELIVERY_20261025.md`|
|12|三户BCE正例权重50改1，3模型|`outputs/accuracy/microwave-source5-loss-preparation-20261025-v4/DETAILED_REPORT.zh.md`|
|13|门控漏报及种子稳定性|`docs/MICROWAVE_GATE_RECALL_AUDIT_20261025.md`|
|14|来源峰值召回保护提高到95%，零训练|`outputs/accuracy/microwave-recall-guard95-preparation-20261025-v2/DETAILED_EXECUTION_REPORT.zh.md`|
|15|三户固定背景增强，3模型|`outputs/accuracy/microwave-source3-background-preparation-20261025-v4/FINAL_EXECUTION_REPORT.md`|
|16|同种子top3检查点概率融合，零训练|`outputs/accuracy/microwave-checkpoint-ensemble-preparation-20261026-v2/final_result.md`|
|17|真实事件覆盖和额外日期审计|`outputs/accuracy/microwave-source-event-coverage-20261026-v1/REPORT.zh.md`|
|18|扩展来源三模型，已完成并来源拒绝|Task18 v3的 `EXECUTION_REPORT.zh.md`|
|19|完整来源拟合/同户时间泛化诊断|上述Task19 v4，最终报告尚未完成|

前置小样本证据：`docs/MICROWAVE_DIAGNOSIS_REPORT_20261025.md`。小集合三seed接近完美，但同户时间留出明显退化；该结果不证明完整来源分布上容量充分。

最接近正式目标的是Task12，但一个seed Recall未过底线；Task14恢复召回后Precision/FP仍未过门槛。三户背景增强、检查点融合和最近扩展来源均来源拒绝。详细数值直接引用上述报告，切勿把来源指标和House17开发指标混在一起。

Task17审计数据不可改；只读入口在 `outputs/accuracy/microwave-source-event-coverage-readonly-20261026-v1/readonly.py`。已批准的本地额外日期和精确计数可查原报告/配置，不需重新扫描全盘或下载。

## 关键对话ID（读记录，不表示恢复）

- Task12：`01a171dd-6b4f-79d1-a163-707286207514`
- Task13：`01a171f9-a63c-75c3-be98-52d05ac297c8`
- Task14：`01a17203-4358-7702-817c-ba6c87571734`
- Task15：`01a172f8-6362-7d20-b1a3-3abc967e20ca`
- Task16：`01a1761b-e786-73d1-9c40-a30cfae5ee01`
- Task17：`01a17633-0a4b-7eb1-9616-55a3c524a60c`
- Task18：`01a1763f-6d35-7b22-8f4b-587b06e298e1`
- Task19：`01a17683-26c5-7b12-ba56-3b4c8d61f5e3`

## 授权、运行和工具注意事项

- Task18的人类训练授权仅3模型/19404更新，已经全部消耗。Task19仅固定模型诊断，未授权新模型或新采样策略训练。
- 原种子42/1019/2026、网络/归一化/标签阈值/数据角色和选择规则以每次冻结版本为准。新增实验必须独立预登记并得到具体新任务及预算授权；自动目标续跑不等于授权。
- 已完成/失败目录都不覆盖，不删除旧证据，不看开发成绩后换候选/调门槛，不改UI默认。
- 读取旧线程时 read_thread 一轮可能非常大，解析JSON仅输出最后状态/必要条目；maxOutputCharsPerItem不是总长度上限。list_threads的limit最大50。
- wait_threads用特定thread和cursor；超时不表示作业停止，不能因此重启。停止后本次已无活session。
- PowerShell读中文文件/JSON用UTF-8。子Python标准输出应显式UTF-8；subprocess capture需errors=replace且stderr可能为None。已有Task18修复，勿退回旧路径。
- 不做大范围rg输出目录扫描；从已知manifest和精确路径找文件。不要猜NPZ键名、报告文件名或集合view值。
- 用`-B`禁bytecode；verify/report默认只读。新证据只能写明确的新文件/目录，不覆盖封存工件。
- 当前工作区存在新实验代码与未提交产物；别假定Git干净，也别为交接提交、清理、移动或删除。

## Suggested skills / 建议技能

下一代理根据任务实际需要调用Skill工具；若环境没有Skill工具，按可用技能目录读取对应SKILL.md。不要仅因关键词启动额外流程。

- **diagnosing-bugs**：若恢复Task19后verify/report/seal仍失败，先建立最小失败反馈再修。位置：`%USERPROFILE%\.agents\skills\diagnosing-bugs\SKILL.md`。
- **code-review**：仅在用户要求审查变更时，检查实验协议与实现一致性；不要为已暂停任务自行启动并行评审。位置：`%USERPROFILE%\.agents\skills\code-review\SKILL.md`。
- **openai-docs**：只在涉及Codex模型、配置、工具或额度产品问题时使用；诊断现有模型指标无需网络检索。位置：`%USERPROFILE%\.codex\skills\.system\openai-docs\SKILL.md`。
- **handoff**：需要再次交接时使用，保存OS临时目录，不在工作区再造状态文档。位置：`%USERPROFILE%\.agents\skills\handoff\SKILL.md`。

本交接未包含API密钥、账号凭据或用户个人身份信息；本地用户目录使用环境变量表示。