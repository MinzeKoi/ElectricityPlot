# House5 同户时间对照试验：代码准备（2026-10-25）

任务10，本版本仅实现独立协议、数组纯函数、选择门禁与合成测试；真实拟合0、真实House5训练池0。没有读取House17/最终8、9、20数据。工作区根与本次新增目录未找到适用AGENTS.md。保留已有修改；不修改默认data.protocol、DATA_CONTRACT、旧模型、UI和旧完成目录。

人类授权原文：“允许这个独立时间对照试验”。授权来源协调对话01a16d7b-478e-7780-a999-88abe71e8c94；本任务初始委派消息携带授权范围及条件：任务8当前损失校正仍未达标，且协调者提供其唯一source预选配方后，才可实际执行。本版本不自行检查开发成绩、不推断条件已满足、不向其他对话发消息。若任务8达标，本试验不执行，保留准备结果。

这是明确批准的同户时间工程试验，两项例外为House5从仅验证改为前缀新增训练，以及该试验来源训练/验证共享House5。它不符合原跨户来源协议。原data.protocol家庭重叠守卫和通用训练入口不变；新permit仅接受House5、aggregate/microwave及精确固定块/角色/边界。原禁用提案继续禁用，新配置另立版本。

## 固定UTC范围

全部左闭右开。训练块0 [2014-11-15,11-17)、1 [11-17,11-19)、2 [11-19,11-21)、3 [11-21,2014-11-22T23:30:00Z)。仅保留验证块4 [2014-11-23,11-25)、5 [11-25,11-27)。原2/19各14天及partial核心259806/849正中心不变。训练末尾至验证有1800秒带；原边缘、11-27及以后均排除，不能按成绩移动。

事件在完整输入时间轴上分段，跨所属角色/块整个阈值段排除。训练partial核心31中心总表/标签有效，至少24真实点且无连续两个缺失；核心不得越块。127外圈保持原mask，遇两连缺、时断或块界停止；缺失保持NaN和观测mask。House5数值索引固定2（House2=0、House19=1）；house字符串、UTC中心、事件身份校验，追加在原数组后，原行逐列相等，归一化沿用原2/19严格来源log1p常数。

预计新增76898/1426，仅为审计支持；将来真实准备必须独立核对。验证仍原C31中心34351/201，禁止partial扩展验证。纯函数要求传入冻结中心/标签及可选原核心输入，逐点比较；最终适配器须绑定冻结文件与原始来源哈希，不能用自行生成的“冻结”数组代替。最后4日已被历史选模接触，只称复用来源时间验证，不称新独立验证。

## 固定训练与选择

等待唯一source配方control_BCE50、importance_BCE50或unit_BCE，不按House17挑配方。真实只新增source5三个seed42/1019/2026一次，各3528更新、Adam .001、batch256、16正/240自然均匀负、CPU4、12个294更新checkpoint、CenterMaskTCN16六层dilation、127上下文、5329参数。同seed初始化与原模型相同；合并池索引流变化，只保持抽样规则/配额/seed，不声称逐批相同。启动前落盘started记录，每步进度及失败保留，最多3启动，禁止静默重试。

importance固定p_ref=849/259806；新Npos/Nneg目标正p_ref/Npos、负(1-p_ref)/Nneg；实际正q/(G E C)、负(1-q)/Nneg，q=.0625。逐点BCE50乘目标/实际概率再batch mean，不截断、不另归一化。目标类别质量仍50*p_ref与1-p_ref，正贡献约14%，避免把新增先验混入覆盖比较。其他配方保持原pos_weight和单位逐点权。

控制只读复用对应旧3模型与12epoch，0重训。控制和新增都在相同block4/5中心及截断输入重新选择公共阈值/checkpoint，逐块R≥.85、非退化，最差P/均P/F1/较早轮次。旧六块阈值不能直接用于本对照。完整两组×三seed来源矩阵才可选；新增组须均F1≥控制、每seed R下降≤.05、至少2seed P升。两组合格者按均P/F1，平分优先控制；允许无候选。唯一source选择必须在开发标签前落盘。

将来完整封存：原始输入/源码/配置/初始化/采样身份、模型、所有状态和阈值、两组全部开发预测、唯一source预选、原W50同支持时间mask及原功率状态不变核验。普通模型加载/推理读回一致后才一次开发评分。验收只评source预选，不允许开发改选：原W50同137523点，均P≥.08、每seed R≥.85、均F1增≥.02（1e-12容差）、至少2seed P/F1同升、任seed F1降≤.02、FP降≥30%、无退化。其他电器和微波炉功率沿原A，不宣称功率改善，不切默认。

## 本版本入口与边界

`scripts/run_microwave_source5_temporal.py` 提供prepare/train/source_select/freeze/evaluate/verify/report/synthetic。prepare仅复制本任务代码、配置、文档、测试并封存哈希；不读任何真实数组。verify校验准备哈希与0拟合状态，report拒绝覆盖。train/source_select/freeze/evaluate当前明确拒绝执行，即使传入模拟授权也不启用真实适配器；这避免条件提前触发。未来收到协调消息后另增真实执行适配器及版本，不修改本准备目录。本版本未交付真实PR曲线/误报/混淆及普通真实预测命令，它们依赖尚未满足的执行条件。

数组助手build_block、append_training、fixed_prior_weights、reselect_checkpoints、select_source独立可测。synthetic仅生成小型随机窗口，执行1步训练后验证、保存并普通load_state_model/predict_windows读回；不是3528步协议运行，也不计为真实拟合。测试覆盖三种配方、角色/块/时间/gap、整段跨界排除、核心缺失/时断、外圈截断、原行与归一化、索引、权重期望、两块选模/早轮次、完整矩阵、低召回/退化/无候选、准备不可覆盖和哈希篡改。

```powershell
.venv-m2/Scripts/python.exe -B -m unittest discover -s tests -p test_microwave_source5_temporal.py -v
.venv-m2/Scripts/python.exe -B -m unittest discover -s tests -p test_data_protocol.py -v
.venv-m2/Scripts/python.exe -B scripts/run_microwave_source5_temporal.py prepare --run outputs/accuracy/microwave-source5-temporal-preparation-20261025-v1
.venv-m2/Scripts/python.exe -B scripts/run_microwave_source5_temporal.py verify --run outputs/accuracy/microwave-source5-temporal-preparation-20261025-v1
.venv-m2/Scripts/python.exe -B scripts/run_microwave_source5_temporal.py report --run outputs/accuracy/microwave-source5-temporal-preparation-20261025-v1
```

权威依据：MICROWAVE_SOURCE_VALIDATION_AUDIT_20261025.md/.json、disabled提案和审计v4；只使用任务8计划理解配方，不读取其开发结果或自行确定最终配方。prepare登记为代码准备完成，不冒称真实数据/模型工作流已完成。

## 条件成立后的独立正式版本

协调者后续明确：“现在条件已成立并授权实际执行”。任务8v2 source_selection唯一选择TCN127，两新损失臂未通过来源保护，控制配方固定为任务7v3 control_BCE50。任务8原开发未达标由协调者确认，本任务不读取任务8开发成绩设计。用户续行原文“额度用完了，现在继续”。只执行固定control_BCE50三seed，不把其他损失测试作为真实候选。

新增microwave_source5_temporal_execution.py及microwave_source5_temporal_execution_20261025.json；原准备目录封存保留，原disabled提案不变。正式目录microwave-source5-temporal-20261025-v1。真实入口使用`--real`：prepare（需--authorization绑定协调条件与task8 source/freeze哈希）、train、source_select、freeze、evaluate、verify、report。各后续阶段使用正式目录内封存的run_microwave_source5_temporal.py和source代码运行。prepare白名单限定数值输入，train/freeze禁止真值及未登记数值路径，evaluate先完整冻结门禁后一次原真值CSV解析，verify/report只复用一次缓存。

新增真实路径测试直接调用同一fit_one，缩小为两checkpoint各一步，验证模型元数据127字段、保存/普通加载、两块重选、曝光和失败持久记录；测试冻结矩阵、推理门禁、数据白名单与source选择不可改写。真实运行只有三次启动许可，失败不重试。正式执行仍须逐项满足本计划前述固定计数与约束。报告分别说明整体相对W50是否达标、新增House5训练是否胜过对称重选旧控制，不能将仅旧控制选模变化归因为新增训练。
