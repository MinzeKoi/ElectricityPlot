# Task16 同种子 top3 等权概率融合执行报告

新训练 **0** 次；优化器更新 **0** 次；Task15 增强模型不纳入。最终8/9/20保持隔离，UI默认未切换。

|seed|合格旧检查点数|固定top3 epoch|
|---|---|---|
|42|12|5, 3, 8|
|1019|12|12, 5, 9|
|2026|12|11, 9, 10|

36个旧单点检查点均从来源缓存重放 source85 资格和排序，并与 history exact 对照。融合为逐seed sigmoid概率 float64求和再转float32，NaN不填充，支持必须exact一致。

根另行独立重放36个旧检查点，确认相同top3；该证据为外部交叉核对，本快照仍以自身 audit/access/identity 为依据。

检查点共享训练、初始化与来源验证块，存在统计依赖和来源验证过拟合风险。House17是重复工程开发，不是独立泛化证据。旧单点错误高度重叠，不能保证融合大幅提升。

[完整固定协议](protocol.md) · [旧来源审计](checkpoint_audit.json) · [来源身份](identity.json) · [准备读取审计](prepare_access.json) · [冻结注册](registration.json)

来源拒绝；唯一预选 `control_guard95`。

[来源完整报告](source_report.md)
