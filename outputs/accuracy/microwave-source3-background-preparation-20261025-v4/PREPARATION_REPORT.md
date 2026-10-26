# Task15 固定模板背景增强：完整实施准备

来源资格通过：214 个合法127点模板，268549 个合法背景，来自 3 个模板家庭。

|来源|原正中心|合法模板|合法背景|因127点未知淘汰|
|---|---:|---:|---:|---:|
|refit_2|466|39|105298|427|
|refit_19|383|44|101378|339|
|refit_5|1426|131|61873|1295|

本目录逐字节复用v2来源池，没有重构池或重新解析原CSV。v2初步报告中关于失败/缺少适配器的固定文案已由本报告更正和取代；v2保持原封存。

当前真实新fit 0、更新0、House17/最终8/9/20数值读取0。authorized=false；只有模板保留的disabled回执，不存在真实授权回执。旧fit授权、Task14零fit规则授权均会被拒绝。

固定增强：每批原16正前8slot替换；另8正及240负exact不变。源家庭/UTC日→事件→合法中心分层均匀模板，独立均匀背景。RNG=SeedSequence([150025,seed])，不触动原采样/模型随机状态。全部先原始W合成，再原log1p归一化；背景mask0仍NaN，无fallback。

复用原fit_one、checkpoint选择、普通加载和gate算法。共享fit只增加默认None的原始W batch_transform，旧None输入路径exact。实际template/background曝光与原索引曝光分开保存。没有真实训练、来源阈值校准或开发候选产生。

固定3528更新/seed（12×294），三seed共10584；每seed 903168个原索引slot，其中28224个合成正slot，三seed84672个合成slot。训练费时仅能参考原模型：同结构原训练成本量级，额外逐batch合成、日志与归一化会增加开销；本阶段不运行计时训练。

授权后的唯一训练命令（此处为待执行说明，不会自动运行）：

```powershell
.venv-m2/Scripts/python.exe -B "outputs\accuracy\microwave-source3-background-preparation-20261025-v4/run_microwave_source3_background.py" train --run "outputs\accuracy\microwave-source3-background-preparation-20261025-v4" --authorization <本任务新增3fit人类回执.json>
```

随后同入口依次 source、freeze、evaluate、verify、report；均使用同一明确授权。source使用固定L95/H95/radius15并与Task14无增强guard95对照；来源不合格则拒绝freeze，不读取开发数据。freeze封存A/W50/control_guard95/aug_guard95×3seed完整12格和普通推理exact证据后，evaluate一次读取已有Task12真值缓存。

普通预测入口 predict --family control_guard95|aug_guard95 --seed 42 --aggregate <总表CSV> --output <新CSV>，使用同一授权并校验来源选择。UI默认不切。

来源/开发验收门槛全部固定在config.json；House17若未来读取仅是重复工程开发。不能把本次池可行性解释为增强效果已验证。

审阅证据：source_pool_audit.json、date_event_support.csv、audit_refit_*.npz、pool_manifest.csv、identity.json、frozen_replay.json、prepare_access.json、registration.json及source/完整源码快照。首次v1列名解析失败已保留，不计训练；v2池资格通过，本快照提供完整实施准备。
