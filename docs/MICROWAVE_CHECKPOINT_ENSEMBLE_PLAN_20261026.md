# Task16：同一种子既有检查点等权概率融合

本快照仅授权准备、旧来源审计、源码实现、合成测试和禁用注册。没有授权真实融合候选推理、融合校准或开发评价。独立 Task16 人类规则授权须绑定 registration.json 的 SHA256；Task14、Task15 历史授权均无效。new_fits=0、optimizer_updates=0，所有阶段禁止优化器与反向传播，无训练入口。

候选仅一个：每个 seed 独立从既有 unit_BCE 的全部12个 epoch，以原 source85 reselect_checkpoints / rank 语义选择最高3个合格且不同的 epoch。资格和排序逐个从原 block4/5 概率缓存、源标签及 plan 重算，与 history exact 对照。排序为 worst_precision、mean_precision、mean_f1 降序，最后早 epoch；每个 epoch 内阈值仍使用原代码完整候选。不足3个立即来源拒绝，无 fallback。绑定全部36权重、72缓存、history、model、complete、源标签、归一化、原计划和冻结 control_guard95 来源身份。Task15 增强模型不进入候选。

固定融合为同 seed 三个 sigmoid 概率，float64 求和除3后转 float32；有限支持逐点完全相同，NaN 原位保留，禁止零填充、权重平均、跨 seed 混合、topK/权重搜索。三个 seed 是复现单位，三个检查点不是独立试验。

只有未来独立授权后，source 才能运行普通原始 aggregate 推理，逐 checkpoint 对照源缓存（rtol=1e-5、atol=1e-7；原单点阈值状态 exact）；均值后按现有 Task14 算法计算 L95/H95、radius15。L 候选为全源 finite p exact 值，H 候选为 finite h 且 h>=L 的 exact 值。阈值排序、非退化与边界保留现有实现。

唯一比较为冻结 control_guard95。必须每 seed 每块 Recall>=.95、不退化，meanF1 不低 control，各 seed 平均 Recall 下降<=.05，至少2个 seed Precision 提高。按 meanP/meanF1 选择，完全 tie 保留 control。任何来源不合格写完整来源报告，不访问开发数据。

来源通过后先唯一预选，再封存 A/W50/control_guard95/ensemble_guard95 ×3 共12格。原 A raw/corrected 全列 exact，其他电器状态 exact，时间、缺失、有效支持 exact。每个新/对照公共普通 CLI 输出 CSV 后 exact 回读。仅封存完整且校验通过后，evaluate 一次读取已有 Task12 truth_once 缓存，不读原真值 CSV；重复评价拒绝。保留逐 seed P/R/F1/混淆、成对新增/移除 FP 与连续片段、原8门槛，verify 用独立标量重算。不得看开发后改变检查点、阈值或规则。

prepare/verify_preparation 无模型构建；准备与 source 的读取有精确数值 allowlist，开发评价 JSON 亦禁止；最终8/9/20始终禁止。准备不读取 House17 数值、hash 或 shape。UI 默认不切换。目录不可覆盖，源码和数据/配置/模型身份冻结；实现修改必须新准备目录。

入口：run_microwave_checkpoint_ensemble.py prepare/source/freeze/evaluate/verify/report/predict。source/freeze/evaluate/predict 需要新的授权；verify/report 在纯准备状态可运行。

风险：训练种子内检查点共享训练数据、初始化和大部分错误；反复使用相同源验证块可能过拟合，House17属于重复工程开发，不是独立泛化证据。旧单检查点错误已有较高重叠，不能承诺融合显著提升。根独立重放36个旧单点结果与 history exact，给出的 top3 仅作为独立审计对照，不参与新候选调参。
