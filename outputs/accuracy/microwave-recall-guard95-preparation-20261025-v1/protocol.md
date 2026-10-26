# 微波炉 Recall95：独立新轮预登记（2026-10-25）

## 状态与授权边界

本轮只准备、旧来源身份重放、合成测试及封存，配置固定 `authorized=false`，0 fit / 0 optimizer update。任务12的三fit预算已用满；本轮不申请也不使用新训练预算。旧H85失败及原协议保持原样。本轮预登记不代表人类已批准新规则；授权原话和会话证据尚无，回执模板保留空值，禁止自造。所有 source_select / freeze / evaluate / verify / report / predict 入口先验证独立人类授权回执，绑定本轮 registration.json SHA256、规则名、三seed、0fit/0更新及原话/证据引用。旧训练授权、配置开关、审计建议均不算批准。prepare 和 verify_preparation 仅可检查准备，不执行新规则。

授权后顺序唯一：source_select → freeze → evaluate（一次）→ verify / report / predict。优化器创建、Tensor.backward、autograd.backward/grad 在所有阶段硬禁。完成目录不覆盖，来源选择和评价不重试，不向其他聊天发消息或创建子代理。

## 固定候选与来源

唯一新候选 guard95，3seed 42/1019/2026，沿用任务12 `outputs/accuracy/microwave-source5-loss-preparation-20261025-v4/source_selection.json` 绑定的 unit_BCE 模型、选定epoch、权重与归一化。旧代码 gate.py 不修改，通过算法hash绑定原 select_threshold / local_max / gate_state。

两个来源块固定 House5 block4（2014-11-23至25 UTC）、block5（2014-11-25至27 UTC），完整aggregate推理p，原34351个C31有效评价中心/201阳性；时间、house、有效标签及原源CSV逐点一致。h对原始p一次中心max，radius15、8秒时间连续、有限score和块界分段；不迭代传播，不跨NaN、时间断点或块界。

L保持旧L95算法：从完整p有限精确值中选候选，每块R≥.95、每块非退化，按最差P、均P、均F1、较高阈值排序；重放必须得到完全相同原L。唯一新常数预先固定为H来源每块R≥.95（旧为.85），H≥L，精确h候选、相同排序。评价时 p<L 的有效点为off，阳性算FN，不能从分母删除；NaN保持missing。无source解即拒绝候选，不放宽、不追加家族，不搜索H90、半径、温度、min-run等。

准备阶段仅重放原L95/H85核验身份，不计算真实H95，不生成新开发候选，不打开House17输入或旧开发缓存，甚至不查询其shape。House17路径与hash仅由旧JSON封存身份引用，不对数值文件重新hash。来源p重放的数组hash封入 source_identity_replay，授权后必须完全重现。

## 原W50与来源预选规则

原W50为 precision-20261024-v1 原MAE（normalized_mae）选定epoch模型，固定200W。从 `predictions/W50/seed_*/source_validation_refit_5_0.npz` 仅显式读stamps/raw，无标签字段、无House17缓存。绑定历史baseline_identity内hash、原model/weights及MAE/50权重/200W元数据；stamps与House5完整aggregate相同，在同两块同34351个中心精确取raw[:,1]。缺字段、时间错位、缺有效预测直接拒绝，不换baseline，不拿W50_CAL阈值冒充原W50。

每seed先对两个块算P/R/F1再算块平均。guard95三个seed必须均有source95解且每块非退化；平均F1≥原W50，各seed平均R最多比W50降5pp，至少2seed平均P严格提高。合格候选与W50按三seed meanP、meanF1排序，完全同分W50优先。不合格即来源预选W50并记录拒绝理由。原H85仅诊断身份重放，不参加本轮候选竞争。这是相对原W50的来源保护，与任务12 parent-relative family guard不同，不能混称。

## 冻结、标签隔离与正式开发验收

来源模型/数据/权重/代码/原W50源身份/规则/source唯一预选先封存。授权后才读取开发aggregate和原A/W50冻结预测。完整冻结新候选、W50、A三seed共9份预测，候选与普通aggregate-only入口逐项精确读回；同一时间轴、原A所有raw功率列精确保持，其他电器状态保持原A。NaN支持与W50/A一致。没有任一种子H95解则保留来源W50 fallback并停止开发阶段，禁止用残缺候选矩阵评价或伪造阈值。若有全矩阵但来源相对保护失败，仍只评价预选W50，候选仅诊断，禁止开发结果后改选。

9份完整预测及读回证据封存后，才允许显式读取任务12已核验 `evaluation/truth_once.npz` 的t/y；hash由旧manifest绑定。绝不重新打开原House17标签CSV。evaluate仅一次；验证使用同一truth缓存复算。矩阵/封存不完整则在任何truth hash或打开之前拒绝。

正式门槛全部保留：meanP≥.08；各seedR≥.85；meanF1相对原W50增益≥.02（仅此门槛1e-12容差）；至少2seed P及F1同时提高；各seedF1下降≤.02；meanFP减少≥30%；无新增全开全关；每seed有效137523点、time/mask与原A和W50精确一致。开发不通过不改阈值、不追加候选。UI默认不切换，微波炉/水壶/洗衣机raw功率原A exact，其他电器状态原A exact。最终8/9/20所有数据保持隔离。

House17为反复使用工程开发集，不是盲测或独立新户泛化验证。source95不是跨户召回保证，不声称本轮实现预测改善，须等待授权后的固定评价。

## 可运行入口与交付

Python为 `.venv-m2/Scripts/python.exe`，入口 `scripts/run_microwave_recall_guard95.py`，prepare / verify_preparation 不需要执行授权；其余每次传 `--authorization` 本轮真实人类回执。普通predict只接受aggregate CSV、seed和新output，按来源预选输出guard95或固定200W W50 fallback，不能强选候选。源码快照、禁用config、空授权模板、输入与模型hash、来源对齐/旧阈值重放证据、prepare_access及注册hash放入新preparation目录。合成测试覆盖阶段授权、旧回执拒绝、标签/最终户隔离、95而非85传参、lowfail FN、NaN/块界、W50精确中心、相对来源保护、冻结矩阵和source→freeze→evaluate→verify→report→predict流程。合成授权均明确标注fixture，不能用于真实执行。
