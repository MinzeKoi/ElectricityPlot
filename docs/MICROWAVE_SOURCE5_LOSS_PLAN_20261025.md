# Task12：House5 单变量损失对照（2026-10-25）

初始阶段仅获实施准备许可，任务10原3fit已用满。准备期间真实新拟合0、真实更新0；prepare禁止构造CenterMaskTCN、优化器和backward。随后根对话人类明确回答“允许追加最多3个模型”，授权本固定试验及门控、封存、一次开发验收和复核。配置仍保留 `authorized=false`，必须用独立回执绑定本run实际registration哈希后方可训练。仅额外3新拟合、10584更新；不得复用旧授权、重命名规避、自动追加或失败重试。授权来源：根对话01a16d7b-478e-7780-a999-88abe71e8c94，经协调者传递的实际本轮人类答复。

## 唯一训练变量与数据身份

唯一变化为 BCE `pos_weight:50→1`。直接引用任务10 v2 冻结 train.npz 的全部 336704 窗口 / 2275 正中心，不重新抽样、不重建窗口。House2/19 原14天和 House5 最早8完整 UTC 日期保持原边界。House5 只允许原训练四块：11月15–17、17–19、19–21、21–22日23:30（2014 UTC），保留1800秒隔离；验证只允许11月23–25和25–27两块，共34351点、201正中心。验证 x/mask/t/y/索引逐字段绑定，无新增日期、无默认 disjoint 保护改动。缺失不填补。

模型固定 CenterMaskTCN127（5329参数），原 House2/19 log1p 正常化常数原字节复用。seed 42/1019/2026 的初始权重文件与任务10完全相同；读取初始 state_dict 对照任务10实际 initial.pt，不在 prepare 构造网络。event 队列直接引用任务10 indices；prepare 重放全部每 seed 3528 个索引，逐项 exact，每批256（16正、240负）。Adam lr=.001、4 CPU threads，每 seed 12次检查×294更新=3528；总10584更新、2709504窗口呈现。不得 importance、focal、增强、新网络、采样搜索或旧控制重训。

## 来源选择与固定 2×2

四家族固定优先序：old_point、old_gate、new_point、new_gate；旧两家族资格来自任务10 source_selection/source_rows 和任务11 v4 source_selection/recipe。只读取来源证据、源配置/代码/输入及封存身份，不读取旧开发指标、标签、预测缓存或其 shape。本阶段也不读取开发 aggregate 或预测。

新 point 完全调用原 reselect_checkpoints/choose：12个 epoch，每个固定两验证块 R≥.85、非退化，原最差P、平均P、F1、阈值和epoch排序保持原实现。无可行 epoch 保留原诊断 fallback，但该家族不合格。

新 gate 在已唯一选择的 point epoch 上，完全复用任务11 local_max/select_threshold/gate_state：L 每块 R≥.95，然后 H 每块 R≥.85；候选阈值和排序同任务11。radius固定15，一次对原 p 做中心 max。连续段仅由有限 p、8秒连续时间及块边界决定，与标签可用性无关；有效 p<L 的点仍为 off，可计 FN。无可行 L/H 时该 seed gate 不可用，整个新 gate 家族不合格，保留 point；不降低95门槛、不搜索半径/minrun/新规则。

新 point 相对旧 point、新 gate 相对旧 gate，各须三 seed 非退化且可行、meanF1不低、每 seed 来源平均R下降≤.05、至少2 seed P更高。不合格的新家族不参与选择。合格者与旧两家族按 meanP、meanF1 降序，完全相同按上列固定优先序。来源资格和选择在开发标签前封存。

## 封存、一次开发评价和普通推理

追加授权后 train → source_select → freeze → evaluate → verify → report，全部采用本 run 注册的源码快照。freeze 在打开任何开发 truth 前生成所有可用候选三 seed 预测，以及原A/W50控制；逐个运行完整 aggregate-only 普通推理，要求时间/分数/状态 exact。旧 point/gate 要求与已有冻结预测 exact，不重校准。所有候选原 raw 矩阵取自原 A，因此微波炉功率及其他电器功率预测逐列保持不变。原 A/W50 有限支持对齐，缺失状态为 -1/空值。

全部矩阵、来源选择、身份和哈希冻结后，才允许一次读取原 House17 标签，缓存用于复核。使用 W50 原有效137523点，和原A及候选支持严格一致。只给来源预选方法正式验收，其他混淆矩阵仅诊断，不按开发改选。

正式门槛固定：meanP≥.08；每seedR≥.85；meanF1比原W50至少+.02（原1e-12比较容差）；至少2seedP/F1同升；每seedF1下降≤.02；meanFP下降≥30%；无新增全开全关；原W50和原A的137523有效时间完全一致。House17为反复使用开发集，不声称独立泛化。最终 House8/9/20 始终隔离，默认UI不切换，不作功率改善声明。

verify 复算 epoch/阈值、权重、实际采样批次、曝光与更新数、封存绑定及普通预测，并以独立标量循环复核混淆计数。普通 predict 入口仅接受 aggregate；未合格家族拒绝加载。该入口输出微波炉状态/分数，完整功率矩阵的原A不变由 freeze 证据约束。

## 授权门和待执行命令

准备：`.venv-m2\Scripts\python.exe scripts/run_microwave_source5_loss.py prepare --run outputs/accuracy/microwave-source5-loss-preparation-20261025-v4`

合成测试：`.venv-m2\Scripts\python.exe -m unittest discover -s tests -p test_microwave_source5_loss.py -v`

追加授权回执必须提供 `authorized=true, approval_origin=human, experiment=microwave_source5_loss_v1, additional_new_fits=3, maximum_updates=10584, seeds=[42,1019,2026], registration_sha256=<本run注册哈希>, human_authorization_quote=允许追加最多3个模型`。原配置仍为默认 disabled；回执是本次特定注册的独立追加许可。缺回执或错误预算/绑定一律拒绝。授权前不创建回执；本轮已收到实际人类追加授权，可在最终核验后创建。

授权后使用准备目录中的快照入口（`$run` 代表上述准备目录，`$receipt` 代表尚不存在的实际人类回执）：

```powershell
.venv-m2\Scripts\python.exe "$run/run_microwave_source5_loss.py" train --run "$run" --authorization "$receipt"
.venv-m2\Scripts\python.exe "$run/run_microwave_source5_loss.py" source_select --run "$run"
.venv-m2\Scripts\python.exe "$run/run_microwave_source5_loss.py" freeze --run "$run"
.venv-m2\Scripts\python.exe "$run/run_microwave_source5_loss.py" evaluate --run "$run"
.venv-m2\Scripts\python.exe "$run/run_microwave_source5_loss.py" verify --run "$run"
.venv-m2\Scripts\python.exe "$run/run_microwave_source5_loss.py" report --run "$run"
.venv-m2\Scripts\python.exe "$run/run_microwave_source5_loss.py" predict --run "$run" --family old_gate --seed 42 --aggregate <aggregate.csv> --output <new-output.csv>
```

计算量：最多3次新拟合、每次3528次更新，三次共10584更新；每 seed 903168训练窗口呈现，12×34351验证打分，三 seed 共1236636验证打分。每次保留12个检查点（共36），0额外拟合用于门控。CPU时间依赖硬件，可参考任务10每seed `complete.json` 的训练用时（仅来源训练记录），不作固定运行时承诺。
