# ElectricityPlot — M3 交接记录

日期：2026-10-20。版本：0.4.0。结论：低交互适配代码闭环已经实现；真实数据效果、CNN 加载/训练路径和 Streamlit 运行验收仍待完成，不将合成测试视为算法成绩。

## 已实现

- 随机、仅不确定性、组合选片三种策略，使用完全相同的候选池、预算、功率骨干、适配算法和评价接口。
- 只从目标适配池总表及基线预测构建完整确认窗口。缺失、时间断裂或模型上下文不足不参与选片；同设备窗口不重叠。
- 会话保存待确认问题、历史、模型版本和预算；支持重新加载。每种设备的一次回答计 1 次，`unsure/skipped` 同样计数，重复/过期回答拒绝。提交采用文件锁和原子替换。
- 离线标注代理必须先看到持久化问题，只返回当前窗口的 `used/not_used/unsure`，来源标记为 `simulated_from_submeter`。标签缺失或覆盖不全不判为关闭。
- 冻结功率主干，在已确认窗口上训练每设备三个系数的状态头。使用窗口最大池化的 MIL 损失；“用过”表示至少一刻用过，不展开成逐点功率或全窗口开机。`unsure/skipped` 不进入损失。
- 自包含适配模型，记录基线指纹、查询快照及校验、目标家庭、种子、配置、预算/回答/来源计数、耗时、训练历史和环境。适配模型禁止用于其他家庭。
- CLI、人工确认界面、前后比较、JSON/CSV/HTML 导出及独立产物复核。变更模型、数据、预算、策略或种子清除旧结果；失败不展示旧适配结果。
- 批量运行先冻结所有家庭/策略/预算/种子的模型，再开启测试评价。若候选不足则失败，不以更少回答冒充相同预算。支持 `--no-test-evaluation` 完全不打开测试标签。

## 方法与边界

初始状态分数为 `sigmoid(log((raw_power_w + 1) / (threshold_w + 1)))`。窗口取最大分数，查询不确定性为 `1 - 2 * abs(score - 0.5)`；这是明确可复现的启发式，未经校准，不是可信概率。

组合分数为 `0.5 * uncertainty + 0.3 * normalized_estimated_energy + 0.2 * answerability_proxy`。电量影响来自预测功率；易回答性代理为 `1 / (1 + aggregate_event_count / 4)`，事件阈值 100 W。没有真人验证该代理，也没有证明三种策略中哪种更好。当前采用固定基线候选分数排序，每次回答后继续取未查询片段，不进行在线重训排序。

状态头输入是偏置、预测功率相对来源阈值的对数、总表相对来源阈值的对数。默认固定 200 次梯度更新、学习率 0.05、L2 0.01；配置在 `configs/m3_adaptation.json`。没有测试集早停、目标强标签归一化或隐蔽功率监督。

状态和功率分别评价：适配后的状态来自独立状态头，评价模块读取显式状态；功率 MAE、分项电量、修正幅度和未解释部分保持基线结果。没有可用回答或预算为 0 时，精确保留原始基线决策。未查询设备保留原始状态；缺失仍为 null；`confidence` 恒为 null。

## 运行

M3 不增加 M2 的训练依赖。CNN 和界面仍需要原 `requirements-m2-cpu.lock` 环境；来源基线、原始数据和权重不在 Git 中，不能只克隆源码就假定已有模型。仅 NumPy 环境可以运行常量基线的 M3 合成契约测试。

完整模拟比较（默认全部预算、全部三个种子、全部三个策略）：

```powershell
python scripts/run_m3.py --model outputs/my-m2/seq2point --output outputs/my-m3
python scripts/verify_m3.py --run outputs/my-m3
```

只冻结模型，不读测试标签：

```powershell
python scripts/run_m3.py --model outputs/my-m2/seq2point --budgets 0 5 --seeds 42 --no-test-evaluation --output outputs/my-m3-frozen
```

手动确认（将 `QUERY_ID` 替换为 `next-query` 实际返回值）：

```powershell
python scripts/project.py start-session --model outputs/my-m2/seq2point --manifest data/processed/refit_m1_protocol/protocol.json --partition target_adaptation_house_3 --budget 5 --strategy proposed --seed 42 --output outputs/my-session
python scripts/project.py next-query --session outputs/my-session
python scripts/project.py confirm-query --session outputs/my-session --query-id QUERY_ID --response used
python scripts/project.py adapt-model --session outputs/my-session --output outputs/my-adapted-model
```

每次 `next-query` 后回答一次；允许 `not_used/unsure/skipped`。模拟回答须明确调用 `simulate-confirmation --session outputs/my-session`，不会伪装为人工回答。若仍有待确认问题，先回答或跳过，再导出适配模型。再次适配选择新目录；从同一基线和查询历史重新拟合，而不累积重复训练。

界面：`python -m streamlit run scripts/app.py`，选择目标适配历史区间，新建/加载会话，逐条回答，预算用完后生成适配模型。冻结测试前后比较只用于评价，不能依据结果继续刷同一测试集调参。

## 验证与证据

本机：CPython 3.14.3、NumPy 2.5.2；与此前 M2 的 Python 3.13.5 锁定环境不同，不宣称重新验证了原训练环境。完整测试 68 项，其中 54 项通过、14 项因缺少 PyTorch/Streamlit 跳过。21 项新增 M3 与产物发布测试已执行；新增的界面交互测试包含在跳过项中。

覆盖预算/重放/过期回答/并发锁、三策略排序、候选缺失、模拟标注边界、MIL 最大池化、独立状态指标、模型/预测指纹、零预算基线一致性、失败记录、导出，以及所有模型冻结后才开始评价的顺序。测试还在禁止打开 oracle/test 标签文件的环境中执行选片与适配，检查隐藏标签指纹不参与随机排序。

最终合成验证运行位于 `outputs/m3-20261020-synthetic-verified/`，明确标记 `synthetic_contract_fixture`，包含单个合成目标家庭的 3 策略 × 3 种子 × 4 预算共 36 组，合计 315 次模拟回答。它证明运行与产物契约，不代表真实 REFIT 表现，也不证明真实用户可用性。每组有 `session/`、`model/`、`prediction/`、`evaluation/`，根目录有 `NOTICE.json` 和 `verification.json`。测试日志位于 `outputs/m3-validation-verified/`，可复核路径及摘要见 `docs/M3_VALIDATION.json`。

当前网络权限拒绝连接，依赖安装和 REFIT 下载均未成功；没有沿用不存在的旧权重，也没有伪造真实数据曲线。此前 M2 结果保持原样。Windows 批量运行中遇到过偶发目录 rename 拒绝访问；产物发布现统一为有上限的短暂重试，保留原子发布、禁止覆盖及永久失败抛错。相关失败记录保留；加入重试后的最终运行和测试均成功。

## 待完成的验收

1. 在具备 M2 依赖、REFIT 工程协议和 CNN 权重的机器上运行真实 M3 比较，保留正负结果；检查状态提升与功率不变分别报告。
2. 执行 Streamlit 交互测试，并人工核验重新加载、预算变化、失败清理和前后报告。
3. 完成正式多户协议、三目标家庭、第二台机器复跑和真实用户可回答性验证。可回答性代理与不确定性均不作为已验证可信度。
4. M4 的校准、未知背景压力实验与可信归因实验不属于本次实现。
