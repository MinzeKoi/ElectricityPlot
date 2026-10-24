# R4 界面统一与 R5 复跑交付

2026-10-24。按用户指示跳过 R3 正式评价；House 8/9/20 保持隔离。黄家豪为用户提供的操作者姓名，本交付不将其解释为未查看标签或独立复现声明。

R4 将界面从旧 M3 学习器接入 `accuracy.live`，复用 A3/R2 的候选池、待回答提交、预算校验、每五次来源初始化拟合、原子模型导出和回放验证。默认载入 R2 稳定性版本 `L1_stable`；同时支持原 A3 `L2_max` 及其匹配来源头/replay。现阶段均标记开发演示，不把 R2 未通过的候选晋升为正式方法。

支持新建、加载、跳过/不确定计费、重复提交拒绝和更新中断恢复。五次更新前的回答保留，但不提前拟合。前后状态比较只读适配总表；没有分表真值时不显示 F1。功率、电量、未解释负荷冻结。导出显示模型 ID、实际状态定义、策略和回答计数；状态分数不称为可靠概率。模拟记录沿用其模拟来源。

导入总表 CSV 后也可进入同一确认流程。导入绑定明确记录“来源权重未经重训转用于用户输入”，新协议只包含总表，不创建虚构分表；默认使用 L1_stable。L2_max 需要与当前来源模型身份一致的冻结来源头/replay。

## 运行

从项目根目录执行 `start.cmd`，选“已有家庭记录”的 House 17 待适配历史区间。高级模型默认是冻结来源 seed 42，窗口及学习器配置保持 R2 稳定性版本。选预算和策略后新建会话，或用同一设置加载原会话目录。提交回答后每五次自动更新；使用“查看当前适配前后状态并导出”查看曲线并下载报告、CSV、JSON。

```powershell
.\.venv-m4\Scripts\python.exe scripts/run_r4_release.py --output outputs/r4-r5-20261024-v1
.\.venv-m4\Scripts\python.exe outputs/r4-r5-20261024-v1/bundle/scripts/replay_release.py --output outputs/r4-r5-replay-20261024-v1
```

首条命令使用既有 House 17/source 42/query 2026/proposed/10 次冻结记录，逐问核对模型版本、回答及最终适配器，禁止访问目标分表文件。不会使用新的真值或重新评价历史成绩。输出已存在时拒绝覆盖。

本次最终源码交付单独封存在 `outputs/r4-r5-delivery-20261024-v1/`；初版 `outputs/r4-r5-20261024-v1/` 保留原始逐问对照证据。由已完成会话重新封装的命令为：

```powershell
.\.venv-m4\Scripts\python.exe scripts/package_release.py --session outputs/r4-r5-20261024-v1/live --output outputs/r4-r5-delivery-20261024-v1 --historical-run outputs/accuracy/r2-stability-20261023-v1
.\.venv-m4\Scripts\python.exe outputs/r4-r5-delivery-20261024-v1/scripts/replay_release.py --output outputs/r4-r5-replay-final-20261024-v1
```

交付包 `bundle/` 含完整源码、实际来源权重和模型清单、协议与唯一适配总表、冻结配置、依赖锁、确认记录、适配器期望、前后预测/HTML/CSV、文件哈希、工作树 diff、文献及许可入口、历史 R2 正负结果。包内不含目标分表、最终家庭文件或第二套虚构指标。协议保留原始身份；不在包内落盘其不可见标签。

复跑必须写到封存包以外的新目录。可以把包复制到另一目录或机器，创建 Python 3.11+ 环境并按 `requirements-m4-cpu.lock` 安装依赖，再运行 `python scripts/replay_release.py --output ../replay-result`。它校验完整文件清单，重新选择每个问题、适配并核对适配器/模型 ID、逐行预测签名、CSV 和重建 HTML，生成 `receipt.json`。在包目录运行 `python -m streamlit run scripts/app.py` 可打开应用；加载 `../replay-result/session` 时选择与包内记录相同的预算、策略、种子。

## 验收边界

R4 的工程验收以 `docs/R4_R5_VALIDATION.json` 和实际测试日志为准。测试包含两种学习器、无目标标签访问、同输入拟合一致、五次更新边界、中断后恢复、重复回答拒绝、输入损坏拒绝、迁移后的包复跑，以及界面无数据/模型失效路径。

R5 的自动化交付与搬迁复跑可以由当前代理完成。另一位成员独立环境的复跑、第二台机器、完整来源训练复跑、真实用户操作试用分别记录；没有实际人员/环境证据时保持待验收，不能用代理本机回放替代独立成员复现。当前没有真实用户确认效果、跨户最终 H1/H2/H3 或实际节电结论。REFIT 数据署名/许可说明见 `REFERENCES.md`，对外发布前按原数据许可保留必要署名。

## 2026-10-24 实际验收结果

- **R3：按用户指示跳过**，未启动正式评价，8/9/20 继续隔离。
- **R4：工程完成**。五项专项及最终全量 155 项回归全部通过（385.802 秒）；旧基线预测/评价、区间展示和失效清理回归保持通过。首轮 154 项的两项失败/五项错误及修复过程保留于原日志，未将首轮记为全通过。
- **真实开发轨迹对照通过**：House 17/source 42/query 2026/proposed/10 次，十个问题的窗口、设备、模型版本及回答逐项一致，最终适配器与模型 ID `a3-state-ffd92f82ea64e97d` 与既有冻结实验一致。模拟标签身份保留，核验禁止访问目标分表。
- **R5：工程交付及搬迁回放完成，独立成员验收待做**。最终包 191 个文件，ZIP 37,170,326 字节（约 35.5 MiB）。解压到另一目录、从 `tmp` 工作目录以 Python `-I` 启动包内脚本，十次确认重放通过；前后逐行预测签名、CSV 和重建 HTML 均匹配。仍为同机同 Python 环境自动化复跑。

证据：`docs/R4_R5_VALIDATION.json`；最终回归日志 `outputs/r4-r5-regression-final-20261024.log`；真实轨迹对照 `outputs/r4-r5-20261024-v1/experiment_parity.json`；搬迁复跑回执 `outputs/r4-r5-replay-final-20261024-v1/receipt.json`。已验证封存的全部 Python 源码与最终测试源码一致。封存包不因后续追加本交接的结果段落而改写；最终验收记录作为独立侧附证据引用其 manifest/ZIP 哈希。

交付包：`outputs/r4-r5-delivery-20261024-v1.zip`。另一位成员实际复跑后，可填写 `docs/R5_MEMBER_REPRODUCTION_TEMPLATE.json` 并附新环境生成的 `receipt.json`，以完成 R5 的人员验收。
