# 数据集扩充与受控重训练

本轮按 2026-10-24 已批准计划执行。实验身份为 `data-expansion-20261024-v1`，来源、验证、开发及最终家庭沿用现有角色；不修改历史训练、既有评价结果或界面默认模型。

## 数据与纳入

原数据 A：REFIT 2/19 原 14 天、150,069 个有效窗口。B：同两户扩至 R0 已选定 28 天内的前 14 天与后 13 天 23 小时，保留中间一小时缺口。C：增加 10/12/15/16/18；使用 R0 已保存的首个覆盖合格区间，不按事件或效果另选日期。

每个家庭—设备—分区单独检查总表/设备覆盖至少 90%、有效观测至少七天、完整阈值事件至少两次。缺失通道不等于关闭。C 实际水壶仍为两户，微波炉和洗衣机各六户：House 12 水壶两段均因完整事件不足而排除，未改日期。所有纳入/拒绝原因保存于实验 `admission.json`。

新增规范化数据为 `data/processed/refit_expansion_additional_20261024_v1/`。首先从原始 CSV 保留所选日期前一观测和结束后的首个观测，再调用原有八秒时间加权重采样；原始完整文件哈希、获取回执、精确切片哈希均保留。没有重置观测年龄、跨缺口填充或更改 Issues 处理。用户要求暂停时中断的 `.prepare-*` 目录保留，未将其视作完成数据。

来源验证继续 REFIT 5 原 14 天；开发评价继续 REFIT 17 原未来分区。最终 8/9/20 未访问；光伏、历史开发、隔离待核家庭未重新分配。

## 训练与评价

沿用 R1 `single_mae` 三个独立网络，31 点上下文、8 通道、32 隐单元、Adam 学习率 0.001、batch 256、开机损失权重 50、固定原始功率阈值。三个种子为 42/1019/2026。

每设备每组固定 7,044 次更新，分 12 个检查点、每 587 步只在来源验证上选择 normalized MAE 最低模型。扩数组中的一次完整数据遍历不等于一个检查点；较短数据遍历结束后按同一确定性洗牌策略继续。不得将该比较描述为各组都训练 12 个完整数据轮次。

每组在所有已纳入来源标签至少一种有效的总表窗口并集上计算公共输入归一化；不使用验证或开发输入拟合归一化。各独立设备只用自身有效标签优化；组合模型逐设备记录训练家庭、窗口数、更新步数、检查点和成本。C 的水壶训练标签未增加新家庭，但公共归一化因其他设备新增来源而变化，不能将其差异描述为水壶新增家庭的收益。

A 组先与 R1 三种子历史权重及确定性探针核对；不一致即停止后续对照。全部 A/B/C 模型和预测封存后才读取开发标签统一评价。预测使用紧凑、可校验的 NPZ（时间、总表、原始/修正功率），保留全时间轴及空值；测试与普通预测逐设备数值对齐。

每个相邻比较预设主设备等权 F1 均值至少 +0.02、至少两种子为正、任一主设备均值下降不超过 0.02、无新增全开/全关。Precision、Recall、运行段 MAE、电量误差分别保存；全部结果与负结果报告，不调整预设门槛。

## UK-DALE

固定 2017-04 发布版，DOI `10.5286/UKERC.EDC.000004`，CC BY 4.0。官方当前数据目录已核实为 CEDA 下 `data/version_0/ukdale.zip`；旧作者链接会跳到上层目录，不应将 HTML 当作数据包。

固定 1/2 训练候选、5 外部测试候选。读取 `mains.dat` 的有功列和独立设备通道，拒绝视在功率、混合通道及 washer_dryer 冒充纯洗衣机。保留原始小数时间戳，以整数微秒计算原生 1 秒/6 秒支持，再时间加权到八秒完整区间。日期为各所需通道最早共同起点后的 28 天，不根据效果挑日期；覆盖不合格则记录失败。

外部测试 5 准备时只核查支持，不计算事件或活动摘要。D 以及既有 C 在该户的零样本预测全部冻结后才评价。下载或资格失败保存原因，D 不冒充已训练，不替换外部测试户，不改取其他数据集。

## 复跑入口

所有新实验与摘要使用新目录；已完成评价不得覆盖。

```powershell
.\.venv-m2\Scripts\python.exe scripts/run_data_expansion.py prepare-additional --additional data/processed/my-additional
.\.venv-m2\Scripts\python.exe scripts/run_data_expansion.py prepare --additional data/processed/my-additional --run outputs/accuracy/my-expansion
.\.venv-m2\Scripts\python.exe scripts/run_data_expansion.py train --run outputs/accuracy/my-expansion
.\.venv-m2\Scripts\python.exe scripts/run_data_expansion.py evaluate --run outputs/accuracy/my-expansion
.\.venv-m2\Scripts\python.exe scripts/run_data_expansion.py external --run outputs/accuracy/my-expansion
# 已有解压数据时可提供 --ukdale <包含 house_1/2/5 的目录>
.\.venv-m2\Scripts\python.exe scripts/run_data_expansion.py train-external --run outputs/accuracy/my-expansion
.\.venv-m2\Scripts\python.exe scripts/run_data_expansion.py evaluate-external --run outputs/accuracy/my-expansion
.\.venv-m2\Scripts\python.exe scripts/run_data_expansion.py verify --run outputs/accuracy/my-expansion
.\.venv-m2\Scripts\python.exe scripts/summarize_data_expansion.py --run outputs/accuracy/my-expansion --output outputs/accuracy/my-expansion-handoff
```

完整输入、训练源码与协议在运行目录中保存。继续被中断的训练时，只复用与冻结协议、设置、代码及权重校验一致的已完成组件，不覆盖历史运行。代码有变化应新建运行，不能绕过源码检查。

## 实际结果与交付

2026-10-24 完成 A/B/C：27 个独立设备网络、九个组合模型、18 份紧凑预测、108 行设备/视图指标。主运行 `outputs/accuracy/data-expansion-20261024-v1/`；易读报告 `outputs/accuracy/data-expansion-20261024-handoff-v1/report.html`；小型结果摘要见 `DATA_EXPANSION_RESULTS_20261024.json`。

| 组 | 水壶有效训练窗口/家庭 | 微波炉有效训练窗口/家庭 | 洗衣机有效训练窗口/家庭 | House 17 主设备平均 F1 |
|---|---:|---:|---:|---:|
| A 原两户14天 | 150,069 / 2 | 150,069 / 2 | 150,069 / 2 | 0.416211 |
| B 延长记录 | 426,204 / 2 | 426,204 / 2 | 426,204 / 2 | 0.409012 |
| C 增加家庭 | 426,204 / 2 | 1,340,422 / 6 | 1,301,717 / 6 | 0.393312 |

B−A 主设备平均 F1 **−0.007199**，仅 1/3 来源种子为正；C−B **−0.015699**，0/3 为正。两个比较均未达到预设内部目标。无新增全开/全关；该检查通过不能代替误报和绝对性能达标。

微波炉 F1 为 0.076913→0.073379→0.071486，Precision 为 0.040128→0.038168→0.037104，Recall 为 0.924282→0.955614→0.974761。增加数据后仍大量误报，未改善微波炉识别。微波炉电量绝对误差均值从 13.990 降到 12.613、12.322 kWh，但运行段 MAE 从 537.84 升到 607.75、673.75 W；不能写成全面功率改善。

结论限定为：**在固定现有训练配方与相同更新步数下，本轮扩数没有提升该开发家庭的识别 F1。** 没有据此证明更多数据普遍无效，也没有延长扩数组的训练预算、调整采样/损失/阈值或状态头来追求通过。后续模型或训练修正需另立实验。

UK-DALE 的导入、原生时间支持重采样及外部隔离入口已实现，但 **D 组尚未训练或评价**。当前官方完整包 3,585,155,959 字节；已验证分段下载 12/428 段，另有未完成片段保留。整包与分段获取经历超时，最后一次限时十分钟的续传达到期限，具体证据见 `data/raw/ukdale_2017_20261024/range_acquisition_failed.json`、主运行 `ukdale_status.json` 和 `tmp/data-expansion-ukdale-*.log`。未把 HTML、部分压缩包或未下载数据计入训练，也未更换测试户。下载进程已结束，不存在隐藏的持续下载任务。

网络恢复后可运行 `.\.venv-m2\Scripts\python.exe scripts/fetch_ukdale.py --max-seconds 600`，复用已校验分段和带哈希的未完成片段；完整归档核验、接入及资格检查成功后再执行 external/train-external/evaluate-external。下载失败不会伪造 D=C 对照；未完成接入时外部训练入口返回错误。

## 验证与边界

- 最新完整回归 **166 项全部通过，278.964 秒**；专项覆盖匹配更新步数、原训练逐参数复现、部分通道与模型组合、普通/紧凑预测一致性、小数时间戳、原生缺口与能量、缺种子冻结屏障及配对支持/效果门槛。
- A 的三个种子与历史 R1 权重逐参数完全一致，最大差值 0；归一化、物理功率缩放及状态视图也完全一致。
- 真实九个组合模型通过普通预测接口读回，与保存的紧凑原始/修正功率一致；验证没有重新打开标签。证据在主运行 `real_readback.json`。
- 模型、输入、原始/修正预测、协议、指标及训练源码均有校验清单，离线 verify 通过。历史权重、既有实验与最终家庭未改，默认界面模型未替换。
- 开发家庭只有一户，三种子不能当作三户泛化验收；没有最终 H1/H2/H3 或跨数据集效果结论。完整计划的外部获取与 D 组执行仍受上述数据获取阻塞。
