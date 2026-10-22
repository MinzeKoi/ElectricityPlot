# ElectricityPlot

低交互可信虚拟分表系统

Windows 可双击根目录的 `start.cmd` 启动界面，详见 [CMD 启动与测试流程](docs/CMD启动与测试.md)。报告查看和 M5 复跑命令见 [运行方式](运行方式.md)。

从家庭总功率曲线估计目标电器的功率与电量，通过少量用户确认适配新家庭，并保留无法可靠解释的耗电。

当前已完成 **M5 冻结实验工程与本机验证**：B0–P2、三种子/四预算、冻结后评价、报告导出及复跑核对已实现。真实 REFIT 前缀的单目标家庭 42 个案例、合成三目标家庭 126 个案例均通过复核，完整环境 91 项测试通过。当前结果未证明拟议策略稳定优于基线，背景误归因与区间覆盖不足仍然存在。正式实测多户盲测和独立成员/双机复跑仍待完成，证据见 [M5 交接](docs/M5_HANDOFF.md)。

## 新增数据（2026-10-22）

已取得 REFIT 20 户完整记录、Plegma 13 户完整清洗数据及 IDEAL 说明/元数据；REFIT 八户各最早 28 天已规范化。新增候选区间尚未满足 A5 覆盖门槛，Plegma 尚未接入八秒模型。实际目录、哈希、覆盖检查、复跑方式与限制见 [数据补齐交接](docs/DATA_EXPANSION_HANDOFF_20261022.md)，不代表已完成新数据训练或最终多户测试。

## 第一版范围

- 输入：无光伏、无储能家庭的带时间戳总有功功率，单位 W。
- 暂定目标：水壶、微波炉、洗衣机；最终可用设备和家庭在数据审计后冻结。
- 输出目标：电器估计功率/电量、运行状态、未解释负荷、可信程度和少量确认问题。
- 用户确认是“这段时间是否使用过该电器”的弱标签，不等于获得逐点功率答案。
- 未解释负荷可能包含未建模设备、测量误差与模型误差，不直接命名为某台未知设备。

## 开始使用

要求 Python 3.11 或更新版本；本机实测为 Python 3.13.5。M1 只使用标准库，无第三方运行依赖。

```powershell
python scripts/project.py status
python scripts/project.py validate-config
python scripts/project.py env-report --output outputs/my_machine.json
python scripts/project.py status --json
python -m unittest discover -s tests -v
```

上述 M1 命令只依赖标准库；完整 M2 测试需要下述环境，否则数值训练测试会明确跳过。安装项目后可使用 `electricityplot` 命令；Python 包目录暂保留 `virtual_submeter`，兼容原入口。

## M1 真实数据闭环

在新检出中运行下列命令。下载约 10 MiB，为真实文件前缀，**不是完整数据集**；已有产物不会被覆盖，重复运行需使用新的输出目录。

```powershell
python scripts/project.py download-refit --output data/raw/refit_m1
python scripts/project.py prepare-refit --input-dir data/raw/refit_m1 --output data/processed/refit_m1
python scripts/project.py audit-data --manifest data/processed/refit_m1/manifest.json
python scripts/project.py freeze-protocol --prepared data/processed/refit_m1 --output data/processed/refit_m1_protocol
python scripts/verify_m1.py --prepared data/processed/refit_m1 --protocol data/processed/refit_m1_protocol
```

打开 `data/processed/refit_m1/quality.html` 查看离线质量视图。真实文件、派生数据和协议均仅保存在本机，不进入 Git。来源、许可、实际通道、覆盖率、局限和下一步见 [M1 交接记录](docs/M1_HANDOFF.md)。当前划分明确标为 `engineering_pilot_only`，不用于正式效果声明。

用户总表 CSV 的最小字段为 `household_id,timestamp,aggregate_power_w`，时间必须带时区，空功率表示缺失，必须显式声明单位 W。合成契约样例可这样检查：

```powershell
python scripts/project.py prepare-aggregate --input tests/fixtures/synthetic_aggregate.csv --household-id demo_home --unit W --output data/processed/synthetic_demo
```

不要将月度账单、累计 kWh、光伏或储能数据作为此入口的总有功功率输入。规范化数据和标签访问接口见 [数据契约](docs/DATA_CONTRACT.md)。

## M2 训练、预测与界面

本机已在 `.venv-m2` 验证 CPython 3.13.5、NumPy 2.3.4、PyTorch 2.9.0+cpu、Streamlit 1.50.0。完整依赖版本见 `requirements-m2-cpu.lock`，第二台机器与 GPU 尚未验证。在新环境中安装：

```powershell
python -m venv .venv-m2
.\.venv-m2\Scripts\python.exe -m pip install -r requirements-m2-cpu.lock
```

启动本地界面（只监听本机）：

```powershell
.\.venv-m2\Scripts\python.exe -m streamlit run scripts/app.py
```

访问 `http://127.0.0.1:8501`，可选择已有数据分区或上传总表 CSV，查看历史曲线，下载 HTML 报告和预测 CSV。已有模型默认在 `outputs/m2-20261020/seq2point`；换机器需先训练，不能依赖未提交的本机权重。

一次复跑单水壶、三设备 CNN、来源均值和零预测四个基线；输出目录必须是新目录：

```powershell
.\.venv-m2\Scripts\python.exe scripts/run_m2.py --output outputs/my-m2
.\.venv-m2\Scripts\python.exe scripts/verify_m2.py --run outputs/my-m2 --manifest data/processed/refit_m1_protocol/protocol.json
.\.venv-m2\Scripts\python.exe -m unittest discover -s tests -v
```

也可单独调用各阶段：

```powershell
.\.venv-m2\Scripts\python.exe scripts/project.py train-baseline --manifest data/processed/refit_m1_protocol/protocol.json --output outputs/my-model
.\.venv-m2\Scripts\python.exe scripts/project.py predict --model outputs/my-model --manifest data/processed/refit_m1_protocol/protocol.json --partition frozen_test_house_3 --output outputs/my-prediction
.\.venv-m2\Scripts\python.exe scripts/project.py evaluate --prediction outputs/my-prediction --manifest data/processed/refit_m1_protocol/protocol.json --output outputs/my-evaluation
```

`train-baseline --appliances kettle` 可跑单设备；`--model-kind constant_mean` / `zero` 可跑简单对照。`predict --input <规范化总表.csv>` 支持无分表答案的新家庭；原始用户 CSV 先通过 `prepare-aggregate` 导入。配置固定在 `configs/m2_baseline.json`，训练和早停选择均不读取目标家庭答案。

交接与实际指标见 [M2_HANDOFF.md](docs/M2_HANDOFF.md) 和 [M2_RESULTS.json](docs/M2_RESULTS.json)。缺失区间不补零，可信度返回缺失；电量指标仅在共同有效区间积分。初始模型误报较多，状态 F1、全时段 MAE、运行片段 MAE 和覆盖率需一并阅读。

## M3 确认与适配

先准备 M1 协议和 M2 基线模型；M3 沿用 M2 环境，不新增训练依赖。窗口回答仅训练独立状态头，功率、电量和未解释负荷保持基线输出。默认预算为 0/5/10/20，比较随机、仅不确定性、组合选片三个策略及三个冻结种子。

批量模拟确认会明确记录 `simulated_from_submeter`，且先冻结全部模型再读取测试答案：

```powershell
python scripts/run_m3.py --model outputs/my-m2/seq2point --output outputs/my-m3
python scripts/verify_m3.py --run outputs/my-m3
```

人工确认支持断点续答：

```powershell
python scripts/project.py start-session --model outputs/my-m2/seq2point --manifest data/processed/refit_m1_protocol/protocol.json --partition target_adaptation_house_3 --budget 5 --strategy proposed --seed 42 --output outputs/my-session
python scripts/project.py next-query --session outputs/my-session
python scripts/project.py confirm-query --session outputs/my-session --query-id QUERY_ID --response used
python scripts/project.py adapt-model --session outputs/my-session --output outputs/my-adapted-model
```

将 `QUERY_ID` 替换为实际返回值。回答可以是 `used/not_used/unsure/skipped`，每个回答都消耗一次预算。没有待回答问题时可生成新模型版本，已有模型不覆盖。界面选择目标适配历史区间后可新建/加载会话、回答问题、生成适配模型并比较前后结果。完整说明与未验证项见 [M3_HANDOFF.md](docs/M3_HANDOFF.md)。

## M4 可信输出与背景压力测试

本次独立 CPU 验收环境为 CPython 3.12.14、NumPy 2.3.4、PyTorch 2.9.0+cpu、Streamlit 1.50.0，完整版本见 `requirements-m4-cpu.lock`。在 Python 3.12 的新虚拟环境安装该锁文件可复跑；它与 M2 的 Python 3.13 环境分别记录，不宣称第二台机器已验证。

M4 分别保留设备原始功率、向下修正功率、修正幅度和未解释部分；不会摊满总表。可选区间使用来源验证残差的经验分位数，原始和修正输出分别估计。来源验证也参与过模型选择，因此区间没有跨家庭覆盖保证，`confidence` 仍为 null。设置见 `configs/m4_trust.json`。

已有真实协议和基线后，一次运行来源校准、固定背景注入、成对预测、评价和产物复核：

```powershell
python scripts/run_m4.py --model outputs/my-m2/seq2point --manifest data/processed/refit_m1_protocol/protocol.json --partition frozen_test_house_3 --output outputs/my-m4
python scripts/verify_m4.py --run outputs/my-m4
```

可加 `--prediction-model outputs/my-adapted-model`，仅允许与基线功率主干相同的 M3 状态适配模型复用区间。报告在 `outputs/my-m4/stress/report.html`，包含有/无背景事件的误差、错误激活、原始超限率、修正幅度、归因比例和“全部留作未解释”对照。注入是明确标记的合成敏感性实验，不是现场未知设备识别；两个预测保存后才打开测试分表。

只需标准库和 NumPy 的独立合成演示（不下载 REFIT，不依赖 PyTorch/Streamlit）：

```powershell
python scripts/run_m4_synthetic.py --output outputs/my-m4-demo
python scripts/verify_m4.py --run outputs/my-m4-demo
```

逐阶段命令如下；在 Streamlit 的高级设置中填写校准目录，也可显示和下载相同区间：

```powershell
python scripts/project.py calibrate-output --model outputs/my-model --manifest data/processed/refit_m1_protocol/protocol.json --output outputs/my-calibration
python scripts/project.py predict --model outputs/my-model --manifest data/processed/refit_m1_protocol/protocol.json --partition frozen_test_house_3 --calibration outputs/my-calibration --output outputs/my-trust-prediction
python scripts/project.py evaluate --prediction outputs/my-trust-prediction --manifest data/processed/refit_m1_protocol/protocol.json --output outputs/my-trust-evaluation
```

所有输出目录都必须为新目录。失败不发布成功结果；切换或损坏模型、输入、校准与导出文件时清除/拒绝旧曲线。详细验收证据和待完成项见 [M4_HANDOFF.md](docs/M4_HANDOFF.md)。

本次真实工程报告位于 `outputs/m4-20261020-real-verified/stress/report.html`，带 5 次模拟确认的模型报告位于 `outputs/m4-20261020-real-adapted-verified/stress/report.html`。真实总表上的背景脉冲仍是人工注入，不冒充现场未知设备。可提交的指标摘要见 [M4_RESULTS.json](docs/M4_RESULTS.json)。界面时间轴与提示均使用 UTC，缺失功率线段断开。

## M5 冻结实验与复跑

M5 实现了 B0/B1/B2/B3/P1/P2、每个种子独立来源训练、相同预算弱标签适配、全矩阵冻结后评价、来源区间与背景压力对照，以及可复核 JSON/CSV/HTML 汇总。详细组定义、冻结边界、复跑方式和正式验收缺项见 [M5_HANDOFF.md](docs/M5_HANDOFF.md)。

```powershell
python scripts/run_m5.py run --manifest data/processed/refit_m4_protocol/protocol.json --output outputs/my-m5
python scripts/run_m5.py verify --run outputs/my-m5
```

可用 `freeze` 和 `evaluate` 分两步运行，或用 `compare --left ... --right ...` 检查两次独立运行的数值差异。完整三目标家庭合成流程：

```powershell
python scripts/run_m5_synthetic.py --output outputs/my-m5-synthetic
```

默认配置为 **pilot**：现有 House 3 前缀测试结果已在 M2/M4 查看过，不能作为新的最终盲测；合成数据也不能作为真实多户成绩。正式模式需要完整实测协议、至少三目标家庭和三个种子、全部预算及未查看测试声明。独立成员复跑仍需另外提供证据。

## A0 开发退化诊断

2026-10-21 已完成本机 A0：24 个真实前缀开发案例、450 行机制探针、98 项回归测试。单个负回答足以触发旧洗衣机状态头全关；查询预算集中和状态阈值口径切换也有证据。当前没有识别提升或最终多户结论。

使用已有 M2 模型和已查看的工程协议，新建开发副本，复跑随机/旧组合选片的三查询种子与 0/5/10/20 户级预算。全部模型和未来预测冻结后，才分析开发分表；不会修改历史协议或学习器。查询种子不代表重新训练基础模型。

```powershell
.\.venv-m2\Scripts\python.exe scripts/run_a0.py run --output outputs/accuracy/my-a0
.\.venv-m2\Scripts\python.exe scripts/run_a0.py verify --run outputs/accuracy/my-a0
.\.venv-m2\Scripts\python.exe scripts/run_a0.py check-regression --run outputs/accuracy/my-a0
.\.venv-m2\Scripts\python.exe scripts/probe_a0.py --run outputs/accuracy/my-a0 --output outputs/accuracy/my-a0-probes
.\.venv-m2\Scripts\python.exe scripts/probe_a0.py --verify --run outputs/accuracy/my-a0-probes
```

`check-regression` 在存在 F1 退化时返回 1，无退化时返回 0，产物无效时返回 2；它是诊断信号，不是工程测试失败。报告为输出目录的 `report.html`，逐案例指标、回答分布、参数和训练历史见 `diagnosis.json` 与 CSV；数据和环境见 `availability.json`。只在开发数据上解释退化，不声明识别提升或最终多户验收。交接见 [A0_HANDOFF.md](docs/A0_HANDOFF.md)。

## A1 来源基础模型

2026-10-21 已冻结 `active50` 的三来源训练种子权重，主设备最低事件检测门槛通过，103 项工程测试通过。开发逐点 F1 仍低于历史模型，微波炉误报多、洗衣机近全开；没有识别精度提升结论。实际冻结设置为 `configs/a1_selected.json`，本机易读报告在 `outputs/accuracy/a1-20261021-handoff/report.html`。

A1 将来源训练与来源验证分别扩展到最早 14 天；House 3 开发记录按训练前的数据质量审计固定为最早 4 天。House 11 未达到主设备完整阈值事件门槛，明确排除。三组预先限定设置先跑单水壶、再跑三设备；配置选定后训练其余来源随机种子，后续确认策略复用冻结权重。

```powershell
.\.venv-m2\Scripts\python.exe scripts/project.py download-refit --households 2 5 11 --sample-bytes 16777216 --output data/raw/my_a1_sources
.\.venv-m2\Scripts\python.exe scripts/project.py download-refit --households 3 --sample-bytes 16777216 --output data/raw/my_a1_development
.\.venv-m2\Scripts\python.exe scripts/project.py prepare-refit --input-dir data/raw/my_a1_sources --households 2 5 11 --output data/processed/my_a1_sources
.\.venv-m2\Scripts\python.exe scripts/project.py prepare-refit --input-dir data/raw/my_a1_development --households 3 --output data/processed/my_a1_development
.\.venv-m2\Scripts\python.exe scripts/run_a1.py run --sources data/processed/my_a1_sources --development data/processed/my_a1_development --historical-model outputs/my-m2/seq2point --output outputs/accuracy/my-a1
.\.venv-m2\Scripts\python.exe scripts/run_a1.py verify --run outputs/accuracy/my-a1
.\.venv-m2\Scripts\python.exe scripts/summarize_a1.py --run outputs/accuracy/my-a1 --output outputs/accuracy/my-a1-handoff
```

历史模型可按上文 M2 流程准备；`--historical-model` 必须指向实际已有三设备模型。也可先运行 `prepare` 检查质量，再用 `train --run <目录>` 启动比较。已有目录不能覆盖；失败准备会保留原因，不能直接重复训练。配置见 `configs/a1_baseline.json`，事件支持、比较表与阶段结果见 [A1 交接](docs/A1_HANDOFF.md)。这仍是前缀开发实验，不是最终多户评价或确认收益证据。

## A2 固定确认学习器比较

A2 实现固定随机确认集合下的 L0（旧 MIL）、L1（有界偏置校准）和 L2（冻结 CNN 时序特征的小状态头）比较。L2 同时保留最大／平滑正窗口聚合、限幅、来源回放与参数正则的逐步消融。所有学习器共用相同确认前缀，各自保留零确认基线；三个来源训练种子复用 A1 冻结权重。功率、电量与未解释部分保持不变，工程验证不等于确认收益达标。

2026-10-21 已跑完 252 个真实前缀开发案例，111 项测试通过。主预算 10 次下 L2_max 的主设备 F1 从 0.1757 升至 0.1838，约提高 0.81 个百分点，无确认新导致的全关／全开；+5 个百分点内部目标未达到。三条随机轨迹均未取得水壶正回答，仅一个已查看开发家庭，不是最终跨户结论。本机报告为 `outputs/accuracy/a2-20261021/report.html`，可提交摘要为 `docs/A2_RESULTS.json`。

使用已有 A1 主运行，新建输出目录：

```powershell
.\.venv-m2\Scripts\python.exe scripts/run_a2.py run --a1-run outputs/accuracy/a1-20261021-v3 --output outputs/accuracy/my-a2
.\.venv-m2\Scripts\python.exe scripts/run_a2.py verify --run outputs/accuracy/my-a2
```

也可用 `freeze --output ...` 与 `evaluate --run ...` 分开执行：全矩阵模型及未来预测先冻结，再打开开发未来分表。默认三来源种子 × 三查询种子 × 七学习器／消融 × 四预算，共 252 案例；0/5/10/20 次是每户三设备共享预算的嵌套前缀，跳过与不确定也计费。来源种子和查询种子不代表独立家庭。

`report.html` 为离线比较报告，`comparison.csv` 保留逐设备／逐种子配对指标；`frozen/` 保存配置、来源头与回放缓存、查询轨迹、可独立调用通用预测入口的模型和紧凑状态数组。学习器无有效回答时严格返回自身来源输出，只有单类回答时使用预先固定的保守偏置规则；旧 L0 和无保护消融保留退化参照。配置与阶段出口见 [A2_HANDOFF.md](docs/A2_HANDOFF.md)。这仅是已查看 House 3 的开发实验，不是最终多户效果或 A3 选片收益。

## A3 动态选片与确认反馈

A3 固定 A2 的 `L2_max` 与来源模型，每 5 次回答从来源重新适配并刷新状态评分。比较随机、不确定性、拟议策略，另保留静态评分、去多样性和去设备覆盖消融。0/5/10/20 是同一条 20 次轨迹的前缀，三设备共享预算；不确定和跳过同样计费。新版本会话支持 pending 幂等、逐轮模型重放和冻结阶段中断恢复。实现与结论边界见 [A3 交接](docs/A3_HANDOFF.md)。

```powershell
.\.venv-m2\Scripts\python.exe scripts/run_a3.py run --output outputs/accuracy/my-a3
.\.venv-m2\Scripts\python.exe scripts/run_a3.py verify --run outputs/accuracy/my-a3
.\.venv-m2\Scripts\python.exe scripts/verify_a3.py --run outputs/accuracy/my-a3
```

可分开使用 `freeze --output ...` 和 `evaluate --run ...`；冻结阶段中断用相同输入执行 `resume --output ...`，完成后再评价。默认三来源种子 × 三查询种子 × 七策略／消融 × 四预算，共 252 案例，全部冻结后才打开开发未来分表。

2026-10-21 已完成 252 案例、756 行设备指标及 114 项测试。主预算 10 次，拟议 F1 为 0.1810，随机为 0.1778，额外增益约 0.32 个百分点；最低开发出口通过，+2 个百分点内部目标未达到。静态不确定性达到 0.1905，动态刷新与完整多样性／覆盖组合没有优势。只有一个已查看开发家庭，不能作为最终跨户 H2。本机完整报告为 `outputs/accuracy/a3-20261021-v3/report.html`，可提交证据见 `docs/A3_RESULTS.json` / `docs/A3_VALIDATION.json`。

本机受限沙箱无法访问 A2 的 42 个已发布派生文件；正常本地权限下 A2 已完整核验通过。A3 主运行沿用完整核验权限并保存来源收据，在新目录生成总表预测。若在受限视图运行，收据明确标记仅核验可访问核心证据；缺失核心证据或保留文件哈希变化均拒绝，不把“不可见”当作文件已删除。

## A4 功率与电量联动验证

A4 在冻结 CNN 的时序特征上新增只用来源开机功率监督的幅度分支，以状态分数与幅度相乘得到功率；目标弱确认仍只更新原有受限状态头。保留联动结构自身零确认基线及同确认的纯状态对照，重新记录修正前后功率、未解释部分与来源残差区间。旧 M4 或其他状态参数版本的区间不能直接复用。A3 状态证据尚弱，本轮为按用户指示启动的开发探索，工程完成不代表 H3 达标。

```powershell
.\.venv-m2\Scripts\python.exe scripts/run_a4.py run --output outputs/accuracy/my-a4
.\.venv-m2\Scripts\python.exe scripts/run_a4.py verify --run outputs/accuracy/my-a4
.\.venv-m2\Scripts\python.exe scripts/verify_a4.py --run outputs/accuracy/my-a4 --output outputs/accuracy/my-a4-validation
```

可分开执行 `freeze --output ...` 与 `evaluate --run ...`。复用 A3 随机、不确定性及拟议策略的原确认前缀；三来源种子 × 三查询种子 × 三策略 × 四预算 × 两种输出，共 216 案例。配置为 `configs/a4_power.json`，主策略 `proposed`、主预算 10 在评价前固定。完整规则、运行证据与效果边界见 [A4_HANDOFF.md](docs/A4_HANDOFF.md)。

2026-10-21 已完成真实开发矩阵、648 行设备指标、117 项测试及独立推理／来源校准核验。主策略相对自身零确认的配对电量误差平均下降 **8.86%**，运行段 MAE 平均恶化 **1.25%**，未达到 H3 的 10% 内部目标；不确定性对照的较好结果保留，不改选主策略。当前仍只有已查看 House 3、理想模拟确认，不能作为最终跨户功率收益。本机报告为 `outputs/accuracy/a4-20261021-v2/report.html`，可提交摘要为 `docs/A4_RESULTS.json` / `docs/A4_VALIDATION.json`。

## A5 冻结方案与最终多户评价

A5 已提供独立数据准备、三策略／四预算多户运行、旧 MIL 学习器参照、全矩阵冻结及逐户配对报告。默认主策略 `proposed`、主预算每户 10 次；功率联动继续单列，状态结果不替代 H3。正式名单暂为空，本机没有三个合格的新家庭完整文件，真实盲测与独立成员复跑尚未完成。范围与完整运行方法见 [A5 交接](docs/A5_HANDOFF.md)。

```powershell
.\.venv-m2\Scripts\python.exe scripts/run_a5.py preflight
.\.venv-m2\Scripts\python.exe scripts/run_a5_synthetic.py --output outputs/accuracy/my-a5-synthetic
.\.venv-m2\Scripts\python.exe scripts/run_a5.py verify --run outputs/accuracy/my-a5-synthetic/experiment
```

合成演示仅验证三户流程，不构成真实跨户识别成绩。正式模式要求至少三户隔离的 `real_complete` 数据、三来源／查询种子、冻结名单及真实的未查看未来标签声明；所有最终预测完成封存后才打开答案。

## 工作入口与目录说明

- [PLAN.md](PLAN.md)：开发目标、日程、成员 A/B 分工、实验与完成条件。
- [协作约定](docs/COLLABORATION.md)：分支、接口、交接和审查方式。
- [数据与结果契约](docs/DATA_CONTRACT.md)：数据划分、弱标签和预测输出定义。
- [默认配置](configs/default.toml)：当前范围与拟定实验预算，不代表已经完成实验。
- [资料来源](docs/REFERENCES.md)：赛事和研究资料。

```text
AIC/
├── PLAN.md
├── configs/                 实验配置
├── data/raw/                原始数据，本地保存，不提交
├── data/processed/          派生数据与划分，本地保存，不提交
├── docs/                    契约、协作与来源
├── models/artifacts/        权重和校准产物，本地保存，不提交
├── outputs/                 运行记录、图表和实验结果，本地保存，不提交
├── scripts/project.py       配置、下载、导入、审计和协议入口
├── scripts/verify_m1.py     已有真实产物的离线复核
├── scripts/run_m2.py        四种基线的顺序复跑入口
├── scripts/verify_m2.py     模型、预测和评价证据复核
├── scripts/app.py           本地 Streamlit 界面
├── src/virtual_submeter/    Python 包
└── tests/                   配置、数据质量和防泄漏边界测试
```

赛事手册保留在根目录，`tmp/` 中已有的手册分析文件保留本地。项目与 FLOWPLOT 独立，不复用其模型、代码或实验成绩。
