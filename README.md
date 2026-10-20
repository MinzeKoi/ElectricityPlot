# ElectricityPlot

低交互可信虚拟分表系统

Windows 可双击根目录的 `start.cmd` 启动界面，详见 [CMD 启动与测试流程](docs/CMD启动与测试.md)。报告查看和 M5 复跑命令见 [运行方式](运行方式.md)。

从家庭总功率曲线估计目标电器的功率与电量，通过少量用户确认适配新家庭，并保留无法可靠解释的耗电。

当前已完成 **M5 冻结实验工程与本机验证**：B0–P2、三种子/四预算、冻结后评价、报告导出及复跑核对已实现。真实 REFIT 前缀的单目标家庭 42 个案例、合成三目标家庭 126 个案例均通过复核，完整环境 91 项测试通过。当前结果未证明拟议策略稳定优于基线，背景误归因与区间覆盖不足仍然存在。正式实测多户盲测和独立成员/双机复跑仍待完成，证据见 [M5 交接](docs/M5_HANDOFF.md)。

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

## 工作入口

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
