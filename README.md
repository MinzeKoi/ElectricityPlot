# ElectricityPlot

低交互可信虚拟分表系统

从家庭总功率曲线估计目标电器的功率与电量，通过少量用户确认适配新家庭，并保留无法可靠解释的耗电。

当前处于 **M2 基线闭环阶段**：已完成单设备到三设备的轻量时序模型训练、常量/零预测对照、未见家庭上的真实预测与评价，以及 Streamlit 导入、回放和报告下载。当前结果来自真实 REFIT 前缀、单目标家庭和单个种子的工程试运行；正式三目标家庭协议与双机环境仍待完成。初始时序模型的全时段 MAE 尚未优于简单对照，不能宣称算法优势。

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
