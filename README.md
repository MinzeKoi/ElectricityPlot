# ElectricityPlot

# 低交互可信虚拟分表系统

从家庭总功率曲线估计目标电器的功率与电量，通过少量用户确认适配新家庭，并保留无法可靠解释的耗电。

当前处于**项目初始化阶段**：已有项目配置、配置检查入口、双人协作约定和开发计划。尚未下载数据、训练模型、实现负荷分解或取得实验结果。

## 第一版范围

- 输入：无光伏、无储能家庭的带时间戳总有功功率，单位 W。
- 暂定目标：水壶、微波炉、洗衣机；最终可用设备和家庭在数据审计后冻结。
- 输出目标：电器估计功率/电量、运行状态、未解释负荷、可信程度和少量确认问题。
- 用户确认是“这段时间是否使用过该电器”的弱标签，不等于获得逐点功率答案。
- 未解释负荷可能包含未建模设备、测量误差与模型误差，不直接命名为某台未知设备。

## 开始使用

要求 Python 3.11 或更新版本。初始化骨架只使用标准库，无需安装训练依赖。

```powershell
python scripts/project.py status
python scripts/project.py validate-config
python scripts/project.py status --json
python -m unittest discover -s tests -v
```

可选：先运行 `python -m venv .venv`，再将上述 `python` 替换为 `.\.venv\Scripts\python.exe`。机器学习和界面依赖在 M1 阶段选定、验证并锁定版本。

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
├── scripts/project.py       初始化阶段的命令入口
├── src/virtual_submeter/    Python 包
└── tests/                   配置和状态声明检查
```

赛事手册保留在根目录，`tmp/` 中已有的手册分析文件保留本地。项目与 FLOWPLOT 独立，不复用其模型、代码或实验成绩。
