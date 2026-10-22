# 数据补齐交接：2026-10-22

本次已实际取得两个完整用电数据源，以及 IDEAL 的说明与元数据。没有启动训练、改变旧模型、冻结最终名单或计算新增家庭的识别成绩。

## 已落盘数据

| 来源 | 本次实际取得 | 本地目录 | 当前可用范围 |
|---|---|---|---|
| REFIT | 20 户完整 CSV，6,943,411,063 字节；官方压缩包 514,265,481 字节 | `data/raw/refit_complete_20261022/households/` | 每户有大小、SHA-256、压缩包成员 CRC 校验及完整来源回执 |
| REFIT 规范化数据 | 2/5/6/8/9/17/19/20 各最早 28 天，共 2,419,200 个八秒区间 | `data/processed/refit_complete_28d_20261022/` | 总表与三设备标签分目录，缺失保留；可供显式使用缺失掩码的开发实验，未获正式 A5 资格 |
| Plegma | 完整清洗包 301,557,385 字节，解压 277 个文件、1,927,690,110 字节，含 13 户 | `data/raw/plegma_20261022/extracted/` | 成员 CRC 和 SHA-256 已核验；原生 10 秒数据，尚未接入当前八秒模型 |
| IDEAL | `documentation.zip`、`metadata_and_surveys.zip` 两个完整包 | `data/raw/ideal_metadata_20261022/` | ZIP 校验通过；仅说明和元数据，尚未下载功率测量文件 |

实际文件、哈希及 REFIT 原始时间范围汇总见 [DATA_EXPANSION_INVENTORY_20261022.json](DATA_EXPANSION_INVENTORY_20261022.json)。完整数据保持本地保存，不进入 Git。两个新原始目录分别包含 `inventory.json`；REFIT 每户还包含 `.receipt.json`。

官方来源和许可：

- [REFIT 完整清洗版](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/)，CC BY 4.0。已同时保存官方清洗版与旧版 README。
- [Plegma 发布页](https://pureportal.strath.ac.uk/en/datasets/plegma-dataset/)，CC BY 4.0。已保存 2024-03-26 README。
- [IDEAL 发布页](https://doi.org/10.7488/ds/2836)，数据页关联的 End-user Licence 为 CC BY 4.0；见本轮来源调研中的许可链接。

本次未取得独立发布的两个 7z 整包校验和；回执明确 `publisher_checksum_verified=false`。验证的是下载后本地 SHA-256、归档成员 CRC 和解压大小，不宣称已有独立发布方摘要验证。

## 覆盖结果与限制

固定先取各户最早 28 天，沿用当前严格规则：每次观测最多维持八秒、不跨缺口补值、完整八秒区间才保留功率。原始时间间隔不规则，存在大量 10–14 秒间隔，因此“取得完整文件”不意味着“八秒派生数据没有缺失”。

下表是按现有 A5 时间规则进行的预检查：前半区适配，后半区扣除 3600 秒隔离带；每个分区要求至少七个有效天、覆盖 90%。数值取三设备标签的共同口径，未计算事件或模型成绩。

| 家庭 | 角色 | 适配标签覆盖 | 未来标签覆盖 | 当前资格 |
|---|---|---:|---:|---|
| 2 | 既有来源 | 94.8% | 22.7% | 不满足两分区门槛；来源身份保持原样 |
| 5 | 既有来源 | 54.3% | 29.8% | 不满足两分区门槛；来源身份保持原样 |
| 6 | 新候选 | 34.7% | 39.6% | 未通过 |
| 8 | 新候选 | 27.1% | 17.4% | 未通过 |
| 9 | 新候选 | 22.4% | 24.2% | 未通过 |
| 17 | 新候选 | 36.1% | 37.0% | 未通过 |
| 19 | 新候选 | 26.8% | 24.8% | 未通过 |
| 20 | 新候选 | 33.9% | 33.5% | 未通过 |

完整机器可读检查见 [DATA_EXPANSION_COVERAGE_20261022.json](DATA_EXPANSION_COVERAGE_20261022.json)。2/5 的表格仅说明此次新 28 天选择对 A5 式分区的支持，不用于推翻旧来源训练协议，也没有重训练旧权重。

因此，当前新增合格 A5 最终家庭数仍为零。没有降低覆盖要求、用零填充缺失或按模型分数挑选家庭。后续应在看成绩前制定日期范围和时间支持规则，利用已取得完整记录进行覆盖审计，再冻结新名单；不把本轮失败区间悄悄换掉后冒充原冻结方案。

官方旧版 README 明确记载 House 11、21 的总表受光伏影响；其他文献对 3/13 的光伏记录存在待复核信息。本轮六个新候选避开这些家庭，但“说明中未标光伏”仍不等于已经证明无光伏/储能，正式纳入前需继续核查。

Plegma 已补充真实独立来源，但原生十秒数据仍需单独的数据契约和适配器；不能改个数据集名称就当作当前八秒模型输入。IDEAL 传感器数据包仍需先按增强监测家庭和有功测量通道筛选再获取。UK-DALE 旧下载链接本次跳转到了 CEDA 的上层目录，未取得数据文件。

## 新增工具与复跑

`configs/refit_channels_complete_20261022.json` 按官方清洗版 README 扩展至 20 户；保留旧通道配置不变。未监测目标的家庭不会虚构相应通道，洗烘一体机未冒充单独洗衣机；House 13 两个微波炉通道明确相加，但未作为本次候选。

`prepare-refit` 新增 `--maximum-days`：只限制派生时间范围，完整原始文件及原始 acquisition receipt 的哈希保持绑定；清单另存 `preparation_selection`。没有该参数时保持原行为。八秒时间加权、缺口和 Issues 处理规则均沿用原实现。

原始包已下载且隔离解压依赖可用时，可以在全新目录复跑：

```powershell
python scripts/ingest_refit_archive.py --archive data/raw/refit_complete_20261022/CLEAN_REFIT_081116.7z --output data/raw/refit_complete_20261022/my_households --tools-dir tmp/refit_archive_tools
python scripts/project.py prepare-refit --input-dir data/raw/refit_complete_20261022/my_households --households 2 5 6 8 9 17 19 20 --catalog configs/refit_channels_complete_20261022.json --maximum-days 28 --output data/processed/my_refit_complete_28d
python scripts/audit_refit_expansion.py --manifest data/processed/my_refit_complete_28d/manifest.json --output outputs/my_refit_coverage.json
```

解压依赖是独立安装在 `tmp/refit_archive_tools` 的 py7zr 1.1.4 及其依赖，未修改 `.venv-m2`。新机器需先准备该解压依赖；数据规范化和覆盖检查只用 Python 标准库。所有输出目录必须全新，已有数据不覆盖。

覆盖检查器只读取时间戳和功率是否为空，不计算开机数量、能量、事件、曲线或识别分数。标准数据准备流程仍会写出原有质量统计；本轮没有用其中的设备活动统计作筛户依据，也未打开新增家庭质量曲线。

## 验证与研究边界

本次 32 项相关测试通过：数据契约 22 项（新增日期上限、边界和完整来源身份保留测试），下载 4 项，项目边界 6 项；脚本语法和差异格式检查通过。未宣称全套训练测试通过。

旧协议、最终家庭名单、未查看声明、旧模型和历史结果未更改。数据准备和覆盖检查不等于训练完成、外部泛化提升或最终盲测达标；本轮未代填任何“未查看”声明。

本轮数据准备代码变更会改变项目源码指纹。需要复核旧封存实验时应使用其对应源码版本；不要绕过旧运行的源码一致性检查。
