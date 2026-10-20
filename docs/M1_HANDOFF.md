# ElectricityPlot · M1 交接记录

日期：2026-10-20（北京时间）。本次承担数据与工程接口实现，供两名成员接续审查。验收范围是真实文件→质量审计→分区产物→访问边界复核，不含训练或预测。

## 已完成的接口和证据

- 项目名称、发行名称和命令入口使用 ElectricityPlot / electricityplot；内部包名保留 virtual_submeter 以兼容 M0。
- Python 标准库实现有界下载、来源回执、局部 SHA-256、完整下载出版方校验、CSV 检查、UTC 8 秒重采样、质量 HTML、划分和预算记录校验。
- 数据导入保留缺失，不把缺失填为关闭。有效区间按时间加权平均，不抽点；一次观测最多沿用 8 秒，不外推最后一点。覆盖不足的区间功率返回 null，仅报告有效区间积分。
- 总表导出完全不使用分表功率或 Issues。Issues 来自分表之和，只影响分表标签的有效性，防止以清洗过程泄露目标家庭答案。
- 来源标签、适配代理答案、冻结测试答案分别存储。加载器按消费者拒绝越界读取，并校验实际文件哈希。窗口回答不是逐点标签；不确定/跳过也计入预算。
- 提供小型合成契约样例和单元测试；下载失败、数据篡改、缺失、隔离、标签泄漏和预算绕过都有回归覆盖。

## 数据获取与许可

下载源为 [Zenodo REFIT 记录 5063428](https://zenodo.org/records/5063428)，API 实际只列出 House 1、2、3、4、5、11 六个文件，并非全部 20 户。记录元数据和[大学原始发布页](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/)均标明 CC BY 4.0。

通道核对依据为[大学发布的清洗版说明](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt)。清洗版已对缺失做前向填补或清零，原始缺失不可恢复；超过 4000 W 的分表尖峰已被上游清零。该限制贯穿 manifest 和质量报告，不把清洗文件的完整覆盖当成传感器原始完整率。

引用：Murray, D., Stankovic, L. & Stankovic, V. (2017). An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study. Scientific Data 4, 160122. DOI: 10.1038/sdata.2016.122。仅采用公开数据及其元数据，项目导入/审计代码自主实现。

本次下载 House 2、3、4、5、11 各 2 MiB 前缀，丢弃末尾不完整行，合计 181,143 行真实观测。前缀只有局部 SHA-256，**不宣称完整出版方 MD5 已通过**。具体哈希、来源、下载字节数、区间与覆盖率保存在 `docs/M1_AUDIT.json` 及本地回执中。

| 家庭 | 水壶通道 | 微波炉通道 | 洗衣机通道 | 总表完整有效区间比例 | 分表完整有效区间比例 |
|---|---|---|---|---|---|
| 2 | Appliance8 | Appliance5 | Appliance2 | 95.16% | 94.47% |
| 3 | Appliance9 | Appliance8 | Appliance6 | 97.46% | 88.54% |
| 4 | Appliance9 | Appliance8 | Appliance4 + Appliance5 | 34.14% | 32.72% |
| 5 | Appliance8 | Appliance7 | Appliance3 | 88.44% | 85.45% |
| 11 | Appliance7 | Appliance6 | Appliance3 | 44.48% | 43.64% |

比例按本项目保守重采样和有效性规则计算，只描述当前文件前缀。三类分表在本样本中的覆盖相同；非零不代表已经识别到运行状态，微波炉等设备可能有待机功率。House 4 洗衣机为两通道类型级合计，不能说是单台设备。House 1 仅有洗衣机目标通道，不包含水壶和微波炉，未下载该户前缀。

## 工程试运行协议

配置为 `configs/m1_split.json`。协议 purpose 明确为 engineering_pilot_only，不得用于正式效果报告。

| 分区 | 家庭 | UTC 起点（含） | UTC 终点（不含） |
|---|---|---|---|
| source_train | 2 | 2013-09-17 22:08:08 | 2013-09-20 14:30:56 |
| source_validation | 5 | 2013-09-26 09:56:08 | 2013-09-28 23:41:20 |
| target_adaptation | 3 | 2013-09-25 19:21:04 | 2013-09-27 02:21:52 |
| frozen_test | 3 | 2013-09-27 03:21:52 | 2013-09-28 09:22:40 |

来源/目标按整户隔离；目标适配池之后留出 3600 秒隔离带。当前允许的最大模型上下文为 3600 秒，确认窗口为 1200 秒。M2 若采用更长上下文，须增加隔离带并生成新协议，不能沿用此划分。预算为 0、5、10、20，预留种子 42、1019、2026；尚未运行查询或随机训练。

最初检查的 House 4/11 前缀未达到预先设定的 50% 覆盖率门槛，未纳入该试运行；没有降低门槛或把缺失改成零以通过。三目标家庭正式协议仍待补充更长数据或其他合格家庭并重新审计。不能从当前单户工程闭环声称跨三户有效。

## 环境和产物

本机实测：Windows AMD64、CPython 3.13.5、AMD Ryzen 5 5600、12 逻辑线程、约 31.89 GiB 物理内存；采样时可用内存约 17.97 GiB。显卡为 AMD Radeon RX 7650 GRE，未验证训练兼容性。M1 以 CPU 和标准库运行，依赖说明为 `requirements-m1.txt`。

第二台开发机未连接，未测量资源。未安装/锁定 NumPy、pandas、scikit-learn、PyTorch 或 Streamlit 为项目依赖；需双机核验后按 CPU/GPU 分别选定 M2 训练及演示版本。不能把此记录说成双机环境已经冻结。

本地可复核产物（默认不入 Git）：

- `data/raw/refit_m1/`：五份真实前缀、逐文件 receipt 和 Zenodo 元数据快照。
- `data/processed/refit_m1/manifest.json`、`catalog.json`、`aggregate/`、`labels/`：规范化数据与质量统计。
- `data/processed/refit_m1/quality.html`：可直接打开的离线质量视图，没有模型曲线。
- `data/processed/refit_m1_protocol/protocol.json`、`plan.json`：固定工程分区及各文件指纹。
- `outputs/m1_environment.json`：本机资源测量。
- `outputs/m1-20261020/`：本次配置、审计/协议指纹、验证结果和无训练状态记录。

复跑命令见 README。已有数据集和协议拒绝覆盖；新下载、不同时间段或改变划分需使用新目录。首次获取需联网，产物复核和单元测试无需联网。

本次验证：30 项单元测试通过，启用 ResourceWarning 错误检查；源码语法检查通过；真实产物离线复核返回 m1_artifacts_verified（5 户审计、4 个工程分区）。ElectricityPlot 0.2.0 wheel 构建通过，并核对了 electricityplot 命令入口和数据模块的打包完整性。上述结果仅为工程验证，没有 MAE、F1 或模型效果声明。

## M1 尚未完成的退出条件

1. 获取更长区间或更多实际合格家庭，审计运行片段、原始缺失风险与时间覆盖，冻结至少三目标家庭的正式实验清单。
2. 完成第二台机器的 env-report 和两人独立复跑，锁定 M2 训练依赖、骨干及其最大上下文。
3. 审查并确认清洗版未知补值的可接受性；必要时改用原始 REFIT，并独立记录清洗方法。不能从清洗文件倒推出不存在的原始缺失标记。

以上都不以模型成绩替代。M2 的训练、预测、指标、Streamlit 交互和 M3 的标注代理没有在本次实现中伪装为已完成。
