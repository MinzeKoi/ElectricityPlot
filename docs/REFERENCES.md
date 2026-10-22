# 资料来源

2026-10-22 数据获取进展：[数据补齐交接](DATA_EXPANSION_HANDOFF_20261022.md)，已落盘 REFIT 完整 20 户、Plegma 完整 13 户和 IDEAL 文档/元数据；其中 REFIT 八户的最早 28 天已规范化，覆盖检查未通过 A5 门槛。

2026-10-22 新增：[训练与测试数据源扩展调研](DATA_SOURCE_EXPANSION_20261022.md)，包含 REFIT 完整版、UK-DALE、IDEAL、ENERTALK、Plegma、DEDDIAG、ECO 的获取入口、许可、适配限制及接入顺序。属于来源调研，尚未取得或导入完整新增数据。

整理日期：2026-10-19。以下支持选题、规则和实验设计，不代表项目已复现其结果。

## 赛事

- 根目录 `【全】AIC·2026赛题赛事规则手册261009.pdf`，AI+能源见印刷页 361–367（PDF 第 365–371 页）。
- [AI+能源官方赛题](https://www.aicomp.cn/tracks/4726.html)。
- [2026-10-10 赛程调整通知](https://www.aicomp.cn/notice/notice-3/5435.html)：AI+能源报名缴费及作品提交为 10 月 30 日 20:00。

## 数据和基线

- [REFIT 清洗数据](https://zenodo.org/records/5063428)：20 户，约 8 秒间隔，获取后审计实际通道、缺失和许可。
- M1 核验（2026-10-20）：上述 Zenodo 记录实际仅列 6 户文件；[大学发布页](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/)和[清洗版通道说明](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt)用于核对 CC BY 4.0、实际通道和补值限制。实际获取范围见 [M1 交接](M1_HANDOFF.md)，不把数据集总体规模当作本项目已下载规模。
- [REFIT 数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5315495/)。
- [UK-DALE 数据说明](https://www.nature.com/articles/sdata20157)。
- [Seq2Point，AAAI 2018](https://ojs.aaai.org/index.php/aaai/article/view/11873)。
- [NILMTK contrib](https://github.com/nilmtk/nilmtk-contrib)：可参考基线接口和评价方法，使用时核对代码许可与依赖。

## 相关研究

- [Transfer Learning for Non-Intrusive Load Monitoring](https://arxiv.org/abs/1902.08835)。
- [An active learning framework for the low-frequency Non-Intrusive Load Monitoring problem](https://doi.org/10.1016/j.apenergy.2023.121078)。
- [A weakly supervised active learning framework for non-intrusive load monitoring](https://doi.org/10.3233/ICA-240738)：2024 在线发表，2025 卷期。
- [Dynamic adaptive modeling for non-intrusive load monitoring with unknown loads](https://doi.org/10.1016/j.enbuild.2024.115246)：2025 卷期。
- [Distribution-Free Uncertainty Quantification and Calibration for Energy Disaggregation](https://doi.org/10.1145/3748334)。

主动学习、弱监督和未知负荷处理并非本项目首创。报告需区分采用方法、自主实现和实验验证的新贡献。
