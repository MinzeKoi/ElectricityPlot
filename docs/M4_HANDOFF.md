# M4 可信输出交接

日期：2026-10-20。负责范围：可信输出工程、评价与演示接口。输入为 M1 冻结协议、M2 未适配功率模型及可选同主干 M3 状态模型；输出为来源校准、成对背景实验、离线报告和复核记录。**工程已实现；已补跑真实 REFIT 前缀单目标家庭的 CNN、5 次模拟确认状态模型、M4 背景实验和 Streamlit。正式多户实验、独立成员复跑及可信效果达标仍未完成。**

## 实现与验收

| M4 要求 | 实现及证据 |
|---|---|
| 原始/修正预测可追溯 | 保留 raw_power_w、power_w、correction_w、raw_excess_w、correction_method；只在超总表时向下缩放 |
| 未解释负荷与可信信息 | 来源验证 raw/corrected 绝对残差经验区间；无匹配校准则区间缺失，confidence 始终 null；质量和外推标记独立展示 |
| 陌生背景压力测试 | 固定 1500 W 脉冲，已知标签不变；基准与注入预测冻结后才打开测试标签；有/无背景组分别报告误差、错误激活 |
| 无“全部未知”虚假优势 | 共同有效标签支持上联合报告归因比例、MAE、过归因和漏掉已知电量；保留全零/全部未解释对照 |
| 展示、恢复与导出 | HTML 原始/修正曲线及区间、Streamlit 质量表、CSV/JSON 下载；导出哈希校验，失败不发布成功目录，切换/损坏输入清除旧结果 |
| 无标签泄漏 | 校准消费者只读来源验证；测试标签仅评价使用；产物复核不重开标签，有文件访问拦截测试 |

校准方法是描述性的来源经验残差分位数。来源验证也用于模型选择，且时间样本相关，不作保形、跨户或同时覆盖保证。未解释项含模型/计量误差，不是真实未知电器电量。状态头适配与功率估计分别报告。

## 复跑

已安装 M2 依赖并准备真实模型/协议后：

```powershell
python scripts/run_m4.py --model outputs/my-m2/seq2point --manifest data/processed/refit_m1_protocol/protocol.json --partition frozen_test_house_3 --output outputs/my-m4
python scripts/verify_m4.py --run outputs/my-m4
```

如需 P1 状态适配模型，可加 `--prediction-model outputs/my-adapted-model`；与来源模型功率指纹不匹配时拒绝校准。已有输出目录不覆盖，所有失败另选目录重跑。

无 REFIT、PyTorch、Streamlit 时可使用标准库 + NumPy 跑合成契约闭环：

```powershell
python scripts/run_m4_synthetic.py --output outputs/my-m4-demo
python scripts/verify_m4.py --run outputs/my-m4-demo
python -m unittest discover -s tests -v
```

Windows 沙箱若默认临时目录不可写，运行前将 `TEMP`、`TMP` 设为项目内已存在的 `tmp` 绝对路径。初始仅用内置 Python/NumPy 跑合成；追加真实验收已在 `.venv-m4` 安装 Python 3.12.14、NumPy 2.3.4、PyTorch 2.9.0+cpu、Streamlit 1.50.0。完整版本为 `requirements-m4-cpu.lock`，`pip check` 通过。与 M2 Python 3.13 环境分别记录，不宣称双机复现。版本和哈希见 [M4_VALIDATION.json](M4_VALIDATION.json)。

## 实际验证与产物

- 全量 85 项测试全部通过，无跳过/失败/错误；其中 M4 17 项全部通过，包含 Streamlit 加载区间、损坏导出/校准清除结果及 UTC/缺失断线图表配置。
- 合成完整演示：`outputs/m4-20261020-demo/stress/report.html`，含基准/注入子报告、输入、预测、指标、设置与协议快照。
- 通用运行入口复跑：`outputs/m4-20261020-cli/run.json`，已运行 `scripts/run_m4.py` 并完成只读复核。
- 最新测试日志、结构化摘要：`outputs/m4-validation-20261020-real/tests.log`、`tests.json`；旧合成验收单独保留。汇总与指纹见 `docs/M4_VALIDATION.json`。
- 真实未适配 CNN：`outputs/m4-20261020-real-verified/stress/report.html`；真实 5 次模拟确认状态模型：`outputs/m4-20261020-real-adapted-verified/stress/report.html`。两份产物均通过只读哈希复核，代码指纹与当前一致；功率与区间不因状态适配变化。最终图表修复未改变先前冻结预测的数值。
- House 2 来源训练、House 5 模型选择/来源区间估计、House 3 目标适配/冻结测试；8 秒采样，3600 秒隔离，固定种子 42。三户各 2 MiB 真实前缀有 CC BY 4.0 来源回执和局部 SHA-256，不声称完整出版方 MD5 通过。5 次回答均标记模拟，状态模型在任何目标测试评价前已冻结。
- 运行产物不提交 Git；可用以上合成命令重新生成。测试保护了校准标签访问、功率指纹、缺数据不当关闭、共同有效支持、错误恢复、导出损坏、合成来源标识及先预测后评价次序。

## 真实试运行结果与局限

未适配模型的有效预测覆盖约 89.35%；原始超限率由基准的 0.45% 升到注入后的 12.79%，向下修正后均为 0。基准总可归因电量比例约 38.46%，注入后约 55.88%。这些是数值一致性和分配结果，不等于识别更准确。

背景注入新增约 10.003 kWh（共同有效预测支持），其中约 88.92% 被额外归给已知设备；该敏感性指标暴露出明显误归因，**未知背景鲁棒性未证明**。三设备修正 MAE 从约 85.73 / 43.26 / 138.59 W 升到 310.00 / 114.54 / 185.50 W。来源标称 90% 区间的目标经验覆盖，基准约为 79.07% / 81.36% / 78.20%，注入后约为 59.65% / 69.22% / 66.60%；不能把标称水平当成实际可信概率。完整 raw/corrected、状态、区间、归因与关闭片段误激活摘要见 [M4_RESULTS.json](M4_RESULTS.json)。

Streamlit 已通过真实浏览器运行和自动化恢复验收；时间轴强制 UTC，缺失处断开，截图与记录见验证目录。导出的本地 `file:` HTML 曾被浏览器工具协议策略拦截，未对该离线文件做浏览器视觉验收；未绕过该限制。离线报告的生成、内容及哈希完整性已验证。

这些是单目标家庭、单种子、真实文件前缀与人工背景注入的工程结果，不代表正式多户、多种子效果或真实用户可用性。尚待第二成员/第二台机器独立复跑和正式多户实验；不得依本次测试结果调区间水平或注入参数后仍称为同一次冻结评价。当前结果支持“M4 链路可运行且负结果可追溯”，不支持“模型可信效果已达标”。
