# House5同户时间对照试验：正式交付（2026-10-25）

**已完成固定3个新模型；整体开发验收未通过。** 新增House5组通过来源保护并被事前唯一预选，但相对W50的Precision、F1增益及误报下降均未达预登记门槛。无开发改选、额外候选或补训。

|开发组|均P|均R|均F1|总FP|
|---|---:|---:|---:|---:|
|W50|0.040128|0.924282|0.076913|25446|
|control|0.037456|0.956484|0.072086|28254|
|source5|0.043832|0.955614|0.083814|23962|

相对W50平均F1增加0.006901（要求≥0.02），误报减少5.83%（要求≥30%）；平均P=0.043832（要求≥0.08）。各seed R≥0.85、非退化、3个seed P/F1同升、各seed F1不下降超过0.02均通过。

来源对称对照：旧控制和新增组均只用原冻结最后两块、同一34351/201中心与输入，重选12epoch checkpoint/公共阈值。旧控制均P=0.162579、均F1=0.275326；新增均P=0.209635、均F1=0.340164，三seed P均提高，来源门槛通过。
冻结后开发诊断：新增组相对对称旧控制均P增加0.006376、均F1增加0.011727、平均FP减少1430.67；该相对收益不足以满足原W50整体验收，不声明达标改善。

新数据支持：原House2/19 partial 259806/849全部行逐列不变；House5新增76898/1426，联合336704/2275，House5索引2。原归一化字节相同；31核心与127外圈不跨各自冻结块，跨界阈值段整段排除，末端1800秒带、NaN及观测mask保留。独立脚本重放三条3528更新采样流并核对实际上下文时间集合隔离。
预算：仅control_BCE50、seed42/1019/2026各一次、每次3528更新，共10584更新；3启动全部完成，0失败拟合，0控制重训。准备v1路径类型错误与空的初次明细报告目录保留，均未引发新拟合；正式有效目录为v2。

人类授权原文：“允许这个独立时间对照试验”。协调者确认任务8终态未达标且source预选TCN127/control_BCE50后执行。两项获批例外为House5仅验证及来源训练/验证整户隔离；本试验不符合原跨户来源协议。默认data.protocol守卫和旧实验、旧CSV、默认UI不变。
House5最后4日已历史选模接触，只称复用来源时间验证；House17此前已查看，只作开发验收，不是全新独立户外验证。最终8/9/20数据未访问。原A微波炉功率及其他电器功率/状态逐点不变，不宣称功率改善。

全部6份开发预测与唯一source预选先封存、普通推理逐点一致后才一次解析开发真值；每seed原W50同137523点。19项本任务合成测试＋25项默认协议回归通过；最终独立脚本复核数值、完整哈希和普通CLI输出。

## 交付文件

- [主报告](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-source5-temporal-20261025-v2/report/REPORT.zh.md>)
- [误报片段与补充明细](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-source5-temporal-20261025-v2/details_v2/DETAILS.zh.md>)
- [逐seed开发混淆](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-source5-temporal-20261025-v2/report/development_comparison.csv>)
- [逐块来源混淆](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-source5-temporal-20261025-v2/details_v2/source_selected_block_confusions.csv>)
- [独立最终复核](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-source5-temporal-20261025-v2/independent_final_verification.json>)
- [完整SHA-256清单](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-source5-temporal-20261025-v2/final_delivery_manifest.json>)
来源/开发PR曲线CSV在report目录；全部误报片段、真实日期/事件曝光及实际正负损失贡献在details_v2。普通可加载权重与元数据在models/source5/seed_42、seed_1019、seed_2026；对称旧控制另在models/control。

## 普通预测命令

```powershell
.venv-m2/Scripts/python.exe -B "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-source5-temporal-20261025-v2/predict_microwave_tcn.py" --model "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-source5-temporal-20261025-v2/models/source5/seed_42" --aggregate "D:\system position\document\真正的文档\AIC\outputs\accuracy\precision-20261024-v1\protocol\data\frozen_test_refit_17_0\aggregate.csv" --output "<新的预测CSV路径>"
```

该命令只输出微波炉开机概率/状态，原功率输出未调整；输出路径必须未存在。seed42普通CLI已实际运行并与封存t/scores/state完全一致，另6份预测均通过普通加载API核对。
