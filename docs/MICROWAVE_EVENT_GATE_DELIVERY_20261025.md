# 固定冻结模型事件高分门控：任务11交付

**来源门控通过，开发整体验收未达标。全程0次新训练、0次优化更新，任务10的3个新增House5模型未重训。**

|开发组|平均P|平均R|平均F1|总FP|
|---|---:|---:|---:|---:|
|W50|0.040128|0.924282|0.076913|25446|
|parent|0.043832|0.955614|0.083814|23962|
|gate|0.045574|0.942559|0.086920|22734|

门控相对W50：平均F1增加0.010007，误报减少10.66%；相对父point规则：平均F1增加0.003106，误报减少5.12%，平均R变化-0.013055。
验收要求平均P≥0.08、F1提升≥0.02及误报下降≥30%，这三项未通过。各seed R≥0.85、至少2seed P/F1同升、任seed F1不下降超过0.02及非退化均通过。来源预选不因开发结果改变；未切默认、未追加候选。

|seed|固定L|固定H|父epoch|父point阈值|
|---|---:|---:|---:|---:|
|42|0.720104992390|0.969567298889|11|0.822113335133|
|1019|0.844448029995|0.995560288429|11|0.911890685558|
|2026|0.754520833492|0.973924815655|10|0.834655284882|

来源只有固定block4/5（Nov23–27 UTC），沿用原34351/201个完整C31评价中心。各seed先L保护每块R≥.95，再固定半径15（120秒）的一次中心max，以H保护每块R≥.85。三个seed全部可行且通过来源保护，来源均P 0.209635→0.220452，均F1 0.340164→0.354818。
每块完整aggregate推理得到19987/15586个有限分数，其中860/362个不属于原评价中心，仍只作为无标签特征；标签缺失不当关闭，也不决定max连续段。连续段只由原始有限score、8秒连续与块边界决定。低于L的原评价点保留off/FN；独立复核逐块核对相应数量。

|seed|开发P|开发R|开发F1|TP|FP|FN|TN|FPR|预测开机段|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|42|0.045712|0.989556|0.087388|379|7912|4|129228|0.057693|817|
|1019|0.047336|0.874674|0.089812|335|6742|48|130398|0.049161|726|
|2026|0.043674|0.963446|0.083560|369|8080|14|129060|0.058918|903|

|seed|对照|新增FP|移除FP|FP净变化|
|---|---|---:|---:|---:|
|42|W50|948|1469|-521|
|42|parent|295|516|-221|
|1019|W50|599|2750|-2151|
|1019|parent|204|1005|-801|
|2026|W50|1119|1159|-40|
|2026|parent|341|547|-206|

## 复核和边界

21项门控合成测试与25项原协议测试通过。独立直方图阈值枚举、填充窗口max、标量TP/FP/FN/TN与来源保护/开发验收均一致。普通CLI的150750行t/score/h/state与封存完全一致，三个seed的普通加载API也逐点重放；开发每seed严格保持原137523点与时间mask。
任务10已完成终态数值复核后才正式执行。prepare/source访问日志不含House17数值输入；6份开发预测、source预选及普通推理重放先封存，才一次读取已核验旧真值缓存。复核只重读本任务封存缓存。确认终态时工具意外返回任务10已有汇总开发结果的情况已在预登记protocol.md记录；未读逐点误报来设计规则。
权重、epoch和归一化保持；加载网络结构以读回冻结权重，不产生新初始化训练。运行期间禁止建立优化器和backward，0新训练/0更新可从入口、访问记录和父权重哈希复核。
源完整aggregate含额外无标签中心，CPU最后一批大小变化使历史分数最多相差3.183×10⁻¹²；使用rtol=1e-5、atol=1e-7核对历史score，同时严格要求父point状态完全相同。新普通入口与本任务封存分数仍严格逐点一致。
准备v1因审计守卫误拦代码文件而停止，v2因上述极小浮点严格比较停止，v3在来源结果JSON序列化时停止；均原样保留、0拟合、未读开发真值。只修复执行/记录检查，算法和参数未变；v3与正式v4的全部L/H曲线、来源混淆及配方阈值已核对完全一致。
最大有效上下文157样本约21分钟，仍含离线未来上下文。House5末4日是历史复用来源验证，House17只是已有开发工程试验，不称独立新户验证；最终8/9/20任何数据未访问。原A功率、其他电器、默认UI与默认协议保持，无功率改善声明。

## 文件与复核命令

- [完整试验报告](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-event-gate-20261025-v4/REPORT.zh.md>)
- [来源逐块指标、L/H曲线与普通recipe](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-event-gate-20261025-v4/source_results/block_metrics.csv>)
- [来源独立复核（含无标签特征中心）](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-event-gate-20261025-v4/independent_source_review.json>)
- [逐seed开发混淆](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-event-gate-20261025-v4/evaluation/development_confusions.csv>)
- [开机预测段](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-event-gate-20261025-v4/evaluation/predicted_on_segments.csv>)
- [误报增加/移除逐点对照](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-event-gate-20261025-v4/evaluation/false_positive_changes.csv>)
- [最终独立复核](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-event-gate-20261025-v4/independent_delivery_review.json>)
- [整包SHA-256清单](<D:/system position/document/真正的文档/AIC/outputs/accuracy/microwave-event-gate-20261025-v4/delivery_manifest.json>)

```powershell
.venv-m2/Scripts/python.exe -B "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-event-gate-20261025-v4/run_microwave_event_gate.py" verify --run "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-event-gate-20261025-v4"
```

公共predict入口只接受合格recipe；只需aggregate输入。已实际运行seed42，其他seed替换recipe文件即可，输出不可覆盖：

```powershell
.venv-m2/Scripts/python.exe -B "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-event-gate-20261025-v4/run_microwave_event_gate.py" predict --run "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-event-gate-20261025-v4" --recipe "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-event-gate-20261025-v4/source_results/recipe_42.json" --aggregate "<aggregate.csv>" --output "<新输出.csv>"
```
