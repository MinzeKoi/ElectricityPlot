# 微波炉固定事件门控试验（任务11）

来源唯一预选：gate；门控来源资格：True；相对W50整体开发验收：False。

本任务0新拟合、0新训练、0优化更新；任务10的3个House5新增模型未被重训。只加载原冻结权重、epoch、归一化和point阈值。

|seed|L|H|seed可行|全组可部署资格|父epoch|父point阈值|
|---|---:|---:|---|---|---:|---:|
|42|0.720104992389679|0.9695672988891602|True|True|11|0.8221133351325989|
|1019|0.8444480299949646|0.9955602884292603|True|True|11|0.9118906855583191|
|2026|0.754520833492279|0.9739248156547546|True|True|10|0.8346552848815918|

来源保护：{'all_seeds_feasible': True, 'mean_f1_not_lower': True, 'each_seed_recall_drop_at_most_005': True, 'at_least_two_seed_precision_up': True}

|来源组|均P|均R|均F1|
|---|---:|---:|---:|
|parent|0.209635|0.916290|0.340164|
|gate|0.220452|0.925980|0.354818|

|开发组|均P|均R|均F1|均FP|
|---|---:|---:|---:|---:|
|A|0.040128|0.924282|0.076913|8482.00|
|W50|0.040128|0.924282|0.076913|8482.00|
|gate|0.045574|0.942559|0.086920|7578.00|
|parent|0.043832|0.955614|0.083814|7987.33|

验收只针对事前source预选，不按开发诊断切换。各项W50验收：{'mean_P_at_least_008': False, 'each_R_at_least_085': True, 'mean_F1_gain_at_least_002': False, 'at_least_two_seed_P_F1_up': True, 'each_F1_drop_at_most_002': True, 'FP_reduction_at_least_30_percent': False, 'nondegenerate': True}

开发对父及W50的逐项差值见evaluation/summary.json的deltas；逐seed P/R/F1、TP/FP/FN/TN、FPR和开机预测段数见evaluation/development_confusions.csv。
来源逐块同样指标见source_results/block_metrics.csv；全部精确L/H阈值曲线在source_results。逐段开机预测见evaluation/predicted_on_segments.csv，误报增加/移除对父及W50逐点对照见evaluation/false_positive_changes.csv。

严格原C31来源34351/201点与开发每seed137523点；低于L的有效点始终为off、计入FN，标签未知中心仅作无标签max特征。原分数有限段连续8秒、单次半径15 max，不跨缺失、断裂、House5验证块；不插补、不递归传播。
普通aggregate推理已完整重放与冻结父/门控逐点一致，原A功率及水壶/洗衣机输出逐点保持；无功率改善声明、无默认切换。未校准sigmoid状态分数不称置信概率。
最大有效上下文157样本，约21分钟，仍含离线未来上下文；House5最后4日是历史复用来源验证，House17是已有开发工程验收，不是独立新户验证。最终8/9/20数据未访问。

候选无来源资格时，公共ruleloader明确拒绝recipe；其冻结开发结果只供诊断。不得部署失败recipe或依据开发结果重新选阈值。

复核命令（只读完整哈希、封存缓存与独立标量计数）：
```powershell
.venv-m2/Scripts/python.exe -B "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-event-gate-20261025-v4/run_microwave_event_gate.py" verify --run "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-event-gate-20261025-v4"
```

合格recipe普通预测命令（输入只有aggregate；输出路径必须未存在）：
```powershell
.venv-m2/Scripts/python.exe -B "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-event-gate-20261025-v4/run_microwave_event_gate.py" predict --run "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-event-gate-20261025-v4" --recipe "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-event-gate-20261025-v4/source_results/recipe_42.json" --aggregate "<aggregate.csv>" --output "<新输出.csv>"
```

本试验的算法和阈值未根据House17改变。终态确认工具意外返回任务10已公开汇总开发指标的审计记录见protocol.md；本任务source阶段数值输入与真值隔离由访问审计记录。
