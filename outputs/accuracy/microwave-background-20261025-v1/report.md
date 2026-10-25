# 微波炉背景组合受控实验

3次新增拟合（复用6个对照）；改善门槛：False。来源时间留出属于开发诊断，不能称独立泛化证明。

|采样|种子|分区|P|R|F1|FP|AP|
|---|---|---|---|---|---|---|---|
|natural|42|refit_2/train|0.0928|0.8870|0.1681|1534|0.1509|
|natural|42|refit_19/train|0.1393|0.7627|0.2356|278|0.3060|
|natural|42|refit_2/validation_0|0.3313|0.8594|0.4783|111|0.2669|
|natural|42|refit_2/validation_1|0.1993|0.9831|0.3314|233|0.2082|
|natural|42|refit_19/validation_0|0.4790|0.9756|0.6426|87|0.7253|
|natural|42|refit_19/validation_1|0.2346|0.8511|0.3678|261|0.4157|
|natural|42|refit_2/outer|0.0890|0.5652|0.1538|399|0.0710|
|natural|42|refit_19/outer|0.0496|0.6667|0.0923|537|0.0588|
|natural|1019|refit_2/train|0.1045|0.9153|0.1876|1388|0.2342|
|natural|1019|refit_19/train|0.1833|0.7797|0.2968|205|0.3344|
|natural|1019|refit_2/validation_0|0.2091|0.8594|0.3364|208|0.3114|
|natural|1019|refit_2/validation_1|0.2043|0.9661|0.3373|222|0.3236|
|natural|1019|refit_19/validation_0|0.4659|1.0000|0.6357|94|0.8176|
|natural|1019|refit_19/validation_1|0.2812|0.8617|0.4241|207|0.5333|
|natural|1019|refit_2/outer|0.0575|0.5507|0.1041|623|0.0853|
|natural|1019|refit_19/outer|0.0535|0.6429|0.0987|478|0.0634|
|natural|2026|refit_2/train|0.0951|0.9209|0.1724|1551|0.1261|
|natural|2026|refit_19/train|0.2010|0.6610|0.3083|155|0.3539|
|natural|2026|refit_2/validation_0|0.2276|0.9531|0.3675|207|0.2765|
|natural|2026|refit_2/validation_1|0.2111|0.9661|0.3465|213|0.1954|
|natural|2026|refit_19/validation_0|0.4472|0.8780|0.5926|89|0.6684|
|natural|2026|refit_19/validation_1|0.3127|0.8617|0.4589|178|0.3321|
|natural|2026|refit_2/outer|0.0472|0.5217|0.0866|726|0.0598|
|natural|2026|refit_19/outer|0.0549|0.5714|0.1002|413|0.0625|
|event|42|refit_2/train|0.0884|0.8531|0.1601|1558|0.0912|
|event|42|refit_19/train|0.1524|0.9322|0.2619|306|0.2891|
|event|42|refit_2/validation_0|0.3146|0.8750|0.4628|122|0.2557|
|event|42|refit_2/validation_1|0.1630|0.8814|0.2751|267|0.1230|
|event|42|refit_19/validation_0|0.5066|0.9390|0.6581|75|0.6345|
|event|42|refit_19/validation_1|0.2032|0.9468|0.3346|349|0.3401|
|event|42|refit_2/outer|0.1043|0.7101|0.1818|421|0.0874|
|event|42|refit_19/outer|0.0436|0.5714|0.0809|527|0.0526|
|event|1019|refit_2/train|0.0919|0.9379|0.1674|1640|0.1089|
|event|1019|refit_19/train|0.2135|0.9661|0.3497|210|0.2958|
|event|1019|refit_2/validation_0|0.2976|0.9531|0.4535|144|0.2934|
|event|1019|refit_2/validation_1|0.1645|0.8644|0.2764|259|0.1271|
|event|1019|refit_19/validation_0|0.4598|0.9756|0.6250|94|0.7387|
|event|1019|refit_19/validation_1|0.2977|0.9787|0.4566|217|0.4237|
|event|1019|refit_2/outer|0.1010|0.8406|0.1804|516|0.1016|
|event|1019|refit_19/outer|0.0685|0.8095|0.1264|462|0.0745|
|event|2026|refit_2/train|0.0963|0.9887|0.1754|1643|0.1181|
|event|2026|refit_19/train|0.2176|0.9661|0.3551|205|0.3531|
|event|2026|refit_2/validation_0|0.2180|0.9062|0.3515|208|0.3106|
|event|2026|refit_2/validation_1|0.1836|0.9492|0.3077|249|0.1550|
|event|2026|refit_19/validation_0|0.4629|0.9878|0.6304|94|0.7641|
|event|2026|refit_19/validation_1|0.2766|0.9681|0.4303|238|0.4356|
|event|2026|refit_2/outer|0.0758|0.8551|0.1393|719|0.1130|
|event|2026|refit_19/outer|0.0660|0.7619|0.1214|453|0.0777|
|event_background|42|refit_2/train|0.0966|0.8588|0.1737|1421|0.1179|
|event_background|42|refit_19/train|0.2202|0.8136|0.3466|170|0.3172|
|event_background|42|refit_2/validation_0|0.3548|0.8594|0.5023|100|0.2659|
|event_background|42|refit_2/validation_1|0.1726|0.9831|0.2937|278|0.2245|
|event_background|42|refit_19/validation_0|0.5417|0.9512|0.6903|66|0.6894|
|event_background|42|refit_19/validation_1|0.2939|0.8723|0.4397|197|0.4769|
|event_background|42|refit_2/outer|0.0862|0.5072|0.1474|371|0.0684|
|event_background|42|refit_19/outer|0.0647|0.6667|0.1179|405|0.0725|
|event_background|1019|refit_2/train|0.1386|1.0000|0.2435|1100|0.2366|
|event_background|1019|refit_19/train|0.2865|0.9322|0.4382|137|0.2795|
|event_background|1019|refit_2/validation_0|0.2113|0.8750|0.3404|209|0.3898|
|event_background|1019|refit_2/validation_1|0.1906|0.8983|0.3145|225|0.2627|
|event_background|1019|refit_19/validation_0|0.5563|0.9634|0.7054|63|0.8267|
|event_background|1019|refit_19/validation_1|0.3374|0.8830|0.4882|163|0.4478|
|event_background|1019|refit_2/outer|0.0600|0.6667|0.1100|721|0.1066|
|event_background|1019|refit_19/outer|0.0825|0.7857|0.1493|367|0.0896|
|event_background|2026|refit_2/train|0.1183|0.8249|0.2069|1088|0.1401|
|event_background|2026|refit_19/train|0.3469|0.8644|0.4951|96|0.3517|
|event_background|2026|refit_2/validation_0|0.3590|0.8750|0.5091|100|0.3479|
|event_background|2026|refit_2/validation_1|0.2014|0.9661|0.3333|226|0.2156|
|event_background|2026|refit_19/validation_0|0.6074|1.0000|0.7558|53|0.8631|
|event_background|2026|refit_19/validation_1|0.3971|0.8617|0.5436|123|0.5093|
|event_background|2026|refit_2/outer|0.1153|0.5362|0.1897|284|0.1027|
|event_background|2026|refit_19/outer|0.0885|0.6429|0.1556|278|0.1187|

完整PR、混淆计数、事后P@R85及事件覆盖见evaluation；数据缺失与支持见support.json；实际曝光与损失比例见models。未满足门槛不追加搜索，不替换旧模型。

源内增强前后曲线见validation_curves.svg/.csv，各检查点按同一选择规则确定阈值；无合格点不绘制，不插补。

完整训练阈值段House2/19仅14/4，House2第二内验证块完整段0；最低块支持不能替代独立事件证据。

种子42：5417次实际增强，22807次回退；正槽实际增强比例9.5964%；仅涉及10个不同阈值段，重复曝光不是新事件。

种子1019：5336次实际增强，22888次回退；正槽实际增强比例9.4529%；仅涉及10个不同阈值段，重复曝光不是新事件。

种子2026：5399次实际增强，22825次回退；正槽实际增强比例9.5646%；仅涉及10个不同阈值段，重复曝光不是新事件。

