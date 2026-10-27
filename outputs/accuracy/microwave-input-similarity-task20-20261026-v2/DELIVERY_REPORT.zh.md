# Task20：冻结模型输入相似性零训练诊断

本轮已完成真实数值分析、独立实现的逐配对复算、固定顺序曲线和封存。没有新增拟合、优化器更新、监督探针、阈值或checkpoint搜索，也没有修改模型输出或默认模型。

## 结论与下一步

在533,565个核心31点完整窗口中，完整127点标准化功率与掩码共形成533,330种有效输入。只有1组完全重复（236窗，235次逐元素确认）；该组均为关闭标签、完整127点无缺失且未跨UTC日，未发现中心200W开关标签相反的精确重复。这不能排除未重复输入的标签或时间对齐问题。

在固定跨日期参考库内，FP与FN的最近正/负距离比中位数均大于1；它们的已观测输入通常仍有更近的同户关闭参考。FP与TP的距离分布也存在接近范围。它们是模型错误来源的线索，不能把一个中位数或近邻相似解释为物理不可辨识、标签错误或可直接提升模型的规则。

在相同完整127点查询与参考的对照中，127点没有表现出对全部错误类别一致的距离比变化；正确TP的距离比中位数还从31点略小于1变成127点大于1。因而本轮不支持“直接扩大上下文就能解决”的结论。距离比只是描述量，未用来构造分类器或计算新验收成绩。

正负参考候选数明显不均衡：负参考每层上限4096，正参考上限256，原支持也不同。更多负候选会影响最近距离；R>1不能直接判定输入可分/不可分，也不是新分类准确率。127点RMS增加背景后改变距离几何，未一致改善该描述量，不能推出网络无法利用长上下文。

**下一轮优先建议：先提出受控的训练/表示诊断，而不是据此大范围改标签。** 可证伪假设为：固定这些错误片段和既有输入后，错误主要来自当前学习/表示不足；先用本轮固定曲线片段复核输入与中心200W标签的对应关系，若发现系统时间错位或标签异常，则推翻训练优先顺序，转为输入/标签修复。任何新增训练或扩大原始标签读取范围须另行授权具体方案与预算；本轮没有执行。

## 固定口径与可复现身份

输入仅限config.json的逐文件白名单：Task19六模型18份冻结概率、阈值/身份/manifest，以及Task18 train、两块validation、normalization和原输入转换源码。未打开模型权重、其他epoch或其他实验成绩。Task19 manifest只核验白名单成员及56项头部，不把“未读取的56项全部重新哈希”当作已完成。报告身份固定为Task19 SHA256 fe5fd289e3129ef7eeb3f99c6e717d93d018aac7335483085edfd3e46a03d58c。

House17、最终House8/9/20未读数值、形状或hash；不增加原始数据读取。control参考仅原336704窗训练前缀（历史3528更新）；expanded参考来自完整615922窗训练角色（历史6468更新）。expanded是此前来源被拒模型，未晋升。追加日期只对control未训练；House5后缀曾参与选模，不能称盲测或独立泛化。

strict31 = 原mask[:,48:79].all；模型输入始终保留原127点功率和127点掩码。功率严格复用float32 (log1p(x)-5.370561599731445)/0.8337093591690063；缺失为功率通道0与掩码0。距离用已冻结功率通道的直接float64逐点RMS，不重估尺度、不减均值、不填缺失。独立合成验证与原tensor_input源码的双通道输出完全相同。

稳定ID为H{物理户号}|{原角色}|{UTC秒}|raw{原行号}，按UTF-8 SHA256升序抽样。同户/真实训练角色/标签参考每层最多256正、4096负；查询按family/seed/house/role/UTC日期/TP、FP、FN、TN每层最多32，错误类别仅用于描述分层。参考与查询同日禁配、中心间隔小于1016秒禁配；无参考为null，不跨户补齐。距离相同按参考stableID字典序。每次最多8个查询×256个参考×127点直接分块，不保存全量平方矩阵。

注册配置和源码先冻结，之后支持清单、参考/查询样本再冻结，最后才计算距离。common31与common127均从已经冻结的strict31列表中筛出完整127点查询和参考，不重新抽样或补样；两个模式的候选完全相同。明细另保存common31选中参考在127点的距离，支持相同配对检查。

## 支持、覆盖与缺失

共650,273个原窗口，strict31支持533,565（82.05%），完整127点支持456,448（70.19%）。抽取31,875条查询记录和60,680条参考记录；记录跨seed/family重复，不是独立物理事件数。

固定查询中完整127点有24,896/31,875（78.11%），不完整的6,979条保持null，未补样。strict31最近正参考可配31,875条；common127可配24,896条。共交付191,250条“查询×模式×参考标签”明细，其中27,916条为明确null。common31与common127都保留不完整查询的空行，以避免隐去覆盖损失。

下表按原house/role聚合，不跨seed或模型重复计数。完整UTC日期、各错误类别原支持、抽样量和完整127覆盖见support_and_coverage.csv；计数不把窗口或可观察片段当独立开机事件。

|house|role|原窗|strict31|完整127|
|---|---|---|---|---|
|19|extended14d|134661|132381|121329|
|19|original14d|132059|56856|54268|
|2|extended14d|144557|141329|129745|
|2|original14d|127747|93213|58047|
|5|house5_prefix_block_0|18899|18063|15628|
|5|house5_prefix_block_1|20716|20518|19237|
|5|house5_prefix_block_2|20714|20641|19767|
|5|house5_prefix_block_3|16569|16213|14623|
|5|house5_validation_block_4|19127|19127|15466|
|5|house5_validation_block_5|15224|15224|8338|

## 固定查询的距离分布

下表跨seed描述性汇总，查询按日期/类别等分层抽样，不能外推为总体发生率；相同窗口可出现在不同seed中。d+、d−为参考库内最近正/负距离的各自中位数；R是逐查询d+/d−的中位数，不能用两个中位数相除代替。

|family|类别|查询n|RMS31 d+|RMS31 d−|R31|
|---|---|---|---|---|---|
|control|FP|5431|0.5743|0.4287|1.2079|
|control|FN|519|0.2638|0.1975|1.4592|
|control|TP|4011|0.4797|0.5384|0.9580|
|control|TN|6067|0.9738|0.0203|42.9742|
|expanded|FP|5361|0.4516|0.3621|1.2302|
|expanded|FN|390|0.2326|0.1950|1.2826|
|expanded|TP|4030|0.4083|0.4854|0.9325|
|expanded|TN|6066|0.8777|0.0186|42.5402|

共同完整127子集对照（同查询、同候选参考；每种长度仍各自找最近参考）：

|family|类别|共同子集有效n|R common31|R common127|
|---|---|---|---|---|
|control|FP|4245|1.2109|1.2285|
|control|FN|448|1.4630|1.2770|
|control|TP|2982|0.9928|1.1735|
|control|TN|4881|39.0740|7.8035|
|expanded|FP|4136|1.2061|1.1750|
|expanded|FN|308|1.2779|1.2261|
|expanded|TP|3011|0.9429|1.0947|
|expanded|TN|4885|41.4101|8.6757|

R在127点的变化不是上下文增益或新分类准确率证据。RMS长度改变会重分配原31点与外围点的权重；参考库是有限抽样而非所有可能输入。任一层完整127支持不足或没有合格参考时只报告null，不改口径。

## 跨日期与角色的错误来源线索

所有参考均来自不同UTC日。下表进一步保留house/role区分；原日期、追加日期和House5前缀/后缀不能混成相同训练角色。表中n为strict31抽样错误窗记录，非事件数；细至每个UTC日、seed的分位数见group_distributions.csv。

|family|house|role|类别|n|R31|R common31|R common127|
|---|---|---|---|---|---|---|---|
|control|19|extended14d|FN|50|1.3884|1.4463|1.2975|
|control|19|extended14d|FP|1344|1.1714|1.1728|1.1260|
|control|19|original14d|FN|8|1.1429|1.1429|1.2069|
|control|19|original14d|FP|584|1.2136|1.2257|1.1795|
|control|2|extended14d|FN|101|1.4400|1.6691|1.6301|
|control|2|extended14d|FP|1220|1.2539|1.2173|1.4372|
|control|2|original14d|FN|16|0.8736|1.7246|1.7206|
|control|2|original14d|FP|1149|1.2381|1.2843|1.3256|
|control|5|house5_prefix_block_0|FN|109|7.9796|8.1405|1.9943|
|control|5|house5_prefix_block_0|FP|192|1.1729|1.2130|1.3489|
|control|5|house5_prefix_block_1|FN|27|1.0219|1.0698|1.1626|
|control|5|house5_prefix_block_1|FP|192|1.1533|1.1434|1.2309|
|control|5|house5_prefix_block_2|FN|103|1.2948|1.2924|1.1481|
|control|5|house5_prefix_block_2|FP|192|1.0806|1.0642|1.3689|
|control|5|house5_prefix_block_3|FN|79|1.4484|1.1337|1.2884|
|control|5|house5_prefix_block_3|FP|192|1.3333|1.3138|1.4045|
|control|5|house5_validation_block_4|FN|7|1.1324|1.5006|1.0384|
|control|5|house5_validation_block_4|FP|192|1.0991|1.1251|1.1260|
|control|5|house5_validation_block_5|FN|19|1.7561|null|null|
|control|5|house5_validation_block_5|FP|174|1.2644|1.5569|3.8835|
|expanded|19|extended14d|FN|18|1.1509|1.1270|1.2177|
|expanded|19|extended14d|FP|1298|1.1728|1.1642|1.1123|
|expanded|19|original14d|FN|15|0.9735|0.9738|1.1523|
|expanded|19|original14d|FP|585|1.1417|1.1667|1.1144|
|expanded|2|extended14d|FN|26|1.2061|1.2033|1.1541|
|expanded|2|extended14d|FP|1207|1.3372|1.3362|1.2321|
|expanded|2|original14d|FN|33|0.8500|0.8429|1.2141|
|expanded|2|original14d|FP|1144|1.3710|1.2149|1.3161|
|expanded|5|house5_prefix_block_0|FN|126|7.9796|8.2446|1.9752|
|expanded|5|house5_prefix_block_0|FP|192|1.2299|1.2464|1.3584|
|expanded|5|house5_prefix_block_1|FN|26|1.0257|1.0295|1.1626|
|expanded|5|house5_prefix_block_1|FP|192|1.2178|1.2608|1.3019|
|expanded|5|house5_prefix_block_2|FN|80|1.2695|1.2560|1.1409|
|expanded|5|house5_prefix_block_2|FP|192|1.1573|1.1489|1.5363|
|expanded|5|house5_prefix_block_3|FN|27|1.4512|1.0749|1.2807|
|expanded|5|house5_prefix_block_3|FP|192|1.2776|1.2501|1.4077|
|expanded|5|house5_validation_block_4|FN|15|1.0900|1.0394|1.0969|
|expanded|5|house5_validation_block_4|FP|192|1.1350|1.1667|1.1319|
|expanded|5|house5_validation_block_5|FN|24|1.8329|null|null|
|expanded|5|house5_validation_block_5|FP|167|1.1880|1.5204|3.8835|

参考库会随family真实训练角色改变；control和expanded的距离差异同时包含参考覆盖差异，不能归因于训练收益。日期与缺失分布也不是随机分配，分组差异仅是工程线索。

## 固定配对曲线

按预先冻结的query SHA顺序，每family/类别最多2条；FP选最近正参考、FN选最近负参考、TP/TN选同标签参考。曲线显示原功率W与完整127点、缺失断线及核心31区间。正例按原event ID，关闭窗按可观察连续片段去重展示；统计计数保持原窗口口径。图示只是“冻结参考库内最近”，不代表全局最近或完整物理事件。

|family|类别|seed|查询中心UTC秒|曲线|
|---|---|---|---|---|
|control|FP|2026|1412322744|[00_control_FP.svg](curves/00_control_FP.svg)|
|control|FP|42|1413403408|[01_control_FP.svg](curves/01_control_FP.svg)|
|control|TP|42|1416833472|[02_control_TP.svg](curves/02_control_TP.svg)|
|control|TP|42|1412186008|[03_control_TP.svg](curves/03_control_TP.svg)|
|control|FN|42|1416506240|[04_control_FN.svg](curves/04_control_FN.svg)|
|control|FN|2026|1413309600|[05_control_FN.svg](curves/05_control_FN.svg)|
|control|TN|42|1412322744|[06_control_TN.svg](curves/06_control_TN.svg)|
|control|TN|42|1414148408|[07_control_TN.svg](curves/07_control_TN.svg)|
|expanded|FP|1019|1412322744|[08_expanded_FP.svg](curves/08_expanded_FP.svg)|
|expanded|FP|42|1413360368|[09_expanded_FP.svg](curves/09_expanded_FP.svg)|
|expanded|TP|42|1416833472|[10_expanded_TP.svg](curves/10_expanded_TP.svg)|
|expanded|TP|42|1412186008|[11_expanded_TP.svg](curves/11_expanded_TP.svg)|
|expanded|FN|1019|1416506240|[12_expanded_FN.svg](curves/12_expanded_FN.svg)|
|expanded|FN|1019|1416337504|[13_expanded_FN.svg](curves/13_expanded_FN.svg)|
|expanded|TN|42|1412322744|[14_expanded_TN.svg](curves/14_expanded_TN.svg)|
|expanded|TN|42|1414148408|[15_expanded_TN.svg](curves/15_expanded_TN.svg)|

## 独立复核与封存

独立复核器没有复用主分析的抽样/近邻/身份实现：重新构造全部支持与SHA前K清单，重新遍历所有合格参考核对最近身份和eligible_n，再以math.fsum标量逐点重算全部163,334条非空交付距离及common31选中配对的127点距离；最大距离绝对误差4.44e-16。核对原行号、UTC、家庭、真实训练角色、原200W标签、冻结概率和阈值、错误类别、无同日/重叠、null和共同候选，全部通过。

7组针对性合成检查通过：分块对标量、跨块全局稳定并列与全不合格null、同日/重叠/跨户/不完整阻断、完整功率与掩码重复语义、与原转换源码完全一致、模型矩阵缺项拒绝、文件与目录不可覆盖。测试中导入主代码仅用于构造被测对象；真实数值复核使用独立实现。

过程说明：v1在读取validation不存在的source_role字段时停止，未产生抽样/距离；v2仅按Task19固定角色赋值修复，v1未覆盖。最初启动过独立review子代理，收到根对话约束后已停止；主对话接管并审阅、修正、运行分离的复核实现。独立性指实现与数值重算，不宣称盲态外部审计。没有扩大路径扫描。重复补充清单首次已成功写出，但复制自身源码被白名单阻断，留下0字节duplicate_inventory.py；保留该失败工件，修复为在审计前读取自身源码，重算确认清单完全相同后另存duplicate_inventory_source_v2.py，未覆盖原输出。

全部48个白名单输入在注册、主分析前后、独立复核前后与封存前后逐文件核对SHA256/size/mtime_ns。最终manifest绑定所有交付配置、脚本、数据身份、抽样、明细、图、报告和独立复核结果。verify为只读，复验完整文件清单以及全部hash/mtime；完成目录以exclusive创建，禁止重跑覆盖。完成后的两遍只读复验记录另存同级receipt，避免修改已封存目录。

## 交付索引

- `config.json` / `registration.json`：规则、授权来源与源码hash。
- `inputs_before.json` / `inputs_after_analysis.json` / `independent_inputs_after.json` / `inputs_after_finalization.json`：原输入只读身份。
- `support_and_coverage.csv` / `reference_sample.json` / `query_sample.json` / `samples_frozen.json`：原支持及固定抽样。
- `pairs.csv`：完整191250条配对、时间/行号/标签/概率/掩码/距离及null。
- `group_distributions.csv`：8760条预定义分组分布。
- `exact_duplicates.json` / `exact_conflict_members.json` / `exact_repeat_groups.json`：完整有效输入重复、日期/掩码明细与冲突。
- `curve_index.json` / `curves/*.svg`：固定顺序、片段去重的16条曲线。
- `independent_verifier.py` / `independent_verification.json` / `independent_synthetic_tests.json`：独立实现、逐对复算与测试。
- `diagnostic.py` / `finalizer.py` / `delivery_manifest.json`：冻结源码与封存清单。

本轮没有提高预测效果的交付结论。精确冲突未发现、跨日期近邻接近或127距离改变，都不能证明标签正确/错误、物理不可分或已实现泛化收益。
