# 微波炉背景增强接口及复用约束

稳定实现：`virtual_submeter.accuracy.microwave_background`。任务3运行的实际执行源码位于`outputs/accuracy/microwave-background-20261025-v1/source`，其登记哈希优先于工作副本。新增候选只有event_background，任务3不扩大训练时间段；下游若用来源全量数据，须另行登记数据范围、归一化和预算，不能称沿用了任务3时间留出的独立验证。

`make_pools(data, mappings, bounds) -> (pools, support)`仅做可用性构建，无训练或模型分数输入。`data`逐行对齐的字段为：`x` float32[N,31]原总表窗、`y` float32[N]中心微波炉瓦数、`t` int64[N] UTC秒、`house` int64[N]（0=refit_2、1=refit_19）、`event` int64[N]原始阈值段身份。原批次采样还可保留`raw_index`。`mappings[house][timestamp]`是训练区域微波炉功率，缺失应不存在或NaN；不得填0。`bounds[house]=(low,high)`为允许的半开训练区间，所有31点上下文必须在内。

池中`power`为原目标31点功率，`residual=x-power`，`template_ok`只允许原正中心，`background_ok`只允许31点全部低于200W的关闭窗，`reason`逐窗说明拒绝原因。非有限、负功率或负残差不合格。下游需把已审核的原始总表/目标CSV哈希按House2/19、aggregate/microwave顺序写入字符串数组`pools['source_hashes']`。冻结npz池和源数据哈希后才能训练，禁止看分数修改池。

`BatchStream(data,'event',seed,plan).next()`返回原始256个索引；`BackgroundStream(data,pools,seed,plan).apply(indices,update)`返回替换背景后的float32[256,31]及8条尝试记录。两对象分别持有独立随机流；必须每步原采样一次、增强一次，更新从1递增。plan沿用固定采样配额16/240，`augmented_positive_per_batch=8`、`augmentation_rng_namespace=310025`。不要为增强失败重抽索引；不完整模板保持原样，原始中心标签不变。`summary()`给出实际比例、回退数、不同阈值段和户对。

背景优先跨户且跨UTC日，然后跨户，再跨日、同日；只在第一个非空支持层内轮换。每个合成窗使用同一原正中心完整目标功率，没有振幅、开机定义或时间改变。数万次合成仍只有原阈值段的重复曝光。

归一化由新登记范围的真实训练窗估计一次；任务3具体沿用任务2归一化。增强后调用既有`tensor_input(meta, augmented_x)`，模型和BCE50/Adam不变。验证/推理永远不增强，模型保存普通`microwave_state_classifier_v1`元数据及weights.pt，沿用`load_state_model`、`predict_windows`和`predict_state_rows`。

任务3实际支持：75,222个训练窗、236个正中心；130个完整31点模板中心（House2/19为105/25），69,151个关闭背景窗（56,083/13,068）。模板来自10个不同阈值段（7/3），其中原审计完整段8个（5/3）；其余2个仅部分可观测。House2/19原训练完整段总数14/4。源内House2第二块完整段0，不能把最低块支持门槛当充分独立事件支持。

任务3真实干跑按种子42/1019/2026，28,224次尝试各成功5,417/5,336/5,399次，回退22,807/22,888/22,825次；正槽实际增强率9.5964%/9.4529%/9.5646%，均实现跨户。后续复用只保证接口，不保证新范围产生相同比例或效果。

独立复核入口`python scripts/verify_microwave_background.py --run <run>`要求先训练完成并冻结，重新从原训练CSV重建全部合格池，逐条重算合成加和，并对照任务2封存采样器及实际event曝光。只能新增复核回执，不能覆盖旧报告或修补冻结权重。
