# Task18：固定扩展来源三模型工程对照（仅准备授权）

用户在根聊天选择“新开并准备3模型方案”。本授权允许数据物化、索引/曝光封存、初始化核验和合成验证；真实训练、候选来源推理/校准与开发评价必须等待新的注册绑定预算回执。旧3fit回执和Task17未授权6fit建议均不可复用。

原Task10 train的336704行、2275正中心、所有原字段字节和顺序不变。仅追加固定House2 UTC [2014-10-14,2014-10-28)与House19 [2014-10-19,2014-11-02)，原始Appliance5/4，Issues=1标签无效，原prepare.resample 8秒hold≤8。缺失保留NaN，31核心至少24观测且无连续2缺失，中心aggregate/标签有限，127上下文不越连续角色边界和缺口。日期完整不等于连续观测，不补值凑覆盖。预期追加144557/404与134661/315，合计615922/2994/612928；不符立即停止查差，不改日期/mask。原House5前缀、验证block4/5及其34351/201支持、checkpoint资格完全不变，不读新House5日期。

窗口ID=(house,source_role,center_time)。旧raw_index/event保持，新增raw_index明确索引各自extra native grid，新增正event以同house旧最大event+1偏移，避免旧/新碰撞。原House5角色从固定block时间映射。索引、角色、事件、初始化、旧模型、源码与配置全部绑定。

只训练expanded候选三种子42/1019/2026，不重训冻结control_guard95（Task14 H95规则/Task12 unit_BCE原3528更新）。原归一化5.370561599731445/.8337093591690063不重估，TCN127/5329参数，原每seed初始化、原event sampler、batch256=16正/240负、Adam .001 CPU4、BCE pos_weight1。每seed6468更新=12checkpoint×539，总19404。负中心期望曝光约2.53263，原约2.53184；这是粗略匹配，不能同时精确匹配正中心、日期和事件曝光。数据和更新量同时改变，不能声称独立证明数据扩展因果。不加背景增强/融合/focal/阈值网格家族，不添加控制fit、不切回3528更新。

三seed deterministic索引完整保存；exposure NPZ为每中心期望/干运行实际计数，CSV按house/role/date/event/正负汇总全部支持及unique覆盖。实际训练后必须逐中心与封存曝光exact。初始化只构建网络比对，不建立优化器、不反传。

执行预算授权是第一个门，在identity、模型、数据、hash读取及输出创建之前完成。新的明确人类回执须绑定experiment、registration sha256、3fit/19404updates、seeds、原话与出处；接口拒绝旧回执。仅单次train，无重启/补训。复用原fit_one/checkpoint worstP/meanP/F1/早epoch和原85% checkpoint资格；终端仅Task14固定L95/H95/radius15校准。

train→source→来源资格通过才freeze（含公共CLI回读）→一次旧Task12 truth cache evaluate→verify→report。source只比较expanded_guard95和冻结control_guard95：每seed每块R≥95且不退化、meanF1≥control、每seed平均R下降≤.05、至少2seed P升；按meanP/F1排序，tie保留control。拒绝只能来源中文报告，0开发读取。通过后冻结唯一来源预选，完整A/W50/control/expanded×3十二格；普通CLI所有格回读exact，原A raw/corrected、其他电器状态、原support均exact。

原137523 mask上一次缓存真值评价：平均P≥.08、各seed R≥.85、meanF1相对W50提升≥.02、至少2seed P和F1同时升、各F1下降≤.02、平均FP下降≥30%、不全开全关、支持exact。独立标量逐点计数与另一套八门槛算术复核，拒绝缺矩阵/重复评价/低R/全关/篡改重hash。报告中文逐seed/来源块/混淆/相对变化/来源选择/八门槛/配对FP片段。运行段MAE/电量/功率只能独立报告，不能虚称改善。

verify/report默认可重复只读，不写旧文件/bytecode。新证据或报告必须显式给全新output路径。准备报告与执行报告分别封存，原Task17 19artifact+manifest20文件hash/mtime不变。本轮House17数值/hash/shape一律不读，最终8/9/20数值/hash/shape一律不读，UI不变。House17反复用于工程开发，不是独立盲测；合成测试通过不代表真实训练改善。
