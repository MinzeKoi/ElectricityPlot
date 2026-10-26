# Task15 三来源微波炉模板与独立来源背景：禁用预登记

本阶段完成来源可行性审计和独立可审阅的完整实施准备，训练保持禁用，真实新fit与更新均为0。v2池已通过固定资格，后续快照逐字节复用，不重新构池。Task12额外3fit预算已经用完，本任务未获额外训练授权。历史授权不可用于本任务。准备目录不可覆盖，代码、配置、输入、索引、初始化与来源模型身份逐项绑定；v2/v3旧快照保持封存。

来源沿Task10冻结train.npz的house/t/x/mask/y/event/raw_index。来源2/19仅原14天；House5仅块0/1/2/3，2014-11-15至17、17至19、19至21、21至22日23:30 UTC，末尾1800秒隔离。127点模板须全部在本训练角色内。House5 CSV虽含验证日期，解析器先检查日期角色，再访问功率字段；验证标签变更不能改变数值池。文件身份hash可随验证内容改变。

模板：冻结正中心，中心至少200W，127点8秒对齐真实标签全部有限、非负。背景：冻结关闭中心，原mask每个可观察点微波炉标签已知、非负，aggregate−micro有限非负。缺失点保留NaN。逐点核对原x是CSV原始W。不得补零、插值、平移、缩放、裁剪、扩充日期或借用验证/开发标签。

来源审计覆盖各户/UTC日期/可观察事件ID的支持、功率、覆盖、拒绝原因以及127未知造成的损失。固定池模板214个，对应三户4/6/6共16个可观察阈值片段；背景268549个。窗口重叠，阈值片段不是独立运行cycle。若未来任何绑定/角色约束失效，立即来源拒绝并停止训练入口，不把非法池变为静默fallback。

完整合成与训练适配器已准备：冻结每批16正240负索引；按正slot顺序前8个合成，另8正与240负原样。模板按家庭/UTC日→事件ID→中心均匀抽样；背景独立全池均匀抽样。增强RNG namespace=150025，隔离原BatchStream和初始化。原始W先合成再用原log1p常数归一化；mask来自背景。原index曝光与真实template曝光分开记录。无额外真实标签集合。

固定对照为Task14 guard95三seed，唯一增强族为aug_guard95。未来如获额外明确授权，最多3fit、10584更新，失败不重试；3528/seed、12×294 checkpoint、BCE1、Adam .001、CPU4；原初始化、索引、模型结构、来源标准化与source85选epoch规则保持。只使用Task14 L95/H95/radius15来源校准，禁止H85搜索。来源资格：每seed每块95且非退化、meanF1不低control、各seed平均R最多降5pp、至少2seed P提升；按meanP/meanF1选择，平分control优先。

只有全部来源资格和唯一选择冻结后，才能冻结两组三seed及A/W50完整矩阵并一次读取已有Task12真值缓存。开发验收逐项沿原门槛，详见disabled配置。House17仅工程开发，不是独立泛化；最终8/9/20隔离；普通aggregate-only推理应与冻结exact；UI默认及所有原始功率/其他电器状态不改。

入口：`scripts/run_microwave_source3_background.py prepare --run <新审计目录>`；已封存v2池不再构建，使用`complete_prepare --pool-run <v2目录> --run <新实施快照>`逐字节复用并记录hash。prepare/complete_prepare/verify/report在未训练状态可审阅；train/source/freeze/evaluate/predict必须持本任务新增明确人类3fit/augmentation回执。历史授权和Task14零fit规则授权不能启用训练。disabled模板不是真实回执。来源拒绝不执行开发冻结。

准备完成后固定待执行顺序为train→source→freeze→evaluate→verify→report，全部从新快照入口运行。train复用fit_one默认None可选batch_transform；source复用Task14 calibrate95和原gate函数，不搜索H85。freeze完整12格，evaluate一次读取Task12真值缓存。普通predict只用总表，不做增强。三seed最多3模型/10584更新/84672合成slot；真实训练费用/耗时仅为原同结构三模型量级估计，另有逐batch合成和日志开销。本阶段不跑真实计时训练。

执行后报告必须给出中文来源资格/唯一预选、来源块阈值与逐seed混淆/P/R/F1、正式八门槛、相对control/W50的变化表。仅在完成唯一冻结和一次评价后制作配对FP新增/移除及连续片段表，缺失或8秒时间断裂分段，不用于调参。verify独立标量重数混淆，重算source资格、精确L95/H95和全部八验收门槛，与metrics/summary/配对片段核对。评价保留本地唯一缓存供复核，verify不重新读父Task12真值缓存。来源拒绝报告不得读取开发数值。
