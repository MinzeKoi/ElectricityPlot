# Task15 v4：固定来源背景增强完整实施准备

截至2026-10-26，本阶段已完成，等待根对话取得明确额外训练授权。**真实新fit 0，真实参数更新0，新模型来源校准0，House17开发数值/预测/指标/真值读取0，最终8/9/20读取0。** 没有创建真实授权回执；authorized=false。本报告及v4取代v3实施准备版本，v2池与v3全部原文件保持封存。

## 来源池与局限

|来源|原正中心|合法127点模板|可观察模板片段数|合法背景|因127点未知淘汰正中心|
|---|---:|---:|---:|---:|---:|
|House2|466|39|4|105298|427|
|House19|383|44|6|101378|339|
|House5|1426|131|6|61873|1295|
|合计|2275|214|16|268549|2061|

214个模板窗来自16个可观察阈值片段，窗口重叠，不能当214个独立事件或16个独立运行周期。本次只证明固定池可行，不能证明增强有效。v4逐字节复用封存v2池，没有重新构池或重新解析原CSV；各户日期/事件分组、功率/覆盖、拒绝矩阵和源索引/窗hash完整保留。模板全127点真实微波炉标签有限非负且本训练角色合法；背景只在原mask=1位置取已知标签并要求aggregate−micro有限非负；mask=0仍NaN。没有补零、插值、clamp、缩窗、扩日或借用验证/开发标签。

## 完整接口与固定行为

prepare/complete_prepare/train/source/freeze/evaluate/verify/report/predict均已实现。每batch原16正slot按原顺序前8个合成，剩余8正和240负exact保留。模板按来源家庭/UTC日→事件→合法中心分层均匀抽取，背景独立均匀；增强RNG为SeedSequence([150025,seed])，不串原BatchStream或初始化RNG。先原始W合成，再使用原归一化。非法配对停止，无fallback。原index曝光和实际模板/背景曝光分别记录。

复用已有fit_one、checkpoint选择和普通模型加载。共享fit只增加默认None的batch_transform，旧None/50/unit路径经回归保持exact。真实训练最多3个seed模型、每seed3528次更新、12×294 checkpoint、BCE1、Adam .001、CPU4，原控制不重训、失败不重试。只用Task14 L95/H95/radius15，未加入H85家族搜索。完整来源资格和唯一预选在开发冻结前锁死；来源拒绝则完整中文报告，禁止进入开发读取。

执行后中文报告包含来源资格、逐seed L/H与块混淆/P/R/F1、相对无增强control变化、唯一预选、正式八门槛，以及A/W50/control_guard95/aug_guard95全部12格结果。配对误报表给出相对control和W50的移除/新增FP、恢复TP、新增FN及P/R/F1变化；连续片段表遇未知/无效mask或8秒时间断裂即分段，只在冻结与唯一缓存评价完成后生成，不能用于调参或改选。

verify除hash外，独立标量逐点重数混淆/P/R/F1，重放精确L95/H95及来源相对资格，重算原全部八验收门槛，与metrics/summary及配对片段表相互比对。evaluate只读一次已有父Task12真值缓存，保留本地唯一副本供后续独立verify，不重开父缓存。冻结与回读要求原A全部raw/corrected/其他电器状态及137523有效mask exact，普通aggregate-only推理exact，UI默认不切。

## 验证结果

- 新增合成测试19项通过；从v4封存源码重跑同19项通过。
- 原时间试验兼容性回归19项、原损失配方回归13项通过，共51项不同测试。
- 新增强训练链路使用全参数不更新的optimizer spy，验证配额、原W公式、掩码、RNG隔离、初始权重、曝光分离和普通模型加载/选中checkpoint回读。旧兼容性回归仅使用生成数组，无真实来源拟合。
- 生成的完整12格矩阵走通冻结→一次评价→独立标量复核→中文报告；可拒绝重复评价、不完整冻结、hash一致但数字错误的summary、错误来源混淆，连续片段缺口分段通过。
- 根指出的生产JSON问题已经修复：qualification_details全部显式转换为Python bool，来源测试真实走过JSON写入和精确阈值/资格重算。另一报告测试夹具已补齐执行阶段标记。两次失败均未产生真实模型或开发候选。
- 独立准备验证核对池计数、127点模板及角色、16个可观察片段、逐batch冻结indices重放、三seed与Task12控制初始参数完全相等、注册hash和访问日志，无禁区读取。
- 实际v4 CLI的train/source/freeze/evaluate/predict五入口在未授权时均前置拒绝，准备文件完全不变。旧授权不能代替Task15额外预算。
- v2注册19个artifact、v3最终清单124个artifact复核无任何差异，旧快照未改。

合成验证不等于真实开发验收。House17未来若读取仍只是重复工程开发，不称独立泛化。

## 待批准固定预算与命令

所需人类授权固定为：**Task15来源模板背景增强，额外最多3个新模型，seed42/1019/2026，共10584次更新，84672个合成正slot，不重训控制，不失败重试。** 必须绑定本v4 registration并有明确人类引用；Task12已用完的3fit和Task14零fit规则批准均不能启用。

历史同结构Task12三模型训练总计356.308秒，约5.94分钟；增强与逐batch日志开销未实测，此数不是耗时上界。当前新增真实训练费用为0；没有可据以估算货币费用的费率。资源上限为上述3模型/10584更新/CPU4。

批准后唯一待执行训练命令（本阶段未执行）：

```powershell
.venv-m2/Scripts/python.exe -B "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-source3-background-preparation-20261025-v4/run_microwave_source3_background.py" train --run "D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-source3-background-preparation-20261025-v4" --authorization <Task15明确新增3fit及augmentation人类回执.json>
```

之后同一入口和同一授权依次运行source、freeze、evaluate、verify、report；源码中每一步均有前置硬门。普通predict需指定--family control_guard95或aug_guard95、--seed、--aggregate和新--output。

准备目录：`D:\system position\document\真正的文档\AIC\outputs\accuracy\microwave-source3-background-preparation-20261025-v4`。

Registration SHA256：`e2dc9fdf165bff1e57dc9456bf6880b2e79d372213de863857fc3f10157f3535`。

审阅入口：PREPARATION_REPORT.md、INDEPENDENT_VERIFICATION.json、BUDGET_GATE_VERIFICATION.json、TEST_VERIFICATION.json、source/完整代码、disabled配置/回执模板、身份/池审计/访问日志及FINAL_PREPARATION_MANIFEST.json。
