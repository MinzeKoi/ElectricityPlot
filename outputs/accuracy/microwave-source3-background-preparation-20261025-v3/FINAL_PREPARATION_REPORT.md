# Task15 完整准备交付：等待额外人类预算

固定来源池可行：House2/19/5模板39/44/131（共214），背景105298/101378/61873（共268549）。模板覆盖的可观察阈值片段仅4/6/6，共16个；214窗口不是214个独立运行周期。127点标签不完整排除2061个原正中心，未知未补零、未插值、未缩窗或放宽角色。当前只能说池可行，不能说增强提高效果。

本阶段真实新fit=0、真实更新=0、真实新模型来源校准=0、House17数值及最终8/9/20读取=0。只创建authorized=false模板；不存在真实新增3fit授权回执。根对话负责后续人类授权，本任务现停止。

v2来源池被完整保留并逐字节复用于v3，没有重构来源池。v1解析失败目录保留（总表列名与标签列名不同）；v2初步报告中不适用于可行池的固定停止文案由v3报告取代。全部完成run源码和权重未修改；共享fit_one仅新增默认None的原始W batch_transform，旧路径行为经回归核对。

实现包含prepare/complete_prepare/train/source/freeze/evaluate/verify/report/predict；真实执行必须持专属microwave_source3_background_v1且绑定本registration的新增3fit/augmentation人类回执。旧Task12训练预算和Task14零fit规则批准均不成立。失败不重启，最多三seed共10584更新。

固定增强在每批原16正中按slot顺序替换前8个，剩余8正240负exact保留；使用独立SeedSequence([150025,seed])，按来源家庭/UTC日→事件→中心抽模板，背景独立均匀抽取。模板全127标签已知；背景mask上标签已知且残差非负；先原W相加再按原log1p常数归一化；背景mask0保持NaN。模板实际曝光与原index曝光分开记录，任何非法配对停止而非fallback。

验证：15项新增合成测试、19项原时间试验回归和13项损失回归共47项通过；另从v3封存源码重跑15项新增测试通过。新增训练链路用全参数不更新的优化器替身；兼容性回归使用生成数组，无真实来源拟合。合成全矩阵测试已走通12格冻结→一次缓存评价，原A全部raw/corrected/其他电器状态exact、137523生成支持mask、重复评价拒绝、不完整冻结拒绝均通过。普通模型加载和选中checkpoint逐值回读通过。不能把合成测试当成真实开发验收。

独立验证已核对来源池计数、模板逐点合法、训练角色范围、冻结索引逐batch重放、三seed初始参数与Task12控制完全一致、访问日志无禁区、注册artifact hashes完整。详见INDEPENDENT_VERIFICATION.json。

来源算法唯一为Task14 L95/H95/radius15，未搜索H85。固定source相对control资格和原W50开发门槛保留。只有来源合格及唯一预选封存后，才允许A/W50/control_guard95/aug_guard95各3seed共12格冻结；随后一次读Task12真值缓存。来源失败完整报告，不读开发数值、不伪造候选。House17未来仍仅重复工程开发，UI默认不切。

训练成本硬界：最多新增3模型、10584更新、CPU4、84672合成正slot；历史Task12同结构三模型训练合计356.308秒（约5.94分钟）。新增合成与日志开销未用真实数据计时，该参考不是耗时上界；没有可据以报价的货币费率，本阶段新增真实训练成本为0。

待根批准后的命令（本阶段没有执行）：

```powershell
.venv-m2/Scripts/python.exe -B "outputs\accuracy\microwave-source3-background-preparation-20261025-v3/run_microwave_source3_background.py" train --run "outputs\accuracy\microwave-source3-background-preparation-20261025-v3" --authorization <Task15明确新增3fit及augmentation人类回执.json>
```

之后依次将train替换为source、freeze、evaluate、verify、report；保持同一--run与--authorization。普通预测为predict，并加--family control_guard95或aug_guard95、--seed 42、--aggregate <总表CSV>、--output <新CSV>。默认UI不切换。

注册SHA256：`4df4d0a63ff32731dd22aa545d7aef56773a6856fbfd20943e5e79c004e728a7`。
