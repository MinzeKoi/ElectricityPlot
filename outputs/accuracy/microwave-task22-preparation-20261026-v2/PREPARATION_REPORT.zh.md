# Task22 容量对照实施准备报告

状态：准备封存，默认关闭，等待新的绑定预算。真实fit 0；优化器构建0；更新0；反传0；House17与最终8/9/20数值、shape、hash读取0。

注册 SHA256：`85bb7de48e8167fb043805e7f37c6ae65bee5191d026dc541b6fea02b74263ef`

唯一未来变量为TCN通道16→32，参数5329→19809。project、六个膨胀conv及head首层输入宽度变化；核3、膨胀1/2/4/8/16/32、残差公式、head隐藏32、中心读出不变。
原训练336704窗/2275正、原House5prefix与block4/5验证34351/201、C127、原归一化/mask/200W、unitBCE/Adam .001/CPU4固定。三seed42/1019/2026每个3528步(12×294)，16正/240均匀负，原3528×256完整索引逐行重放exact。未使用Task21困难采样、Task18追加数据、背景增强、融合或新特征/损失。
旧16通道三seed初始化与Task10/Task12历史逐张量exact；旧控制在原批边界来源普通预测逐点exact。新32通道按同规则/同seed可重复，19809参数，完整127训练mask前向与严格core31公开推理支持验证；新旧张量不声称相同，不裁剪或拼接旧权重。
已实现显式channels接口，复用原fit_one训练引擎；旧缺省16及旧元数据行为保持。32通道模型元数据必须明确tcn_channels=32/parameter_count=19809，不能按缺省16加载。
执行入口为本目录run_microwave_task22.py。训练、来源筛选、freeze、evaluate和诊断均先验证新的Task22注册SHA回执；缺失或Task21旧回执在训练数据、模型训练、优化器/反传前拒绝。真实执行强制从本次不可变source快照加载；数据/旧控制身份继续核对，不依赖未来工作区源文件不变。
来源仍按原85保护、worstP/meanP/F1/早epoch选择完整36checkpoint；复核读取完整event及日期边界。L95/H95/r15与Task14 control_guard95固定；每seed每块R≥.95非退化、meanF1≥control、每seed meanR下降≤.05、至少2seed P提高。失败则0devread。
通过后唯一候选wide32与A/W50三seed完整九格冻结，公开普通推理exact后，仅单次原137523有效点开发验收。八门槛为meanP≥.08、各seedR≥.85、meanF1比W50≥+.02、至少2seed P/F1同升、各seedF1下降≤.02、meanFP减少≥30%、非全开全关、支持与其他设备原状态exact。不得开发后改选、改宽、延训；最终8/9/20隔离、默认模型不切换。
新增diagnostics入口在训练后可对固定选中模型及旧控制的原完整训练/来源支持与共同strict_core31两视图报告AP、FPR、固定点混淆；只解释、不调阈值或改选，来源失败仍不读开发。AP用同分分组的第二实现复核。
完整合成测试构建独立静态训练支持及带event/跨UTC日期边界的来源数组，用fakefit写出3seed×12检查点、实际格式批日志/初始化/曝光。真实走train→source_select→freeze→evaluate→verify/report全部分支；verify/report本身未mock，包含两套checkpoint数值算法、L95/H95、独立单调deque时域最大、标量混淆与开发八门槛。静态九格有137523个生成点，不含真实House17。
合成拒绝路径覆盖缺预算/旧预算、缺九格矩阵、低Recall/全关、重复评价、严格core31缺失、错误32宽度元数据；fixture全程禁止优化器/反传。这里只fakefit，不运行任何真实或合成优化训练。Task21工作区复核器遗漏event已修复，Task21 v2封存文件未修改。
独立实现复核在本对话完成：原采样使用另一套deque消费与独立日期/事件分组；新旧初始化和模型结构单独重建，来源日期/事件隔离重查。未新增subagent；协调者可做外部核验。
精确逐文件白名单、输入hash/mtime、源数据/控制/代码身份、完整批矩阵、新初始化、中文报告和测试日志均封存。future_identity只从既有身份元数据登记未来路径，未打开、hash或解析对应开发文件；不读取混有开发结果的旧报告。
完成目录不可覆盖；已有完整prepare转只读verify。finalize外部只读回执记录两遍live与snapshot prepare/verify/report共12次及所有输入/交付hash和mtime不变。

待预算：最多3个新模型、总10584更新仅提议；必须另行由用户批准并绑定本注册SHA。Task21预算已花完，不能复用。
