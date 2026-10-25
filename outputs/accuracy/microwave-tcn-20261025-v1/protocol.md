# 微波炉中心TCN与长上下文对照：训练前登记

2026-10-25。准备阶段零真实拟合。新模型默认不启用，原默认行为与原始CSV不变。

固定TCN31/TCN127，各seed 42/1019/2026，最多6次新真实拟合，不重训旧partial_mask。模型2→16点投影；6个单Conv1d(16,16,k3)+ReLU残差块，dilation=1/2/4/8/16/32，对称same padding，残差乘1/√2；投影和每块输出均乘真实观测mask。中心Linear16→32→1，ReLU，末偏置−3。无BatchNorm、Dropout或搜索。感受野127点，8秒采样下首末相隔1008秒，约17分钟；这是使用未来上下文的离线模型。

输入是真实功率log1p按原strict来源统计标准化及观测mask。仅NaN编码为标准化空间0，绝非0W填补。TCN31逐点复用任务5 v2 partial核心集259806中心/849正中心；TCN127同中心扩至±63点，仅允许原训练分区观测。沿两侧遇连续≥2缺失格或时间断裂后截断该方向更远上下文，越界未知。核心31至少24真实点/最多1连缺，不因外圈不足删中心。保存每窗新增点数、比例、左右截断原因及packbits mask身份；0正中心获得新真实上下文则准备失败、0拟合。

两臂每seed使用同一保存的3528×256中心索引数组；16正/240自然均匀负，原事件/日期轮换BatchStream。相同初始化、Adam .001、BCE正权50、12×294更新、CPU4线程。归一化沿用strict来源原统计，不从含NaN或新外圈重算。两臂相同参数数，仅输入长度不同。

House5验证逐字节复制任务4/5固定六个双日块的完整核心31时间点和标签。127外圈不得超出各块边界；缺外圈仍保留点。每个受支持块R≥.85且非退化；单阈值按最差P、均P、均F1，跨轮次同分取较早。候选TCN相对冻结旧partial_mask需均F1不降，各seedR降≤.05，至少2seedP提高。合格旧partial和两新臂按均P/F1排序，同分旧partial/TCN31/TCN127。允许无候选。唯一来源选择在开发标签读取前落盘；不得见开发后改选。

准备只访问House2/19训练与House5验证。House17标签和逐点误报不用于设计。后续先冻结六模型、唯一选择、全部开发预测和旧partial参考，复核普通推理、完整核心31共同支持、原功率/其他电器不变，才允许开发评分。House17核心31不完整仍None；核心完整时外圈不足用mask表示；原W50有效评价点137523不减少。旧partial必须使用旧mask模型加载器，不能把权重装入新网络。

开发验收保持原目标：相对W50均P增≥.08、各R≥.85、均F1增>.02（原浮点比较容差1e-12）、至少2seed P/F1同升、各F1降<.02、FP降≥30%，且非退化。新旧网络比较不能全部归因为上下文；只有TCN31对TCN127为配对上下文对照。旧partial预选仅表示新结构/上下文未胜出。不称功率改善，微波炉功率沿用原A。最终House8/9/20隔离。

独立入口：`.venv-m2/Scripts/python.exe scripts/run_microwave_tcn.py prepare --run outputs/accuracy/microwave-tcn-20261025-v1`。随后使用运行目录封存入口，命令为verify/benchmark/report/train/freeze/evaluate。普通推理：`predict_microwave_tcn.py --model <TCN模型目录> --aggregate <规范CSV> --output <新CSV>`。旧partial使用`predict_microwave_mask.py`。

train需要后续协调消息绑定registration SHA256及maximum_fits=6；evaluate还需绑定freeze SHA256。启动标记与models目录禁止重启、覆盖或隐式增加拟合；失败保留不完整运行。准备不提供训练授权。合成CPU基准固定每臂8个更新、8批推理，不使用来源样本，不据此调参；估计另列验证/序列化/CPU争用开销。源码、原数据、源块、mask、索引、初始化、模型和预测均以哈希封存，已有运行不覆盖。
