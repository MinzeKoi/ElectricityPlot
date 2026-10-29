# Task26 实施准备状态

本目录只含隔离的候选输入组件、合成测试、禁用预算及原来源身份核对。不是训练入口，也不是可部署模型；真实拟合、候选推理、来源阈值选择及开发评价均未执行。

`bind_source.py` 校验原三户池等注册哈希并核对所有逐批索引；`source_bindings.json` 保存结果。测试用 `PYTHONPATH=src;task26_centered` 与 `.venv-m2/Scripts/python.exe -m unittest discover -s task26_centered -p test_model.py -v`。新增通道与模型细节见 `docs/MICROWAVE_NEXT_IMPROVEMENT_DESIGN_20261029.md`。

仍待完成：真实训练及持久化接线、检查点来源选择、完整时间轴门控、条件开发与普通推理入口，以及端到端合成测试和独立复核。`registration.disabled.json` 不是执行授权或完整注册；没有获得新增训练预算之前不可启动真实训练。
