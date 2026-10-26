# Task17 可重复只读入口

此目录补充封存的 `microwave-source-event-coverage-20261026-v1`，没有修改原目录任何文件。入口默认只读、仅向标准输出展示结果，代码内及命令均禁用 bytecode，不创建报告、缓存或验证结果文件。

从项目根目录运行：

```powershell
.venv-m2/Scripts/python.exe -B outputs/accuracy/microwave-source-event-coverage-readonly-20261026-v1/readonly.py verify
.venv-m2/Scripts/python.exe -B outputs/accuracy/microwave-source-event-coverage-readonly-20261026-v1/readonly.py report
.venv-m2/Scripts/python.exe -B outputs/accuracy/microwave-source-event-coverage-readonly-20261026-v1/readonly.py report --link
```

`verify` 验证固定 manifest 身份、原20个文件及源码依赖哈希，并重新汇总逐日、逐片段计数，与封存 summary 和 independent_verification 相互核对。它是快照复核，不重扫原始数据，也不冒称重新完成原始区间积分审计。原始数值独立复核已在原目录完成并封存。

`report` 核对原报告身份后原样展示中文报告；`--link` 只输出绝对路径 Markdown 链接。两种入口均检查运行前后封存文件的 SHA256、mtime_ns 和大小未变化。

验证证据在 `readonly_validation.json`：verify、report、report --link 各连续执行两遍，共6次成功。原20个已完成文件的哈希、修改时间和大小全部不变，未生成 __pycache__。证据文件由独立交付检查保存，入口本身不写文件。

本补充不改变日期、计数、训练协议或禁用提案，0训练、0推理、0校准、0开发评价。
