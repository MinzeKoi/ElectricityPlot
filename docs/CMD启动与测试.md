# Windows CMD 启动与测试

## 启动

双击项目根目录的 `start.cmd`。脚本会定位项目目录，按 `.venv-m4`、`.venv-m2`、`.venv` 的顺序选择第一个存在的 Python 环境，检查依赖，然后启动 ElectricityPlot 并尝试打开浏览器。无需先激活环境，项目路径含空格、中文也可运行。

保持窗口开启，访问 `http://127.0.0.1:8501`。停止时在窗口按 `Ctrl+C`；如果 CMD 询问是否终止批处理，输入 `Y`。

在项目根目录的 **CMD** 中也可运行：

```bat
start.cmd
```

仅检查环境（不启动服务、不暂停，成功返回 0，失败返回 1）：

```bat
start.cmd --check
echo %ERRORLEVEL%
```

8501 端口被占用时，可传入 Streamlit 参数覆盖端口，此时访问 `http://127.0.0.1:8502`：

```bat
start.cmd --server.port 8502
```

没有虚拟环境时，先安装 Python 3.12，再在项目根目录执行（安装依赖需要联网）：

```bat
py -3.12 -m venv .venv-m4
.venv-m4\Scripts\python.exe -m pip install -r requirements-m4-cpu.lock
.venv-m4\Scripts\python.exe -m pip check
start.cmd --check
```

脚本不会自动安装依赖或训练模型。若优先选中的环境损坏，按错误提示修复该环境。

## 测试流程

以下以当前本机已存在的 `.venv-m2`、M2 模型和 M1 协议为例。其他机器需要先准备对应产物；源码检出不包含数据和模型。若脚本选用了其他环境，自动化测试命令也要改用同一个 Python。

| 步骤 | 操作 | 通过标准 |
|---|---|---|
| 1. 环境检查 | 执行 `start.cmd --check`，随后执行 `echo %ERRORLEVEL%` | 显示 `Environment check passed.`，退出码为 0 |
| 2. 双击启动 | 双击 `start.cmd`，打开默认地址 | 显示 ElectricityPlot 标题，没有 Python 导入异常 |
| 3. 基本分析 | 模型目录填写 `outputs/m2-20261020/seq2point`；选择“已有家庭记录”；数据协议填写 `data/processed/refit_m1_protocol/protocol.json`；选 House 3 测试区间；点击“开始分析” | 出现历史曲线、覆盖率和未解释负荷，无“分析未完成”错误；未填校准目录时，区间不可用是正常结果 |
| 4. 交互 | 切换设备、回放进度及显示时长 | 图表随选择更新；缺失时段保留空段 |
| 5. 下载与评价 | 下载 HTML、CSV、JSON；点击“生成分表评价报告”，再次下载 HTML | 文件可打开，预测文件含对应设备数据；评价成功后显示 MAE、F1 等指标 |
| 6. 错误处理 | 将模型目录改成不存在的路径 | 明确提示模型不可用，先前分析结果清除；恢复正确路径后可重新分析 |
| 7. 停止与重启 | `Ctrl+C` 停止，再次运行脚本 | 服务可正常重启，默认端口可重新使用 |
| 8. 换端口 | 停止服务后执行 `start.cmd --server.port 8502` | 8502 可访问 ElectricityPlot |

自动化回归测试在项目根目录的 CMD 中执行：

```bat
if not exist tmp mkdir tmp
set "TEMP=%CD%\tmp"
set "TMP=%TEMP%"
.venv-m2\Scripts\python.exe scripts/project.py validate-config
.venv-m2\Scripts\python.exe -m unittest discover -s tests -v
echo %ERRORLEVEL%
```

配置检查应成功，测试最终应为 `OK`、退出码为 0。记录运行数量和跳过数量；如有跳过，不能算相关功能已通过。界面测试会在 `outputs/ui/` 生成新的分析结果，不覆盖既有模型和实验。

## 本次验证记录（2026-10-20）

- 当前 `.venv-m2` 环境检查、配置检查通过。
- 通过 CMD 脚本在 18501 端口实际启动，健康检查返回 `200 ok`，网页返回 HTTP 200；验证后已停止测试服务。
- 完整自动化回归：91 项测试全部通过，无跳过，耗时约 49 秒。
- 上表中的浏览器手工操作作为验收流程提供，本次未逐项人工点击验证。
