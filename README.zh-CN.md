# ForeContract

[English](README.md)

ForeContract 是一个 Python 实验框架，用契约约束模型提出的动作，并验证动作产生的效果。Core 提供执行、验证、追踪与记忆。六个包包括 Core、Fake／Web／Coding 三个适配器和 Scripted／LLM 两个规划器。桌面客户端由独立项目 ForeContract-Desktop 提供。

## 运行第一个契约

在 Windows x64 安装 Python 3.12，打开 PowerShell，进入框架仓库根目录。示例使用三个包、脚本规划器和模拟环境。只有安装依赖需要联网。

```powershell
$fw = (Resolve-Path .).Path
$run = Join-Path $env:LOCALAPPDATA ("ForeContract-demo-" + [guid]::NewGuid())
py -3.12 -m venv "$run\venv"
$python = "$run\venv\Scripts\python.exe"
$env:PYTHONPATH = ''
& $python -m pip install setuptools==84.0.0
if ($LASTEXITCODE) { throw 'Build dependency installation failed' }
& $python -m pip install --no-build-isolation "$fw\packages\core" "$fw\packages\adapters\fake" "$fw\packages\planners\scripted"
if ($LASTEXITCODE) { throw 'Framework installation failed' }
& $python -B "$fw\examples\offline_runtime_demo.py" --output-dir "$run\demo"
if ($LASTEXITCODE) { throw 'Demo failed' }
```

成功时，结果 JSON 中的 `terminal_status` 为 `completed`，`reason_code` 为 `TASK_ACHIEVED`，并给出执行轨迹和情景记忆文件的位置。Web 自动化需要另装浏览器。原生 Coding 适配器面向 Windows。示例覆盖一次确定性动作，通用编码任务仍处于实验阶段。

Copyright 2026 Zale Pan · Apache-2.0. [LICENSE](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).
