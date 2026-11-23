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

## 恢复中断的 wheel 下载

保留原 PowerShell 窗口和虚拟环境。从 pip 的下载输出中找到包名、版本和完整 wheel 文件名，在 [PyPI](https://pypi.org/) 打开对应版本的 **Download files**，将同名文件完整下载到新位置。例如，Windows x64 上的 CPython 3.12 可使用 `cp312-cp312-win_amd64.whl`。以本次 pip 选中的完整文件名为准。出现提示时粘贴文件绝对路径，两端不用加引号。

```powershell
$wheel = Get-Item -LiteralPath (Read-Host 'Full path to the downloaded .whl')
if ($wheel.Extension -ne '.whl') { throw 'Expected a wheel file' }
$dist, $version, $tags = $wheel.BaseName -split '-', 3
if (!$tags -or $dist -notmatch '^[A-Za-z0-9_.]+$' -or $version -notmatch '^[A-Za-z0-9.!+_]+$') { throw 'Invalid wheel filename' }
$release = Invoke-RestMethod "https://pypi.org/pypi/$dist/$version/json" -ErrorAction Stop
$file = @($release.urls | Where-Object { $_.filename -ceq $wheel.Name })
if ($file.Count -ne 1 -or (Get-FileHash -LiteralPath $wheel.FullName -Algorithm SHA256).Hash -ne $file[0].digests.sha256) { throw 'PyPI wheel identity or SHA-256 mismatch' }
& $python -m pip install $wheel.FullName
if ($LASTEXITCODE) { throw 'Wheel installation failed' }
```

本地 wheel 安装成功后，重新运行快速开始中失败的那一行 `pip install`，再按顺序完成余下命令。如果摘要不匹配，重新获取文件后再继续。

Copyright 2026 Zale Pan · Apache-2.0. [LICENSE](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).
