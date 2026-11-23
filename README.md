# ForeContract

[简体中文](README.zh-CN.md)

ForeContract is an experimental Python framework for running a model-proposed action under a contract and checking its effect. Core supplies execution, verification, tracing and memory. The six packages include Fake, Web and Coding adapters, plus Scripted and LLM planners. The desktop client lives in the separate ForeContract-Desktop project.

## Run your first contract

On Windows x64 with Python 3.12 and PowerShell, open the framework repository root. This example installs three packages and uses a scripted planner and a fake environment. Only dependency installation needs a network connection.

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

Look for `terminal_status: completed` and `reason_code: TASK_ACHIEVED` in the JSON result. The output also gives the paths to the trace and episodic memory. Web automation needs a separate browser installation. The native Coding adapter targets Windows. The example exercises one deterministic action; general coding reliability is still experimental.

Copyright 2026 Zale Pan · Apache-2.0. [LICENSE](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).
