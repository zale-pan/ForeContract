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

## Recover an interrupted wheel download

Keep the same PowerShell session and virtual environment. Find the package name, version and full wheel filename in pip’s download output. Open the corresponding release on [PyPI](https://pypi.org/), select **Download files**, and save the matching wheel to a new location. For example, CPython 3.12 on Windows x64 can use a `cp312-cp312-win_amd64.whl` wheel. Use the exact filename selected by pip. Paste its full path without surrounding quotes when prompted.

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

After the wheel installs, repeat the failed `pip install` line from the quick start, then run the remaining lines in order. If the digest check fails, obtain a fresh copy before continuing.

Copyright 2026 Zale Pan · Apache-2.0. [LICENSE](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).
