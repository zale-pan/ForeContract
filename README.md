# ForeContract

[简体中文](README.zh-CN.md)

I started ForeContract with a web testing problem. Change a name in a form and click Save. The browser can tell you the click happened. If you open the record tomorrow and the old name is still there, what did yesterday’s test actually establish?

There was a decision to make *before* clicking: **what would count as a successful save?** Seeing a confirmation toast and reopening the same record to find the new name are two different expectations. Pick one after the action and it’s easy to find an explanation that fits. ForeContract records the expectation early enough for the result to prove it wrong.

### The OFCEVR Loop

The original design names this execution loop **Observe → Forecast → Commit → Execute → Validate → Recover**. A useful starting point for readers familiar with ReAct is the idea that an agent learns from the environment as it acts. ForeContract asks a further question between choosing an action and seeing the next observation: what consequence of this action is the model prepared to predict?

In the Save example, the model sees an existing record (Observe). It proposes saving the change and predicts what it should read on reopening the record (Forecast). The runtime binds the action, expectation and check into a **Contract** for this step (Commit). The browser clicks (Execute), and the verifier compares a fresh observation with the expectation (Validate). A mismatch can inform a retry, a new plan or a decision to stop (Recover). Commit here means making the step’s contract definite. Git commits and user approval for local commands are separate concepts.

I think of this as a way of reasoning about execution. Alongside a proposal for what to do next, the model leaves behind a judgment that the world can contradict. Whether that extra work pays off is an empirical question, and I don’t have evidence of a general advantage over other agent loops.

### What changed when I tried this with code

A saved name is relatively cheap to check. What observation tells you a bug is fixed? During development I saw a test command exit successfully after discovering zero tests. I also made the opposite mistake: I specified one pytest node so narrowly that a viable plan from the model failed my acceptance criteria. In another trial the model supplied the wrong test path. Human review caught these problems. They illustrate why evidence matters and where the product still relies on a reviewer.

These experiences helped me separate three decisions. **Permission** belongs to the user: which workspace, which files, which local command. The **method** can be proposed by the model, including a better test than the one I had in mind. A **claim of success** needs independent support: what actually ran, and how much does its result establish?

As models get stronger, I’m less sure it’s worth prescribing how they should solve a problem. I still want to know what I authorized, and I want something I can inspect when an assistant tells me it’s done. **The more interesting boundary may be whether an outcome can be observed independently at a reasonable cost.** Web UI tests offer a natural place to start. For an open-ended coding task, choosing the observation and deciding who bears its cost are harder. That open question is part of why I want other people to be able to study the framework on its own. I’d particularly like to hear about tasks where expressing a meaningful expectation turned out to be the hardest part.

### Try the application or explore the runtime

The Windows app brings these questions into a coding conversation, with approvals, diffs and test results you can revisit. [ForeContract Desktop](https://github.com/zale-pan/ForeContract-Desktop)

To study the runtime, start small. A scripted planner proposes `mark_done`, a simulated adapter performs the action, and the verifier checks whether a fresh observation reports `done` as `true`. It runs without a model, so you can follow the contract alongside its trace. [Core](packages/core) defines execution and verification, [planners](packages/planners) propose actions, and [adapters](packages/adapters) interact with environments. [Offline example](examples/offline_runtime_demo.py) · [Web runtime contract (Chinese)](docs/WEB_RUNTIME_CONTRACT.zh-CN.md)

## Run your first contract from source

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

## A note from the author

While building ForeContract, I kept wondering which tasks actually benefit from asking an agent to state its expectations up front. I can only try so many on my own. I want to open up the framework so other people can try it on problems they know well.

I’m still learning, and I’m sure I’ve overlooked things. If a contract helped you catch something, or made a simple task harder, I’d love to hear what happened. You can [open an issue](https://github.com/zale-pan/ForeContract/issues) or email me directly.

Zale Pan · <hashuuki@gmail.com>

Copyright 2026 Zale Pan · Apache-2.0. [LICENSE](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).
