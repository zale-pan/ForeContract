# ForeContract

[English](README.md)

我最早做 ForeContract，是想让一个 Agent 测网页。比如在表单里改一个名字，点“保存”。浏览器很容易告诉我们按钮点到了。可要是产品第二天重新打开，名字还是旧的，昨天那次测试究竟测到了什么？

这个小例子里，有一个很容易被跳过去的决定：**点按钮之前，我们认为怎样才算保存成功？** 如果关心的是数据存下来了，那么“看到了成功提示”和“重新打开同一条记录，读到了新名字”就是两种不同的预期。等到点击之后才挑一种解释，几乎总能把刚刚发生的事说圆。ForeContract 的做法是把预期提前写下来，再给真实结果一次推翻它的机会。

### 这套设计叫 OFCEVR Loop

原始设计给这个执行循环起了一个名字：**Observe → Forecast → Commit → Execute → Validate → Recover**。如果你熟悉 ReAct，可以从它们共同关心的事开始理解：模型要一边行动，一边从环境中获得新的信息。ForeContract 想再往前问一步：在这次行动和下一次观察之间，模型愿意对什么结果作出可以检查的预测？

在刚才的例子里，模型看到的是一条已有记录（Observe）。它提出保存，并预测重新打开后应读到什么（Forecast）。运行时把动作、预期和检查方式一起确定下来（Commit）；这份绑定就是这一步的 **Contract**。浏览器随后执行点击（Execute），验证器拿新的观察与预期比较（Validate）。如果不一致，这次偏差会成为后续重试、重新规划或停止的依据（Recover）。这里的 Commit 指运行时确定本次契约，和 Git 提交无关；涉及写文件或本机命令时，用户审批又是另一件事。

我愿意把它称为一种面向执行的推理思路：模型除了提出“接下来做什么”，还要留下一个可能被事实证伪的判断。这是本项目探索的设计，并没有实验证明它对所有任务都比其他 Agent 循环有效。

### 从网页走到写代码，问题变难了

网页里的名字通常可以重新读一次。换成编程助手，“修好了”究竟对应哪个观察？开发中我遇到过一个测试命令，退出码是 0，发现的用例数也是 0。我也走过另一个极端：我把验收写死为一个 pytest 节点，模型明明提出了可行的验证方案，还是过不了我定的条件。后来还有一次，模型写错了测试文件路径。这些问题是在人的审批和复核中发现的。拿它们说明为什么需要证据很合适；把它们算作框架已经能够自动识别的错误，就夸大了实现。

这几次经历让我把三件事分得更清楚。**权限**来自用户：让 Agent 在哪个工作区动手、能改哪些文件、是否同意运行这条本机命令。**方法**适合留给模型：它可以选择实现，也可以提出比我预想更合适的测试。**完成的依据**则要经得起独立核对：至少能说清实际跑了什么，那个结果支持多大的结论。

模型越强，我越怀疑把“具体怎么做”规定得很细是否值得。与此同时，我仍然希望知道自己批准了什么，也希望在助手说“完成了”时有东西可以回看。**真正值得探索的边界，也许是结果能否以合理成本被独立观察。** Web UI 测试是很自然的起点。对于一个开放式的编码任务，怎样选择观察、由谁负担验证成本，我还没有通用答案。这也是我想把框架单独开放给其他人研究的原因。如果你尝试过一个场景，发现最难的部分恰好是写出有意义的预期，我很想听听那个例子。

### 想体验产品，还是研究运行时？

桌面应用把同一个问题放进真实的编程会话：人可以先讨论，决定动手后再看审批、Diff 和测试结果。[ForeContract Desktop 中文介绍](https://github.com/zale-pan/ForeContract-Desktop/blob/main/README.zh-CN.md)

研究运行时可以从一个很小的例子开始。脚本规划器提出 `mark_done`，模拟适配器执行动作，验证器重新读取状态，检查 `done` 是否为 `true`。整个例子不调用模型，适合把 Contract 和执行轨迹对着读。再往下，[Core](packages/core) 定义运行和验证的语义，[规划器](packages/planners) 提出动作，[适配器](packages/adapters) 负责与环境交互。[离线示例](examples/offline_runtime_demo.py) · [Web 运行契约](docs/WEB_RUNTIME_CONTRACT.zh-CN.md)

## 从源码运行第一个契约

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

## 作者的话

我做 ForeContract 时一直很好奇：让 Agent 提前说清楚预期，到底在哪些任务上值得？一个人能试的场景有限，我想把框架开源，让别人也能拿自己的问题试一试。

我自己还在学，很多地方考虑得不够周全。如果你试过后觉得某个契约帮上了忙，或者反而让一件简单的事变麻烦了，我都很想听听。 你可以在 [Issues](https://github.com/zale-pan/ForeContract/issues) 留言，也可以直接写信给我。

Zale Pan · <hashuuki@gmail.com>

Copyright 2026 Zale Pan · Apache-2.0. [LICENSE](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).
