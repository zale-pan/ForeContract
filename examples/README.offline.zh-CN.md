# 离线 Runtime 最小示例（P2-T08）

`offline_runtime_demo.py` 用**真实已安装组件**跑一次闭环成功场景：不联网、不需要密钥、不加载模型，也不启动真实浏览器或 Coding Adapter。它是一次可复查的**离线**演示，不是新的分发物，也不改任何 Core CLI 或通用场景运行器。

## 安装与运行

离线流程只有一条真实路径，且**四步全部发生在仓库外**：外部源码副本 → 外部 `dist` 里的 wheel/sdist → 全新 venv 用 `--no-index` 安装这些 wheel → 用该 venv 的 python 运行示例。仓库内的源树只被**读取和复制**，**从不**在原地 `build`，也**从不**按路径 `pip install` 源码树。

下列命令在仓库根目录执行（复制源与示例的相对路径都相对该目录）；`$outside` 是任意的仓库外目录。

```powershell
# 0. 构建解释器：已装有 build 与声明的后端 setuptools>=84,<85（离线构建不需要联网）
$py = "<仓库之外的虚拟环境>/Scripts/python.exe"

# 1. 仓库外工作目录，并把三个源码树复制成外部副本。
#    setuptools 会往被构建的源码树里写 egg-info/build，所以绝不在原 packages 目录内构建。
$outside = "<仓库之外的临时目录>"
New-Item -ItemType Directory -Force "$outside\src", "$outside\dist", "$outside\demo-out" | Out-Null
Copy-Item -Recurse -Force ".\packages\core"              "$outside\src\core"
Copy-Item -Recurse -Force ".\packages\adapters\fake"     "$outside\src\fake"
Copy-Item -Recurse -Force ".\packages\planners\scripted" "$outside\src\scripted"
# 每个包自带 pyproject.toml（setuptools.build_meta），构建只以该包的副本为源码树。
# 仓库根目录没有 pyproject.toml 或根构建配置，因此外部副本不需要额外的根构建文件；
# 若将来新增根构建配置，必须同样复制到外部副本，而不是回到原地构建。

# 2. 从外部副本构建 wheel 与 sdist，产物只写进仓库外的 dist
& $py -B -m build --no-isolation --wheel --sdist --outdir "$outside\dist" "$outside\src\core"
& $py -B -m build --no-isolation --wheel --sdist --outdir "$outside\dist" "$outside\src\fake"
& $py -B -m build --no-isolation --wheel --sdist --outdir "$outside\dist" "$outside\src\scripted"

# 3. 全新 venv，只从本地 wheel 目录安装，--no-index 保证不会访问网络
& $py -B -m venv "$outside\venv"
& "$outside\venv\Scripts\python.exe" -B -m pip install --no-index `
  --find-links "$outside\dist" `
  --find-links "<你的本地 wheelhouse>" `
  forecontract-core==0.1.0 forecontract-fake==0.1.0 forecontract-scripted==0.1.0

# 4. 用该 venv 的 python 运行示例；输出目录是仓库外的新目录或空目录
& "$outside\venv\Scripts\python.exe" -B .\examples\offline_runtime_demo.py `
  --output-dir "$outside\demo-out"
```

- 第 3 步的两个 `--find-links` 都是本地目录：第一个是上一步构建出的三个 `forecontract-*` wheel，第二个是仓库外的离线依赖 wheelhouse（`pydantic` 及其传递依赖）；如果目标 venv 已经自带这些依赖，第二个 `--find-links` 可以省略。`--no-index` 使安装只能从这些本地目录解析，不会从网络取任何东西。
- 示例**不自动安装依赖**、**不改 `sys.path`**，也不执行任何安装、构建或复制：它只是被安装好的发行版的普通消费者（宿主先安装，示例后运行）。
- 本轮**没有新增分发物**：没有新增或修改任何分发 metadata 与依赖（各 `pyproject.toml`、`requires-python`、`dependencies` 保持只读），示例本身不构成新的 distribution。
- 不带 `--output-dir` 时，示例在系统临时目录下新建一个目录。显式路径会被 `resolve()` 后检查：**指向仓库内部会被拒绝**，已存在的目标只允许是**空目录**，因此不会覆盖既有产物。运行或宿主投影失败时示例**非零退出**，并且**不打印成功摘要**。

## 合成任务目标与预期动作数

- 目标 `demo-goal`：观测状态中的 `done` 必须为真。
- 初始观测 `done=false`，因此必须真实执行**恰好一次**动作 `mark_done`。
- 脚本只提供一个 Forecast，所以 `planning_calls=1`、`steps_started=1`、`attempts_started=1`。
- 预期成功事实：`terminal_status="completed"`、`reason_code="TASK_ACHIEVED"`。

stdout 是**一个 JSON 对象**，顶层字段固定为三个：

| 字段 | 内容 |
|---|---|
| `result` | 真实 `RunResult.model_dump(mode="json")`，含 `run_id`、`terminal_status`、`reason_code` 与各计数/关联字段 |
| `trace_path` | 真实 JSONL Trace 文件路径 |
| `episodic_memory_path` | 宿主投影出的 Episodic Memory 文件路径 |

## 产物如何反查

1. **Trace**：逐行读 `trace_path`。每行是一条真实记录，`sequence` 从 0 递增；预期顺序为 `run_started`、`step_started`、`contract_committed`、`attempt_started`、`attempt_finished`、`verification_recorded`（步骤范围）、`verification_recorded`（运行范围）、`run_finished`。
2. **唯一终态**：`run_finished` 只有一条；它的 `verification_record_id` 必须指向同一文件里那条 **run 范围** 的 `verification_recorded`。
3. **Working Memory**：示例另写 `working-memory.json`，内容是宿主从 `WorkingMemory` **分离读回**的已发布安全状态（不是 Runtime 自己的对象）。
4. **Episodic Memory**：逐行读 `episodic_memory_path`。它由宿主**先用 `TraceSourceResolver` 解析已确认来源、再写入**；来源不可解析时不会产生投影。

## 责任划分

- **Fake**：生命周期（`start`/`stop`）、观测（`observe`）、动作校验（`validate_action`）与执行（`execute`）全部是真实 Fake 在做事。
- **领域绑定**（`DemoDomain`，宿主提供）：`to_evidence` 声明证据是否完整；`prepare_contract` 声明验证绑定与恢复策略；`declare_action_safety` 声明动作的重试安全性；`assess_effect` 只依据拿到的证据报告效果是否已应用。
- **能力声明**：Fake **不**自动声明 `json_rule` 领域绑定，示例用一个**薄代理**在能力声明中加入 `json_rule`，其余调用原样转发给真实 Fake。声明与实际提供的领域支持一致。代理**不读取 Fake 的私有 actual_state**，也**不代替** TaskCompletionOracle 判定任务完成。
- **宿主 Episodic 投影**：Runtime **不**写 Episodic Memory，本轮也**不新增**该耦合或 API。投影完全在宿主侧、来源已确认之后完成。

## 这个成功示例**不能**证明什么

- 不能证明真实 Web/Coding Adapter、真实浏览器、真实文件系统或任何 LLM/模型能力。
- 不能证明通用场景运行器能力：示例只提供**一个**成功场景，没有参数化场景、没有 CLI 子命令体系。
- 不能证明硬实时、同步挂起抢占、强杀下必落盘、断电/OS 隔离或崩溃恢复。
- 不能证明同 Run 续跑或跨进程恢复。
- 不能替代独立验收：本示例是**可运行演示**，不是验收证据。

## 公开测试选择集

公开离线默认命令不直接运行整个 `tests` 目录。下面的完整块在仓库根目录执行，先把冻结的 54 个
文件放入 `$files` 数组，再用 `@files` 传给 pytest；只有一条 `--deselect`。

```powershell
$venvRoot = '<outside-the-repo>/forecontract-venv'
$python = Join-Path $venvRoot 'Scripts/python.exe'
$env:FORECONTRACT_TEST_SCRATCH = '<outside-the-repo>/forecontract-scratch'
$files = @(
  'tests/integration/test_llm_api_smoke.py'
  'tests/integration/test_llm_planner_runtime.py'
  'tests/integration/test_llm_retrieval_evidence.py'
  'tests/integration/test_llm_retrieval_runtime.py'
  'tests/integration/test_local_test_site_http.py'
  'tests/integration/test_offline_runtime_budget_cleanup.py'
  'tests/integration/test_offline_runtime_example.py'
  'tests/integration/test_offline_runtime_recovery.py'
  'tests/integration/test_offline_runtime_success.py'
  'tests/integration/test_offline_runtime_trace_memory.py'
  'tests/integration/test_offline_runtime_verification.py'
  'tests/unit/test_adapter_protocol.py'
  'tests/unit/test_contract_builder.py'
  'tests/unit/test_contract_models.py'
  'tests/unit/test_core_models.py'
  'tests/unit/test_deepseek_provider.py'
  'tests/unit/test_episodic_memory.py'
  'tests/unit/test_fake_adapter.py'
  'tests/unit/test_json_rule_verification.py'
  'tests/unit/test_llm_planner.py'
  'tests/unit/test_llm_planner_codec.py'
  'tests/unit/test_llm_planner_lifecycle.py'
  'tests/unit/test_llm_planner_models.py'
  'tests/unit/test_llm_retrieval_codec.py'
  'tests/unit/test_llm_retrieval_lifecycle.py'
  'tests/unit/test_llm_retrieval_models.py'
  'tests/unit/test_llm_retrieval_planner.py'
  'tests/unit/test_llm_retrieval_reader.py'
  'tests/unit/test_local_test_site.py'
  'tests/unit/test_playwright_web_adapter.py'
  'tests/unit/test_playwright_web_observation.py'
  'tests/unit/test_recovery_engine.py'
  'tests/unit/test_recovery_models.py'
  'tests/unit/test_recovery_verifier_binding.py'
  'tests/unit/test_runtime_contract_hooks.py'
  'tests/unit/test_runtime_engine.py'
  'tests/unit/test_runtime_fake_flow.py'
  'tests/unit/test_runtime_finalization.py'
  'tests/unit/test_runtime_models.py'
  'tests/unit/test_runtime_state.py'
  'tests/unit/test_runtime_timing.py'
  'tests/unit/test_scripted_planner.py'
  'tests/unit/test_task_completion_oracle.py'
  'tests/unit/test_trace_models.py'
  'tests/unit/test_trace_store.py'
  'tests/unit/test_web_capabilities.py'
  'tests/unit/test_web_models.py'
  'tests/unit/test_web_observation_target_budget.py'
  'tests/unit/test_web_rules.py'
  'tests/unit/test_web_runtime_hooks.py'
  'tests/unit/test_web_runtime_support.py'
  'tests/unit/test_web_validation.py'
  'tests/unit/test_web_verification.py'
  'tests/unit/test_working_memory.py'
)
& $python -B -m pytest -q -p no:cacheprovider --import-mode=importlib @files `
  --deselect=tests/integration/test_llm_retrieval_evidence.py::test_the_demo_never_touches_the_governance_event_log
```

- 被排除的节点依赖内部治理事件日志，公开目录中不存在该日志；
- 旧的两个安装测试不在公开集内；Web 组单列并需要单独安装的 Chromium；
- 未安装浏览器的机器上，Web 用例应如实报告**未运行**，不得写成通过。
