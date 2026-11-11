# Web Runtime 示例运行说明（P3-T05）

本目录的两个示例在**固定本地测试站**上运行真实 Web Runtime 闭环：

| 文件 | 作用 |
| --- | --- |
| `web_runtime_demo.py` | 装配真实 Runtime / Builder / ScriptedPlanner / PlaywrightWebAdapter / Web hooks / Web RuntimeDomainSupport / WebStepVerifier / WebTaskCompletionOracle / RecoveryEngine / Trace / Memory，运行八种 profile |
| `web_runtime_evidence.py` | 仅宿主使用的纯转发记录代理与 `evidence-index.json` 旁文件；不改变 Core trace 格式 |

示例不访问生产站点，不使用 API key，不做远端发布，也不执行任意 JS / Shell / Python。

## 运行前

- 使用已配置好的 Python 环境（Playwright 1.64.1、Chromium 156.0.8078.4 / rev 1248）。
- `test_site`、`forecontract-core`、`forecontract-web`、`forecontract-scripted` 可从外部受控副本导入；示例本身不修改 `sys.path`。
- 输出目录必须是**仓库外的新目录或空目录**；示例拒绝覆盖非空目录或写入仓库。

## 命令

```console
python examples/web_runtime_demo.py --profile normal --output-dir <外部新目录>
```

`--profile` 只接受以下八种之一（默认 `normal`），`--output-dir` 为必需参数。成功时 stdout
只输出一个 JSON 对象，恰好五个字段：

```json
{"result": null, "terminal": {"terminal_status": "cancelled", "reason_code": "USER_CANCELLED", "source": "last_finalization"},
 "trace_path": "...", "evidence_index_path": "...", "working_memory_path": "..."}
```

`result` 只在真实返回 `RunResult` 时是完整 JSON，否则为 `null`；`terminal.source` 严格为
`run_result` 或 `last_finalization`。没有返回的取消执行在旁文件记录 `call_interrupted`，
不伪造 `ActualOutcome` 或终结事件。

退出码：

| 退出码 | 含义 |
| --- | --- |
| 0 | 真实 `completed`，且证据写入与资源收尾成功 |
| 2 | 正常运行但任务未完成（含预期 aborted / timeout）；argparse 参数错误也沿用 2 且不执行 |
| 130 | 取消经 Core 传播到宿主后记录并以 130 退出 |
| 1 | 设施、证据、目录或资源收尾失败 |

故障 profile 符合预期仍保留 2 / 130，不把演示符合预期当作任务完成。

## 八种 profile

| profile | 站点/宿主时序 | 期望结果 | 请求/提交计数 |
| --- | --- | --- | --- |
| `normal` | NORMAL | completed / TASK_ACHIEVED | login=1, commits=1；4 Steps/4 Attempts |
| `wrong_credentials` | NORMAL，固定错误合成密码 | aborted / POLICY_ABORT | login=1, auth_failure=1, commits=0；4/4 |
| `delayed_release` | DELAYED_LOGIN，LOGIN_RECEIVED 后确认零提交再 release | completed / TASK_ACHIEVED | 释放前 0 提交，最终 1/1；4/4，不重放 |
| `delayed_deadline` | DELAYED_LOGIN，不 release，Run deadline 先到 | timeout / DEADLINE_EXCEEDED | login=1, commits=0；4/4，LOGIN_RECEIVED 先发生 |
| `overlay_blocked` | BLOCKED_SUBMIT，保持遮挡 | aborted / POLICY_ABORT | login=0, commits=0；4/4 |
| `overlay_released` | 上一 Run 完成清理后显式新 Run；idle reset 新代并 release | completed / TASK_ACHIEVED | 新代 1/1；4/4 |
| `commit_unknown` | COMMIT_THEN_STALL，不 release；Adapter 超时先到 | aborted / POLICY_ABORT，同时 Recovery EFFECT_UNRESOLVED | login=1, commits=1；submit outcome=unknown；4/4，零重放 |
| `cancel_after_received` | DELAYED_LOGIN；LOGIN_RECEIVED 后取消 Run task | CancelledError 传播；last_finalization cancelled / USER_CANCELLED | login=1, commits=0；4 Steps/4 Attempts 已启动 |

四步脚本固定为 `navigate /login`、`fill username`、`fill password`、`click submit`。每步最多
一个 Attempt。全部 profile 使用冻结的标准恢复政策 `bounded`、`max_retries=0`、
`allow_replan=False`；预算也保持冻结值（普通 Run 30s / operation 3s / 站点 request 15s，
`delayed_deadline` 为 15s / 25s / 30s）。任务成功只由独立的 `WebTaskCompletionOracle` 依据宿主
criteria（dashboard URL + welcome 标题）判定。

唯一两个新增的可选配置是 `PlaywrightWebConfig(..., observation_target_timeout_ms=None)` 与
仅关键字的 `PlaywrightWebConfig(..., *, startup_timeout_ms=None)`：`None` 都保留既有行为。
`observation_target_timeout_ms` 只有 `commit_unknown` 显式声明 500ms；它只给**单个目标整组**一个
**绝对 deadline**，同时覆盖该组 native 读取与必要的取消收尾；内部只为收尾预留该额度内的有界部分，
不重置、不追加第二个窗口，也不延长整次 observation 或任何冻结预算。收尾在额度内确认时，局部
无法完成的目标返回既有 `unavailable`/`TARGET_QUERY_FAILED`（count 与属性为 `None`），不伪造
成 0 匹配；收尾无法在额度内确认时保留晚任务所有权并让整次 observation 失败。适配器自身的
普通取消与观察到的 `SessionLost`、内部协议缺陷、`KeyboardInterrupt`/`SystemExit` 严格区分，
后者传播而不是被写成 `TARGET_QUERY_FAILED`；成功快照返回前重新校验会话可用性。

`startup_timeout_ms` 是**仅启动**额度：八个 profile 都显式声明 `10000`。它只约束 `start` 的
**一次**浏览器资源获取，使用一个绝对 deadline，不重置阶段、不重试、不预热、不复用旧会话。
`None` 时保持历史行为（由 `operation_timeout_ms` 约束 start）。它不改变 `observe`/`execute` 的
operation 额度、单目标额度、Run 额度、站点 request 额度或 cleanup 额度，也不要求大于或小于
operation 额度。配置校验与其他 timeout 相同：只接受有限正 JSON 数值，拒绝 bool、字符串、零、
负数、NaN/Inf；伪造或改写的槽位在重校验时以固定 `WEB_BROWSER_CONFIG_INVALID` 拒绝，不带原始值
或异常链。

注意时长语义：`diagnostics.boundaries[].elapsed_ms` 是**整个宿主调用边界**的时长；`start` 失败时
它还包含有界的失败清理，因此**不能**据此宣称原生浏览器启动本身耗时多少毫秒，也不证明原生启动的
根因。`10000` 是拟议配置，不是从任何一次失败时长推导出的保证值；真实冷启动仍须独立验收，首次
失败照实保留，不自动加额度、不预热、不重试到绿。

### 两处实现说明

- **commit_unknown 保留两个真实事实**：Adapter 在 3s operation 预算内先返回 `unknown`；提交后的
  真实观察仍受冻结的 3s operation 约束，但每个目标只有 500ms，因此它在预算内返回一个真实新
  快照（停滞页面的目标通常为 `unavailable`）。Runtime 用该真实新快照判步骤 FAIL，Recovery
  请求的第二次真实观察同样有界返回，`effect_state=unknown`、`safe_to_repeat=false`，恢复得到
  `EFFECT_UNRESOLVED`；Runtime 终态同时为 `POLICY_ABORT`。两者并存而非二选一，计数仍是一次
  提交、零重放。Run 30s / operation 3s / 站点 request 15s 全部保持冻结值，冲突只报告、不调预算。
- **Recovery 验证器身份是通用配置**：`RecoveryEngine(step_verifier_name="web_step_verifier")`
  显式声明宿主实际组合的验证器；Core 不再硬编码旧标签，也不从判词自动学习身份。真实
  `web_step_verifier` 结果因此原生可用，示例**不重写任何标签**，默认 JSON/Fake 路径仍为
  `json_step`。该通用绑定由 `tests/unit/test_recovery_verifier_binding.py` 独立证明。
- **criteria 启动前预检**：宿主显式注入 `WebTaskCriteriaValidator`，Runtime 在 trace 写入、
  `start` 与 `observe` 之前完成全部 Web 规则（含 optional）预检；非法规则在资源启动前被拒绝。

## 输出文件

每个 Run 的输出目录包含：

- `trace.jsonl`：既有严格 Core JSONL trace，格式不变；
- `working-memory.json`：Runtime 发布的 safe Working Memory 状态（含 observe 调用序号）；
- `evidence-index.json`：真实 run/step/contract/attempt/event/observation 关联，每份快照的
  相对路径、原字节 SHA-256 与字节数，并与真实 trace / Working Memory 交叉核对；验证器拒绝
  缺清单/伪造清单/多余文件/重复序号/错误终态/错误关联，正常启动前失败的真实零快照必须有
  trace 终态依据；索引还带 `diagnostics`：真实 adapter `start`/`observe`/`execute`/`stop` 边界
  耗时与允许的固定错误码（在 Core 包裹之前）、宿主 `LOGIN_RECEIVED` 等待、声明阶段的
  login/auth/commit/active 计数、adapter/Runtime cleanup 事实。只记录封闭码表与固定计数器，
  不记录任意异常文本；Core trace schema 不变；
- `snapshots/`：完整 `Observation`、`Forecast`、已提交 `Contract`、`ActualOutcome`、步骤/任务
  验证及恢复输入/结果的 JSON 快照（Observation 以每 Run 调用序号关联，重复
  `observation_id` 仍可区分）。

`overlay_released` 的上一 Run 位于 `previous-run/` 子目录，主结果只描述显式新 Run。

## 边界

- 只证明固定本地 fixture 与真实 headless Chromium 上的下列闭环，不证明生产可靠性、所有网站、
  LLM 规划或跨进程恢复。
- 示例不会修改 `docs/**` 以外的产品文档；`docs/WEB_RUNTIME_CONTRACT.zh-CN.md` 是本任务唯一
  产品说明。
- 记录代理没有环境决策权；真实底层被调用且返回的内容原样转发。序列化/写入失败显式非零，
  不打印成功摘要，也不吞掉原始运行失败。
- 真实浏览器冷启动可能仍超出可用的启动额度。每个 profile 显式声明 `startup_timeout_ms=10000`，
  该额度**只**约束 `start` 的一次资源获取；若真实 start 仍超时，`diagnostics` 保留实际 `start`
  边界、`WEB_BROWSER_START_FAILED` 与真实耗时。该 `elapsed_ms` 是整个调用边界时长，含失败清理，
  不是原生启动耗时证据，也不证明根因；首次失败照实报告，不放宽任何其他预算、不自动预热、不重试
  到绿。

## 浏览器依赖与本地测试边界

浏览器自动化需要**单独安装** Chromium 构建，不属于离线最小安装。先设置两个变量：

```powershell
$venvRoot = '<仓库之外的虚拟环境>'
$python = Join-Path $venvRoot 'Scripts/python.exe'
$env:PLAYWRIGHT_BROWSERS_PATH = '<仓库之外的浏览器目录>'
& $python -m playwright install chromium
```

- 八个 profile 的复现命令见 `examples/web_*_demo.py` 与对应测试，运行时需要已安装的 Chromium；
- 该目录与版本必须在运行前显式声明，不得从个人路径隐式继承；
- 进程隔离沿用已验证的 Windows Job 机制，不使用会影响当前会话的控制台中断路径；
- 未安装 Chromium 时，`tests/integration/test_web_runtime_trace.py` 的浏览器用例应如实报告
  **未运行**，不得写成通过；
- 本地测试边界：所有输出、截图与浏览器缓存都必须位于仓库之外。
