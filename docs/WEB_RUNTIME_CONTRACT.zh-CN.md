# Web Runtime 合同（P3-T05，本地 v0.1 候选）

状态：本地可复现候选说明；不是生产可靠性、全站点、LLM 规划或跨进程恢复证明。

本文描述 Web Runtime 的产品接口；历史轮次名称仅用于区分接口演进。独立验证范围及限制见 [发布验证说明](RELEASE_VALIDATION.zh-CN.md)。

## 1. Core：可选的合约校验 hooks factory

`RuntimeEngine.__init__` 新增唯一可选关键字 `contract_hooks_factory: object | None = None`。
原参数顺序、返回模型、终态、错误码集合和默认 JSON/Fake 路径不变。

`forecontract.runtime.protocols.ContractHooksFactory` 是结构协议：

```python
class ContractHooksFactory(Protocol):
    def create(self, *, evidence: VerificationEvidence) -> ContractValidationHooks: ...
```

- 未提供 factory 时，构建合约与执行前重检继续各自使用
  `RuleValidationHooks(state_complete=current_evidence.state_complete)`。
- 提供 factory 时，两个边界分别使用**各自当前证据**创建 hooks，不跨阶段缓存完整性或观察：
  1. `RuntimeEngine._build`（Builder 提交前）；
  2. `RuntimeEngine._check_preconditions`（执行前重检）。
- 返回对象必须提供既有四个同步方法：`validate_precondition`、`validate_expected_outcome`、
  `validate_verification_binding`、`check_precondition`；不是新 Builder 接口。
- factory 收到的证据是分离副本；Runtime 在调用返回后同时比较副本与自身保存证据的完整
  签名，覆盖标量真实类型、映射键类型、嵌套容器、未知/隐藏存储字段和显式完整性声明。
- 执行前重检的 `check_precondition` 也使用分离的 condition/observation，并在调用前后比较
  副本与保存值；可注入实现不继承默认 hooks 的“可信”假设。
- 合法返回必须是 `VerificationStatus`；缺方法、错误返回类型、awaitable、普通属性读取/调用
  异常均被明确拒绝。BaseException（取消/中断）按既有传播语义处理，不转成 PASS/FAIL。
- malformed factory/return → 既有 `DEPENDENCY_FAILURE`；输入或保存证据被修改 →
  `EVIDENCE_INVALID`；Builder 域拒绝沿用 `INVALID_DOMAIN_PAYLOAD` 等映射。stage 仅为
  `build` 或 `check_preconditions`，不新增自由文本 reason code。
- 回调返回后立即消耗既有 Run 单调 deadline；返回时已过期的 build 不提交合约，已过期的
  前置重检不返回 PASS，因此不会执行。预算不重置，cleanup/finalize 边界不变。
- factory/hook 是同步、无 I/O 的受信任扩展；不承诺抢占任意同步挂起。
- **Task criteria 始终预检**：无论是否注入 factory，Runtime 都在 trace 写入、`start` 与
  `observe` 之前预检 criteria。默认继续用 JSON-rule 的 `prevalidate_criteria`；宿主也可注入
  独立的 `TaskCriteriaValidator.validate(*, criteria: TaskCompletionCriteria) -> None`，由它
  验证领域规则。预检只决定 criteria 是否可用，不评价任务是否完成、不返回 verdict、不调用
  task oracle；非法 criteria → `INPUT_INVALID`，缺方法/普通异常/awaitable/非 None 返回 →
  `DEPENDENCY_FAILURE`，修改输入 → `EVIDENCE_INVALID`，BaseException 保持取消/中断身份。
- 本节新增的 factory/hooks 与 criteria 入口在异常处理块之外抛固定设施错误：固定诊断、既有
  错误码与 BaseException 语义保留，普通错误的可达 `__cause__`/`__context__` 链不携带原始文本。

Core 仍为 45 个源码成员，包根导出仍只有 `__version__`；Core 不导入 Web/Playwright/DOM。

## 2. Web：合约校验 hooks

`forecontract_web.runtime_hooks`（新子模块，不加入根 19 导出）：

```python
WebContractValidationHooks(*, state_complete: bool)
WebContractHooksFactory()  # create(evidence=...) 读取该证据的显式完整性
WebTaskCriteriaValidator()  # validate(criteria=...) -> None，复用同一 criteria/五规则解析
```

- 四方法匹配既有 `ContractValidationHooks`。五类规则沿用 `web.url_match`、
  `web.element_visible`、`web.element_hidden`、`web.text_contains`、`web.value_equals`。
- 规则解析与判定复用既有 `WebRuleEvaluator`/`_prepare_rule`；不复制更宽松解析器，不读取活页，
  不只验 required 项。
- verification binding 固定 `verifier_type="web_rule"`、`config={}`；范围/唯一性仍由既有
  Builder 与 `WebStepVerifier` 检查。
- `validate_*` 成功返回 `None`；`check_precondition` 返回 PASS/FAIL/INCONCLUSIVE。非法结构/
  未知类型/错误 binding 走既有 `InvalidActionPayloadError` 域拒绝，不泄漏原始普通异常。
- 合法缺字段证据仍按旧规则得到 INCONCLUSIVE；结构损坏与缺字段分开；显式 false 完整性不会
  被强制为 true。无 I/O、无 browser import、无回调执行能力。
- `WebTaskCriteriaValidator` 复用既有 criteria 结构重验与五规则解析，验证包括 optional 在内的
  全部条件；非法规则转换为 Core 既有 `VerificationError(VERIFICATION_INPUT_INVALID)`，由
  Runtime 在资源启动前映射为 `INPUT_INVALID`。它只做预检，不评价任务完成。

## 3. Web：RuntimeDomainSupport

`forecontract_web.runtime_support`（新子模块，不加入根 19 导出）提供
`WebActionPolicy`、`WebRuntimeDomainSupport`、`WebRuntimeSupportError`。

`WebActionPolicy` 是冻结、严格、未知字段拒绝的记录，恰七个字段：

1. `action: ActionSpec`（六种真实 Web 动作之一的合法 payload）；
2. `preconditions: list[Precondition]`；
3. `retry_safety: read_only | idempotent | non_idempotent | unknown`；
4. `applied_when: list[ExpectedOutcome]`；
5. `not_applied_when: list[ExpectedOutcome]`；
6. `recovery_policy: RecoveryPolicy`；
7. `timeout_ms`：非 bool、有限正数毫秒，不转换数字字符串。

两组 effect 条件可为空；空表示“不提供该判断”，不是空集 PASS；非空组每项必须是 required
Web 规则。嵌套 Core/Web 记录从实际存储字段重验，不用 `model_dump` 丢弃注入字段。

`WebRuntimeDomainSupport(*, adapter_name, required_targets, require_url, action_policies)`：

- 配置分离保存；`adapter_name` 非空；`require_url` 严格 bool；targets 唯一；策略非空且动作
  不重复；不接受动态回调/脚本/regex/网页对象/oracle 句柄。
- 动作匹配是整个 `ActionSpec` 的类型敏感数据匹配（含 `metadata`），不以 action_type 名称兜底，
  1/True、1/1.0、隐藏字段或自定义相等都不能误匹配；缺失策略明确拒绝。
- 固定错误码仅 `WEB_RUNTIME_SUPPORT_INVALID`、`WEB_RUNTIME_POLICY_NOT_FOUND`；只带固定诊断
  与有界结构位置，无原始异常 cause/context/marker。
- `to_evidence`：完整校验 `web.snapshot` 并携带本次实际 Observation；`state_complete` 仅由
  显式 required URL（已取得）与 required targets（均有合法 `collected` 记录）决定；属性为
  null 仍按旧规则给 INCONCLUSIVE。
- `prepare_contract`：匹配 `forecast.proposed_action`，返回策略的前置条件、恢复政策、timeout
  及每个 expected_outcome 恰好一个 `web_rule`/空 config 绑定；保持顺序与 optional 项，不修改
  候选动作/预期结果；提交仍由 Builder 独立决定。
- `declare_action_safety`：重验真实已提交合约、adapter_name 与策略，返回绑定
  contract_id/adapter_name/action_type 的声明；不从执行状态或动作名推断安全。
- `assess_effect`：仅用传入 evidence 与策略。先独立严格验证完整 evidence 及其 Observation，
  再判断两个 effect 组，因此空组不会绕过证据检查；合法无声明仍为 `unknown`，损坏结构是固定
  设施错误而不是 unknown。非空 applied_when 全 PASS 且 not_applied_when 未全 PASS →
  `applied`；反向 → `not_applied`；两组都全 PASS、都未全 PASS 或无判断 → `unknown`。不会因
  outcome=failed/unknown 或仍在登录页就断定未提交；不据此直接判定任务成功。
- 所有方法纯同步、无 I/O，不读私有站点状态/宿主计数器，不调用 Adapter，不读活页，不写
  trace/文件。任务成功只由独立 `WebTaskCompletionOracle` 与宿主 criteria 决定。

## 4. 八种真实 profile

### 4.1 可选单目标读取上限（`observation_target_timeout_ms`）

`PlaywrightWebConfig` 新增可选字段 `observation_target_timeout_ms=None`；原四个位置参数与
默认调用不变。`None` 完整保留既有行为：每个目标只受整次 observation 的 operation deadline
约束。非 None 只接受既有配置规则认可的有限正数（拒绝 bool、字符串、NaN/Inf、零/负数），且必须
小于 `operation_timeout_ms`；构造与 `revalidated_config` 都验证并分离保存，未知字段继续拒绝，
默认 `repr` 不变。

- 该上限是**一个目标的唯一绝对 deadline**：它同时覆盖该目标整组 native 读取（计数、元素句柄、
  属性）**和必要的取消收尾**，不会为每个属性重新计时，也不会在本地 deadline 之后追加第二个
  `target_seconds` 窗口；整次 observation 仍受原 operation deadline 约束，不延长、不重置 Run 或
  操作 deadline。内部只在该额度内为收尾预留有界部分，总机会不增加。
- 收尾在同一绝对 deadline 内确认时，返回既有 `unavailable`/`TARGET_QUERY_FAILED` 记录，`count`
  与属性均为 `None`；不得伪造成 0 匹配、`false`、空串或读取成功。成功完成的独立目标保留真实
  内容，普通属性读取失败规则不变。收尾无法在额度内确认时，保持既有晚任务所有权并让整次
  observation 失败（`WEB_OBSERVE_TIMEOUT`），不继续采集、不发布部分成功。
- 适配器自身的普通任务取消与观察到的 `SessionLost`、内部协议缺陷、`KeyboardInterrupt`/
  `SystemExit`/`CancelledError` 严格区分：后几类照实传播或保持原始 BaseException 身份，绝不
  被记录成 `TARGET_QUERY_FAILED`；任何成功快照返回前都重新校验会话仍可用，已失联的会话不能
  支撑 collected 记录。取消收尾不重放、不复用旧属性。
- 页面关闭、browser 断连、整次 operation 预算耗尽、内部协议损坏、取消/中断仍按既有设施错误或
  `BaseException` 传播，不以“部分快照”掩盖。目标读取子任务有明确所有权并复用 Adapter 的有界
  等待与晚完成任务管理；取消未及时收敛时不再继续采集，也不声称成功。
- 只有 `commit_unknown` 在宿主显式声明 500ms；其余 profile 维持 `None`。该值只缩短单目标读取
  机会，不改变普通 Run 30s / operation 3s / site 15s、`delayed_deadline` 15s/25s/30s 以及
  cleanup/finalize/host wait 的冻结值。

### 4.2 独立启动额度（`startup_timeout_ms`）

`PlaywrightWebConfig` 另新增**仅关键字**参数 `startup_timeout_ms=None`，并新增同名只读属性与
只读的有效启动额度属性：

```python
PlaywrightWebConfig(allowed_origins, observation_targets, operation_timeout_ms,
                    cleanup_timeout_ms, observation_target_timeout_ms=None, *,
                    startup_timeout_ms=None)
config.startup_timeout_ms            # 保留 None 或显式值
config.effective_startup_timeout_ms  # None 时为 operation_timeout_ms，否则为显式值
```

- 原四个必填位置参数与第五个可选目标额度参数的调用兼容性保留；`startup_timeout_ms` 只能按关键字
  传入，第六个位置参数被 Python 拒绝。`None` 保留历史行为：`start` 由既有
  `operation_timeout_ms` 约束。
- 显式值沿用同一严格正有限 JSON 数值校验（拒绝 bool、字符串、零、负数、NaN、Inf），不要求它
  大于或小于 operation 额度；构造、`revalidated_config`、伪造实例与改写槽位都以既有
  `WEB_BROWSER_CONFIG_INVALID` 拒绝，无原始值/异常链，控制信号不吞没。
- **仅 `start` 的浏览器资源获取**使用有效启动额度：一次调用计算一份绝对截止时间，无阶段重置、
  重试、预热或共享旧会话；启动超时不能迟到转为 `RUNNING`，随后仍走既有有界失败清理。
- `observe`/`execute` 继续使用 `operation_timeout_ms`；单目标额度继续只由
  `operation_timeout_ms` 约束并保留 F08（一个绝对目标 deadline 覆盖取消收尾）与 F09（设施/
  内部缺陷/进程控制结果不被写成 `TARGET_QUERY_FAILED`）。cleanup 仍使用既有独立
  `cleanup_timeout_ms` 与 Core finalization 限制，不借启动额度扩大清理额度；Core Run 截止时间
  继续约束 start 等待，不延后计时、不增加 Core 接口、不修改 engine。
- 宿主诊断（`examples/web_runtime_evidence.py` 的 `RecordingAdapter`）只增加可选
  `startup_timeout_ms` 关键字：`start` 记录实际有效启动额度，`observe`/`execute` 记录 operation
  额度，`stop` 记录 cleanup 额度；默认（未声明）诊断沿用 operation 值，历史记录不变。不修改
  Core trace schema，不包装或吞没原异常。
- **时长语义**：`diagnostics.boundaries[].elapsed_ms` 是整个宿主调用边界的时长；`start` 失败时
  可能包含失败清理。因此 4157ms 这类边界总时长**不能**被表述为“原生浏览器启动独自耗时 4157ms”，
  也不证明原生启动的根因。只记录固定码、既有真实阶段/`LOGIN_RECEIVED`/计数/清理事实，不记录
  任意异常文本，不增加无依据的根因标签。
- 八个 profile 都显式声明 `startup_timeout_ms=10000`。这只是**启动**额度：普通 profile 的
  operation 仍 3000、`commit_unknown` 的目标额度仍 500、`delayed_deadline` 的 operation 仍
  25000，各 profile 的 run_ms/request_ms/click_timeout 及全部 cleanup/finalize/host wait 预算
  保持原值。10000 是拟议配置，不是从 4157 推导出的保证值；真实冷启动仍须独立验收，首次失败
  保留，不自动扩到更长额度、不预热、不重试到绿。

示例 `examples/web_runtime_demo.py` 仅支持八种 profile；每 profile 使用真实新 browser、独立
run_id 和仓库外新目录。脚本四步最多各一个 Attempt，全部 profile 使用冻结的标准恢复政策
`bounded/max_retries=0/allow_replan=False`。宿主以
`RecoveryEngine(step_verifier_name="web_step_verifier")` 显式声明真实 Web 验证器身份，不重写
标签、不加白名单、不绕过政策。Run 上限 `max_steps=4/planning=4/retries=0/replans=0`。

| profile | 站点模式/时序 | 终态 | 计数 | 动作 |
| --- | --- | --- | --- | --- |
| normal | NORMAL | completed / TASK_ACHIEVED | login=1, commits=1 | 4/4 |
| wrong_credentials | NORMAL + 错误合成密码 | aborted / POLICY_ABORT | login=1, auth_failure=1, commits=0 | 4/4 |
| delayed_release | DELAYED_LOGIN，确认零提交后 release | completed / TASK_ACHIEVED | 释放前 0 提交，最终 1/1 | 4/4 |
| delayed_deadline | DELAYED_LOGIN，不 release，Run deadline 先到 | timeout / DEADLINE_EXCEEDED | login=1, commits=0 | 4/4 |
| overlay_blocked | BLOCKED_SUBMIT | aborted / POLICY_ABORT | login=0, commits=0 | 4/4 |
| overlay_released | 上一 Run 清理后显式新 Run；reset 新代 + release | completed / TASK_ACHIEVED | 新代 1/1 | 4/4 |
| commit_unknown | COMMIT_THEN_STALL，不 release，Adapter 超时先到 | aborted / POLICY_ABORT，同时 Recovery EFFECT_UNRESOLVED | login=1, commits=1，submit unknown | 4/4，零重放 |
| cancel_after_received | DELAYED_LOGIN，LOGIN_RECEIVED 后取消 Run task | CancelledError 传播；last_finalization cancelled / USER_CANCELLED | login=1, commits=0 | 4 Steps/4 Attempts 已启动 |

普通预算：Run 30s、Adapter operation 3s、站点 request 15s；`delayed_deadline`：
Run 15s、operation 25s、request 30s；cleanup/finalize 各 5s、Adapter cleanup 5s、站点 cleanup
3s。另有**仅启动**额度 10000ms，只约束 `start` 的一次浏览器资源获取。**全部非启动预算保持冻结
值，不因首次失败或实时冲突而放宽**。`commit_unknown` 同样使用 3s Adapter operation 与 15s 站点
request：提交后请求仍停滞，但每个目标只获 500ms 读取机会，因此提交后的真实观察仍能在 3s
operation 内返回一个**真实新快照**（停滞页面的目标通常以 `unavailable`/`TARGET_QUERY_FAILED`
记录，绝不伪造 0 匹配）。Runtime 用该真实新快照判步骤 FAIL，Recovery 请求观察后再次得到真实新
快照，`effect_state=unknown`、`safe_to_repeat=false`，恢复引擎给出 `EFFECT_UNRESOLVED`；Runtime
终态为 `POLICY_ABORT`，二者并存而非二选一，且始终一次提交、零重放。若真实新证据导致
INCONCLUSIVE、观测整体失败或其他结果，保存实际事实并判该项未完成，不修改 oracle、不扩预算、
不重试到绿。若真实浏览器在 10000ms 启动额度内仍无法完成，证据索引的 `diagnostics` 保留实际
`start` 边界、`WEB_BROWSER_START_FAILED` 与整个调用边界时长（含失败清理）；该时长不是原生启动
耗时证据，首次失败照实报告，不通过放宽其他预算、自动预热或重试至绿来改变它。

任务 criteria 独立声明 dashboard URL + welcome 可见；脚本只提供候选动作，不设置成功结果。
Web 宿主显式注入 `WebTaskCriteriaValidator`，在资源启动前完成领域语法预检。

## 5. 完整证据旁文件

Core 严格 JSONL trace 不含完整 Observation/Forecast/Contract/ActualOutcome。示例通过
`examples/web_runtime_evidence.py` 的宿主纯转发代理捕获真实调用输入/输出，写到仓库外
快照与 `evidence-index.json`：

- 代理不改变调用次数、参数或返回值，不额外 observe/execute，不读取隐藏 oracle；不包装纯
  DomainSupport/hooks 去 I/O。
- 快照含完整 Observation、Forecast、已提交 Contract、ActualOutcome、步骤/任务验证、恢复
  输入/结果；Observation 用每 Run 调用序号关联，重复 `observation_id` 仍可区分。
- 索引记录 run/step/contract/attempt/event/observation 关联、相对路径、原字节 SHA-256、字节数，
  并与真实 trace、Runtime 发布的 Working Memory/call provenance 交叉核对；不虚构 trace 中不
  存在的 snapshot 字段。
- 合同、attempt、verification 关联行必须与其引用的 snapshot 条目逐项一致（含 path、kind、
  ordinal、bytes/SHA-256 及适用的 run/goal/step/contract/attempt/outcome/observation 标识），
  再与真实 trace 对应事件核对，而不是只验证“路径在清单里且类型相同”。
- `result` 的 step/attempt 计数与真实 trace 计数交叉核对，其余计数与 Runtime 发布的 Working
  Memory 交叉核对；`terminal`、`result`、`last_finalization` 在所有 source 模式下校验主诊断
  一致，cleanup 错误作为次要事实可与主原因不同，不机械要求所有错误字段相同。
- Runtime 发布完整 `observation_calls`：真实返回的观测必须与快照唯一对应；未返回的调用保留
  其真实 `None` 记录且没有快照，不为了数量相等删除调用；重复 `observation_id` 仍按调用序号关联。
- 每个篡改测试先让未篡改的重定位副本通过（trace/Working Memory 路径重定位到副本内），再只修改
  目标字段并断言该目标被拒；源绝对路径逃逸导致的拒绝不算关联验证通过。
- 序列化/写入失败显式非零、不打印成功摘要，不吞掉原始运行失败；写入在真实调用返回后同步
  完成，不逃出原调用预算。
- **宿主边界诊断**：索引的 `diagnostics` 在 Core 包裹异常之前记录每次真实 adapter
  `start`/`observe`/`execute`/`stop` 边界、实际耗时，以及**逐操作分开的预算**：`start` 记录有效
  启动额度（声明了 `startup_timeout_ms` 时用它，否则沿用 operation 值以保持历史诊断不变）、
  `observe`/`execute` 记录 operation 额度、`stop` 记录 cleanup 额度；失败时还记录该边界允许的
  封闭设施错误码（如 `WEB_BROWSER_START_FAILED`、`WEB_OBSERVE_TIMEOUT`）。`elapsed_ms` 是整个
  调用边界时长，`start` 失败时包含失败清理，不是原生启动耗时证据。同时记录宿主
  `LOGIN_RECEIVED` 等待（阶段、是否收到、耗时）、声明阶段的 login/auth/commit/active 计数，以及
  adapter/Runtime cleanup 事实。只接受封闭码表与固定计数器，不记录任意异常文本、异常链或调用
  者值；Core trace schema 不变。重放索引时重新校验这套完整结构，篡改即拒绝。
- 不改 Core tracing/Memory 模型；无截图、网络抓包、知识提取、Episodic 自动写入或崩溃恢复承诺。
  只有固定公开合成凭据出现在本地证据中，无 API key。

## 6. CLI 与退出码

`--profile` 八选一（默认 normal），`--output-dir` 必需，必须是仓库外新目录或空目录。成功写
运行记录时 stdout 仅一个 JSON 对象，五字段：`result`、`terminal`、`trace_path`、
`evidence_index_path`、`working_memory_path`。`terminal.source` 严格为 `run_result` 或
`last_finalization`；没有真实 RunResult 时 `result=null`。

退出码：真实 completed 且证据/资源收尾成功 0；正常返回但任务未完成 2；取消经 Core 传播并
记录 130；设施/证据/收尾失败 1；参数错误沿用 argparse 2 且不执行。故障 profile 符合预期仍
保留 2/130。

## 7. 打包与本地候选

- `forecontract-core`/`forecontract-web`/`forecontract-fake`/`forecontract-scripted` 均为
  `0.1.0`；Core 仍 45 源码成员，Web 从 15 精确增加到 17（新增 `runtime_hooks.py`、
  `runtime_support.py`），Web 根导出仍 19 项。
- 两个 `pyproject.toml` 只改 description；版本、依赖、backend、Python 约束与 browser extra
  不变。`forecontract-web` 仍只直接依赖 Core 与 Pydantic，`browser` extra 仍为
  `playwright==1.64.1`。
- 一次 canonical 正常构建在既有 package-boundary fixture 中完成；显式
  `FORECONTRACT_RELEASE_EXPORT_DIR` 指向本轮仓库外新空目录时，fixture 在清理前导出当前八份
  wheel/sdist 与 `release-index.json`（名称、相对路径、字节、SHA-256、完整构建输入清单、构建
  工具版本与实际来源）。默认未设置时旧生命周期不变；不重复构建同指纹正常包。
- 本任务只做本地 v0.1 候选：不上传 PyPI、不操作远端/tag、不更改既有版本号，不引入生产动作、
  任意 JS/Shell/Python 执行、Coding/LLM/Training/Serving。

## 8. 通用扩展、证据索引与未覆盖

- **Recovery 验证器身份是通用配置**：`RecoveryEngine(*, step_verifier_name="json_step")` 接受
  内建非空白 1..128 字符的 str；非法配置以既有 `POLICY_INVALID` 固定诊断拒绝，且不携带原值/
  异常链。构造时保存，每次 `assess` 重新检查；不从判词或 evidence 自动学习身份。Web 宿主显式
  构造 `step_verifier_name="web_step_verifier"`，真实 `WebStepVerifier` 结果因此原生可用；不
  重写任何标签，默认 JSON/Fake 路径保持 `json_step`。同一构造器仍保留 scope、contract/goal/
  observation ID、完整性、outcome 索引/required 聚合等全部关联校验。
- **Task criteria 预检独立于 factory**：默认 JSON 预检与显式 `TaskCriteriaValidator` 都发生在
  trace 写入/`start`/`observe` 之前；factory 不再关闭预检，Web oracle 仍完整重验 criteria。
- **证据索引严格验证**：`verify_evidence_index` 不接受只含 `kind` 的索引、缺失/伪造清单或
  调用方自选文件列表。它核对完整顶层结构、快照真实文件清单（缺文件/多文件均拒绝）、按 kind
  的唯一连续序号、每个快照原字节/大小/SHA-256、快照内真实 ID 与 payload、trace 与 Working
  Memory 原字节及观察调用序号、run/step/contract/attempt/observation/verification/event 关联、
  唯一终态与 `result`/`last_finalization`/trace 一致性。正常启动前失败允许真实零快照，但必须
  有可解析的 trace 终态依据；缺清单不等于零快照。
- **发布索引不声明未经输入支持的身份**：`release-index.json` 只记录真实产物、构建源、工具与
  输入哈希；实际执行角色由各角色的命令证据与独立导出目录绑定，不写死 `actor`。
- 未覆盖：生产站点、任意网站安全声明、并发 Run、崩溃恢复、跨进程恢复、远端发布、LLM 规划、
  全量 Core 功能套件；`overlay_released` 明确是两次 Run，不是同 Run 续跑。
