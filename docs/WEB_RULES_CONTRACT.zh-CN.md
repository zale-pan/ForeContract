# Web 五类纯规则与步骤/任务验证产品接口说明（forecontract-web 0.1.0 / P3-T03 round-001）

> **文档性质**：本文说明 `forecontract-web` 0.1.0 的五类纯 Web 规则、独立验证器和能力接口。章节保留接口演进顺序；纯规则层不执行浏览器。实际独立验证范围和限制见 [发布验证说明](RELEASE_VALIDATION.zh-CN.md)。

## 1. 范围与非目标

本轮在既有 Web 合同之上只新增四个源码成员：

| 源码成员 | 内容 |
|---|---|
| `rule_models.py` | 五个严格规则 payload 与 `WebRuleEvaluation` |
| `rules.py` | 纯同步 `WebRuleEvaluator`、证据重验与五规则真值表 |
| `verification.py` | 独立的 `WebStepVerifier` 与 `WebTaskCompletionOracle` |
| `rule_capabilities.py` | 显式五列表能力声明构造器 |

`PlaywrightWebAdapter.capabilities()` 只做**能力接线**：声明六个动作、`web.snapshot` 与五个 Web condition/outcome 类型。生命周期、动作执行、观察采集、网络策略、预算与清理实现均未改变。

明确的非目标：

- 不访问真实浏览器、不查询活页 DOM、不等待页面、不修复证据、不调用模型或网络；
- 不把动作执行 `succeeded`、Planner 提示、步骤结果或 forecast 当作验证依据；
- 不实现整段 `RuntimeDomainSupport`、安全/副作用判定、重试策略、完整故障站或 Runtime E2E；
- 不新增 CSS/XPath/text/`nth`/frame/shadow/任意 JS/正则/回调等入口；
- 不扩根包 19 项导出，不新增依赖，不修改 Core/Runtime/Planner/Fake 或既有五个 Web 合同源码。

## 2. 公共入口

```python
from forecontract_web.rules import WebRuleEvaluator
from forecontract_web.verification import WebStepVerifier, WebTaskCompletionOracle
from forecontract_web.rule_capabilities import build_web_rule_capabilities
```

```python
class WebRuleEvaluator:
    def evaluate(self, *, rule_type: str, payload: object,
                 evidence: VerificationEvidence) -> WebRuleEvaluation: ...

class WebStepVerifier:
    def verify_step(self, *, contract: ActionContract,
                    evidence: VerificationEvidence) -> VerificationResult: ...

class WebTaskCompletionOracle:
    def evaluate_completion(self, *, criteria: TaskCompletionCriteria,
                            evidence: VerificationEvidence) -> VerificationResult: ...

def build_web_rule_capabilities(*, adapter_name: str,
    supported_actions: list[str],
    supported_observation_types: list[str],
    supported_condition_types: list[str],
    supported_outcome_types: list[str]) -> AdapterCapabilities: ...
```

三个评价入口都是**同步、确定、无状态**的；构造函数无参数，不接受回调、插件或动态注册，不做 I/O。同一请求重复调用产生相等结果。

## 3. 五类规则与闭合 payload

规则类型是闭集，命名空间为 `web.`，与 Core 通用 JSON 规则区分：

| `rule_type` / condition / outcome | 唯一允许的 payload 字段 | PASS 条件 |
|---|---|---|
| `web.url_match` | `url: str` | 已获得的页面 URL 与指定完整字符串逐字符相同 |
| `web.element_visible` | `target: WebTarget` | 目标唯一匹配且 `visible=true` |
| `web.element_hidden` | `target: WebTarget` | 明确成功查询且 `match_count=0`，或唯一匹配且 `visible=false` |
| `web.text_contains` | `target: WebTarget`, `text: str`（非空） | 唯一匹配、`text` 已获得且包含区分大小写的原始子串 |
| `web.value_equals` | `target: WebTarget`, `value: str` | 唯一匹配、`value` 已获得且逐字符相等；空字符串合法 |

约束：

- payload **不重复** `rule_type`、`schema_version` 或 `required`：规则类型由调用方参数或外层 `ExpectedOutcome.outcome_type` 决定，版本与 `required` 由既有 Core envelope 负责；
- 严格类型、拒绝未知字段与缺失字段、不做隐式转换；字段不允许嵌套任意规则定义；
- `target` 只能使用既有两个定位模型：`TestIdTarget`（`strategy="test_id"` + `test_id`）或 `RoleTarget`（`strategy="role"` + `role` + `name`）；
- `text_contains.text` 必须是**非空**字符串（空子串会恒真）；仅空白字符是允许的，因为不做 trim；
- `value_equals.value` 允许空字符串，用于“字段已清空”的真实值。

## 4. 真值表（已获得的结构有效证据）

| 结构有效证据中的情况 | `element_visible` | `element_hidden` | `text_contains` | `value_equals` |
|---|---|---|---|---|
| `collected` 且 `match_count=0` | FAIL / `TARGET_NOT_FOUND` | PASS / `RULE_MATCH` | FAIL / `TARGET_NOT_FOUND` | FAIL / `TARGET_NOT_FOUND` |
| `collected` 且 `match_count=1`，所需属性满足 | PASS / `RULE_MATCH` | FAIL / `RULE_MISMATCH`（`visible=true`） | PASS / `RULE_MATCH` | PASS / `RULE_MATCH` |
| `collected` 且 `match_count=1`，所需属性不满足 | FAIL / `RULE_MISMATCH`（`visible=false`） | PASS / `RULE_MATCH`（`visible=false`） | FAIL / `RULE_MISMATCH` | FAIL / `RULE_MISMATCH` |
| `collected` 且 `match_count>1` | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| 对应属性为 `null` | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| 目标记录未提供 | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| `unavailable`（查询失败） | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |

### 4.1 `web.url_match`

- 只比较**整段原始页面 URL 字符串**；支持既有页面 URL 合同，包括精确的 `about:blank`；
- 不做 regex/glob/前缀匹配、大小写折叠、scheme/host/端口归一化，也不隐式忽略 query 或 fragment；
- `url=null`（本次未获得）→ INCONCLUSIVE；已获得且不相等 → FAIL。

### 4.2 目标证据语义

- **零匹配**是成功查询后的存在性否定证据：`element_hidden` 通过，其余目标规则 `TARGET_NOT_FOUND` 失败；
- **多匹配**是明确歧义证据，**不**“任选一个”，也不合并不同节点属性；
- **`unavailable`** 是结构有效快照中的查询失败，**不能**伪造成零匹配；
- 实际空字符串 `text`/`value` 与 `visible=false` 是**已获得**证据，不是 `null`；
- `text_contains` 只读 `text`，`value_equals` 只读 `value`，两者不互相替代；
- 不以 `visible` 作为 text/value 规则的隐式门槛：隐藏但已获得文本/值仍按原值判定；
- 目标按定位模型**全部声明字段**精确选择，与 `targets` 列表下标无关、与对象键顺序无关；重复定位记录违反既有 `WebObservationState` schema，是设施错误。

### 4.3 `state_complete`

`state_complete` 是调用方显式提供的严格布尔声明：

- 不认证来源、不从字段数量或 metadata 推导；
- `true` **不能**补出快照中未记录的目标或属性；
- 明确 `collected` 的零匹配在 `state_complete=false` 时仍是局部有效的否定证据；
- 规则不猜测、不改写该值，只把它原样带入结果 metadata。

## 5. `WebRuleEvaluation`

`rule_models.py` 中的单一结果记录，字段**恰好**为：

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | 字面量 `"0.1"` | 与既有 Core envelope 一致 |
| `rule_type` | 五个闭集类型之一 | 本次判定使用的规则 |
| `status` | Core `VerificationStatus` | `PASS` / `FAIL` / `INCONCLUSIVE` |
| `reason` | 闭集 | `RULE_MATCH` / `RULE_MISMATCH` / `TARGET_NOT_FOUND` / `EVIDENCE_INSUFFICIENT` |
| `observation_id` | 非空白严格字符串 | 输入快照的身份，用于回查原始证据 |
| `target_index` | 非负严格整数或 `null` | 命中记录的位置；URL 规则与记录缺失时为 `null` |

一致性由模型自身强制：`PASS` 只配 `RULE_MATCH`；`FAIL` 只配 `RULE_MISMATCH` 或 `TARGET_NOT_FOUND`；`INCONCLUSIVE` 只配 `EVIDENCE_INSUFFICIENT`。

结果**不复制** URL、文本或 value；需要原始值时，调用方按 `observation_id` 与 `target_index` 读取输入 Observation。记录是 frozen 的，未知字段拒绝。

## 6. 证据完整性与设施错误

评价前先完整重验 `VerificationEvidence`、其嵌套 `Observation` 与 Web `WebObservationState`：

- 必须是 `VerificationEvidence` 实例；已有实例、`model_copy`、`model_construct` 与隐藏的未声明存储键都按**实际存储内容**重验，序列化不能把未知字段洗掉；
- `Observation.environment_type` 必须为 `web`；
- `WebObservationState` 的版本、字段形状、额外字段、URL 语法、目标证据一致性与定位去重全部按既有 Web 合同重验；
- 读取调用方自有对象（`Mapping.items`、模型存储字段等）抛出的普通异常，转换为本轮固定设施错误；
- 取消、`KeyboardInterrupt`、`SystemExit` 始终传播，不被吞掉；
- 不修改输入，不查询活页 DOM，不修复或补齐证据。

两类固定设施错误都复用既有 `WebValidationError`：

| code | 触发场景 |
|---|---|
| `WEB_RULE_INVALID` | 规则类型不受支持，或 payload 不符合所选规则 schema |
| `WEB_VERIFICATION_INPUT_INVALID` | 外层 contract / criteria / evidence / binding 不可用，或结构损坏 |

两者是既有 `WebValidationError` 的子类（因此 `isinstance(error, WebValidationError)` 仍成立），把本轮两个固定 code 加入错误词汇，而不修改既有 `errors.py`。诊断只含固定句子、声明字段名与有界列表索引；不含输入值、未知键或可达 `__cause__`/`__context__`。

## 7. `WebStepVerifier`

`verify_step` 只判断当前 `ActionContract.expected_outcomes`：

1. 先完整重验 evidence 与 contract；
2. 校验 contract 至少有一个 `required=true` 的 outcome；
3. 每个 outcome 恰好一个 `VerificationBinding`，覆盖全部索引，无重复、无越界；
4. `binding.verifier_type` 必须为 `web_rule`，`binding.config` 必须为空；
5. **所有** outcome（含 optional）的 `outcome_type` 必须是五类 Web 规则之一，payload 必须先完整校验；
6. 全部校验通过后才逐项评价，按输入顺序返回 `OutcomeVerification`。

返回原 Core `VerificationResult`：

- `verifier = "web_step_verifier"`；
- `outcome_results` 的 `outcome_index` 保持输入顺序；每项 `has_actual_value=false`、`actual_value=null`（值留在证据里），`reason` 为固定规则码；
- `metadata` 固定包含 `scope="step"`、`contract_id`、`goal_id`、`observation_id`、`state_complete`，以及按项顺序的 `rule_evidence_refs`（每项 `{rule_type, target_index}`）；
- 聚合只看 `required=true` 项：`FAIL` 优先，其次 `INCONCLUSIVE`，全部 `PASS` 才 `PASS`；整体 `reason` 为 `REQUIRED_FAILED` / `REQUIRED_INCONCLUSIVE` / `REQUIRED_PASSED`；
- optional PASS **不能**抵消 required FAIL；optional 的未知/畸形规则**不会**因 `required=false` 被跳过。

## 8. `WebTaskCompletionOracle`

`evaluate_completion` 只判断调用方独立传入的 `TaskCompletionCriteria.conditions`：

- 不读取 forecast、`task_completion_hint`、执行状态、步骤结果或 action 合同；
- 与步骤入口共享同一五规则真值表和同一证据重验，但接口与 metadata scope 独立；
- `criteria.conditions` 必须非空且至少有一个 `required=true`；空集或仅 optional 是设施错误；
- 所有 condition（含 optional）先完整校验，再按顺序逐项评价；
- 每项同样的 `OutcomeVerification` 形状，`verifier = "web_task_completion_oracle"`；
- `metadata` 固定包含 `scope="task"`、`goal_id`、`observation_id`、`state_complete` 与按项顺序的 `rule_evidence_refs`，**没有** `contract_id`；
- 聚合规则与步骤入口相同：required 内 `FAIL` > `INCONCLUSIVE` > `PASS`。

步骤通过不等于任务完成；反过来，步骤失败也不会阻止独立任务条件成立。两者都只是“所给规则在所提供的证据上成立”，不是 Runtime 终态或完整任务成功证明。

## 9. 能力声明与适配器接线

`build_web_rule_capabilities` 要求五个显式关键字列表：

- `supported_actions`：既有六动作的**显式子集**；
- `supported_observation_types`：仅 `web.snapshot` 的显式子集；
- `supported_condition_types`、`supported_outcome_types`：五个 Web 规则类型的显式子集；
- 每个列表必须是真正的 `list`，无重复、无闭集外取值；允许显式空列表；
- 不提供“默认全开”，不修改调用方列表，每次返回 detached 的 Core `AdapterCapabilities`；
- `environment_type` 固定 `web`；
- 能力错误沿用既有 `WEB_CAPABILITIES_INVALID`。

`PlaywrightWebAdapter.capabilities()` 改为通过该构造器声明：

- 六个动作全部；
- `web.snapshot`；
- 五个 `supported_condition_types` 与五个 `supported_outcome_types`。

除导入与 `capabilities()` 方法体之外，适配器源码不变；既有三参数 `build_web_capabilities` 与 `capabilities.py` 保持字节不变（其 condition/outcome 列表仍为空）。

## 10. 分发、依赖与导入边界

- Web 源码成员由 11 个增至**恰好 15 个**：新增 `rule_models.py`、`rules.py`、`verification.py`、`rule_capabilities.py`；
- 根包公开导出**仍为恰好 19 项**：本轮四个模块只通过子模块导入，不扩根导出；
- 版本仍 `0.1.0`；直接依赖仍为 `forecontract-core>=0.1,<0.2` 与 `pydantic>=2,<3`；可选 extra 仍为 `playwright==1.64.1`；
- 不新增依赖，不修改 `pyproject.toml`；
- 四个新模块在 **schema-only** 环境（无 Playwright、无浏览器二进制）可导入且无 I/O；`verification.py` 只依赖 Core 类型与同包纯模块。

## 11. 测试与覆盖

| 文件 | 覆盖 |
|---|---|
| `tests/unit/test_web_rules.py` | 五规则完整真值表、每一种 INCONCLUSIVE 形状、显式零匹配、URL 精确语义、text/value 的空值/`false`/非 trim、无效证据与恶意输入、设施错误诊断、冻结 payload 与普通映射/模型正向 |
| `tests/unit/test_web_verification.py` | 步骤/任务入口签名与无状态、唯一绑定、`verifier_type=web_rule`、空 config、required 聚合（FAIL > INCONCLUSIVE > PASS）、optional 不抵消 required FAIL、未知 optional 不被忽略、metadata 定位信息、值不进入结果、输入不被修改 |
| `tests/unit/test_playwright_web_adapter.py` | 仅能力接线断言：六个动作、`web.snapshot`、五个 condition/outcome |
| `tests/integration/test_web_rule_verification.py` | 真实 headless Chromium 观察本地合成页面，关闭浏览器后纯规则仍可判定 PASS/FAIL/INCONCLUSIVE；另以合成结构有效快照覆盖 zero/unique/many/unavailable/null/false/empty 全矩阵 |
| `tests/unit/test_package_boundaries.py` | Web 15 成员、schema-only 与 browser-extra 安装、四个新子模块导入；Core/Fake/Planner 与全部既有负向检查保留 |

这些测试证明本轮纯规则与两个上层入口的行为；它们**不**证明完整故障站、P3-T05 Runtime E2E、生产站点自动化或整个 `G-WEB-001-AC01` 已完成。

## 12. 边界声明

- 本轮不修改 Core、Runtime、Planner、Fake、既有五个 Web 合同源码、浏览器后端/动作/观察/生命周期模块、根 19 项导出或旧三个 Web 纯合同单测；
- 不接收、不输出真实密钥；测试只使用合成标记；
- 规则评价不是执行授权，`WebRuleEvaluation.status` 不是任务终态；
- Runtime 中已存在的注入点只是“可用”，不构成本轮完成整段 Runtime 闭环的证明。

## 13. round-002 同范围修正（F01/F02/F03）

本节只记录 round-002 对既有 W01–W12 合同的三组原范围修正，不新增规则类型、字段或能力。第 1–12 节仍是主合同；本节与第 1–12 节冲突时，以本轮修正后的行为为准，且不改写既有产品边界与验收结论。

### 13.1 F01 实际存储重验与序列化不可信

`WebRuleEvaluator`、`WebStepVerifier` 与 `WebTaskCompletionOracle` 重验输入时，读取的是记录**实际存储结构**：

- 通过 Pydantic 基类的 `__dict__` / `__pydantic_extra__` 存储描述符读取真实存储槽，不经过调用方可能覆盖的属性视图；
- 声明字段映射从存储字典的**真实底层表**枚举，额外存储同样按真实条目枚举；不再信任调用方的 `keys`/`items`/`__iter__`/`__bool__`/`__len__`；
- 非空但 `__bool__` 返回 false 的额外存储、`__bool__` 抛异常的额外存储、以及从映射视图隐藏键的存储字典，都按实际内容判为设施错误；任何未知/额外存储条目同样是设施错误；
- 重验用的纯文档由**实际存储字段**自行构建，不再调用调用方自有的 `model_dump()`。因此合法的证据/条件/合同子类即使覆盖 `model_dump()`，也不能把实际不匹配的 URL 或实际失败的 goal 改写成通过；
- 上述失败仍复用既有固定设施错误（`WEB_RULE_INVALID` / `WEB_VERIFICATION_INPUT_INVALID`），诊断不含输入值，且不带可达 `__cause__`/`__context__`；
- 正向保持：普通 payload/证据/条件子类、`__pydantic_extra__` 为 `None` 或空字典的普通记录、普通 Mapping/str 与已提交 `FrozenMapping` 输入仍被接受；取消、`KeyboardInterrupt`、`SystemExit` 继续传播。

### 13.2 F02 期望 URL 必须符合既有页面 URL 语法

`web.url_match` 的 `url` 在 payload 校验阶段就按**观察状态既有的页面 URL 语法**校验（绝对 `http`/`https`，或精确 `about:blank`）：

- 空串、相对串、其他 scheme、含 userinfo、非法端口、前导空白、控制字符或反斜杠等**规则定义错误**在评价前抛出固定 `WEB_RULE_INVALID`，不再退化为普通 `FAIL`/`RULE_MISMATCH`；
- 不修改 `models.py`，不归一化、不 trim、不做大小写折叠；校验只读取字符串真实存储的字符，返回原对象；
- 合法期望 URL 仍按**整段原始字符串**比较：query、fragment、路径/host 大小写差异仍是 `FAIL`/`RULE_MISMATCH`，`about:blank` 与 `url=null` 语义不变。

### 13.3 F03 URL 规则结果必须携带 `target_index=null`

`WebRuleEvaluation` 在普通结果 schema 入口强制：`rule_type="web.url_match"` 时 `target_index` 必须为 `null`。URL 规则没有目标记录，非空索引（如 `0`）在构造时即被拒绝；目标规则的合法非负索引、严格类型与既有字段/状态/原因闭集不变，不新增字段。

## 14. round-003 剩余 F01 修正（字典实际条目、映射异常边界与嵌套 abort 语义）

本节只记录 round-003 对同一 F01 范围的剩余三组修正，不新增规则类型、字段、能力或默认行为。第 1–13 节仍是主合同；本节与它们冲突时，以本轮修正后的行为为准。

### 14.1 实际字典条目不可被视图隐藏

`_payload_fields` / `_target_fields` 对 **dict 支撑**的 payload 与 locator 不再读取调用方的 `items()`/`__iter__`/`keys()` 或拷贝视图，而是用 `dict.items(mapping)` 枚举映射真实底层表；真实存储但未声明的条目（即使 `items()` 只报告声明字段）一律是固定 `WEB_RULE_INVALID`。合法的普通 dict 子类、已提交 payload 模型、`FrozenMapping` 与普通通用 `Mapping`（非 dict，按其正常协议读取）仍被接受。

### 14.2 实际文档重建的调用方普通失败全部受限

由 `_plain_mapping` / `_plain_value` / `_plain_sequence` / `_plain_document` 组成的实际文档重建，对调用方自有的读取、`hash`、相等比较、迭代、插入与转换失败统一转换为固定设施错误：外层 evidence/contract/criteria 为 `WEB_VERIFICATION_INPUT_INVALID`，规则 payload 为 `WEB_RULE_INVALID`；错误不含原始值或合成标记，且不带可达 `__cause__`/`__context__`，`KeyboardInterrupt`/`SystemExit`/取消仍传播。实现先在真实类型层判断键是否定义了自己的 `__hash__`/`__eq__`：会执行调用方钩子的键（含非 `str` 键）直接以固定设施错误拒绝且不执行该钩子；只有自带实现且未覆盖任何钩子的普通 `str` 子类键才按精确内容重建并通过。因此 PASS 仅出现在可证明零调用方钩子、键/值身份保留、输入未被修改且字段无丢失的重建上。

### 14.3 嵌套 abort 恢复策略先校验实际值再做合法投影

嵌套 `RecoveryPolicy` 在投影为合法的两字段 abort 形状**之前**，先检查实际存储的 `max_retries` / `allow_replan`：abort 模式下任一非 `null` 实际值都是固定 `WEB_VERIFICATION_INPUT_INVALID`，不再被静默丢弃后判为 PASS。普通默认 abort 记录（两个控制均为 `null`/缺失）与合法 `bounded` 策略保持不变；Core 仍只读，本节不扩展恢复功能。

## 15. round-004 剩余 F01 修正（调用方类型判断不执行调用方属性读取）

本节只记录 round-004 对同一 F01 范围的最后一组修正：三个公开参数边界（evidence、`ActionContract`、`TaskCompletionCriteria`）与嵌套模型重建中的**调用方类型/模型判断**。不新增规则类型、字段、能力或默认行为。第 1–14 节仍是主合同；本节与它们冲突时，以本轮修正后的行为为准。

- `_revalidated_model` 不再对传入记录执行 `isinstance(record, model)`。`isinstance` 在真实类型不是模型子类时会回退读取实例的 `__class__`，该查找会执行调用方的 `__class__` property 或 `__getattribute__`；外部对象因此可以把原始异常和合成标记泄漏给调用方。现在先用内建 `type(record)` 读取真实 `Py_TYPE` 槽，再用 `issubclass(真实类型, 受信模型类)` 判断，整个过程不执行任何调用方属性读取。
- `_plain_document` 中识别嵌套 `RecoveryPolicy` 的分支同样从 `isinstance(record, RecoveryPolicy)` 改为真实类型判断，因此结构合法但带恶意 `__class__` property 的 `VerificationEvidence` / `Observation` 子类不会再在该分支触发原始 `ValueError`。
- 外部非模型对象（属性读取抛异常、或 `__class__` property 抛异常并携带 Pydantic 外观属性）在三个入口统一成为固定 `WEB_VERIFICATION_INPUT_INVALID`：诊断不含原始值或标记，且不带可达 `__cause__`/`__context__`。
- 合法的**真实**模型子类仍被接受（不整体禁止子类）：其实际存储内容按前文规则重建并判定；仅当精确存储内容保留、输入未被修改、且独立观测到**零次**恶意属性调用时才可以 PASS。其他普通读取失败仍转换为固定设施错误；`KeyboardInterrupt`/`SystemExit`/取消继续传播。
- 既有 F01/F02/F03 修正全部保留，包括实际字典条目、额外存储、序列化不可信、键钩子边界、嵌套 abort 语义、URL 语法与结果索引约束；Core、Runtime、Planner、Fake 与既有 Web 合同源码不变。
