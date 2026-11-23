# Web 领域合同产品接口说明（forecontract-web 0.1.0 / P3-T01 round-001，round-002 至 round-007 加固）

> **文档性质**：本文说明 `forecontract-web` 0.1.0 的产品接口。各章节保留接口演进顺序；早期章节的“尚未包含”仅指该层次或当时范围，不代表整个当前包缺少后续能力。独立验证范围及限制见 [发布验证说明](RELEASE_VALIDATION.zh-CN.md)。

## 1. 范围

本轮交付的是一层**纯粹的 Web 领域合同**，由五个源码成员组成：

| 源码成员 | 内容 |
|---|---|
| `__init__.py` | 公开导出集合与包说明 |
| `models.py` | 六个动作 payload 模型、两个定位模型、`web.snapshot` 证据模型 |
| `validation.py` | 四个公共函数中的三个纯校验入口 |
| `capabilities.py` | 显式能力声明构造器 |
| `errors.py` | 两个公开错误类与固定错误码 |

它复用 Core 已有的 `ActionSpec`、`Observation`、`AdapterCapabilities` 以及 Core 适配器校验异常，**不新增 Core envelope、不新增 Contract 类型、不修改 Forecast/Planner 接口**。

### 1.1 明确非目标

以下内容本轮**没有实现，也不得据此声称**：

- 没有浏览器适配器、浏览器生命周期、页面执行、真实站点操作；
- 没有五条验证规则、没有规则裁决，不返回 `PASS`/`FAIL`/`INCONCLUSIVE`；
- 没有完整的 `RuntimeDomainSupport`，也不从字段数量推导 `state_complete`；
- 没有 Playwright、Selenium 或任何浏览器自动化/模型/训练依赖；
- 不访问网络、不做 DNS 解析、不读取或写入真实凭据；
- 不提交、不调度、不执行任何动作；解析成功只证明该动作匹配声明的 schema；
- 能力构造器不证明任何能力已 implemented / exercised / verified，声明由调用方负责。

## 2. 动作类型与 payload

动作名是**闭集**，`payload` 不再重复写动作名；由 `action_type` 唯一选择模型。

| `action_type` | payload 模型 | 字段（全部必需） |
|---|---|---|
| `web.navigate` | `NavigatePayload` | `url: str` |
| `web.click` | `ClickPayload` | `target: WebTarget` |
| `web.fill` | `FillPayload` | `target: WebTarget`, `text: str` |
| `web.press` | `PressPayload` | `target: WebTarget`, `key: 闭集字符串` |
| `web.select` | `SelectPayload` | `target: WebTarget`, `value: str` |
| `web.wait` | `WaitPayload` | `target: WebTarget`, `state: 闭集字符串` |

规则与语义：

- **严格类型**：字符串字段只接受 `str`，不接受数字、布尔、`bytes`、`None`；不做隐式类型转换。
- **未知字段一律拒绝**，不会被静默丢弃；字段缺失同样拒绝。
- `web.fill.text` 允许空字符串：它表示“清空”，是真实值，原文保留。
- `web.select.value` 允许空字符串：它是明确的 option value，不隐式转换 label 或 index。
- `web.press.key` 首期闭集：`Enter`、`Tab`、`Escape`、`Space`、`ArrowUp`、`ArrowDown`、`ArrowLeft`、`ArrowRight`、`Backspace`、`Delete`、`Home`、`End`。大小写敏感，无组合键、无脚本、无操作系统命令。
- `web.wait.state` 闭集：`attached`、`detached`、`visible`、`hidden`。等待的是目标状态，不是固定睡眠。
- payload **不提供** `timeout`、`sleep`、`evaluate`、`script`、`javascript`、`shell`、`python` 等代码入口，也没有 `retry_safe` 字段；动作不能在后续执行中受本合同的漏洞绕开既有 Runtime 预算。不能依据动作名或文本内容自动推断“无副作用”或“可重试”。

## 3. 定位模型（WebTarget）

`WebTarget` 是按 `strategy` 判别的两种模型之一：

- `TestIdTarget`：`strategy="test_id"`，`test_id` 为非空白严格字符串。
- `RoleTarget`：`strategy="role"`，`role` 属于闭集 `button`、`link`、`textbox`、`combobox`、`checkbox`、`radio`、`option`、`listbox`、`heading`、`dialog`、`status`、`alert`，`name` 为非空白严格字符串。`name` 是**精确匹配**：不 trim、不做大小写折叠、不做 Unicode 归一化、不做模糊回退。

明确不支持的写法：`css`/`xpath`/`text` 策略、`nth`/`first`/`index`、把 `selector` 字符串透传给运行时、`frame`/`shadow` 配置、“任选一个匹配”的语义。以上任何字段都会因“未知字段”被拒绝；未知策略会因判别失败被拒绝。

本轮只校验定位**结构**。它不宣称已验证页面上唯一匹配；零匹配、多匹配与歧义处理属于后续执行合同（P3-T02）的范围。

## 4. `web.navigate.url` 的语法校验

只做语法校验：不发起网络请求、不做 DNS、不重写 URL，返回给调用方的字符串与输入逐字一致。

必须满足：

1. 是绝对 URL，scheme 为 `http` 或 `https`（本轮实现只接受小写的这两个字面量），然后是 `//`；
2. host 非空；允许 IPv4、常规域名、带方括号的 IPv6 字面量与端口；
3. 端口为本轮可解析的十进制数字且位于 1–65535。

必须拒绝：

- 相对地址、缺少 host、只有 scheme 的地址；
- 带 userinfo 的地址（`user@host`、`user:secret@host`）——本合同不从 URL 推断任何授权信息；
- 非法端口（空端口、非数字、越界、超长数字）；
- 控制字符与任何空白；
- 反斜杠；
- `javascript:`、`data:`、`file:`、`ftp:` 等其他 scheme。

实际允许的 origin、重定向和本地站点限制由后续执行合同独立约束；本轮语法校验不构成网络访问授权或隔离证明。

## 5. 观察证据（`web.snapshot`）

Core `Observation.environment_type` 必须为 `web`；`Observation.state` 使用 `WebObservationState`。外层 `observation_id`/`timestamp`/`artifacts`/`metadata` 继续使用 Core 原字段——不复制身份、不另造时钟。

### 5.1 `WebObservationState`（全部字段必需）

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | 字面量 `"0.1"` | 未知版本拒绝 |
| `url` | 严格字符串或 `null` | 字符串限绝对 `http`/`https` 页面地址或精确 `about:blank`；`null` 表示本快照**未获得**页面 URL，不得解释为空 URL；字段缺失是 schema 错误 |
| `targets` | 严格 `list[WebTargetEvidence]` | 允许空列表；相同定位目标的重复记录拒绝 |

`targets` 为空表示本轮**未提供**目标证据，不等于目标不存在；也不能由列表数量推断证据完整性或 `state_complete`。

重复目标按两个定位模型的明确字段**精确比较**，与对象键顺序无关。

### 5.2 `WebTargetEvidence`（全部字段必需，用 `null` 表达未获得）

| 字段 | 类型 |
|---|---|
| `target` | `WebTarget` |
| `status` | `collected` 或 `unavailable` |
| `match_count` | 非负严格整数或 `null`（`bool`、`float`、数字字符串都不算整数） |
| `visible` | 严格 `bool` 或 `null` |
| `text` | 严格字符串或 `null` |
| `value` | 严格字符串或 `null` |
| `error_code` | `null` 或闭集值 `TARGET_QUERY_FAILED` |

### 5.3 一致性约束

| status | match_count | visible/text/value | error_code |
|---|---|---|---|
| `collected` | 必须为非负整数 | `count=1` 时各字段可独立为已获得值或 `null`；`count=0` 或 `>1` 时必须全部为 `null` | 必须为 `null` |
| `unavailable` | 必须为 `null` | 必须全部为 `null` | 必须为 `TARGET_QUERY_FAILED` |

语义区别必须保持：

- **零匹配**（`collected` + `count=0`）是成功查询之后的存在性证据；
- **多匹配**（`collected` + `count>1`）是明确的歧义证据，**不归并**多个元素的属性；
- **不可用**（`unavailable`）是结构有效快照中的查询失败，**不能伪造** `match_count=0`；
- `text`/`value` 的空字符串是**已获得的真实空值**，不等于 `null`；
- `visible=false` 是**明确不可见**，不等于 `null`。

整份 Observation/state 损坏、未知版本或未知字段、非法 target、非有限数值或违反一致性约束时，公共入口一律抛设施输入错误，**绝不返回 `PASS`/`FAIL`/`INCONCLUSIVE`**。

## 6. 四个公共函数

### 6.1 `parse_web_action(action: ActionSpec | Mapping) -> ActionSpec`

重新校验 Core envelope 与由 `action_type` 选中的 Web payload，返回一个**与输入隔离**的 Core `ActionSpec`（payload/metadata 都是新对象）。不是新合同，不提交、不执行动作。

### 6.2 `validate_web_action(action: ActionSpec | Mapping) -> None`

与 `parse_web_action` 相同的判定顺序与异常语义；无 I/O、无状态修改、不修改输入，返回 `None`。

### 6.3 `parse_web_observation(observation: Observation | Mapping) -> WebObservationState`

重验外层 Core `Observation`、`environment_type` 与 Web state，返回**与输入隔离**的领域快照。不伪造、不复制、不替换外层身份（`observation_id`/`timestamp`/`artifacts` 只被校验，不出现在返回值里）。

### 6.4 `build_web_capabilities(*, adapter_name, supported_actions, supported_observation_types) -> AdapterCapabilities`

显式声明构造器：

- 两个列表都是**必需关键字参数**，严格 `list`（tuple/字符串/集合/映射一律拒绝），无重复，顺序保留，输入不被修改；
- `supported_actions` 只能是六个动作的子集；`supported_observation_types` 只能是 `["web.snapshot"]` 的子集；
- `environment_type` 固定 `web`；`supported_condition_types` 与 `supported_outcome_types` 本轮固定为空，不提前声明 P3-T03 的规则能力；
- 每次调用返回一个 detached 的 Core 模型；
- 不创建 Adapter、不启动环境、不打开浏览器，也不证明任何能力已实现、已演练或已验证。

### 6.5 严格重验的共同要求

所有入口都**不信任**传入实例：

- 读取实例实际存储的字段，拒绝未声明的存储键（`model_copy`/`model_construct`/直接属性注入都不能夹带隐藏键或被篡改的字段）；
- 从原始存储字段重新走一遍严格 schema，嵌套内容的原地污染同样会被发现；
- 接受合法候选 `ActionSpec` 与 Builder 提交后的 `FrozenActionSpec`；对 Core 已知的 `FrozenMapping`/`FrozenList` 只做**有依据的容器解包**并重新严格校验，不把任意 tuple、`bytes`、集合或错误值普遍强制转换成合法 JSON；
- 普通映射入口与 JSON 往返保持同样的语义；调用后输入不变，返回值与输入不共享可变容器；
- 冻结字段（`frozen=True`）只是普通 API 级不可赋值，不等于深层不可变，也不阻止上述重验。

## 7. 错误码、优先级与安全诊断

### 7.1 判定顺序

1. 先确认 Core envelope 结构合法；
2. 再判定 `action_type` 是否受支持；
3. 最后校验对应领域 payload。

因此：结构合法但 `action_type` 未知时，抛 Core 的 `UnsupportedActionError`（先于任何 payload 判定）；受支持动作但 payload 非法时，抛 `WebActionPayloadError`；其他 Web snapshot / capability / envelope 错误抛 `WebValidationError`。

### 7.2 异常类层次与固定错误码

- `WebValidationError(ValueError)`
- `WebActionPayloadError(WebValidationError, InvalidActionPayloadError)`——同时兼容 Core 的 `InvalidActionPayloadError`。

固定错误码只有四个：

| code | 触发场景 |
|---|---|
| `WEB_ACTION_INVALID` | Core 动作 envelope 结构非法（版本、动作名、payload/metadata 结构、未知键等） |
| `WEB_PAYLOAD_INVALID` | 动作受支持，但对应 Web payload 不符合领域 schema |
| `WEB_OBSERVATION_INVALID` | 外层 Observation、`environment_type` 或 Web snapshot state 非法 |
| `WEB_CAPABILITIES_INVALID` | 能力声明的名称、列表、重复项或闭集外取值非法 |

### 7.3 安全诊断

每个错误提供 `code`、`path`、`message`：

- `message` 是**固定语句**，不含任何输入值；
- `path` 只由**已知字段名**和有界列表索引组成，未知键会被替换为常量占位符，不回显用户键；
- 公共函数抛出异常的 `str`/`repr`/公开属性以及**可达异常链**都不包含输入对象、URL、填写文本、未知键或原始 Pydantic 错误（不挂 `__cause__`/`__context__`）；
- 底层直接构造模型仍按普通 schema 校验使用，其原始 `ValidationError` **不是**公共安全诊断；
- 不接收、不输出真实密钥；测试只使用合成标记。

## 8. 公开导出集合（根包 `__all__`，恰好 19 项）

```
__version__                  WEB_ACTION_TYPES            WEB_OBSERVATION_TYPES
TestIdTarget                 RoleTarget                  NavigatePayload
ClickPayload                 FillPayload                 PressPayload
SelectPayload                WaitPayload                 WebTargetEvidence
WebObservationState          WebValidationError          WebActionPayloadError
parse_web_action             validate_web_action         parse_web_observation
build_web_capabilities
```

`__version__` 保持 `"0.1.0"`。除此之外不导出任何名字：`WEB_ACTION_TYPES`、`WEB_OBSERVATION_TYPES` 之外没有 `capabilities()` 全开默认值，也没有可运行的 `WebAdapter` 替身。

## 9. 分发依赖

`packages/adapters/web/pyproject.toml` 保持名称、版本、构建后端与 Core metadata 不变，依赖精确声明为：

```toml
dependencies = ["forecontract-core>=0.1,<0.2", "pydantic>=2,<3"]
```

Web 直接使用 Pydantic，因此直接声明其主版本区间；没有 Playwright 或其他自动化/模型依赖。分发包的源码成员恰好为第 1 节列出的五个文件。

## 10. 与验收矩阵的对应关系

本仓库新增的单元测试按 W01–W07 组织，全部只驱动公开入口或普通模型构造：

| 文件 | 覆盖 |
|---|---|
| `tests/unit/test_web_models.py` | W01（六动作、两定位、闭集、未知/额外/缺失/混合字段）、W04（证据一致性与空值语义） |
| `tests/unit/test_web_validation.py` | W02（URL 语法与无 I/O）、W03（往返、`FrozenActionSpec`、污染与所有权）、W06（优先级与安全诊断） |
| `tests/unit/test_web_capabilities.py` | W05（显式能力声明）、W07（根导出集合） |
| `tests/unit/test_package_boundaries.py`（仅 Web 相关声明） | W07（五个源码成员、两条精确依赖、隔离离线安装与外部导入） |

需要强调：这些测试证明的是**本轮的 schema 与纯校验行为**，不证明六个动作在真实站点上可执行，也不证明五条验证规则或整个 `G-WEB-001-AC01` 已完成。

## 11. round-002 入口加固（F01/F02/F03）

round-001 的独立验收确认了正常 Web 模型、证据矩阵、冻结动作兼容、分发成员与依赖、隔离安装等行为。round-002 只修正三个同范围缺口，不新增任何动作、字段或能力。

### 11.1 F01：额外字段存储区必须重验

Core 的 `ActionSpec`/`Observation` 如果被调用方以 `extra="allow"` 的派生 schema 构造，未知键不会进入 `__dict__`，而是保存在 `__pydantic_extra__`。只检查 `__dict__` 会**接受并把未知键静默丢弃**，这违反"不得把未知字段静默丢弃"。

现在两个入口都同时检查：

- `__dict__` 中的键必须全是该模型声明的字段；
- `__pydantic_extra__` 只要非空即拒绝。

任一情况都抛对应的固定 code（动作入口 `WEB_ACTION_INVALID`，观察入口 `WEB_OBSERVATION_INVALID`），`path` 只给出常量占位符 `<field>`，不回显未知键名或值。

### 11.2 F02：嵌套 Core 模型必须按存储内容重验

`Observation.artifacts` 的元素是 Core `ArtifactRef`。Pydantic 对**既有嵌套模型实例**只做浅检查，因此一个被 `model_copy(update=...)` 污染过的嵌套实例（如 `artifact_id=int`、`artifact_type=bytes`、`uri=None`、`uri=123`、`schema_version="0.2"`）会原样通过外层重验。

现在观察入口会先把受支持的嵌套 Core 模型（当前为 `ArtifactRef`）拆成其**真实存储字段**，再交给外层 schema 重新校验：

- 嵌套记录里出现未声明的存储字段 → 拒绝；
- 字段类型非法（含 bool/float 冒充整数、bytes、null）→ 拒绝；
- 不通过 JSON 序列化"洗白"类型，也不改写外层 `observation_id`/`timestamp`/`environment_type` 身份；
- 合法嵌套 `ArtifactRef` 仍然通过，外层身份保持不变。

拒绝时 `path` 形如 `artifacts[0].<field>`（声明过的字段名保留，未声明键折叠为 `<field>`），code 为 `WEB_OBSERVATION_INVALID`。

### 11.3 F03：输入读取与字符串操作异常必须转为有界诊断

公共入口会执行**调用方自己的**代码：读取一个 `Mapping` 会调用它的 `items`/`__iter__`/`__getitem__`；非空校验会调用传入 `str` 子类的 `strip`。这些调用可能抛异常，而异常文本由调用方决定，可能携带输入值（URL、填写文本、合成标记）。

现在每个公共入口在自己的读取/校验边界把这些**可预期的依赖异常**转换为该入口的固定 code 与固定句子：

| 入口 | 转换后的 code |
|---|---|
| `parse_web_action` / `validate_web_action` | `WEB_ACTION_INVALID` |
| `parse_web_observation` | `WEB_OBSERVATION_INVALID` |
| `build_web_capabilities` | `WEB_CAPABILITIES_INVALID` |

转换后：

- 错误链被剥离——`__cause__` 与 `__context__` 都为 `None`，原始异常文本不会出现在 `str`/`repr`/公开属性里；
- 本包自己抛出的有界错误（含 `WebActionPayloadError`、Core `UnsupportedActionError`）**原样保留**，因此"未知动作先于 payload"的优先级不变，`WebActionPayloadError` 与 Core `InvalidActionPayloadError` 的兼容性也不变；
- 只转换普通 `Exception`：`KeyboardInterrupt`、`SystemExit`、取消等进程控制/中断信号**绝不吞掉**。

### 11.4 仍未改变的非目标

本节三项加固**没有**新增浏览器生命周期、页面执行、五规则验证器、真实站点访问或整段 `RuntimeDomainSupport`。公开导出集合仍是恰好 19 项，源码成员仍是恰好五个，依赖仍是 `forecontract-core>=0.1,<0.2` 与 `pydantic>=2,<3`。本文档只是产品接口说明，不是治理事实源；本轮仍然只贡献 `G-WEB-001-AC01` 的 schema 基础，不构成该 AC、Goal 或 P3 的整体验收。

## 12. round-003 剩余 F03 收口：调用方来源与 artifacts 读取边界

round-002 修好 F01/F02 与原 9 个失败输入；其独立复核另发现两个 F03 复现机制，共 10 个负向输入。本节说明 round-003 的收口，仍不新增任何动作、字段或能力。

### 12.1 机制一：异常“来源”不靠异常类判定

公共入口会执行**调用方自己的**代码：读取 `Mapping` 调用其 `items`/`__iter__`/`__getitem__`；非空校验调用传入 `str` 子类的 `strip`。调用方可以**自己构造** `WebValidationError` 或 Core `ActionValidationError` 并抛出，把任意 `code`、`path` 与消息（含输入标记）带出公共边界——若按异常类型直接重抛，就等于把本包的安全诊断权限交给调用方。

现在的规则是**按来源**而不是按类型：

- 只有调用方自己的读取在受控调用内执行，其中抛出的**任何** `Exception` 都转换为该入口的固定 code 与固定句子；
- 本包自身的语义拒绝**不在**受控调用内产生——读取先返回数据，判定再在读取之后进行——因此内部 `UnsupportedActionError`、`InvalidActionPayloadError`、`WebActionPayloadError` 的类型与优先级完全保持，而伪造的同名异常仍会被拒绝；
- 只捕获 `Exception`：取消、`KeyboardInterrupt`、`SystemExit` 继续传播；
- 转换后的错误在处理器之外抛出，`__cause__` 与 `__context__` 都为 `None`。

对应 9 个负向输入（3 个入口 × 伪造 Web 错误与伪造 Core 错误，加上 name/两条声明列表的伪造字符串操作）现在都得到固定 code、空 `path`、空异常链，且不回显标记。

### 12.2 机制二：`artifacts` 容器读取进入转换边界

`Observation.artifacts` 是调用方提供的容器。此前只有在容器**取出之后**才做嵌套改写，容器自身的迭代发生在转换边界之外，因此一个迭代即抛异常的列表派生对象会让原始 `ValueError` 直接逃逸。

现在先把容器在受控调用内**物化为普通 `list`**，再在这个普通列表上做嵌套 `ArtifactRef` 改写。于是：

- 容器迭代失败 → 固定 `WEB_OBSERVATION_INVALID`、空 `path`、空异常链；
- round-002 的嵌套校验强度不变：被 `model_copy` 污染的嵌套引用仍以 `artifacts[0].<field>` 路径拒绝；
- 合法嵌套引用与合法空 `targets`/`null` url 语义不变。

### 12.3 仍未改变

公开导出仍是恰好 19 项，源码成员仍是五个，依赖仍是 `forecontract-core>=0.1,<0.2` 与 `pydantic>=2,<3`；没有浏览器执行、五规则验证或整段 `RuntimeDomainSupport`。本文档是产品接口说明，不是治理事实源；本轮仍只贡献 `G-WEB-001-AC01` 的 schema 与纯校验基础，不构成该 AC、Goal 或 P3 的整体验收。

## 13. round-004 剩余 F03 收口：键哈希与条目相等比较

round-003 已收口“异常来源”与 `artifacts` 容器读取；其独立复核另发现两种机制、共 5 个负向输入。本节说明 round-004 的收口，仍不新增任何动作、字段、类型支持或强制转换。

### 13.1 机制一：物化容器不等于信任容器内的键（F03-C）

`Mapping.items()` 物化之后，返回的**键**仍可能是调用方提供的 `str` 派生对象。随后“键是否为本 envelope 声明字段”的判断会对键做哈希与相等比较，也就是执行调用方自己的 `__hash__`/`__eq__`；若其抛异常，三个 action/observation 公共入口此前会直接透传原始 `ValueError`。

现在键的成员判断本身作为**受控读取**执行：`isinstance(key, str) and key in allowed` 在转换边界内运行，抛出的任何普通异常都变成该入口的固定 code；返回的布尔答案再由本包判定，因此非字符串键与未知键仍以 `path ('<field>',)` 安全拒绝。

### 13.2 机制二：物化并 strip 之后，条目相等比较仍在边界外（F03-D）

能力列表物化并完成受控 `strip` 之后，条目仍是 `str` 派生对象。`item in declared`（重复判断）与 `item in allowed`（闭集判断）都会调用其 `__eq__`，两个列表此前都会直接透传原始 `ValueError`。

现在两次比较在同一次受控读取内执行，返回 `(repeated, known)` 两个布尔答案；真正的拒绝仍由本包在读取之后发出，因此：

- 伪造异常 → 固定 `WEB_CAPABILITIES_INVALID`、空 `path`、空异常链；
- 真实重复 → 仍为 `path ('supported_actions', 1)`；
- 真实闭集外条目 → 仍为 `path ('supported_actions', 0)`。

### 13.3 判别边界与不变量

- 普通 `str` 派生声明、普通 `Mapping`（含普通 `str` 派生键）与未知字段安全拒绝三个对照保持通过；
- 内部 `UnsupportedActionError`、`InvalidActionPayloadError`、`WebActionPayloadError` 的类型、优先级与具体 `path` 未变；
- 取消、`KeyboardInterrupt`、`SystemExit` 继续传播，绝不吞掉；
- 不新增类型接受范围或隐式转换：一个哈希即抛的键仍然被**拒绝**，只是拒绝方式变为有界诊断。

公开导出仍是恰好 19 项，源码成员仍是五个，依赖仍是 `forecontract-core>=0.1,<0.2` 与 `pydantic>=2,<3`。没有浏览器执行、五规则验证或整段 `RuntimeDomainSupport`。本文档是产品接口说明，不是治理事实源；本轮仍只贡献 `G-WEB-001-AC01` 的 schema 与纯校验基础。

## 14. round-005 剩余 F03-C 收口：键处理整条链（复制与保留键查找）

round-004 已把“键的成员判断”纳入受控读取；其独立复核发现同一条键处理链上还有两处调用方代码，共 6 个负向输入。本节说明 round-005 的收口，仍不新增字段、类型支持、强制转换或功能。

### 14.1 机制一：成员判断之后的存储再次哈希同一个键（key-copy）

`key in allowed` 只是键的**第一次**使用。随后 `fields[key] = item` 会把同一个调用方键对象插入内部字典，字典插入**再次调用其 `__hash__`**。若首次哈希成功、第二次抛异常，三个 action/observation 公共入口此前会在成员判断通过之后原样透传。

现在这一次存储同样在**受控读取**内执行：第二次哈希抛出的普通异常被转换为该入口的固定 code、空 `path`、空异常链。探针断言**恰好发生两次哈希**，因此可以证明失败点落在产品自身的复制步骤，而不是夹具。

### 14.2 机制二：保留在内部字典里的键在后续查找时再次比较（key-lookup）

重建字典并不会移除其中的调用方行为。一旦键被保留，本包随后“取 payload / metadata / state / artifacts”时，`name in fields` 与 `fields[name]` 仍会调用该保留键的 `__eq__`。若首次比较成功、第二次抛异常，三个入口此前会透传。

现在每次后续字段查找都在**各自的受控读取**内执行，并用一个内部哨兵区分“字段不存在”与“字段值为 `None`”，避免为了判断存在性而多做一次查找。探针断言**恰好发生两次比较**。

### 14.3 判别边界与不变量

- 普通 `str` 派生键（两个 envelope）与普通 `Mapping` 仍正常解析；未知字段仍以 `path ('<field>',)` 拒绝且不回显键名；
- 内部 `UnsupportedActionError`、`InvalidActionPayloadError`、`WebActionPayloadError` 的类型、优先级与具体 `path` 未变；
- 取消、`KeyboardInterrupt`、`SystemExit` 继续传播；
- 不直接拒绝所有 `str`/`Mapping` 派生对象，也不通过转换“洗白”非法内容：被拒绝的是**执行失败的调用方操作**，不是合法子类。

公开导出仍是恰好 19 项，源码成员仍是五个，依赖仍是 `forecontract-core>=0.1,<0.2` 与 `pydantic>=2,<3`。没有浏览器执行、五规则验证或整段 `RuntimeDomainSupport`。本文档是产品接口说明，不是治理事实源；本轮仍只贡献 `G-WEB-001-AC01` 的 schema 与纯校验基础。

## 15. round-006 剩余 F03-C 收口：读取之后的写回

round-005 已把成员判定、存储与后续查找全部纳入受控读取；其独立复核发现同一条键链上**读取之后的写回**仍会触碰保留键，共 6 个负向输入。本节说明 round-006 的收口，仍不新增字段、类型支持、强制转换或功能。

### 15.1 机制：写回再次比较同一个保留键

受控读取不会让之后的**写入**安全。`_take` 完成“字段是否存在”与“取值”两次成功查找之后，本包还要把改写后的值写回同一份文档：

- action：`raw[payload] = _thaw_frozen(value)`、`raw[metadata] = ...`
- observation：`raw[state] = ...`、`raw[artifacts] = _artifact_items(...)`

字典赋值会重新哈希被写入的字段名，并与文档中**保留的调用方键**比较。若该键前三次比较成功、第四次抛异常，两个 action 入口 × 两个字段与 observation 的两个字段共 6 个负向输入此前会原样透传。

现在写回同样在**独立受控读取**内执行，因此那第四次比较抛出的普通异常被转换为该入口的固定 code、空 `path`、空异常链、不回显输入。探针断言**恰好四次回调**，确保失败点落在产品自身的写回步骤。

### 15.2 判别边界与不变量

- 普通 action/observation 键（含 `metadata` 与 `artifacts` 同时出现）仍正常解析；
- 内部 `UnsupportedActionError`、`InvalidActionPayloadError`、`WebActionPayloadError` 的类型、优先级与具体 `path` 未变；
- 取消、`KeyboardInterrupt`、`SystemExit` 继续传播；
- 不直接拒绝普通 `str`/`Mapping` 派生对象，也不通过转换洗白非法内容：被拒绝的是**执行失败的调用方操作**。

公开导出仍是恰好 19 项，源码成员仍是五个，依赖仍是 `forecontract-core>=0.1,<0.2` 与 `pydantic>=2,<3`。没有浏览器执行、五规则验证或整段 `RuntimeDomainSupport`。本文档是产品接口说明，不是治理事实源；本轮仍只贡献 `G-WEB-001-AC01` 的 schema 与纯校验基础。

## 16. round-007 剩余 F03 收口：冻结容器的实参求值

round-006 已把写回纳入受控存储；其独立复核发现同一字段链上还有一处：**归一化结果作为实参求值**发生在进入受控存储之前，共 10 个负向输入。本节说明 round-007 的收口，仍不新增字段、类型支持、强制转换或功能。

### 16.1 机制：`FrozenMapping` 的底层读取发生在边界之前

Core 的 `FrozenMapping`/`FrozenList` 只是包装：读取它会执行其**底层映射**的 `__iter__` 与 `__getitem__`，而底层映射由调用方提供。此前写回是

    _store_field(raw, name, _thaw_frozen(value), message, code)

即 `_thaw_frozen(value)` 作为**实参**先被求值，早于 `_store_field` 内部的受控调用。于是五个入口/字段组合（action 的 payload、metadata × 两个入口，observation 的 state）各配两种回调（迭代、按下标取值）共 **10 个**负向输入，会把原始 `ValueError` 直接透传。该缺口在原始下发基线中同样存在，不是 round-006 引入的回归。

现在归一化本身成为**独立的受控读取**，随后的存储是第二段受控读取：

- 底层读取抛出的普通异常 → 该入口固定 code、空 `path`、固定 message、空异常链、不回显输入；
- 合法的 `FrozenMapping`/`FrozenList`/`FrozenActionSpec` 仍按原语义解包，普通 `str`/`Mapping` 子类不会被一刀切拒绝；
- 取消、`KeyboardInterrupt`、`SystemExit` 继续传播。

### 16.2 未受影响的部分

`artifacts` 早已有各自的容器边界，本轮**没有**再包一层：第二轮定义的嵌套污染路径 `artifacts[0].<field>` 与其迭代边界都保持不变。内部 `UnsupportedActionError`、`InvalidActionPayloadError`、`WebActionPayloadError` 的类型、优先级与具体 `path` 也保持不变。

公开导出仍是恰好 19 项，源码成员仍是五个，依赖仍是 `forecontract-core>=0.1,<0.2` 与 `pydantic>=2,<3`。没有浏览器执行、五规则验证或整段 `RuntimeDomainSupport`。本文档是产品接口说明，不是治理事实源；本轮仍只贡献 `G-WEB-001-AC01` 的 schema 与纯校验基础。

## 17. P3-T02 round-001：Playwright 浏览器适配器与生命周期

本节在产品接口层面说明 `forecontract-web` 0.1.0 新增的 **Playwright Web 适配器**。它建立在第 1–16 节已冻结的六动作、两定位与 `web.snapshot` 纯合同之上，**不修改**那五个既有源码成员、根包 19 项导出、两个直接依赖或 Core/Runtime/Planner 的任何接口。本节是产品接口说明，不是治理事实源，也不是验收结论；当前独立验证范围见 [发布验证说明](RELEASE_VALIDATION.zh-CN.md)。

### 17.1 公开入口与无 I/O 构造

```python
from forecontract_web.playwright_adapter import PlaywrightWebAdapter
from forecontract_web.browser_config import PlaywrightWebConfig
```

- `PlaywrightWebAdapter(config: PlaywrightWebConfig)` 只接受一个配置参数，**不做 I/O**：不启动浏览器、不导入 Playwright、不访问网络、不创建任务。它会严格重验并隔离保存传入配置（包括以 `model_construct`/`model_copy` 伪造的实例），之后调用方修改原列表或模型不会改变适配器行为。
- `name` 固定为 `"playwright_web"`。
- 完整实现 Core `Adapter`：同步无 I/O 的 `capabilities()`/`validate_action()`，以及异步的 `start()`/`stop()`/`observe()`/`execute(action)`；不新增 deadline、context、retry 或取消参数。
- 一个实例拥有一个 Playwright driver、一个 Browser、一个独立 BrowserContext 与一个活动 Page；不接管 CDP、既有浏览器或用户 profile，不做全局 context/profile/cookie 共享，也不设置全局 test-id 配置。
- 浏览器子模块在 **Playwright 未安装** 时仍可导入、构造、声明能力与执行纯校验；`start()` 才以固定的 `WEB_BROWSER_DEPENDENCY_MISSING` 失败，而不是抛出原始 `ImportError`。

### 17.2 `PlaywrightWebConfig`：四个必需字段

| 字段 | 类型 | 约束 |
|---|---|---|
| `allowed_origins` | `list[str]` | 非空、去重；只接受 `http`/`https` origin；host 仅 `localhost`、`127.0.0.1`、`::1`；effective port 由显式端口或 scheme 缺省端口确定并进入规范化比较；无 userinfo、查询、片段，路径仅允许一个尾斜杠 |
| `observation_targets` | `list[TestIdTarget \| RoleTarget]` | 显式、去重、允许为空；未知 strategy 与额外字段拒绝 |
| `operation_timeout_ms` | 严格有限正数 | 每一次 `start`/`observe`/`execute` 的整次调用上限 |
| `cleanup_timeout_ms` | 严格有限正数 | 整次 cleanup 尝试的独立上限 |

- 没有隐藏默认值；`bool`、字符串强转、`NaN`、`inf`、零与负数一律拒绝；额外字段拒绝。
- `allowed_origins` 的授权比较使用规范化的 `(scheme, host, effective port)` 三元组，而不是字符串前缀或子串；默认端口按 `http=80`、`https=443` 计算。单 origin 与显式多 origin 均支持，**没有隐式通配**。
- 配置不保存调用方的可变对象：访问器返回 detached 副本。
- 配置错误统一为固定 `code`、固定 `message`、固定 operation 标签的 `WebBrowserError`，不包含原始配置、URL、填充值或合成标记，且异常链为空。

### 17.3 生命周期状态表

状态为 `NEW`、`STARTING`、`RUNNING`、`STOPPING`、`STOPPED`、`FAILED`：

| 状态/调用 | 结果 |
|---|---|
| `NEW` / `start` | 进入 `STARTING`；每获得一个资源立即登记；全部就绪后才进入 `RUNNING` |
| `RUNNING` / `start` | 幂等；不创建第二套资源 |
| `STARTING` / `start` | 在第二次后端操作之前拒绝 `WEB_ADAPTER_BUSY` |
| `STARTING` / `observe` 或 `execute` | 在第二次后端操作之前拒绝 `WEB_ADAPTER_BUSY` |
| 非 `RUNNING` / `observe` 或 `execute` | 在任何 I/O 之前拒绝 `WEB_ADAPTER_NOT_RUNNING` |
| `NEW` / `stop` 或 `STOPPED` / `stop` | 幂等 `STOPPED`；不创建资源 |
| `RUNNING` / `stop` | 进入 `STOPPING`；按有限预算释放；成功则 `STOPPED` |
| `STARTING` / `stop` | 取消本次 acquisition 并释放已获得的部分资源；不会重新启动 |
| `FAILED` / `stop` | 对剩余句柄发起一次新的、宿主授权范围内的有限 cleanup |
| `STOPPING` / `stop` | 加入**同一次** cleanup；不重置其 deadline，不重复 close |
| `STOPPED` / `start` | 拒绝 `WEB_ADAPTER_STOPPED`；新的 Run 需要新实例 |
| cleanup 失败 | 进入 `FAILED`，并通过只读 `cleanup_error` 暴露固定诊断 |

- 同一实例上的并发 `observe`/`execute` 在进入浏览器之前被拒绝为 `WEB_ADAPTER_BUSY`。
- `start` 被取消或失败后**不会**留下 `RUNNING` 实例：适配器在有限 cleanup 中释放已获得资源，保持原始 `CancelledError`/启动失败的优先级，不会被 cleanup 失败掩盖；cleanup 失败通过 `cleanup_error` 保留可见。
- 资源释放顺序为 Page → Context → Browser → Playwright；一个普通 close 失败不会阻止其余独立 close；超时的 close 保留句柄以便后续诚实重试；只关闭自身拥有的句柄，不枚举或终止无关浏览器进程。该承诺不包含强杀、driver 失联、断电或存储故障后的 OS 级释放保证。

### 17.4 六动作与两定位

- 只复用既有 `web.navigate`、`web.click`、`web.fill`、`web.press`、`web.select`、`web.wait` 六个 payload，不新增字段、动作或 selector 入口。
- 定位仅 `data-testid` 与 `role + 精确 name`；使用 Playwright 原生 `get_by_test_id` / `get_by_role(exact=True)`，**不使用** `first`/`nth`/`force`/CSS/XPath/text/`frame`/`shadow`/任意 evaluate。
- `role` 查询在采集证据时包含隐藏节点，隐藏不会被误报为不存在；动作仍由原生 actionability 决定。
- `wait` 语义：`attached`/`visible` 需要最终唯一且达到状态；`detached`/`hidden` 的零匹配是既定的成功语义；多匹配从不“任选一个”。动作前会再次核对匹配数，不把旧计数当作永久 DOM 身份。
- 空 `fill.text` 与空 `select.value` 是合法值并原文传递。

### 17.5 本地网络策略

- 导航前先校验目标 URL；BrowserContext 对每个 HTTP(S) 请求执行同一套规范 origin 校验。
- Playwright 的路由回调不会收到浏览器内部重定向的下一跳，因此适配器在路由内以 `max_redirects=0` 自行取回响应，逐跳解析并校验 `Location`；只有最终**获准**的响应会交给浏览器。未获准的 scheme/host/port、缺失 `Location` 或超过上限的重定向链都会被 abort；未获准 origin 永远不被连接。
- 跨 origin 的获准重定向不会继承上一跳的 `Cookie`/`Authorization` 等敏感请求头。
- 固定 headless Chromium；不转发代理、CDP、profile、下载或任意 launch 参数；不关闭 TLS 校验；`service_workers="block"`；WebSocket 连接一律关闭。
- 不支持多页工作流：新弹出页面被跟踪、阻止其未获准请求并立即关闭，不切换到第二个活动页。该边界是应用层测试边界，不冒充 OS/网络协议隔离。

### 17.6 观察证据

`observe()` 返回既有 Core `Observation`，其中 `observation_id`/`timestamp` 真实生成、`environment_type="web"`、`state` 符合既有 `WebObservationState`；只查询 `config.observation_targets`。

- 0 匹配 → `collected`、`match_count=0`、全部属性 `null`；
- 多匹配 → `collected`、真实计数、全部属性 `null`（不合并不同节点）；
- 恰好 1 个 → `collected`、`match_count=1`，`visible`/`text`/`value` 从同一个捕获句柄独立读取；`null` 表示本次未获得，`""`/`false`/`0` 是真实证据；
- 单目标查询失败 → `unavailable`、`TARGET_QUERY_FAILED`，计数与属性均为 `null`，不会被伪造成零匹配。

属性不跨快照复用；缺失的目标记录不表示目标不存在；不宣称原子快照、完整 DOM、截图或任务完成裁决。整页/会话失联抛设施错误，而不是返回空目标列表或伪造的零匹配。

### 17.7 执行结果、UNKNOWN 与预算

- 动作先经过既有纯校验，再进入浏览器；校验异常保持原有 `UnsupportedActionError` / `InvalidActionPayloadError` / `WebActionPayloadError` 层次与安全诊断。
- `ActualOutcome` 只记录执行事实，不裁决 `PASS`/`FAIL` 或最终任务成功；不伪造 `state_before`/`state_after`/`artifacts`/服务端回执/证据。
- 确定性的交互前拒绝为 `failed`：`WEB_TARGET_NOT_FOUND`、`WEB_TARGET_AMBIGUOUS`、`WEB_TARGET_NOT_ACTIONABLE`。
- 已进入交互后，超时、断连或响应丢失无法确认结果时为保守 `unknown`：`WEB_ACTION_TIMEOUT` 或 `WEB_ACTION_INTERRUPTED`；绝不自动重试/重发 click、submit、fill 或 press，也不启动新的 Run。
- 每次公共调用只计算一个单调 deadline，向每个原生 wait/query 传递剩余时间；不按目标重置，不把 0 当作无限超时。cleanup 使用独立的整次上限，加入 `stop` 不重置该预算。Runtime 仍是 Run deadline、剩余预算与最终终态的唯一拥有者。

### 17.8 设施错误码

`WebBrowserError` 只定义在 `browser_errors.py`，包含固定 `code`、固定 `message` 与固定 operation 标签：

`WEB_BROWSER_CONFIG_INVALID`、`WEB_BROWSER_DEPENDENCY_MISSING`、`WEB_ADAPTER_NOT_RUNNING`、`WEB_ADAPTER_BUSY`、`WEB_ADAPTER_STOPPED`、`WEB_ORIGIN_BLOCKED`、`WEB_BROWSER_START_FAILED`、`WEB_BROWSER_UNAVAILABLE`、`WEB_OBSERVE_FAILED`、`WEB_OBSERVE_TIMEOUT`、`WEB_BROWSER_CLEANUP_FAILED`、`WEB_BROWSER_CLEANUP_TIMEOUT`。

公共诊断不复制后端原文、URL、输入值、未知键、`repr` 或 `cause`/`context`；不从后端消息文本推断错误码。`asyncio.CancelledError`、`KeyboardInterrupt`、`SystemExit` 始终传播。

### 17.9 分发与依赖

- 版本保持 `0.1.0`；根包公开导出保持**恰好 19 项**，既有五个合同源码成员字节不变。
- 新增六个浏览器源码成员，Web 源码成员因此为**恰好 11 个**。
- 直接依赖保持 `forecontract-core>=0.1,<0.2` 与 `pydantic>=2,<3`；只新增一个可选 extra：

```toml
[project.optional-dependencies]
browser = ["playwright==1.64.1"]
```

对应的安装元数据 requirement 带正确的 `extra == "browser"` marker。schema-only 安装/导入不依赖 Playwright 或浏览器二进制；browser extra 的 Chromium 由同版本 Playwright 管理。

### 17.10 测试与证据

- `tests/unit/test_playwright_web_adapter.py`：配置严格性、无 I/O 构造/声明/校验、生命周期状态表、部分 acquisition 释放、close 顺序/失败/超时/并发、预算递减、UNKNOWN 不重放、诊断安全与进程控制异常传播。
- `tests/unit/test_playwright_web_observation.py`：0/多/单匹配、`null`/`false`/空字符串/零、单目标不可用、属性不跨快照复用、角色隐藏节点、整会话失败与观察超时。
- `tests/integration/test_playwright_web_adapter.py`：测试内启动临时 loopback HTTP 服务器，以**服务端计数器与可观察 DOM 为独立 oracle**，用真实 headless Chromium 驱动六动作、唯一/歧义/隐藏定位、本地资源/重定向拒绝、UNKNOWN 不重复提交、正常/超时/取消/失败启动/stop 的真实资源释放，以及实例间 cookie 隔离。

这些测试证明本轮的适配器与浏览器生命周期行为；它们不证明五条验证规则、整段 `RuntimeDomainSupport`、完整故障站、P3-T05 E2E 或任何生产站点自动化。

## 18. P3-T02 round-002：F01–F06 修正

本节记录同一 12 条实施/测试/产品说明路径内、针对 round-001 复核所提 F01–F06 的产品接口级修正。它**不**新增动作、字段、网络权限、下载入口或全局浏览器参数，**不**改变第 1–16 节的纯合同、根包 19 项导出、五个旧合同源码成员、两个直接依赖或 11 个 Web 源码成员。第 17 节中关于重定向实现方式与预算执行的叙述，以本节为准；其余内容保持不变。本节仍是产品接口说明，不是治理事实源，也不是验收结论。

- **F01 配置读取原始身份。** 配置重验按调用方**原始存储的字符和键**取值，绝不通过调用方的 `__str__` 转换：伪造的 origin 字符串无法借 `__str__` 洗成合法 origin，未知键也无法被改名成 `strategy`。调用方自有的读取（如 `Mapping.items`）或 `strategy` 相等比较若抛出普通异常，在该真实边界转成固定 `WEB_BROWSER_CONFIG_INVALID`，异常链为空且不含原始值；普通 `str`、`str` 子类、`Mapping` 与定位模型的正向输入继续可用，保存的 origin 始终是 detached 的普通 `str`。
- **F02 整次操作与整次清理时限真实生效。** 每次 `start`/`observe`/`execute` 只计算**一个**单调 deadline，并由一个受控任务覆盖该次调用路径上的**全部** `await`，包括 `count` 读取、属性读取与 driver 调用；到点即以 `WEB_BROWSER_START_FAILED`/`WEB_OBSERVE_TIMEOUT`/`unknown + WEB_ACTION_TIMEOUT` 结束，绝不等待“取消本身被延迟”的收尾。这样的迟到工作被取消并登记为自有任务，不能再把实例变成 `RUNNING`，也不能再产生可见成功；启动迟到取得的句柄在剩余清理预算内关闭，超出该预算时尽力关闭并保留以便后续重试。整次 cleanup 尝试只有一个绝对上限，`stop()` 加入同一次尝试时不重置该上限，也不重复 close；主取消与启动失败的优先级和类型保持不变。
- **F03 交互后保守 UNKNOWN。** 交互一旦进入，任何无法确认结果的普通失败（超时、断连、响应丢失）一律返回 `unknown`，固定为 `WEB_ACTION_TIMEOUT` 或 `WEB_ACTION_INTERRUPTED`；不推断 `not_applied`，不构成 `failed`，不重放 click/submit/fill/press，也不改抛设施错误。交互前的确定性拒绝仍是独立的 `failed` 路径与固定目标错误码；交互前的设施/后端普通失败仍是独立的有界设施诊断；宿主取消仍直接传播。
- **F04 后端普通异常只在真实调用边界转换。** 只有单个原生 Playwright 调用的构造与 await 处于转换边界内：普通异常在该点变成安全固定诊断（交互前 `WEB_BROWSER_UNAVAILABLE`，交互后保守 UNKNOWN），不携带后端原文、标记、`repr` 或 `cause`/`context`。公共方法整体不被宽包，纯校验异常与内部语义错误不被吞掉，原生方法解析发生在边界之外；`asyncio.CancelledError`、`KeyboardInterrupt`、`SystemExit` 一律传播（进程控制异常只做“交还给调用任务”的传递，不转成任何公开诊断）。
- **F05 重定向保留原生语义。** 每个请求与每个服务端重定向跳仍然逐跳做 loopback origin 校验，但校验装在**本实例自有页面**的拦截层上，浏览器因此原生跟随重定向：真实最终文档 URL、最终 DOM/响应体、相对资源与 base-origin 解析都由浏览器自己产生，不再以“在策略层 fulfill 原始请求”的方式替换。未获准目的地在该跳发出之前即被拒绝（服务端计数为 0）；获准重定向每个 URL 只产生一次服务端请求，不重复动作、不重复请求。该拦截只作用于适配器自身的浏览器/页面，不接管外部 CDP、不连接既有浏览器、不转发 launch 参数或 CDP 设置；Playwright 路由回调继续为其它自有页面（如被立即关闭的 popup）执行同一 origin 策略。
- **F06 显式禁止下载。** BrowserContext 以 `accept_downloads=False` 创建，合成附件下载被 Chromium 显式拒绝：真实 headless Chromium 测试证明下载事件带非空 `failure()`、页面不会被导航到附件、`tmp_path` 与默认下载目录都不产生任何文件。不新增下载字段、公开参数或产品入口。

以上修正对应的单元证据在既有与新加节点中（配置原始身份与有界调用方失败、整次 observe/execute/start 预算、加入 `stop` 不重置清理预算、取消延迟启动不进入 `RUNNING`、交互后断连 UNKNOWN、交互前后端普通错误、内部语义错误不被吞、取消与进程控制异常传播）；真实浏览器证据在中止/获准重定向的真实最终 URL、最终 DOM、相对资源与未授权目的地 0 次计数，以及合成附件下载被拒绝且无文件落盘。

## 19. P3-T02 round-003：剩余 F01/F02/F05 修正

本节记录同一 12 条实施/测试/产品说明路径内、针对 round-002 复核所剩 F01/F02/F05 的产品接口级修正。它**不**新增动作、字段、模块、依赖、网络权限、下载入口、外部浏览器/CDP 接管或多页工作流，也**不**改变第 1–16 节的纯合同、根包 19 项导出、五个旧合同源码成员、11 个 Web 源码成员、两个直接依赖或 `playwright==1.64.1` 可选 extra。第 17、18 节中与本节不一致的叙述以本节为准；其余内容保持不变。本节仍是产品接口说明，不是治理事实源，也不是验收结论。

- **F01 存储模型字段/extra 的有界读取。** 当调用方以一个既有 `TestIdTarget`/`RoleTarget` 实例作为 observation target 时，适配器不信任该实例，而是重新读取它实际存储的字段：`__dict__`、`__pydantic_extra__`、extra 存储的真假判断以及去重身份的读取/哈希都在同一条调用方读取边界内执行。一个在属性、槽位、extra 存储读取或 `__bool__` 中抛出普通异常的伪造实例，统一转成固定 `WEB_BROWSER_CONFIG_INVALID`、operation `config`、空 `cause`/`context` 且不含任何原始值或合成标记；普通 `str`、派生字符串、`Mapping`、普通定位模型以及“无 extra 键”（`None` 或空映射）的正向输入继续可用。配置仍按调用方原始存储的字符和键取值，不做 `__str__` 洗白或强制转换。
- **F02 取消优先级、在途 owned await 与迟到关闭。** 宿主取消 `start` 时，即使并发 `stop` 已认领实例，原 `CancelledError` 的类型与身份仍直接传播，绝不改写成 `WEB_BROWSER_START_FAILED`，有界清理仍在原预算内执行。`stop` 会在同一次 cleanup 预算内取消并收束适配器自有的在途 `observe`/`execute` await：停在可取消门上的 fixture 操作被取消，不会在 `stop` 返回后迟到变成 `SUCCEEDED`；这**不**承诺撤回已经交给外部服务的副作用。退出 cleanup 尝试之后才失败或被取消的关闭不再被 consume 后丢弃：句柄保留，固定 `WEB_BROWSER_CLEANUP_FAILED` 通过 `cleanup_error` 可见，状态为 `FAILED` 而非 `STOPPED`，后续宿主授权的 `stop` 可在新预算内重试该句柄；迟到失败不扩展原 cleanup 上限。原有整次操作/清理上限、非重置 join、主错误优先级与 `BaseException` 传播保持不变。
- **F05 所有自有页面的逐跳 origin 策略。** 活动页仍在本实例页面的拦截层上逐跳校验，保留原生最终文档 URL、最终 DOM/响应体、相对资源与 base-origin 语义，且不重复请求或动作。每个 unsupported popup 在其 page 事件到达后立即安装同一逐跳策略，再按原逻辑关闭；被 Playwright 在页面/帧存在之前报告的 popup 初始导航请求无法归属到任何自有页面，也无法在该请求发出前安装逐跳拦截，因此一律在发出前拒绝，而不是放行后靠关闭页面补救。真实 Chromium 测试以正常关闭与受控延迟关闭两种方式证明：未获准目的地的服务端计数为 0；受控延迟关闭下，popup 已开放且已安装同一策略，获准首跳可被放行（服务端计数 1），但其重定向跳在发出前被拒绝（目的地计数仍为 0），因此保护不依赖关闭时机，也不依赖“先请求后关闭”。

上述修正对应的单元与真实浏览器证据均在既有及新加节点中：存储模型 extra 真值/槽位读取的有界拒绝与普通输入正向对照；宿主取消与并发 `stop` 的原类型传播；`stop` 对在途 `observe`/`execute` 的有界收束与无迟到副作用；`stop` 自身被宿主取消时类型保持且同一次 cleanup 仍完成；迟到失败 driver/page 关闭的句柄保留、错误可见与下一次 `stop` 重试；以及真实 popup 重定向在正常/受控延迟关闭下的未授权目的地 0 次计数。

## 20. P3-T02 round-004：F01 实际存储结构与实际条目

round-003 已把存储模型的 extra 真值与槽位读取纳入调用方读取边界；其独立复核证明同一 F01 仍有两类“字段洗白”机制：调用方的真假判断可以宣称存储为空，复制/枚举覆写可以让真实未知字段消失在复制视图之外。本节说明 round-004 的收口，仍**不**新增动作、字段、类型支持、隐式转换或功能，也**不**改变第 1–19 节的纯合同、根包 19 项导出、五个旧合同源码成员、11 个 Web 源码成员、两个直接依赖或 `playwright==1.64.1` 可选 extra。第 19 节中把 extra 的“真假判断”当作存储读取之一的叙述以本节为准；其余内容保持不变。本节仍是产品接口说明，不是治理事实源，也不是验收结论。

### 20.1 机制一：`__bool__`/`__len__` 返回假不能证明 extra 存储为空

一个既有 locator 的 `__pydantic_extra__` 可以是一个真实保存未知键的 `dict` 派生存储，同时覆写 `__bool__` 返回 `False`（或 `__len__` 返回 `0`）。round-003 之前以 `bool(extra)` 的真假作为“是否存在 extra 键”的证明，于是这个**真实存在**的未知键被当作不存在，配置重建时被静默丢弃。

现在 extra 存储的存在性按**结构**判定，而不是按调用方的真假答案：

- `None` 表示没有 extra 存储；
- `dict`（含派生类）按其**真实底层条目**判定：真实条目为空才是空；
- 任何真实条目都作为未知字段拒绝；非 `None` 且非 `dict` 的对象不是合法存储形态，同样拒绝；
- 判定过程不调用调用方的 `__bool__`/`__len__`，也不经过可覆写的 `keys`/`items` 视图。

### 20.2 机制二：`dict(stored)` 复制会丢掉真实存储的未知项

一个既有 locator 的 `__dict__` 可以是保存 `strategy`、`test_id` 与 `hidden` 的 `dict` 派生实例，但覆写 `keys`/`__iter__` 只宣布前两项。round-003 用 `dict(stored)` 复制后再重验，复制按调用方宣布的视图进行，`hidden` 被丢掉，于是配置重建错误接受了一个实际携带未知字段的实例。

现在两个存储都按真实结构与实际条目重验：

- 字段映射通过模型基类的真实存储描述符读取**实例的实际存储对象**，再用 `dict.items` 直接读取其底层条目，不调用被覆写的 `keys`/`items`/`__iter__`，也不以 `dict(stored)` 这类可能偏离真实存储的复制为准；
- 键仍按**原始存储字符**读取，值原样交给 schema 重验，因此未知键不能借复制或改名隐藏，也不会被“洗白”；
- 把 `__dict__`/`__pydantic_extra__` 替换成子类属性或另一个视图，不能顶替实例的真实存储。

### 20.3 判别边界与不变量

- 任一调用方读取失败（属性、槽位、存储或键读取）或任何真实未知条目，统一转成固定 `WEB_BROWSER_CONFIG_INVALID`、operation `config`、空 `cause`/`context`，且不包含原始值、键名或合成标记；
- 正向输入保持可用：干净普通 `dict` 与干净 `dict` 派生的存储映射、真实空 extra（含 `__bool__` 为假的空派生存储）、`None`/缺席 extra、普通定位模型、干净模型子类，以及既有的普通 `str` 派生与 `Mapping` 输入；
- 不削弱 round-002/round-003 的控制：原始字符读取、调用方读取异常的有界转换、模型/`str`/`Mapping` 正向与既有 F02/F03/F04/F05/F06 行为均不变；
- 只转换普通 `Exception`：取消、`KeyboardInterrupt`、`SystemExit` 继续传播；
- 不新增支持类型：被拒绝的是“执行失败的调用方操作”或“真实存在的未知条目”，不是合法子类。

对应单元证据在既有及新加节点中：真假为假但真实携带未知键的 extra 存储被有界拒绝（且真假回调从未被调用）；真实携带 `hidden`、复制视图缺失该键的字段映射被有界拒绝；干净存储映射、真实空/缺席 extra、干净模型子类与普通 `Mapping` 仍被接受。本文档仍只是产品接口说明，本轮仍只贡献 `G-WEB-001-AC01` 的 schema 与纯校验基础，不构成该 AC、Goal 或 P3 的整体验收。
