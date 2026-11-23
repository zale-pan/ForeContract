# Core 验证扩展说明：组合验证与离线模型结果解释

本文档说明 `forecontract.verification` 新增的两个公共扩展接口：`CompositeVerifier` 与 `ModelVerifier`。二者均为无状态、离线、确定性的三态验证设施，返回既有 `schema_version="0.1"` 的 `VerificationResult`，不改变 `Runtime` 终态裁决权、不替代独立的 `TaskCompletionOracle`、不引入任何外部网络或模型 SDK 依赖。

## 1. 公共导出与架构边界

`forecontract.verification` 新增且仅新增两个公共导出：

- `CompositeVerifier`：在确定性关键门禁（critical gate）优先的前提下，对确定性验证项与模型验证项按 `all`、`any`、`weighted` 模式进行三态组合。
- `ModelVerifier`：对外部提供的模型请求与响应记录进行完整结构校验、身份绑定核对与三态转换。

`forecontract.verification.extension_models` 仅承载内部校验结构与辅助函数，不作为公共 API 导出。

## 2. `CompositeVerifier` 接口与语义

### 2.1 调用签名

```python
class CompositeVerifier:
    def verify(
        self,
        *,
        items: list[dict[str, object]],
        mode: str,
        threshold: float | None = None,
    ) -> VerificationResult: ...
```

### 2.2 输入条目与校验顺序

`items` 必须为包含 `1`–`128` 个普通字典的 `list`。每个条目必须精确包含以下 7 个键（不得缺失或包含额外键）：

- `id`：长度 `1`–`128` 的非空白字符串（在单次调用内必须唯一）。
- `status`：`"pass"`、`"fail"` 或 `"inconclusive"`。
- `kind`：`"deterministic"` 或 `"model"`。
- `critical`：严格布尔值（`bool`）；仅允许 `kind="deterministic"` 的条目设置 `critical=True`。
- `weight`：非 `bool` 的有限非负数值（`int | float`），在所有模式下均执行校验。
- `source_id`：长度 `1`–`128` 的非空白字符串。
- `reason`：长度 `1`–`1024` 的非空白字符串。

校验严格按以下顺序执行：
1. **完整结构与字段校验**（触发 `VERIFICATION_INPUT_INVALID`）：先校验 `items` 容器及全部子项的结构、类型与取值范围。
2. **绑定校验**（触发 `VERIFICATION_BINDING_INVALID`）：拒绝重复 `id` 以及 `kind="model"` 且 `critical=True` 的条目。
3. **聚合规则校验**（触发 `VERIFICATION_RULE_INVALID`）：
   - `mode` 必须为 `"all"`、`"any"` 或 `"weighted"`。
   - `all` / `any` 模式下 `threshold` 必须为 `None`。
   - `weighted` 模式下 `threshold` 必须为 `(0, 1]` 内的有限非布尔数值，且 `items` 总权重必须大于 `0`。

### 2.3 关键门禁与三态聚合规则

1. **关键门禁优先（C01）**：
   - 若任一 `critical=True` 条目为 `fail`，最终结果为 `FAIL`（`reason="CRITICAL_FAILED"`）。
   - 若无关键 `fail` 但任一 `critical=True` 条目为 `inconclusive`，最终结果为 `INCONCLUSIVE`（`reason="CRITICAL_INCONCLUSIVE"`）。
   - 仅当所有关键条目均为 `pass`（或无关键条目）时，才应用 `mode` 聚合规则。
2. **模式规则**：
   - `all`：存在 `fail` 则 `FAIL`；全部 `pass` 则 `PASS`；其余为 `INCONCLUSIVE`。
   - `any`：存在 `pass` 则 `PASS`；全部 `fail` 则 `FAIL`；其余为 `INCONCLUSIVE`。
   - `weighted`：`pass` 贡献 `[w, w]`，`fail` 贡献 `[0, 0]`，`inconclusive` 贡献 `[0, w]`；按最大有限权重缩放后以总权重归一化得到区间 `[lower, upper]`。若 `lower >= threshold` 则 `PASS`，若 `upper < threshold` 则 `FAIL`，否则为 `INCONCLUSIVE`。组合规则对应的 `reason` 为 `COMPOSITE_PASSED`、`COMPOSITE_FAILED` 或 `COMPOSITE_INCONCLUSIVE`。

返回 `VerificationResult` 的 `verifier="composite"`，`outcome_results=[]`，`metadata` 精确包含 `scope="evidence_only"`、`mode`、`threshold`、`items`（按输入顺序深复制）、`lower`、`upper`。

## 3. `ModelVerifier` 接口与语义

### 3.1 调用签名

```python
class ModelVerifier:
    def verify(
        self,
        *,
        request: dict[str, object],
        response: dict[str, object],
    ) -> VerificationResult: ...
```

### 3.2 请求与响应结构

- `request` 必须精确包含 4 个身份键：`request_id`、`model_id`、`prompt_version`、`evidence_id`（均为长度 `1`–`128` 的非空白字符串）。
- `response` 必须精确包含上述 4 个身份键及 `state`、`response_id`、`status`、`confidence`、`reason`：
  - `state="ok"`：`response_id` 为 `1`–`128` 非空白字符串；`status` 为 `"pass"` / `"fail"` / `"inconclusive"`；`confidence` 为 `None` 或 `[0, 1]` 内的有限非布尔数值。`inconclusive` 不会因高 `confidence` 被提升为 `PASS`。
  - `state="refused"`：`response_id` 为 `1`–`128` 非空白字符串；`status` 必须为 `"inconclusive"`；`confidence` 必须为 `None`。
  - `state="unavailable"` 或 `"timeout"`：`response_id`、`status`、`confidence` 必须全部为 `None`（离线上游状态记录，不伪造响应标识或分数）。
- 先完成 `request` 与 `response` 的完整结构校验（`VERIFICATION_INPUT_INVALID`），再核对 4 个身份键完全一致（不一致触发 `VERIFICATION_BINDING_INVALID`）。

### 3.3 返回结果

- `verifier="model_offline"`，`outcome_results=[]`。
- `reason` 映射为 `MODEL_PASSED`、`MODEL_FAILED`、`MODEL_INCONCLUSIVE`、`MODEL_REFUSED`、`MODEL_UNAVAILABLE` 或 `MODEL_TIMEOUT`。
- `metadata` 精确包含 `scope="evidence_only"`、`request`、`response`（深复制已验证输入）；`VerificationResult` 模型本身不新增 `confidence` 字段。
