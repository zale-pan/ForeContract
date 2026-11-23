# 派生只读记忆索引说明：`DerivedMemoryIndex`

本文档说明 `forecontract.memory.derived.DerivedMemoryIndex` 的接口契约、宿主目录装载方式、来源局限、固定离线评估结论及明确非目标。本模块仅提供只读、来源可核对的标签派生索引，**不修改既有 `JsonlEpisodicMemory` 存储格式、不更改 `forecontract.memory` 根导出（保持 14 个既有导出）、不引入新的持久存储后端**。

## 1. 接口边界与导入方式

- 公共入口（唯一新增公共类）：`from forecontract.memory.derived import DerivedMemoryIndex`
- 内部校验模块：`forecontract.memory.derived_models` 仅承载内部数据结构与边界校验，不作为公共 API 导出。
- 构造与查询签名：

```python
from datetime import datetime
from forecontract.memory import JsonlEpisodicMemory
from forecontract.memory.derived import DerivedMemoryIndex

class DerivedMemoryIndex:
    def __init__(
        self,
        *,
        episodic_memory: JsonlEpisodicMemory,
        entries: list[dict[str, object]],
    ) -> None: ...

    def query(
        self,
        *,
        tags: list[str],
        now: datetime,
        kind: str | None = None,
        limit: int = 5,
    ) -> dict[str, object]: ...
```

无 `append`、`delete`、`save` 或 `execute` 接口；构造时不读取或修改磁盘文件；查询显式接收带时区的 `now`，不调用系统墙钟，不修改输入、索引、环境变量、GC 或持久化文件。

## 2. 宿主目录装载示例（普通 JSON 由宿主管理）

宿主可将外部授权的语义说明（`semantic`）或高层步骤目录（`procedural`）保存为普通 JSON 文件，并在需要时重新构建只读 `DerivedMemoryIndex`（Core 本身不读写该 JSON 文件）：

```python
import json
from datetime import datetime, timezone
from pathlib import Path
from forecontract.memory import JsonlEpisodicMemory, TraceSourceResolver
from forecontract.memory.derived import DerivedMemoryIndex
from forecontract.tracing import JsonlTraceStore

trace = JsonlTraceStore(Path("external-scratch/trace.jsonl"))
episodic = JsonlEpisodicMemory(
    Path("external-scratch/memory.jsonl"),
    source_resolver=TraceSourceResolver(trace),
)
entries = json.loads(Path("external-scratch/host_catalog.json").read_text(encoding="utf-8"))
index = DerivedMemoryIndex(episodic_memory=episodic, entries=entries)

result = index.query(
    tags=["login", "form"],
    now=datetime(2026, 11, 14, 0, 0, tzinfo=timezone.utc),
    kind=None,
    limit=5,
)
```

每个条目须严格包含 11 个字段：`schema_version`（`"0.1"`）、`entry_id`、`kind`（`"semantic"` / `"procedural"`）、`memory_id`、`run_id`、`source_record_id`、`source_level`（`"run"` / `"attempt"`）、`tags`（`1..32` 个唯一小写标签 `[a-z0-9][a-z0-9_-]{0,31}`）、`text`、`steps`、`expires_at`（`None` 或带时区的 ISO8601 字符串，最长 40 字符）。单次索引最多容纳 64 条唯一 `entry_id` 记录。

## 3. 确定性来源校验、筛选优先级与排序规则

每次调用 `query()` 都会在校验参数后重新向传入的 `episodic_memory` 发起校验：
1. 调用 `episodic_memory.query()` 获取经真实 `TraceSourceResolver` 校验的当前链头（heads）。
2. 对目录中每个不同的 `memory_id` 至多调用一次 `episodic_memory.get(memory_id)`；无跨查询缓存。
3. 校验全部条目的 `memory_id`、`run_id`、`source_record_id`、`source_level` 与真实 `MemoryItem` 完全一致。即使存在 `limit=0`、`no_match` 或已过期条目，也不得跳过全目录来源校验：`MemoryNotFoundError` 与身份不匹配统一抛出有界 `MemorySourceError`；既有 `MemoryCorruptionError` 与 `MemoryStorageError` 保持原异常类别。

所有来源通过校验后，按以下固定优先级给出排除理由（多原因命中时仅取首项）：
1. `superseded`：该 `memory_id` 已非当前链头。
2. `source_not_success`：对应 `MemoryItem.result != "success"`。
3. `source_in_future`：`source.created_at > now`。
4. `expired`：`expires_at is not None and now >= expires_at`。
5. `kind_mismatch`：指定 `kind` 与条目的 `kind` 不一致。
6. `no_match`：条目标签与查询标签交集为空。

候选得分定义为 `score = 匹配的查询标签数 / 查询标签总数`（位于 `(0.0, 1.0]`），按 `(-score, entry_id)` 确定性排序，取前 `limit` 条作为 `hits`（超限候选记录为 `excluded` 且 `reason="limit"`），`excluded` 最终按 `entry_id` 升序排列。返回值对全部嵌套容器深复制，不返回任何 `is_fact`、`verified`、`COMPLETED` 或执行权字段。

## 4. 来源局限与错误边界

- **`source_not_success` 与 `success` 的含义局限**：`source.result == "success"` 仅表示底层 `MemoryItem` 对应的 `run` 或 `attempt` 事实在 Trace 映射中为 `"success"`；它**不证明**宿主所写 `text` 或 `steps` 客观为真，也**不能**将某次 `attempt` 成功提升为整个任务成功（`COMPLETED`）。
- **建议性质**：返回结果仅作为只读检索建议与注释；`procedural.steps` 为人类可读的高层步骤描述，永远不会被解释或执行。
- **有界错误**：结构或参数错误抛出 `MemoryValidationError`，所有公开异常消息固定有界（`<= 240` 字符），不回显未知字段名、原始输入值、完整文件路径或底层异常文本。

## 5. 固定合成查询离线评估摘要

在控制集合与集成测试使用的固定 4 组合成查询 oracle 上，开启标签重叠度排序与来源有效性过滤后的离线对比如下：

| 指标 | 按 `entry_id` 排序基线（相同合法过滤） | 规定规则（来源过滤 + `(-score, entry_id)` 排序） |
| :--- | :--- | :--- |
| Exact-list accuracy（4 组查询） | `3/4` | `4/4` |
| Top-1 accuracy（3 组非空 oracle） | `2/3` | `3/3` |
| 无效来源（非 `success` / `superseded` / 过期 / 未来）命中数 | `0` | `0` |

> **说明**：上述指标仅刻画该小型合成目录与固定查询集合下的确定性排序与过滤行为，**不用于宣称具备 embedding 语义检索能力、自动知识归纳效果或真实大模型收益**。

## 6. 明确非目标（本轮不包含）

- 不新增向量库、embedding 索引或近似语义搜索。
- 不新增自动知识抽取、摘要生成、自动删除/过期清理或链头改写。
- 不自动执行 `procedural` 步骤，不接入 `Planner` 或 `Runtime` 自动消费。
- 不新增持久化目录文件读写 API、后台缓存或网络/LLM API 调用。
