# test_site — 本地可控测试站点

`test_site` 是一个**独立、仅标准库**的本地 HTTP 测试站点，不是 Adapter，也不是 Core 组件。
它只绑定 `127.0.0.1:0`（由操作系统选择临时端口），不接受固定端口、公网接口、真实账号、
API key、任意文件路径映射、JavaScript 动作、插件、回调或外部资源。导入 `test_site` 不会
启动服务、创建线程、绑定端口或写文件。

它提供固定的合成登录页、四种服务端模式、五个验收场景，以及**仅宿主可调用**的 oracle 与
故障控制。HTTP 上不存在 `/oracle`、`/admin`、`/reset`、`/release` 等控制入口；这些路径与
其他未知路径一样返回 404。

本组件不实现运行时联动、完整 Runtime 流程或 P3-T05 的验收；它不把整个 Goal/AC 或 P3 阶段
标为完成。

## 启动命令

```text
python -B -m test_site --scenario normal --request-timeout-ms 15000 --cleanup-timeout-ms 5000
```

三个参数均必填，`--scenario` 只接受下表四个值。启动成功后**只输出实际 origin**
（例如 `http://127.0.0.1:54321`），随后保持服务直到 Ctrl+C；退出时执行同一套有界清理。
参数或启动错误返回非零；清理不完整返回非零且只输出固定诊断。`--help` 不绑定端口、不启动
线程。

## 固定公开合成凭据

以下值只用于本地测试，是公开的合成数据，不是 secret；oracle 和日志不记录提交的密码或
请求正文。

```text
用户名: test-user
密码:   test-password
```

## 四种模式

| `Scenario` | 服务端行为 |
| --- | --- |
| `normal` | 正确凭据立即提交、建立当前代 session、`303 /dashboard`；错误凭据立即返回错误页。 |
| `delayed_login` | 正确凭据通过结构与代校验后，在提交**之前**等待宿主 `release()`；放行后只提交一次。自身请求 deadline 先到则 `503` 且零提交。 |
| `blocked_submit` | 登录页用 CSS 遮挡层覆盖提交按钮并拦截指针，不覆盖输入框；服务端认证合同不变，直接合法 HTTP POST 不被遮挡阻止。`release()` 后在重新加载、重新填表时可成功一次。 |
| `commit_then_stall` | 正确凭据先提交并登记 session，再关闭响应门；门释放前不发送成功响应头或 cookie。自身 deadline 先到则直接关闭连接，已提交事实不撤销。 |

错误凭据在四种模式下都立即失败：`login_requests + 1`、`auth_failures + 1`、零提交、零
session，不进入任何延迟/响应门。

## 五个验收场景

| 场景 | 独立断言 |
| --- | --- |
| 正常登录 | `login_requests=1`、`auth_successes=1`、`auth_failures=0`、`commits=1`；成功页可认证访问。 |
| 错误凭据 | `login_requests=1`、`auth_failures=1`、`commits=0`、有效 session 为 0；不能访问成功页，伪造/旧代 cookie 不认证。 |
| 提交前延迟 | `LOGIN_RECEIVED` 后可等到 `RESPONSE_BLOCKED`；放行前 requests=1、commits=0、successes=0；放行后一次提交；自身超时 `503`、零提交。 |
| 按钮遮挡 | 真实 Adapter 点击被遮挡层拦截，服务端 requests/commits 均 0；`release()` + 重新加载/填表后一次成功；直接 HTTP POST 仍按认证合同处理。 |
| 提交后故障 | `LOGIN_COMMITTED` 与 `RESPONSE_BLOCKED` 后可独立证明确实提交一次而客户端未确认；Adapter 预算先到时为既有 `unknown`；观察与清理不再 POST，commits 仍为 1；有意重复 POST 使 commits 变为 2。 |

“请求到达”“认证通过”“提交发生”“响应可见”是四个不同事实。响应错误或 Adapter 的
`succeeded` 不能反推服务端是否提交；**执行结果 unknown 不等于未提交**。客户端操作预算先
到时，观察和清理都不得再次 POST。

## 固定页面定位点

| `data-testid` / 角色 | 用途 |
| --- | --- |
| `username` | 用户名输入框，带可访问标签“用户名”。 |
| `password` | 密码输入框，带可访问标签“密码”。 |
| `submit` | 提交按钮；角色 `button`，可访问名称固定为“登录”。 |
| `login-error` | 固定错误态标记，平时 `hidden`，错误凭据时可见。 |
| `overlay` | 遮挡层标记，`blocked_submit` 未释放时可见，释放后 `hidden`。 |
| `welcome` | 成功页标题，角色 `heading`，固定成功文案“登录成功”。 |

页面没有 JavaScript、没有外部资源、没有随机 session 标识、内部计数、故障模式或 oracle
字段。`generation` 只用于拒绝旧表单，不是成功判据。

## 宿主接口示意

```python
from test_site import LocalTestSite, Scenario, SiteConfig, SiteEvent

site = LocalTestSite(
    SiteConfig(
        scenario=Scenario.DELAYED_LOGIN,
        request_timeout_ms=15000,
        cleanup_timeout_ms=5000,
    )
).start()
try:
    print(site.origin)
    assert site.wait_for(event=SiteEvent.LOGIN_RECEIVED, timeout_ms=5000)
    assert site.wait_for(event=SiteEvent.RESPONSE_BLOCKED, timeout_ms=5000)
    before = site.snapshot_oracle()
    assert before.login_requests == 1 and before.commits == 0
    site.release()
    assert site.wait_for(event=SiteEvent.LOGIN_COMMITTED, timeout_ms=5000)
    after = site.snapshot_oracle()
    assert after.commits == 1
finally:
    result = site.stop()
    assert result.complete, result
```

`stop()` 幂等且终止后不能重启；需要新服务时创建新实例。`reset(scenario=...)` 仅在运行且
`active_requests=0` 时成功，否则 `SITE_BUSY` 且所有状态不变；成功时 generation 递增、计数与
事件清空、场景门重新关闭、旧 session 全部撤销。旧表单和旧 cookie 不能污染新代。
`wait_for` 是宿主侧有界等待，不使用固定 sleep 猜测服务端阶段。

## HTTP 合同

| 请求 | 行为 |
| --- | --- |
| `GET /` | `303` 到 `/login`。 |
| `GET /login` | `200 text/html; charset=utf-8`，固定表单，含当前 generation。 |
| `POST /login` | 仅 `application/x-www-form-urlencoded`；字段恰为 `username`、`password`、`generation` 各一次。 |
| `GET /dashboard` | 当前代有效 cookie 时 `200` 成功页，否则 `303` 到 `/login`。 |
| 未知路径 | `404` 固定文本。 |
| 已知路径的不支持方法 | `405`，无认证/提交副作用。 |

严格拒绝：非法表单结构、非法或旧 generation、非法 UTF-8/表单编码为 `400`；不支持
`Content-Type` 为 `415`；请求体超过 4096 字节为 `413`；缺失/重复/非法 `Content-Length` 或
任何 `Transfer-Encoding` 为 `400`。这些拒绝都不改变 `login_requests`、`auth_*` 或 `commits`。
`generation` 只接受 ASCII 十进制正整数，无正号、空白或前导零。

## 预算、代隔离与清理判据

- `request_timeout_ms` 与 `cleanup_timeout_ms` 都是严格整数毫秒，范围 `1..60000`；`bool`、
  字符串数字和越界值都不接受。
- 一次请求从读取请求行与头部、读取请求体、门等待到发送响应，共用一个绝对 monotonic 请求
  deadline，不会为每个阶段重新计时，也不会在读完头部后重置。合法但分片到达的请求体在预算内
  经过短暂轮询后继续读取；提交会在状态锁边界内复核同一 deadline，响应发送前若预算已到期也
  只发送固定 `503`，不会发出正常的 `200`/`303`。deadline 到期后只保留提交前固定 `503` 或
  提交后直接关闭连接的语义。
- `stop()` 先拒绝新工作并唤醒门与事件等待，再关闭监听、在**一个总 monotonic cleanup
  deadline** 内关闭所有自有连接并 join 所有自有线程；实例、状态与连接跟踪等所有内部协调锁
  的获取以及资源计数同样受这一 deadline 约束，被其他方暂时持有的协调锁不会让 `stop()` 越过
  预算等待。同一 deadline 也约束端口释放证明，不会另用固定的探测预算。`complete=True` 仅在监听端口不可连接、自有线程/连接全部关闭且
  `active_requests=0` 时为真。短预算下无法在剩余时间内完成或证明时会如实返回
  `complete=False` 与真实剩余资源数量；调用方、CLI 和测试必须显式检查，不能无限等待。
- 站点不使用 daemon 线程或只置 `stopped` 标记来伪装清理完成。

## 限制

站点是本机 HTTP 上的测试工具，不声明生产认证强度；cookie 只用于本机测试（`HttpOnly`、
`SameSite=Lax`、`Path=/`，不设置 `Secure`）。它不加载外部资源、不执行站点 JavaScript、不
提供任意代码执行或文件读取能力。
