"""Deterministic fixed pages of the local test site.

Nothing here loads an external resource, executes JavaScript or maps a caller path onto a file.
The login page carries exactly one ``username``, one ``password`` and one ``generation`` form
field, with the fixed ``data-testid`` anchors and accessible labels the Web observation vocabulary
can address. The error marker and the overlay marker are always present in the DOM and are hidden
until their state applies, so "hidden" and "absent" remain different observations.

The synthetic fixture credentials below are public test data. They are deliberately not a
secret, are never written to an oracle, a cookie name or a log, and only the server compares
them. The submitted password never appears in a response body.
"""

from __future__ import annotations

__all__ = [
    "FIXTURE_PASSWORD",
    "FIXTURE_USERNAME",
    "SUCCESS_TEXT",
    "render_dashboard_page",
    "render_login_page",
]

#: The one fixed public synthetic login identity of this local site.
FIXTURE_USERNAME = "test-user"
FIXTURE_PASSWORD = "test-password"

#: The fixed success text of the dashboard heading.
SUCCESS_TEXT = "登录成功"

_LOGIN_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>本地测试登录</title>
<style>
html { color-scheme: light; }
body { margin: 0; font-family: system-ui, sans-serif; background: #f4f5f7; color: #1d2430; }
main { width: 22rem; margin: 4rem auto; padding: 2rem; background: #ffffff; border-radius: .5rem; }
h1 { margin: 0 0 1.25rem; font-size: 1.35rem; }
.row { margin: 0 0 1rem; }
label { display: block; margin: 0 0 .35rem; font-weight: 600; }
input[type="text"], input[type="password"] {
  box-sizing: border-box; width: 100%; padding: .55rem; border: 1px solid #9aa4b2;
  border-radius: .3rem; font: inherit;
}
.error { margin: 0 0 1rem; padding: .55rem; border: 1px solid #c0392b; color: #8e2418; }
.submit-row { position: relative; display: inline-block; }
button { padding: .6rem 1.5rem; border: 0; border-radius: .3rem; background: #1f5fbf; color: #fff; font: inherit; }
.overlay { position: absolute; inset: 0; z-index: 2; cursor: not-allowed; background: rgba(40, 50, 65, .55); }
</style>
</head>
<body>
<main>
<h1>本地测试登录</h1>
<form method="post" action="/login" novalidate>
<div class="row">
<label for="username">用户名</label>
<input id="username" name="username" data-testid="username" type="text" autocomplete="username" required>
</div>
<div class="row">
<label for="password">密码</label>
<input id="password" name="password" data-testid="password" type="password" autocomplete="current-password" required>
</div>
<input type="hidden" name="generation" value="__GENERATION__">
<p data-testid="login-error" role="alert" class="error"__ERROR_HIDDEN__>用户名或密码错误</p>
<div class="submit-row">
<button type="submit" data-testid="submit">登录</button>
<div data-testid="overlay" class="overlay" aria-hidden="true"__OVERLAY_HIDDEN__></div>
</div>
</form>
</main>
</body>
</html>"""

_DASHBOARD_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>本地测试站点</title>
<style>
html { color-scheme: light; }
body { margin: 0; font-family: system-ui, sans-serif; background: #f4f5f7; color: #1d2430; }
main { width: 22rem; margin: 4rem auto; padding: 2rem; background: #ffffff; border-radius: .5rem; }
h1 { margin: 0 0 1rem; font-size: 1.35rem; }
</style>
</head>
<body>
<main>
<h1 data-testid="welcome">__SUCCESS_TEXT__</h1>
<p>已建立当前代的测试会话。</p>
</main>
</body>
</html>"""


def _with_hidden(template: str, generation: int, *, error: bool, overlay: bool) -> str:
    """Fill the one fixed template; only the generation integer is substituted."""
    return (
        template.replace("__GENERATION__", str(generation))
        .replace("__ERROR_HIDDEN__", "" if error else " hidden")
        .replace("__OVERLAY_HIDDEN__", "" if overlay else " hidden")
    )


def render_login_page(*, generation: int, error: bool = False, overlay: bool = False) -> str:
    """Return the fixed login page for ``generation`` in the requested visible state."""
    if type(generation) is not int or generation < 1:
        raise ValueError("generation must be a positive integer")
    return _with_hidden(_LOGIN_PAGE, generation, error=error, overlay=overlay)


def render_dashboard_page() -> str:
    """Return the fixed authenticated page; it carries no session identifier or counter."""
    return _DASHBOARD_PAGE.replace("__SUCCESS_TEXT__", SUCCESS_TEXT)
