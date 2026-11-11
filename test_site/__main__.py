"""Standard-library CLI for manual browsing of the local test site.

All three arguments are required and explicit. There is no host, port, path, credential or
arbitrary configuration option: the listener always binds ``127.0.0.1:0``. ``--help`` parses and
exits without binding a port or starting a thread. After a successful start only the actual
origin is printed; the process keeps serving until Ctrl+C and then performs the same bounded
cleanup the Python host uses. Incomplete cleanup is a non-zero exit.

The oracle and the failure controls are host-only: this CLI intentionally exposes no HTTP admin,
oracle, reset or release route.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from test_site.models import LocalSiteError, Scenario, SiteConfig
from test_site.server import LocalTestSite

__all__ = ["main"]

_EXIT_CONFIG = 2
_EXIT_START = 1
_EXIT_CLEANUP = 1


def _budget(text: str) -> int:
    """Parse one explicit whole-millisecond budget in ``1..60000``."""
    if not text or len(text) > 5 or not text.isascii() or not text.isdigit():
        raise argparse.ArgumentTypeError("a whole millisecond budget is required")
    value = int(text)
    if value < 1 or value > 60000:
        raise argparse.ArgumentTypeError("the budget must be in 1..60000")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -B -m test_site",
        description="Run the controllable local test site on an ephemeral loopback port.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=[scenario.value for scenario in Scenario],
        help="the fixed server mode of this process",
    )
    parser.add_argument(
        "--request-timeout-ms",
        required=True,
        type=_budget,
        help="one absolute per-request budget in milliseconds (1..60000)",
    )
    parser.add_argument(
        "--cleanup-timeout-ms",
        required=True,
        type=_budget,
        help="one absolute cleanup budget in milliseconds (1..60000)",
    )
    return parser


def _wait_for_interrupt(site: LocalTestSite) -> None:
    """Keep the process alive until Ctrl+C; the site itself owns all server state."""
    while site.snapshot_oracle().running:
        time.sleep(0.2)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its process exit code; no exception text is printed."""
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else _EXIT_CONFIG
    try:
        config = SiteConfig(
            scenario=Scenario(arguments.scenario),
            request_timeout_ms=arguments.request_timeout_ms,
            cleanup_timeout_ms=arguments.cleanup_timeout_ms,
        )
    except LocalSiteError:
        print("test_site: invalid configuration", file=sys.stderr)
        return _EXIT_CONFIG
    site = LocalTestSite(config)
    try:
        site.start()
    except LocalSiteError:
        print("test_site: could not start", file=sys.stderr)
        return _EXIT_START
    print(site.origin, flush=True)
    try:
        try:
            _wait_for_interrupt(site)
        except KeyboardInterrupt:
            pass
    finally:
        result = site.stop()
    if not result.complete:
        print("test_site: cleanup incomplete", file=sys.stderr)
        return _EXIT_CLEANUP
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
