"""
TOVAH v14 tools/browser.py — Browser automation via Playwright.

v14-RC2 HARDENING:
  - parse_browser_command() is pure parsing, no side effects, no imports
  - browser_action() has explicit mode: parse_only / dry_run / runtime_required
  - Missing playwright returns fast typed failure, never hangs
  - Chromium install cached after first success
"""
from __future__ import annotations

import logging
import sys
import time as _time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

# Module-level chromium install state
_playwright_browser_ready = False


@dataclass
class BrowserCommand:
    """Parsed browser command — pure data, no side effects."""
    action: str
    url: str = ""
    selector: str = ""
    text: str = ""
    raw: str = ""


def parse_browser_command(
    action: str,
    url: str = "",
    selector: str = "",
    text: str = "",
) -> BrowserCommand:
    """Parse a browser command string. PURE FUNCTION — no I/O, no imports.

    Handles pipe-delimited compound actions:
      "extract_text|http://example.com" -> BrowserCommand(action="extract_text", url="http://example.com")
      "click|http://x|#btn" -> BrowserCommand(action="click", url="http://x", selector="#btn")
    """
    raw = action
    if url == "" and "|" in action:
        parts = [p.strip() for p in action.split("|") if p.strip()]
        action = parts[0] if parts else action
        url = parts[1] if len(parts) > 1 else ""
        selector = parts[2] if len(parts) > 2 else selector
        text = parts[3] if len(parts) > 3 else text
    return BrowserCommand(action=action, url=url, selector=selector, text=text, raw=raw)


def _check_playwright_available() -> Tuple[bool, str]:
    """Check if playwright is importable. Does NOT install anything."""
    try:
        __import__("playwright")
        return True, "available"
    except ImportError:
        return False, "playwright not installed"


def _ensure_chromium() -> Tuple[bool, str]:
    """Install chromium browser if not already done. Cached globally."""
    global _playwright_browser_ready
    if _playwright_browser_ready:
        return True, "cached"
    try:
        import subprocess
        r = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            timeout=180, capture_output=True, text=True,
        )
        if r.returncode != 0:
            return False, f"chromium install failed: {r.stderr[:200]}"
        _playwright_browser_ready = True
        return True, "installed"
    except Exception as e:
        return False, f"chromium install error: {e}"


def browser_action(
    action: str,
    url: str = "",
    selector: str = "",
    text: str = "",
    timeout: int = 30,
    ensure_package: Optional[Callable[[str, Optional[str]], Tuple[bool, str]]] = None,
    *,
    mode: str = "runtime_required",
) -> Dict[str, Any]:
    """Browser automation via Playwright.

    Modes:
      parse_only: parse the command and return immediately (for testing)
      dry_run: parse + check playwright availability, but don't launch browser
      runtime_required: full execution (default)

    Returns dict: {ok, action, url, summary, data?, error?, latency_ms?}
    """
    t0 = _time.monotonic()
    cmd = parse_browser_command(action, url, selector, text)

    if mode == "parse_only":
        return {
            "ok": True, "action": cmd.action, "url": cmd.url,
            "summary": "parsed (parse_only mode)",
            "data": {"selector": cmd.selector, "text": cmd.text, "raw": cmd.raw},
            "error": None, "latency_ms": 0,
        }

    logging.info(f"[BROWSER] action='{cmd.action}' url='{cmd.url}' mode={mode}")

    # Check playwright availability
    if ensure_package is not None:
        pw_ok, pw_msg = ensure_package("playwright", "playwright")
    else:
        pw_ok, pw_msg = _check_playwright_available()

    if not pw_ok:
        return {
            "ok": False, "action": cmd.action, "url": cmd.url,
            "summary": f"browser unavailable: {pw_msg}",
            "error": pw_msg, "latency_ms": int((_time.monotonic() - t0) * 1000),
        }

    if mode == "dry_run":
        return {
            "ok": True, "action": cmd.action, "url": cmd.url,
            "summary": "dry_run: playwright available, browser not launched",
            "error": None, "latency_ms": int((_time.monotonic() - t0) * 1000),
        }

    # runtime_required: full execution
    try:
        cr_ok, cr_msg = _ensure_chromium()
        if not cr_ok:
            return {
                "ok": False, "action": cmd.action, "url": cmd.url,
                "summary": cr_msg, "error": cr_msg,
                "latency_ms": int((_time.monotonic() - t0) * 1000),
            }

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, timeout=timeout * 1000)
            page = browser.new_page()
            page.set_default_timeout(timeout * 1000)

            if cmd.action == "extract_text" and cmd.url:
                page.goto(cmd.url, wait_until="domcontentloaded")
                tc = page.evaluate("document.body.innerText")
                result = {
                    "ok": True, "action": "extract_text", "url": cmd.url,
                    "summary": f"Extracted {len(tc)} chars",
                    "data": {"text": tc[:10000]},
                }
            elif cmd.action == "navigate" and cmd.url:
                page.goto(cmd.url, wait_until="domcontentloaded")
                result = {
                    "ok": True, "action": "navigate", "url": cmd.url,
                    "summary": "OK", "data": {"title": page.title()},
                }
            elif cmd.action in ("click", "fill") and cmd.selector:
                page.goto(cmd.url or page.url, wait_until="domcontentloaded")
                if cmd.action == "click":
                    page.click(cmd.selector, timeout=8000)
                    result = {"ok": True, "action": "click", "summary": f"Clicked {cmd.selector}"}
                else:
                    page.fill(cmd.selector, cmd.text or "")
                    result = {"ok": True, "action": "fill", "summary": f"Filled {cmd.selector}"}
            else:
                result = {"ok": False, "action": cmd.action, "summary": f"Unsupported: {cmd.action}"}

            browser.close()

        result.setdefault("error", None)
        result["latency_ms"] = int((_time.monotonic() - t0) * 1000)
        return result

    except Exception as e:
        logging.error(f"[BROWSER] ERROR: {e}")
        return {
            "ok": False, "action": cmd.action, "url": cmd.url,
            "summary": f"Error: {str(e)[:200]}", "error": str(e)[:200],
            "latency_ms": int((_time.monotonic() - t0) * 1000),
        }
