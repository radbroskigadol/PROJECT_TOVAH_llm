"""
TOVAH v14 tools/layer.py — ToolResult and ToolLayer base.

SEMANTIC PRESERVATION:
  ToolResult shape is identical to v13.
  ToolLayer session management is identical.

ToolLayer is the composition root for all builtin tools.
Individual tool implementations are in separate files but
composed here for backward compatibility with v13's self.tools interface.
"""
from __future__ import annotations

from typing import Any, List

import requests

from tovah_v14.config.constants import USER_AGENT
from tovah_v14.tools.result import ToolResult
from tovah_v14.tools.search import tool_web_search
from tovah_v14.tools.builtins import (
    tool_fetch_url,
    tool_robots_ok,
    tool_wikipedia_summary,
    tool_arxiv_search,
    tool_rss_fetch,
    tool_json_api_fetch,
    tool_sitemap_fetch,
)
from tovah_v14.tools.github import tool_github_repo, tool_github_file
from tovah_v14.tools.browser import browser_action as _browser_action_impl
from tovah_v14.tools.extraction import extract_text as _extract_text_impl


class ToolLayer:
    """Composition root for all builtin tools.

    Owns the requests.Session used by all HTTP tools.
    Individual tool methods delegate to module-level functions,
    passing the session and timeout.
    """

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    @property
    def builtins(self) -> List[str]:
        return [
            "web_search", "fetch_url", "github_repo", "github_file", "robots_ok",
            "wikipedia_summary", "arxiv_search", "rss_fetch", "json_api_fetch",
            "sitemap_fetch", "browser_action", "extract_text",
        ]

    def web_search(self, query: str) -> ToolResult:
        """Search using DuckDuckGo HTML endpoint."""
        return tool_web_search(self.session, query, self.timeout)

    def fetch_url(self, url: str) -> ToolResult:
        """Fetch raw content from a URL."""
        return tool_fetch_url(self.session, url, self.timeout)

    def github_repo(self, repo: str) -> ToolResult:
        """Fetch GitHub repository metadata."""
        return tool_github_repo(self.session, repo, self.timeout)

    def github_file(self, repo: str, path: str, branch: str = "") -> ToolResult:
        """Fetch a file from GitHub with automatic branch detection."""
        return tool_github_file(self.session, repo, path, branch, self.timeout)

    def robots_ok(self, base_url: str) -> ToolResult:
        """Check robots.txt for a given URL."""
        return tool_robots_ok(self.session, base_url, self.timeout)

    def wikipedia_summary(self, topic: str) -> ToolResult:
        """Fetch Wikipedia summary with search fallback."""
        return tool_wikipedia_summary(self.session, topic, self.timeout)

    def arxiv_search(self, query: str) -> ToolResult:
        """Search arXiv with retry and fallback endpoints."""
        return tool_arxiv_search(self.session, query, self.timeout)

    def rss_fetch(self, url: str) -> ToolResult:
        """Fetch and parse RSS/Atom feed."""
        return tool_rss_fetch(self.session, url, self.timeout)

    def json_api_fetch(self, url: str) -> ToolResult:
        """Fetch and parse JSON from an API endpoint."""
        return tool_json_api_fetch(self.session, url, self.timeout)

    def sitemap_fetch(self, base_url: str) -> ToolResult:
        """Fetch and parse sitemap.xml."""
        return tool_sitemap_fetch(self.session, base_url, self.timeout)

    def browser_action(
        self,
        action: str,
        url: str = "",
        selector: str = "",
        text: str = "",
        timeout: int = 30,
        ensure_package: Any = None,
    ) -> ToolResult:
        """Browser automation via Playwright. Wraps tools.browser.browser_action.

        Returns ToolResult for v13 interface parity with self.tools.browser_action().
        """
        rd = _browser_action_impl(action, url, selector, text, timeout, ensure_package=ensure_package)
        return ToolResult(
            ok=rd.get("ok", False),
            tool="browser_action",
            summary=rd.get("summary", ""),
            payload=rd.get("data", {}),
            url=url,
        )

    def extract_text(
        self,
        url: str,
        ensure_package: Any = None,
    ) -> ToolResult:
        """Extract text from web page. Wraps tools.extraction.extract_text.

        Returns ToolResult for v13 interface parity with self.tools.extract_text().
        """
        rd = _extract_text_impl(self.session, url, self.timeout, ensure_package=ensure_package)
        return ToolResult(
            ok=rd.get("ok", False),
            tool="extract_text",
            summary=rd.get("summary", ""),
            payload=rd.get("text", ""),
            url=url,
        )
