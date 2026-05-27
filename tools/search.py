"""
TOVAH v14 tools/search.py — Web search via DuckDuckGo.

SEMANTIC PRESERVATION: implementation identical to v13 ToolLayer.web_search.
"""
from __future__ import annotations

import re
from typing import List
from urllib.parse import quote_plus

import requests

from tovah_v14.tools.result import ToolResult


def tool_web_search(session: requests.Session, query: str, timeout: int = 15) -> ToolResult:
    """Search using DuckDuckGo HTML endpoint with instant-answer fallback."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        text = r.text[:50000]
        items: List[str] = []
        for match in re.finditer(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|span|td)', text, re.DOTALL
        ):
            snippet = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            if snippet and len(snippet) > 20:
                items.append(snippet[:500])
        for match in re.finditer(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', text, re.DOTALL
        ):
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if title:
                items.append(title[:300])
        seen: set = set()
        items = [x for x in items if not (x in seen or seen.add(x))][:12]  # type: ignore
        if not items:
            r2 = session.get(
                f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1",
                timeout=timeout,
            )
            if r2.ok:
                d = r2.json()
                if d.get("AbstractText"):
                    items.append(d["AbstractText"])
                for t in d.get("RelatedTopics", [])[:8]:
                    if isinstance(t, dict) and "Text" in t:
                        items.append(t["Text"])
        return ToolResult(True, "web_search", f"{len(items)} search snippets", items, url)
    except Exception as e:
        return ToolResult(False, "web_search", f"search failed: {e}")
