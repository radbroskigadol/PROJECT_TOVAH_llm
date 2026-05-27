"""
TOVAH v14 tools/builtins.py — Standard HTTP-based builtin tools.

SEMANTIC PRESERVATION: all implementations identical to v13 ToolLayer methods.
"""
from __future__ import annotations

import re
import time
from typing import List
from urllib.parse import quote_plus, urlparse

import requests

from tovah_v14.tools.result import ToolResult


def tool_fetch_url(session: requests.Session, url: str, timeout: int = 15) -> ToolResult:
    """Fetch raw content from a URL."""
    if urlparse(url).scheme not in {"http", "https"}:
        return ToolResult(False, "fetch_url", "invalid scheme", url=url)
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return ToolResult(True, "fetch_url", f"fetched {len(r.text[:25000])} chars", r.text[:25000], url)
    except Exception as e:
        return ToolResult(False, "fetch_url", f"fetch failed: {e}", url=url)


def tool_robots_ok(session: requests.Session, base_url: str, timeout: int = 15) -> ToolResult:
    """Check robots.txt for a given URL."""
    p = urlparse(base_url)
    if not p.scheme or not p.netloc:
        return ToolResult(False, "robots_ok", "invalid url")
    return tool_fetch_url(session, f"{p.scheme}://{p.netloc}/robots.txt", timeout)


def tool_wikipedia_summary(session: requests.Session, topic: str, timeout: int = 15) -> ToolResult:
    """Fetch Wikipedia summary with search fallback."""
    clean = topic.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean}"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 404:
            sr = session.get(
                f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(topic.strip())}&limit=3&format=json",
                timeout=timeout,
            )
            if sr.ok:
                titles = sr.json()[1] if len(sr.json()) > 1 else []
                if titles:
                    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{str(titles[0]).replace(' ', '_')}"
                    r = session.get(url, timeout=timeout)
                else:
                    return ToolResult(False, "wikipedia_summary", f"not found: {topic}")
        r.raise_for_status()
        d = r.json()
        payload = {
            "title": d.get("title"),
            "extract": d.get("extract", "")[:4000],
            "content_urls": d.get("content_urls", {}),
        }
        return ToolResult(True, "wikipedia_summary", f"summary for {d.get('title', topic)}", payload, url)
    except Exception as e:
        # Fallback to action API
        try:
            fb = (
                f"https://en.wikipedia.org/w/api.php?action=query&titles={clean}"
                f"&prop=extracts&exintro=true&explaintext=true&redirects=1&format=json"
            )
            fr = session.get(fb, timeout=timeout)
            if fr.ok:
                for pid, page in fr.json().get("query", {}).get("pages", {}).items():
                    if pid != "-1" and page.get("extract"):
                        return ToolResult(
                            True, "wikipedia_summary",
                            f"summary for {page.get('title', topic)} (action API)",
                            {"title": page.get("title", topic), "extract": page["extract"][:4000]},
                            fb,
                        )
        except Exception:
            pass
        return ToolResult(False, "wikipedia_summary", f"wikipedia failed: {e}", url=url)


def tool_arxiv_search(session: requests.Session, query: str, timeout: int = 15) -> ToolResult:
    """Search arXiv with retry and fallback endpoints."""
    endpoints = [
        f"https://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}&start=0&max_results=5",
        f"https://arxiv.org/api/query?search_query=all:{quote_plus(query)}&start=0&max_results=3",
    ]
    last_err = None
    for url in endpoints:
        try:
            r = session.get(url, timeout=25)
            if r.status_code == 429:
                time.sleep(3)
                continue
            r.raise_for_status()
            text = r.text[:20000]
            parsed: List[dict] = []
            for e in re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL)[:5]:
                t = re.search(r"<title>(.*?)</title>", e, re.DOTALL)
                s = re.search(r"<summary>(.*?)</summary>", e, re.DOTALL)
                l = re.search(r"<id>(.*?)</id>", e, re.DOTALL)
                parsed.append({
                    "title": (t.group(1).strip() if t else "")[:500],
                    "summary": (s.group(1).strip() if s else "")[:1500],
                    "id": l.group(1).strip() if l else "",
                })
            return ToolResult(True, "arxiv_search", f"{len(parsed)} arXiv results", parsed, url)
        except Exception as e:
            last_err = e
    return ToolResult(False, "arxiv_search", f"arxiv failed: {last_err}")


def tool_rss_fetch(session: requests.Session, url: str, timeout: int = 15) -> ToolResult:
    """Fetch and parse RSS/Atom feed."""
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        items = re.findall(r"<item>(.*?)</item>", r.text[:25000], re.DOTALL) or \
                re.findall(r"<entry>(.*?)</entry>", r.text[:25000], re.DOTALL)
        parsed: List[dict] = []
        for item in items[:10]:
            title = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
            link = re.search(r"<link>(.*?)</link>", item, re.DOTALL) or re.search(r'href="([^"]+)"', item)
            parsed.append({
                "title": (title.group(1).strip() if title else "")[:300],
                "link": link.group(1).strip() if link else "",
            })
        return ToolResult(True, "rss_fetch", f"{len(parsed)} feed items", parsed, url)
    except Exception as e:
        return ToolResult(False, "rss_fetch", f"rss failed: {e}", url=url)


def tool_json_api_fetch(session: requests.Session, url: str, timeout: int = 15) -> ToolResult:
    """Fetch and parse JSON from an API endpoint."""
    if urlparse(url).scheme not in {"http", "https"}:
        return ToolResult(False, "json_api_fetch", "invalid scheme", url=url)
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return ToolResult(True, "json_api_fetch", "json fetched", r.json(), url)
    except Exception as e:
        return ToolResult(False, "json_api_fetch", f"json fetch failed: {e}", url=url)


def tool_sitemap_fetch(session: requests.Session, base_url: str, timeout: int = 15) -> ToolResult:
    """Fetch and parse sitemap.xml."""
    p = urlparse(base_url)
    if not p.scheme or not p.netloc:
        return ToolResult(False, "sitemap_fetch", "invalid url")
    url = f"{p.scheme}://{p.netloc}/sitemap.xml"
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        locs = re.findall(r"<loc>(.*?)</loc>", r.text[:25000], re.DOTALL)[:50]
        return ToolResult(True, "sitemap_fetch", f"parsed {len(locs)} sitemap URLs", locs, url)
    except Exception as e:
        return ToolResult(False, "sitemap_fetch", f"sitemap failed: {e}", url=url)
