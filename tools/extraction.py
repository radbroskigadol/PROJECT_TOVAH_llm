"""
TOVAH v14 tools/extraction.py — Text extraction from web pages.

SEMANTIC PRESERVATION: implementation identical to v13 extract_text method.
Uses _ensure_package pattern for bs4 availability.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Optional, Tuple

import requests


def extract_text(
    session: requests.Session,
    url: str,
    timeout: int = 15,
    ensure_package: Optional[Callable[[str, Optional[str]], Tuple[bool, str]]] = None,
) -> Dict[str, Any]:
    """Extract readable text from a web page using BeautifulSoup.

    Returns dict with {ok, url, summary, text?}.
    The kernel is responsible for translating this into bilateral state updates.
    """
    try:
        # Check bs4 availability
        bs4_available = False
        if ensure_package is not None:
            ok, msg = ensure_package("beautifulsoup4", "bs4")
            if not ok:
                return {"ok": False, "url": url, "summary": f"bs4 unavailable: {msg}"}
            bs4_available = True
        else:
            try:
                __import__("bs4")
                bs4_available = True
            except ImportError:
                return {"ok": False, "url": url, "summary": "bs4 not installed"}

        from bs4 import BeautifulSoup

        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))[:12000]
        return {"ok": True, "url": url, "summary": f"Extracted {len(text)} chars", "text": text}

    except Exception as e:
        return {"ok": False, "url": url, "summary": f"Error: {str(e)[:150]}"}
