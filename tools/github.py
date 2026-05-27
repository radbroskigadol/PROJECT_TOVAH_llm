"""
TOVAH v14 tools/github.py — GitHub repository and file tools.

SEMANTIC PRESERVATION: implementations identical to v13.
Includes branch fallback via repo metadata (v12.2 fix [12]).
"""
from __future__ import annotations

import requests

from tovah_v14.tools.result import ToolResult


def tool_github_repo(session: requests.Session, repo: str, timeout: int = 15) -> ToolResult:
    """Fetch GitHub repository metadata."""
    repo = repo.strip().strip("/")
    if repo.count("/") != 1:
        return ToolResult(False, "github_repo", "repo must be owner/name")
    url = f"https://api.github.com/repos/{repo}"
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        payload = {
            "full_name": d.get("full_name"),
            "description": d.get("description"),
            "default_branch": d.get("default_branch"),
            "stargazers_count": d.get("stargazers_count"),
            "html_url": d.get("html_url"),
        }
        return ToolResult(True, "github_repo", f"repo metadata for {repo}", payload, url)
    except Exception as e:
        return ToolResult(False, "github_repo", f"github failed: {e}", url=url)


def tool_github_file(
    session: requests.Session,
    repo: str,
    path: str,
    branch: str = "",
    timeout: int = 15,
) -> ToolResult:
    """Fetch a file from GitHub with automatic branch detection.

    Branch fallback: tries provided branch, then repo default, then main, then master.
    """
    repo = repo.strip().strip("/")
    path = path.strip().lstrip("/")
    branches = [branch] if branch else []
    if not branches:
        meta = tool_github_repo(session, repo, timeout)
        if meta.ok and isinstance(meta.payload, dict):
            branches.append(meta.payload.get("default_branch") or "main")
        branches.extend(["main", "master"])
    seen: set = set()
    for b in branches:
        if not b or b in seen:
            continue
        seen.add(b)
        url = f"https://raw.githubusercontent.com/{repo}/{b}/{path}"
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return ToolResult(True, "github_file", f"fetched {repo}/{path}@{b}", r.text[:25000], url)
        except Exception:
            pass
    return ToolResult(False, "github_file", f"failed all branches: {list(seen)}")
