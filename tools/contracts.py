"""
TOVAH v14 tools/contracts.py — Tool capability contracts.

Each builtin tool has a CapabilityContract that declares its inputs,
outputs, failure modes, cost, side effects, and required permissions.

These contracts are used by:
- the kernel's _perform_tool_action for dispatch validation
- the planner for cost-aware tool selection
- the critic for tool-use auditing
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ToolContract:
    """Capability contract for a single tool."""
    name: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    failure_modes: List[str] = field(default_factory=list)
    cost: str = "low"  # "low", "medium", "high"
    side_effects: List[str] = field(default_factory=list)
    required_permissions: str = "safe_autonomous"
    budget_resource: str = ""  # which budget to check


TOOL_CONTRACTS: Dict[str, ToolContract] = {
    "web_search": ToolContract(
        name="web_search",
        inputs={"query": "str"},
        outputs={"payload": "List[str]", "url": "str"},
        failure_modes=["network_error", "rate_limit", "empty_results"],
        cost="low",
        budget_resource="web_search",
    ),
    "fetch_url": ToolContract(
        name="fetch_url",
        inputs={"url": "str"},
        outputs={"payload": "str (raw content)", "url": "str"},
        failure_modes=["network_error", "invalid_scheme", "timeout"],
        cost="low",
        budget_resource="fetch_url",
    ),
    "github_repo": ToolContract(
        name="github_repo",
        inputs={"repo": "str (owner/name)"},
        outputs={"payload": "dict (metadata)"},
        failure_modes=["network_error", "not_found", "rate_limit"],
        cost="low",
        budget_resource="fetch_url",
    ),
    "github_file": ToolContract(
        name="github_file",
        inputs={"repo": "str", "path": "str", "branch": "str (optional)"},
        outputs={"payload": "str (file content)"},
        failure_modes=["network_error", "not_found", "branch_miss"],
        cost="low",
        budget_resource="fetch_url",
    ),
    "robots_ok": ToolContract(
        name="robots_ok",
        inputs={"base_url": "str"},
        outputs={"payload": "str (robots.txt content)"},
        failure_modes=["network_error", "invalid_url"],
        cost="low",
    ),
    "wikipedia_summary": ToolContract(
        name="wikipedia_summary",
        inputs={"topic": "str"},
        outputs={"payload": "dict (title, extract)"},
        failure_modes=["not_found", "network_error"],
        cost="low",
        budget_resource="web_search",
    ),
    "arxiv_search": ToolContract(
        name="arxiv_search",
        inputs={"query": "str"},
        outputs={"payload": "List[dict] (title, summary, id)"},
        failure_modes=["network_error", "rate_limit", "empty_results"],
        cost="low",
        budget_resource="web_search",
    ),
    "rss_fetch": ToolContract(
        name="rss_fetch",
        inputs={"url": "str"},
        outputs={"payload": "List[dict] (title, link)"},
        failure_modes=["network_error", "parse_error"],
        cost="low",
        budget_resource="fetch_url",
    ),
    "json_api_fetch": ToolContract(
        name="json_api_fetch",
        inputs={"url": "str"},
        outputs={"payload": "Any (parsed JSON)"},
        failure_modes=["network_error", "invalid_json", "invalid_scheme"],
        cost="low",
        budget_resource="fetch_url",
    ),
    "sitemap_fetch": ToolContract(
        name="sitemap_fetch",
        inputs={"base_url": "str"},
        outputs={"payload": "List[str] (URLs)"},
        failure_modes=["network_error", "invalid_url"],
        cost="low",
        budget_resource="fetch_url",
    ),
    "browser_action": ToolContract(
        name="browser_action",
        inputs={"action": "str", "url": "str", "selector": "str", "text": "str"},
        outputs={"payload": "dict (result data)"},
        failure_modes=["playwright_unavailable", "chromium_install_fail", "page_timeout", "selector_not_found"],
        cost="high",
        side_effects=["installs chromium on first use"],
        required_permissions="safe_logged",
        budget_resource="browser_action",
    ),
    "extract_text": ToolContract(
        name="extract_text",
        inputs={"url": "str"},
        outputs={"payload": "str (extracted text)"},
        failure_modes=["bs4_unavailable", "network_error", "parse_error"],
        cost="medium",
        budget_resource="fetch_url",
    ),
}
