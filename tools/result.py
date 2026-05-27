"""
TOVAH v14 tools/result.py — Tool result types.

ToolResult: v13-compatible shape, preserved for backward compat.
ToolActionResult: v14 typed contract with structured fields for reliability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ToolResult:
    """Result from any tool invocation. Shape preserved from v13."""
    ok: bool
    tool: str
    summary: str
    payload: Any = None
    url: str = ""


@dataclass
class ToolActionResult:
    """v14 typed tool-result contract. Stable structure for all tool calls.

    This is the target contract for new tool integrations.
    Existing tools still return ToolResult; this provides a migration path.
    """
    ok: bool
    tool: str
    action: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    retryable: bool = False
    provenance: Tuple[str, ...] = ()
    latency_ms: Optional[int] = None
    summary: str = ""

    def to_tool_result(self) -> ToolResult:
        """Convert to v13-compatible ToolResult."""
        return ToolResult(
            ok=self.ok,
            tool=self.tool,
            summary=self.summary or (self.error or ""),
            payload=self.data or None,
            url=self.data.get("url", ""),
        )
