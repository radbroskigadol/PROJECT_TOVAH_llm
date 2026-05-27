"""
TOVAH v14 — Runtime settings.

Free service registry defaults, ShadowHoTT advisor system context,
and curated tool templates.
"""
from typing import Any, Dict, List

# --- Free services (v13 compat: exact entries preserved) ---
DEFAULT_FREE_SERVICES: List[Dict[str, Any]] = [
    {"name": "duckduckgo", "type": "search", "url": "https://html.duckduckgo.com/html/", "auth": None, "status": "active"},
    {"name": "arxiv", "type": "academic_search", "url": "https://export.arxiv.org/api/", "auth": None, "status": "active"},
    {"name": "wikipedia", "type": "knowledge", "url": "https://en.wikipedia.org/api/rest_v1/", "auth": None, "status": "active"},
    {"name": "github_public", "type": "code_repository", "url": "https://api.github.com/", "auth": None, "status": "active"},
    {"name": "jsonplaceholder", "type": "test_api", "url": "https://jsonplaceholder.typicode.com/", "auth": None, "status": "available"},
    {"name": "httpbin", "type": "test_api", "url": "https://httpbin.org/", "auth": None, "status": "available"},
    {"name": "open_library", "type": "book_search", "url": "https://openlibrary.org/api/", "auth": None, "status": "available"},
    {"name": "crossref", "type": "academic_metadata", "url": "https://api.crossref.org/", "auth": None, "status": "available"},
    {"name": "semantic_scholar", "type": "academic_search", "url": "https://api.semanticscholar.org/graph/v1/", "auth": None, "status": "available"},
    {"name": "gutenberg", "type": "text_corpus", "url": "https://www.gutenberg.org/", "auth": None, "status": "available"},
    {"name": "pypi", "type": "package_registry", "url": "https://pypi.org/pypi/", "auth": None, "status": "available"},
    {"name": "stackexchange", "type": "qa_knowledge", "url": "https://api.stackexchange.com/2.3/", "auth": None, "status": "available"},
    {"name": "hacker_news", "type": "tech_news", "url": "https://hacker-news.firebaseio.com/v0/", "auth": None, "status": "available"},
]

# --- Curated tool templates (v13 compat: exact code preserved) ---
CURATED_TOOL_TEMPLATES: Dict[str, str] = {
    "json_schema_probe": '''TOOL_SPEC = {"name": "json_schema_probe", "description": "Inspect JSON payload shapes and top-level keys."}
def run(kernel, **kwargs):
    action = kwargs.get("action", {})
    payload = action.get("payload")
    if isinstance(payload, dict):
        sample = {k: type(v).__name__ for k, v in list(payload.items())[:20]}
        return {"type": "dict", "keys": sorted(list(payload.keys()))[:100], "value_types": sample, "size": len(payload)}
    if isinstance(payload, list):
        return {"type": "list", "length": len(payload), "first_type": type(payload[0]).__name__ if payload else "empty"}
    return {"type": type(payload).__name__, "repr": repr(payload)[:500]}
''',
    "paper_digest": '''TOOL_SPEC = {"name": "paper_digest", "description": "Summarize paper/article search payloads into compact notes."}
def _clean(text): return " ".join(str(text).split())
def run(kernel, **kwargs):
    action = kwargs.get("action", {})
    payload = action.get("payload")
    if not isinstance(payload, list): return {"error": "paper_digest expects a list payload"}
    notes = []
    for item in payload[:10]:
        if not isinstance(item, dict): continue
        title = _clean(item.get("title", ""))[:300]
        summary = _clean(item.get("summary", ""))[:800]
        ident = _clean(item.get("id", "") or item.get("url", ""))[:300]
        if title or summary: notes.append({"title": title, "summary": summary, "id": ident})
    compact = [n["title"] + (" — " + n["summary"][:220] if n["summary"] else "") for n in notes[:5]]
    return {"count": len(notes), "items": notes[:5], "digest": compact}
''',
    "repo_readme_digest": '''TOOL_SPEC = {"name": "repo_readme_digest", "description": "Extract repository metadata and compact README notes."}
def _clean(text): return " ".join(str(text).split())
def run(kernel, **kwargs):
    action = kwargs.get("action", {})
    payload = action.get("payload")
    if isinstance(payload, dict):
        return {"full_name": payload.get("full_name"), "description": _clean(payload.get("description", ""))[:500], "default_branch": payload.get("default_branch"), "stars": payload.get("stargazers_count"), "url": payload.get("html_url")}
    if isinstance(payload, str):
        lines = [ln.strip() for ln in payload.splitlines() if ln.strip()]
        return {"line_count": len(lines), "preview": lines[:20], "summary": " ".join(lines[:6])[:700]}
    return {"error": "expects dict metadata or README text"}
''',
    "web_note_compiler": '''TOOL_SPEC = {"name": "web_note_compiler", "description": "Compile search/web results into a short note artifact."}
def _clean(text): return " ".join(str(text).split())
def run(kernel, **kwargs):
    action = kwargs.get("action", {})
    payload = action.get("payload")
    lines = []
    if isinstance(payload, list):
        for item in payload[:10]:
            if isinstance(item, dict):
                title = _clean(item.get("title", "") or item.get("name", ""))[:200]
                summary = _clean(item.get("summary", "") or item.get("snippet", "") or item.get("description", ""))[:300]
                line = f"{title} — {summary}" if title and summary else (title or summary)
                if line: lines.append(line)
            elif isinstance(item, str):
                t = _clean(item)[:300]
                if t: lines.append(t)
    elif isinstance(payload, str):
        lines = [s[:300] for s in _clean(payload).split(". ")[:8] if s.strip()]
    else: return {"error": "expects list or string payload"}
    return {"note_count": len(lines), "notes": lines[:8], "joined": "\\n".join(lines[:8])}
''',
}

# --- ShadowHoTT advisor system context (updated for v14) ---
SHADOWHOTT_SYSTEM_CONTEXT = """You are advising TOVAH v14, an autonomous self-improving AI kernel running ShadowHoTT architecture.

ARCHITECTURE:
- ShadowHoTT = Shadow Homotopy Type Theory. A bilateral (dual-channel) neural transformer.
- Every proposition has BOTH a truth value t in [0,1] AND a falsity value f in [0,1] (independent).
- Four semantic lanes: A (classical), B (paraconsistent/glut-tolerant), C (paracomplete/gap-tolerant), D (deterministic).
- BilateralValue(t, f) is the core data type. glut=min(t,f), gap=min(1-t,1-f), delta=t-f.
- bilateral_or(a,b) and bilateral_recover(v, truth_gain, falsity_decay) update belief state.
- refresh_state(s) recomputes the gamma cache from beta values. MUST be called after beta mutations.
- Lanes are VIEWS over bilateral state. Never collapse bilateral core to serve a single lane.

KERNEL STRUCTURE:
- ProtozoanKernel composes subsystems: tools, memory, tasks, planner, patcher, observer.
- Every patchable method has a MethodContract in CONTRACT_REGISTRY.
- _shadow_score_text(text, *extra_parts) returns a DICT: {entropy, divergence, lane_weights, top_bytes}.
  It is NEVER a scalar. Use _shadow_score_scalar(text) if a float is needed.

v14 PRINCIPLES:
1. Contract-first: patches validated against MethodContract before staging.
2. Promotion ladder is the ONLY path to live deployment.
3. Contradiction governance: detect, preserve, dampen, or quarantine — never silently overwrite.
4. Gate-like updates (structure-preserving) vs measurement-like updates (reset/collapse) are distinct.
5. Layer firewall: bilateral state, semantic decisions, control logic, and external IO do not blur.
6. Degraded safe mode when regression drops below threshold.

PATCH GENERATION:
Return ONLY JSON: {"patch_name": "...", "target": "method_name", "code": "def method_name(self, ...): ...", "rationale": "..."}
- target must be in ALLOWED_PATCH_TARGETS and satisfy CONTRACT_REGISTRY[target].
- Must use BilateralValue and call refresh_state(self.state).
- No subprocess/eval/exec/socket/ctypes. No global/nonlocal.
"""
