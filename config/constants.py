"""
TOVAH v14 — Constants.

Numeric limits, thresholds, budget defaults, and structural constants.
"""
from typing import Any, Dict, List

VERSION = "14.3.2a"
USER_AGENT = "TOVAH/14.3.2a (autonomous research agent; david.betzer@yahoo.com)"

# --- Size limits ---
PATCH_CODE_MAX_CHARS = 30000
TOOL_CODE_MAX_CHARS = 25000
MAX_RESEARCH_RESULTS_STORED = 300
MAX_TRACES_STORED = 500
MAX_PATCH_HISTORY = 200
MAX_PDF_TEXT_CHARS = 120000
MAX_PDF_NOTE_CHARS = 4000
MAX_SNAPSHOTS_DISK = 10
MAX_SNAPSHOTS_MEMORY = 5
MAX_MEMORY_PER_KIND = 500
MAX_TASK_QUEUE = 100
MAX_ESCALATION_LOG = 50
MAX_EXPERIENCE_RECORDS = 1000
MAX_METRICS_FILES = 200
MAX_KERNEL_PACKET_LOG = 2000  # rolling cap on kernel_packet_log; was 200 (S-2)

# --- Timing ---
PDF_RETRY_COOLDOWN_SECONDS = 600
SANDBOX_TIMEOUT = 30
BUDGET_RESET_INTERVAL = 3600
PLAN_MAX_AGE_SECONDS = 14400  # 4 hours

# --- Thresholds ---
DEGRADED_MODE_REGRESSION_THRESHOLD = 0.70
BOOT_VALIDATION_MAX_FAILURES = 3
STALE_BELIEF_MAX_CYCLES = 150
GAMMA_THETA_T = 0.55
GAMMA_THETA_F = 0.55

# --- Promotion stages (ordered) ---
PROMOTION_STAGES: List[str] = [
    "proposed", "static_approved", "sandbox_passed", "regression_passed",
    "shadow_deployed", "live_promoted", "revertable",
]

# --- Action privilege levels ---
ACTION_PRIVILEGES: Dict[str, set] = {
    "safe_autonomous": {
        "web_search", "fetch_url", "arxiv_search", "wikipedia_summary",
        "github_repo", "github_file", "rss_fetch", "json_api_fetch",
        "sitemap_fetch", "extract_text", "robots_ok",
    },
    "safe_logged": {"browser_action"},
    "sandbox_only": {"run_code", "exec_patch", "test_tool"},
    "approval_required": {
        "apply_patch", "install_package", "store_credential",
        "activate_service", "promote_tool", "inject_method",
    },
    "forbidden": {"os_system", "os_popen", "shutil_rmtree"},
}

# --- Resource budgets ---
DEFAULT_BUDGETS: Dict[str, Dict[str, Any]] = {
    "web_search": {"limit": 60, "used": 0, "reset_at": 0.0},
    "fetch_url": {"limit": 40, "used": 0, "reset_at": 0.0},
    "browser_action": {"limit": 10, "used": 0, "reset_at": 0.0},
    "advisor_call": {"limit": 30, "used": 0, "reset_at": 0.0},
    "pip_install": {"limit": 5, "used": 0, "reset_at": 0.0},
    "patch_apply": {"limit": 8, "used": 0, "reset_at": 0.0},
    "disk_write_mb": {"limit": 100, "used": 0, "reset_at": 0.0},
}

# --- Model profiles ---
# AUDIT FIX (P0-4, v14.1.2): max_len raised across the board. The previous
# 320-byte limit truncated 47% of typical experience records and 61% of
# total text bytes in the corpus before they reached the model.
MODEL_PROFILES: Dict[str, Dict[str, int]] = {
    "debug": {"d_model": 192, "d_hidden": 768, "n_heads": 6, "n_blocks": 4, "max_len": 512},
    "standard": {"d_model": 224, "d_hidden": 896, "n_heads": 7, "n_blocks": 5, "max_len": 1024},
    "heavy": {"d_model": 256, "d_hidden": 1024, "n_heads": 8, "n_blocks": 6, "max_len": 1024},
    # Large profile for actual pretraining runs. ~50M params; not for live operation
    # on CPU. Select with TOVAH_PROFILE=large. max_len=1024 is the sweet spot for
    # 12-block attention activation memory on a consumer GPU.
    "large": {"d_model": 512, "d_hidden": 2048, "n_heads": 8, "n_blocks": 12, "max_len": 1024},
}

# --- Curriculum ---
DEFAULT_CURRICULUM: List[Dict[str, Any]] = [
    {"domain": "tool_mastery", "lessons": ["web_search_patterns", "fetch_chain", "github_navigation", "arxiv_mining"], "mastery": 0.0, "test_count": 0, "last_tested": 0.0},
    {"domain": "api_integration", "lessons": ["rest_basics", "auth_patterns", "rate_limiting", "error_handling"], "mastery": 0.0, "test_count": 0, "last_tested": 0.0},
    {"domain": "file_handling", "lessons": ["pdf_extraction", "json_processing", "text_parsing", "csv_handling"], "mastery": 0.0, "test_count": 0, "last_tested": 0.0},
    {"domain": "data_transformation", "lessons": ["schema_extraction", "normalization", "filtering", "aggregation"], "mastery": 0.0, "test_count": 0, "last_tested": 0.0},
    {"domain": "testing_discipline", "lessons": ["unit_test_writing", "regression_design", "sandbox_usage", "coverage_analysis"], "mastery": 0.0, "test_count": 0, "last_tested": 0.0},
    {"domain": "environment_mgmt", "lessons": ["package_management", "path_handling", "config_management", "state_persistence"], "mastery": 0.0, "test_count": 0, "last_tested": 0.0},
    {"domain": "browsing_quality", "lessons": ["selector_targeting", "js_rendering", "content_extraction", "navigation_chains"], "mastery": 0.0, "test_count": 0, "last_tested": 0.0},
    {"domain": "code_review", "lessons": ["ast_analysis", "pattern_detection", "security_checking", "style_consistency"], "mastery": 0.0, "test_count": 0, "last_tested": 0.0},
    {"domain": "planning_quality", "lessons": ["goal_decomposition", "dependency_ordering", "resource_estimation", "fallback_design"], "mastery": 0.0, "test_count": 0, "last_tested": 0.0},
]
