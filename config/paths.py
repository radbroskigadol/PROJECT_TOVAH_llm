"""
TOVAH v14 — Filesystem paths.

Every path used by v13 is preserved exactly for live asset compatibility.
New v14 paths are added without disturbing existing layout.
All paths use pathlib for Windows/Unix portability.
"""
from pathlib import Path

# Root is the working directory where TOVAH runs.
# All v13 assets live relative to this.
ROOT = Path(".")

# --- Core state files (v13 compat: exact names preserved) ---
STATE_FILE = ROOT / "tovah_state.json"
SHADOW_FILE = ROOT / "tovah_shadow.pt"
MIRROR_FILE = ROOT / "tovah_mirror.py"
PATCH_LOG = ROOT / "tovah_mutations.py"

# --- Directories (v13 compat: exact names preserved) ---
PATCH_DIR = ROOT / "tovah_patches"
TRACE_DIR = ROOT / "tovah_traces"
REPORT_DIR = ROOT / "tovah_reports"

LAB_ROOT = ROOT / "tovah_lab"
LAB_STAGED = LAB_ROOT / "staged_tools"
LAB_ACTIVE = LAB_ROOT / "active_tools"
LAB_REJECTED = LAB_ROOT / "rejected_tools"
LAB_MATH = LAB_ROOT / "math_exports"
LAB_TRACES = LAB_ROOT / "traces"
LAB_REPORTS = LAB_ROOT / "reports"

LEVBEL_DIR = ROOT / "levbel"
LEVBEL_STATE_FILE = ROOT / "tovah_levbel_state.json"

CAPABILITIES_DIR = ROOT / "tovah_capabilities"
FREE_SERVICES_FILE = ROOT / "tovah_free_services.json"
PLANS_DIR = ROOT / "tovah_plans"
SNAPSHOT_DIR = ROOT / "tovah_snapshots"
MEMORY_DIR = ROOT / "tovah_memory"
SANDBOX_DIR = ROOT / "tovah_sandbox"
TASKS_DIR = ROOT / "tovah_tasks"
WORKBENCH_DIR = ROOT / "tovah_workbench"

# --- Human interface files (v13 compat) ---
COMMAND_FILE = ROOT / "david_says.txt"
RESPONSE_FILE = ROOT / "david_response.txt"
NEEDS_FILE = ROOT / "tovah_needs.txt"

# --- Credential / notes / model files (v13 compat) ---
CREDENTIALS_FILE = ROOT / "tovah_credentials.json"
MODEL_NOTES_FILE = ROOT / "tovah_model_notes.json"
CURRICULUM_FILE = ROOT / "tovah_curriculum.json"
SELF_MODEL_FILE = ROOT / "tovah_self_model.json"
ESCALATION_FILE = ROOT / "tovah_escalations.json"
WORLD_STATE_FILE = ROOT / "tovah_world_state.json"
REGRESSION_RESULTS_FILE = ROOT / "tovah_regression.json"

# --- v14 new paths ---
EXPERIENCE_DIR = ROOT / "tovah_experience"
COMPETENCE_FILE = ROOT / "tovah_competence.json"
MODULE_REGISTRY_FILE = ROOT / "tovah_modules.json"
METRICS_DIR = ROOT / "tovah_metrics"
BOOT_VALIDATION_FILE = ROOT / "tovah_boot_validation.json"
BASELINE_FILE = ROOT / "tovah_baseline.json"  # last-known-good runtime state

# --- v16 ecology persistence paths ---
KERNEL_ECOLOGY_FILE = ROOT / "tovah_kernel_ecology.json"
PACKET_LOG_FILE = ROOT / "tovah_packet_log.json"
MEMORY_PROVENANCE_FILE = ROOT / "tovah_memory_provenance.json"
BRANCH_CHECKPOINT_DIR = ROOT / "tovah_branches"
CLUSTER_REGISTRY_FILE = ROOT / "tovah_cluster_registry.json"
CLUSTER_TRUST_FILE = ROOT / "tovah_cluster_trust.json"
NODE_IDENTITY_FILE = ROOT / "tovah_node_identity.json"

# --- v14.1 training corpus paths ---
CORPUS_DIR = ROOT / "tovah_corpus"
CORPUS_STREAM_DIR = CORPUS_DIR / "stream"

# --- All directories that must exist ---
ALL_DIRS = [
    PATCH_DIR, TRACE_DIR, REPORT_DIR,
    LAB_ROOT, LAB_STAGED, LAB_ACTIVE, LAB_REJECTED, LAB_MATH, LAB_TRACES, LAB_REPORTS,
    LEVBEL_DIR, CAPABILITIES_DIR, PLANS_DIR, SNAPSHOT_DIR,
    MEMORY_DIR, SANDBOX_DIR, TASKS_DIR, WORKBENCH_DIR,
    EXPERIENCE_DIR, METRICS_DIR, BRANCH_CHECKPOINT_DIR,
    CORPUS_DIR, CORPUS_STREAM_DIR,
]


def ensure_directories() -> None:
    """Create all required directories. Safe to call repeatedly."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
