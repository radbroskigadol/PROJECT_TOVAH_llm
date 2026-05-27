"""
TOVAH v14 mutation/analysis.py — Static patch analysis.

SEMANTIC PRESERVATION:
  analyze_patch_code is identical to v13: same blocked imports,
  same blocked calls, same blocked attrs, same dunder/protected checks.

v14 ADDITION:
  analyze_patch_with_contract combines static analysis with
  MethodContract validation for test-first patching.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Set, Tuple

from tovah_v14.config.constants import PATCH_CODE_MAX_CHARS
from tovah_v14.core.contracts import (
    ALLOWED_PATCH_TARGETS,
    ALLOWED_INJECT_TARGETS,
    PROTECTED_METHODS,
    CONTRACT_REGISTRY,
    verify_patch_contract,
)


# --- Blocked constructs (v13 compat: exact sets preserved) ---
BLOCKED_IMPORT_ROOTS: Set[str] = {
    "subprocess", "ctypes", "multiprocessing", "socket", "ssl",
    "telnetlib", "asyncio", "pexpect", "paramiko",
}
BLOCKED_CALL_NAMES: Set[str] = {"eval", "exec", "compile", "__import__", "input", "breakpoint"}
BLOCKED_ATTR_CALLS: Set[frozenset] = {
    frozenset(("os", "system")),
    frozenset(("os", "popen")),
    frozenset(("shutil", "rmtree")),
}

# Keep mutable set versions for v13 command compat (REMOVE_BLOCK, etc.)
BLOCKED_IMPORT_ROOTS_MUTABLE = set(BLOCKED_IMPORT_ROOTS)
BLOCKED_CALL_NAMES_MUTABLE = set(BLOCKED_CALL_NAMES)
BLOCKED_ATTR_CALLS_TUPLES = {("os", "system"), ("os", "popen"), ("shutil", "rmtree")}


@dataclass
class PatchDescriptor:
    """Describes a patch candidate. Shape preserved from v13."""
    patch_name: str
    target: str
    code: str
    rationale: str
    source: str


def analyze_patch_code(code: str) -> Tuple[bool, List[str], List[str]]:
    """Static analysis of patch code. Returns (ok, function_names, errors).

    BEHAVIOR (preserved exactly from v13):
      - Rejects code exceeding PATCH_CODE_MAX_CHARS
      - Rejects syntax errors
      - Blocks forbidden imports, calls, attrs
      - Blocks global/nonlocal
      - Requires at least one function definition
      - Blocks protected methods and dunders
    """
    errors: List[str] = []
    fn_names: List[str] = []
    if len(code) > PATCH_CODE_MAX_CHARS:
        return False, [], [f"patch exceeds max length {PATCH_CODE_MAX_CHARS}"]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [], [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_IMPORT_ROOTS_MUTABLE:
                    errors.append(f"blocked import: {alias.name.split('.')[0]}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in BLOCKED_IMPORT_ROOTS_MUTABLE:
                errors.append(f"blocked from-import: {mod}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALL_NAMES_MUTABLE:
                errors.append(f"blocked call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                pair = (node.func.value.id, node.func.attr)
                if pair in BLOCKED_ATTR_CALLS_TUPLES:
                    errors.append(f"blocked call: {pair[0]}.{pair[1]}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append("global/nonlocal not allowed")
        elif isinstance(node, ast.FunctionDef):
            fn_names.append(node.name)
    if not fn_names:
        errors.append("patch must define at least one function")
    for fn in fn_names:
        if fn in PROTECTED_METHODS:
            errors.append(f"protected method: {fn}")
        if fn.startswith("__"):
            errors.append(f"dunder not allowed: {fn}")
    return len(errors) == 0, fn_names, errors


def analyze_patch_with_contract(
    target: str,
    code: str,
) -> Tuple[bool, List[str], List[str], bool]:
    """Static analysis + contract validation.

    Returns (analysis_ok, function_names, errors, contract_ok).

    v14 ADDITION: runs analyze_patch_code then verify_patch_contract.
    Both must pass for a patch to be stageable.
    """
    # Static analysis first
    analysis_ok, fn_names, analysis_errors = analyze_patch_code(code)

    # Contract validation (even if static fails, collect all errors)
    contract_ok = False
    contract_errors: List[str] = []
    if target in CONTRACT_REGISTRY:
        contract_ok, contract_errors = verify_patch_contract(target, code)
    elif target in ALLOWED_PATCH_TARGETS or target in ALLOWED_INJECT_TARGETS:
        # Target allowed but no contract yet — pass with warning
        contract_ok = True
        contract_errors = [f"no contract for {target} (allowed but uncontracted)"]
    else:
        contract_errors = [f"target '{target}' not in any allowed target set"]

    all_errors = analysis_errors + contract_errors
    overall_ok = analysis_ok and contract_ok and (target in fn_names if fn_names else False)

    return overall_ok, fn_names, all_errors, contract_ok
