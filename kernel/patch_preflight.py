"""
TOVAH v14 kernel/patch_preflight.py — Single authoritative patch validation.

Both stage_patch() and direct_inject_method() route through this.
Create-new requires EXPLICIT allow_create_new=True AND target in EXTENSION_TARGETS.
Being in EXTENSION_TARGETS alone does NOT authorize creation.
"""
from __future__ import annotations
import ast, inspect, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from tovah_v14.core.contracts import (
    ALLOWED_PATCH_TARGETS, ALLOWED_INJECT_TARGETS, PROTECTED_METHODS,
    CONTRACT_REGISTRY, EXTENSION_TARGETS, ALLOWED_TARGETS_UNIFIED,
    verify_patch_contract,
)
from tovah_v14.mutation.analysis import analyze_patch_code

OBSOLETE_PATTERNS = [
    "ShadowScoreCompat", "float(self._shadow_score_text",
    "score = self._shadow_score_text", "__float__", "__lt__", "__gt__",
    "class ShadowScoreCompat",
]

@dataclass
class PatchPreflightReport:
    accepted: bool = False
    target: str = ""
    patch_kind: str = "patch_existing"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    analysis_ok: bool = False
    contract_ok: bool = False
    signature_compatible: bool = True
    obsolete_patterns: List[str] = field(default_factory=list)
    missing_state_fields: List[str] = field(default_factory=list)

def validate_patch_preflight(
    target: str, code: str, kernel_class: type,
    state_beta_keys: set | None = None,
    *, allow_create_new: bool = False,
) -> PatchPreflightReport:
    report = PatchPreflightReport(target=target)
    if target in PROTECTED_METHODS:
        report.errors.append(f"target '{target}' is PROTECTED")
        return report
    if target not in ALLOWED_TARGETS_UNIFIED:
        if not allow_create_new:
            report.errors.append(f"target '{target}' not in any allowed target set")
            return report
    existing = getattr(kernel_class, target, None)
    target_exists = existing is not None and callable(existing)
    if not target_exists:
        report.patch_kind = "create_new"
        if not allow_create_new:
            report.errors.append(f"target '{target}' absent; requires explicit create_new=True")
            return report
        if target not in EXTENSION_TARGETS:
            report.errors.append(f"target '{target}' not in EXTENSION_TARGETS")
            return report
        report.warnings.append(f"create-new: '{target}' is extension slot")
    analysis_ok, fn_names, analysis_errors = analyze_patch_code(code)
    report.analysis_ok = analysis_ok
    if not analysis_ok:
        report.errors.extend(analysis_errors)
    if target not in fn_names:
        report.errors.append(f"code does not define function '{target}'")
    if target in CONTRACT_REGISTRY:
        contract_ok, contract_errors = verify_patch_contract(target, code)
        report.contract_ok = contract_ok
        if not contract_ok:
            report.errors.extend(contract_errors)
    else:
        report.contract_ok = True
        report.warnings.append(f"no contract for '{target}' (uncontracted)")
    if target_exists and callable(existing):
        try:
            old_sig = inspect.signature(existing)
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == target:
                    old_required = [p for p in old_sig.parameters.values()
                        if p.name != "self" and p.default is inspect.Parameter.empty
                        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
                    new_params = [a.arg for a in node.args.args if a.arg != "self"]
                    new_defaults = len(node.args.defaults)
                    if len(new_params) - new_defaults > len(old_required):
                        report.signature_compatible = False
                        report.errors.append(f"sig: {len(new_params)-new_defaults} required > {len(old_required)} old")
                    break
        except Exception as e:
            report.warnings.append(f"sig check failed: {e}")
    for pat in OBSOLETE_PATTERNS:
        if pat in code:
            report.obsolete_patterns.append(pat)
    if report.obsolete_patterns:
        report.errors.append(f"obsolete patterns: {report.obsolete_patterns}")
    if state_beta_keys is not None:
        refs = re.findall(r'self\.state\.beta\[[\"\']([^\"\']+)[\"\']\]', code)
        for ref in refs:
            if ref not in state_beta_keys and not ref.startswith("_"):
                report.missing_state_fields.append(ref)
        if report.missing_state_fields:
            report.warnings.append(f"beta refs not in state: {report.missing_state_fields}")
    report.accepted = len(report.errors) == 0
    return report
