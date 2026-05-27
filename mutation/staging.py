"""
TOVAH v14 mutation/staging.py — Patch staging via authoritative preflight.
"""
from __future__ import annotations
import datetime as dt, json, logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from tovah_v14.config.paths import PATCH_DIR
from tovah_v14.core.contracts import ALLOWED_PATCH_TARGETS, ALLOWED_TARGETS_UNIFIED
from tovah_v14.invariants.certification import CertificationLayer
from tovah_v14.mutation.analysis import PatchDescriptor, analyze_patch_code, analyze_patch_with_contract
from tovah_v14.kernel.action_model import PatchProposal
from tovah_v14.kernel.packet import KernelPacket

@dataclass
class StagingResult:
    ok: bool
    message: str
    patch_name: str = ""
    target: str = ""
    certificate: Dict[str, Any] | None = None
    record: Dict[str, Any] | None = None

def stage_patch(
    raw_json: str, source: str = "advisor",
    staged_patches: Dict[str, Dict[str, Any]] | None = None,
    certs: CertificationLayer | None = None,
    kernel_class: type | None = None,
    state_beta_keys: set | None = None,
    allow_create_new: bool = False,
) -> StagingResult:
    if staged_patches is None:
        staged_patches = {}
    if certs is None:
        certs = CertificationLayer()
    try:
        obj = json.loads(raw_json)
    except Exception as e:
        return StagingResult(False, f"invalid json: {e}")
    desc = PatchDescriptor(
        patch_name=str(obj.get("patch_name", f"patch_{int(dt.datetime.now().timestamp())}")),
        target=str(obj.get("target", "")).strip(),
        code=str(obj.get("code", "")).strip(),
        rationale=str(obj.get("rationale", "")).strip(),
        source=source,
    )
    patch_kind = "patch_existing"
    if kernel_class is not None:
        from tovah_v14.kernel.patch_preflight import validate_patch_preflight
        report = validate_patch_preflight(
            desc.target, desc.code, kernel_class,
            state_beta_keys=state_beta_keys,
            allow_create_new=allow_create_new,
        )
        if not report.accepted:
            return StagingResult(False, " | ".join(report.errors[:5]), desc.patch_name, desc.target)
        patch_kind = report.patch_kind
    else:
        if desc.target not in ALLOWED_TARGETS_UNIFIED and not allow_create_new:
            return StagingResult(False, f"target not allowed: {desc.target}")
        overall_ok, fn_names, all_errors, contract_ok = analyze_patch_with_contract(desc.target, desc.code)
        if not overall_ok:
            return StagingResult(False, " | ".join(all_errors), desc.patch_name, desc.target)
        if desc.target not in fn_names:
            return StagingResult(False, f"must define: {desc.target}", desc.patch_name, desc.target)
    cert = certs.certify_patch_contract(desc.target, True, True, [], source)
    cert_ok, cert_msg = certs.check(cert)
    if not cert_ok:
        return StagingResult(False, cert_msg, desc.patch_name, desc.target)
    rec = {
        **asdict(desc), "status": "staged", "patch_kind": patch_kind,
        "allow_create_new": allow_create_new,
        "staged_at": dt.datetime.now().isoformat(timespec="seconds"),
        "certificate": asdict(cert),
    }
    try:
        patch_file = PATCH_DIR / f"{desc.patch_name}.json"
        with open(patch_file, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
    except Exception as e:
        logging.warning(f"patch file write failed: {e}")
    staged_patches[desc.patch_name] = rec
    logging.info(f"PATCH STAGED: {desc.patch_name} -> {desc.target} (kind={patch_kind})")
    return StagingResult(True, f"staged {desc.patch_name}", desc.patch_name, desc.target, asdict(cert), rec)


def stage_patch_proposal(
    proposal: PatchProposal | Dict[str, Any],
    *,
    source_kernel_id: str = "hub",
    packet: KernelPacket | None = None,
    staged_patches: Dict[str, Dict[str, Any]] | None = None,
    certs: CertificationLayer | None = None,
    kernel_class: type | None = None,
    state_beta_keys: set | None = None,
    allow_create_new: bool = False,
) -> StagingResult:
    if isinstance(proposal, dict):
        proposal = PatchProposal(**proposal)
    raw = json.dumps({
        "patch_name": proposal.patch_name,
        "target": proposal.target,
        "code": proposal.code,
        "rationale": proposal.rationale,
    })
    source = f"{proposal.source}:{source_kernel_id}" if proposal.source else source_kernel_id
    result = stage_patch(
        raw,
        source=source,
        staged_patches=staged_patches,
        certs=certs,
        kernel_class=kernel_class,
        state_beta_keys=state_beta_keys,
        allow_create_new=allow_create_new,
    )
    if result.ok and staged_patches is not None and proposal.patch_name in staged_patches:
        rec = staged_patches[proposal.patch_name]
        rec["ecology"] = {
            "source_kernel_id": source_kernel_id,
            "risk_level": proposal.risk_level,
            "risk_notes": proposal.risk_notes,
            "expected_state_changes": list(proposal.expected_state_changes),
            "approval_required": proposal.approval_required,
            "packet_id": packet.packet_id if packet is not None else "",
            "packet_kind": packet.packet_kind if packet is not None else "",
            "packet_provenance": dict(packet.provenance) if packet is not None else {},
        }
        result.record = rec
    return result
