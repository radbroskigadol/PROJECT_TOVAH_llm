"""
TOVAH v14 invariants/certification.py — Certification layer.

SEMANTIC PRESERVATION:
  CertificationLayer.certify_state, certify_report, and check
  are identical to v13.

Added: certify_patch_contract for v14 contract-based validation.
"""
from __future__ import annotations

import datetime as dt
from typing import Tuple

from tovah_v14.core.state import ShadowState
from tovah_v14.core.cache import is_cache_coherent
from tovah_v14.invariants.schemas import Certificate
from tovah_v14.invariants.state_invariants import InvariantReport


class CertificationLayer:
    """Issues and checks certificates for state, reports, patches, and capabilities."""

    def certify_state(self, s: ShadowState, profile: str = "default") -> Certificate:
        return Certificate(
            "StateCert", "14.2.6", profile,
            {"coherent": is_cache_coherent(s), "cache_size": len(s.nu), "step": s.pi.step},
            {"profile": profile},
            dt.datetime.now().isoformat(timespec="seconds"),
        )

    def certify_report(self, report: InvariantReport, profile: str = "default") -> Certificate:
        return Certificate(
            "ReportCert", "14.2.6", profile,
            {"coherent": report.coherent, "hist": report.cache_histogram,
             "mean_glut": report.mean_glut, "mean_gap": report.mean_gap},
            {"profile": profile},
            dt.datetime.now().isoformat(timespec="seconds"),
        )

    def certify_patch_contract(
        self,
        target: str,
        analysis_ok: bool,
        contract_ok: bool,
        errors: list[str],
        source: str = "unknown",
    ) -> Certificate:
        """Issue a certificate for a patch that has passed contract validation."""
        return Certificate(
            "PatchContractCert", "14.2.6", "default",
            {
                "target": target,
                "analysis_ok": analysis_ok,
                "contract_ok": contract_ok,
                "errors": errors[:10],
            },
            {"source": source},
            dt.datetime.now().isoformat(timespec="seconds"),
        )

    def check(self, cert: Certificate) -> Tuple[bool, str]:
        """Verify a certificate's structural validity."""
        if not cert.cert_kind or not cert.version or not cert.created_at:
            return False, "missing metadata"
        if cert.cert_kind == "StateCert":
            ok = isinstance(cert.witness.get("coherent"), bool) and isinstance(cert.witness.get("cache_size"), int)
            return ok, "state certificate accepted" if ok else "state certificate malformed"
        if cert.cert_kind == "ReportCert":
            hist = cert.witness.get("hist", {})
            ok = all(k in hist for k in ("T", "F", "B", "G"))
            return ok, "report certificate accepted" if ok else "report certificate malformed"
        if cert.cert_kind == "PatchCert":
            ok = cert.witness.get("target_allowed", False) and cert.witness.get("analysis_ok", False)
            return ok, "patch certificate accepted" if ok else "patch certificate rejected"
        if cert.cert_kind == "PatchContractCert":
            ok = cert.witness.get("analysis_ok", False) and cert.witness.get("contract_ok", False)
            return ok, "patch contract certificate accepted" if ok else "patch contract certificate rejected"
        if cert.cert_kind == "CapabilityCert":
            ok = cert.witness.get("module_loaded", False)
            return ok, "capability certificate accepted" if ok else "capability certificate rejected"
        return False, f"unknown cert kind: {cert.cert_kind}"
