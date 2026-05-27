"""Patch-certificate smoke eval: invariant-breaking patch must block."""
from __future__ import annotations

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.evals.common import emit, result
from tovah_v14.hott.patch_certificates import InvariantProbe, Patch, certify_patch, verify_certificate


def run() -> dict:
    source = {"tool_permissions": ("read", "write"), "node_id": "alpha"}
    target = {"tool_permissions": ("read",), "node_id": "alpha"}
    patch = Patch(
        name="drop_write_permission_eval",
        source_state=source,
        target_state=target,
        diff_witness={"remove": "write"},
        bilateral=BilateralValue(0.9, 0.05),
    )
    probe = InvariantProbe(
        "tool_permission",
        lambda s: tuple(s.get("tool_permissions", ())),
        protected=True,
        describe="protected tool permissions should transport unchanged",
    )
    cert = certify_patch(patch, [probe])
    ok, reason = verify_certificate(cert)
    return result(
        "patch_certification_eval",
        (not ok) and cert.verdict == "block_refuted",
        verdict=cert.verdict,
        reason=reason,
        protected_failed=cert.protected_failed,
    )


if __name__ == "__main__":
    emit(run())
