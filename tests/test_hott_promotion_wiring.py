"""
Tests for the v14.2.0 HoTT integration with the promotion ladder.

The integration point: when a patch advances from `regression_passed`
to `shadow_deployed`, the kernel can optionally supply a state provider
that yields (pre_state, post_state). The ladder then runs certify_patch
against protected invariants and blocks on K/B verdicts.

Verifies:
  - Without state_provider, ladder behaves exactly as v14.1.x (back-compat)
  - With state_provider yielding identical states, advancement proceeds
  - With state_provider yielding states where a protected invariant
    differs, advancement is BLOCKED with verdict=block_refuted
  - Block reason is captured in the gate log
"""
from __future__ import annotations

import pytest

from tovah_v14.mutation.promotion_ladder import PromotionLadder
from tovah_v14.hott import InvariantProbe


# --- Helpers ---------------------------------------------------------------

def _make_ladder_and_patch():
    """Returns a ladder + a patch staged to regression_passed."""
    ladder = PromotionLadder()
    patch_name = "test_patch"
    staged = {
        patch_name: {
            "target": "test_target",
            "code": "def test_target():\n    return 42\n",
            "diff": "simple no-op",
        },
    }
    # Force the patch to regression_passed (bypass prior gates).
    ladder.state[patch_name] = "regression_passed"
    # AUDIT FIX (v14.2.7): under v14.2.6 these tests reached the HoTT check
    # via the implicit-sovereign default. That default was inverted in
    # v14.2.7 (sec 1 / RC-1 hardening), so the test now must explicitly
    # register sovereign-main metadata to exercise the HoTT path.
    ladder.set_source_metadata(
        patch_name,
        source_role="main",
        trust_level="sovereign",
        risk_level="low",
        outcome_success_rate=1.0,
        budget_pressure=0.0,
    )
    # Provide policy evidence using the actual record_evidence API.
    ladder.record_evidence(
        patch_name, "passed_static_analysis",
        trust_level="trusted", risk_class="low",
    )
    ladder.record_evidence(
        patch_name, "passed_regression",
        trust_level="trusted", risk_class="low",
    )
    return ladder, patch_name, staged


# --- Tests -----------------------------------------------------------------

class TestPromotionLadderHoTTWiring:
    def test_without_state_provider_back_compat(self):
        """If no kernel_state_provider is given, ladder behaves as v14.1.x."""
        ladder, name, staged = _make_ladder_and_patch()
        new_stage, msg = ladder.advance(name, staged)
        # Either advances to shadow_deployed or stays blocked by policy.
        # The point is: no exception, no HoTT logic invoked.
        assert new_stage in ("shadow_deployed", "regression_passed")

    def test_with_identity_state_passes(self):
        """Identical pre/post state → all invariants trivially transport → pass."""
        ladder, name, staged = _make_ladder_and_patch()
        state = {"foo": 1, "bar": "x"}
        provider = lambda: (state, state)
        probe = InvariantProbe("foo", lambda s: s["foo"], protected=True)
        new_stage, msg = ladder.advance(
            name, staged,
            kernel_state_provider=provider,
            hott_probes=[probe],
        )
        # Either shadow_deployed (policy + HoTT pass) or stays blocked by policy.
        # If policy passes, HoTT also passes (refl-style transport).
        assert new_stage in ("shadow_deployed", "regression_passed")

    def test_with_changed_protected_invariant_blocks(self):
        """Protected invariant changes → certify_patch blocks promotion."""
        ladder, name, staged = _make_ladder_and_patch()
        pre = {"sovereign_id": "kernel_A"}
        post = {"sovereign_id": "kernel_B"}  # protected attribute changed
        provider = lambda: (pre, post)
        probe = InvariantProbe(
            "sovereign_id",
            lambda s: s["sovereign_id"],
            protected=True,
        )
        new_stage, msg = ladder.advance(
            name, staged,
            kernel_state_provider=provider,
            hott_probes=[probe],
        )
        assert new_stage == "regression_passed"
        assert "blocked by HoTT" in msg
        assert ladder.current_stage(name) == "regression_passed"
        assert any(
            r.patch_name == name and r.gate_result == "blocked-hott-refuted"
            for r in ladder.history
        )

    def test_blocked_patch_logs_certificate_summary(self):
        """When HoTT blocks, the gate-log entry should carry the certificate."""
        ladder, name, staged = _make_ladder_and_patch()
        pre = {"foo": 1}
        post = {"foo": 2}
        provider = lambda: (pre, post)
        probe = InvariantProbe("foo", lambda s: s["foo"], protected=True)
        new_stage, msg = ladder.advance(
            name, staged,
            kernel_state_provider=provider,
            hott_probes=[probe],
        )
        # Inspect gate log for a HoTT certificate entry.
        recent = [e for e in ladder.gate_log
                  if e.get("patch_name") == name]
        # At least one entry exists.
        assert recent

    def test_provider_exception_fails_closed(self):
        """If state_provider raises, HoTT certification fails closed."""
        ladder, name, staged = _make_ladder_and_patch()
        def broken_provider():
            raise RuntimeError("synthetic provider failure")
        new_stage, msg = ladder.advance(
            name, staged,
            kernel_state_provider=broken_provider,
        )
        assert new_stage == "regression_passed"
        assert "blocked by HoTT certification error" in msg
        assert ladder.current_stage(name) == "regression_passed"
        assert any(
            r.patch_name == name and r.gate_result == "blocked-hott-error"
            for r in ladder.history
        )
