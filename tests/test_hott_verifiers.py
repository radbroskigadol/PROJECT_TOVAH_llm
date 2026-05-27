"""
Tests for the HoTT verifier layers above the core.

Covers:
  - patch_certificates: certify_patch, default_probes, verdict logic
  - memory_identity: classify_pair, find_genuine_conflicts
  - module_equivalence: can_substitute, substitution_witness
  - obstruction: cocycle_check, coboundary, is_trivializable, globalize,
                 lifting_obstruction
"""
from __future__ import annotations

import pytest

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.hott import (
    # patch certificates
    Patch, InvariantProbe, InvariantValue, PatchCertificate,
    TransportWitness, certify_patch, verify_certificate, default_probes,
    KernelStateType,
    # memory identity
    MemoryReferent, build_referent, identity_path,
    PairDiagnosis, classify_pair, is_genuine_conflict, find_genuine_conflicts,
    # module equivalence
    ModuleProperty, ModuleContract, can_substitute,
    substitution_witness, build_equiv, make_probe,
    # obstruction
    AbelianGroup, IntGroup, ModGroup, NonAbelianGroup,
    LocalFragment, Overlap, TransitionSymmetry,
    Cocycle, cocycle_check, ObstructionClass,
    coboundary, is_trivializable, obstruction_class,
    globalize, GlobalizationResult,
    LiftingObstruction, lifting_obstruction,
    # paraconsistent
    IdentityClass,
)


# --- Patch certificates ----------------------------------------------------

class TestPatchCertificates:
    def test_refl_patch_passes(self):
        """A patch that doesn't change anything must pass certification."""
        state = {"foo": 1, "bar": 2}
        patch = Patch(
            name="no_op", source_state=state, target_state=state,
            diff_witness="empty",
            bilateral=BilateralValue(1.0, 0.0),
        )
        probe = InvariantProbe(
            "test_invariant",
            probe=lambda s: s.get("foo"),
            protected=True,
        )
        cert = certify_patch(patch, [probe])
        assert cert.verdict == "pass"
        ok, _ = verify_certificate(cert)
        assert ok

    def test_protected_change_blocks(self):
        """Protected invariant change → block_refuted."""
        pre = {"foo": 1}
        post = {"foo": 2}
        patch = Patch("change_foo", pre, post, "diff", BilateralValue(0.8, 0.1))
        probe = InvariantProbe("foo_invariant", lambda s: s.get("foo"), protected=True)
        cert = certify_patch(patch, [probe])
        assert cert.verdict == "block_refuted"
        assert "foo_invariant" in cert.protected_failed
        ok, _ = verify_certificate(cert)
        assert not ok

    def test_non_protected_change_warns(self):
        pre = {"audit_count": 0}
        post = {"audit_count": 1}
        patch = Patch("bump", pre, post, "diff", BilateralValue(0.8, 0.1))
        probe = InvariantProbe(
            "audit", lambda s: s.get("audit_count"), protected=False,
        )
        cert = certify_patch(patch, [probe])
        assert cert.verdict == "warn"
        ok, _ = verify_certificate(cert)
        assert ok

    def test_probe_error_is_refutation(self):
        """A probe that raises is treated as refuted (fail-safe)."""
        def broken(_state):
            raise RuntimeError("probe failed")
        pre = {"x": 1}
        post = {"x": 1}
        patch = Patch("p", pre, post, "diff", BilateralValue(0.8, 0.1))
        probe = InvariantProbe("broken", broken, protected=True)
        cert = certify_patch(patch, [probe])
        assert cert.verdict == "block_refuted"

    def test_default_probes_callable(self):
        """default_probes() returns a non-empty list of InvariantProbes."""
        probes = default_probes()
        assert len(probes) >= 6  # six from the brief
        for p in probes:
            assert isinstance(p, InvariantProbe)

    def test_certificate_to_dict_serializes_cleanly(self):
        patch = Patch("p", {"a": 1}, {"a": 1}, "d", BilateralValue(0.9, 0.05))
        probe = InvariantProbe("a", lambda s: s["a"], protected=True)
        cert = certify_patch(patch, [probe])
        d = cert.to_dict()
        assert d["verdict"] == "pass"
        assert d["patch_name"] == "p"
        assert len(d["witnesses"]) == 1


# --- Memory identity -------------------------------------------------------

class TestMemoryIdentity:
    def test_same_module_same_test_same_version_conflict(self):
        """Brief's example: 'module failed' vs 'module succeeded' on
        same module/version/test → SAME_OBJECT_CONFLICT."""
        m1 = {"subject": "GateModule", "version": "v3", "test": "sandbox_run",
              "bilateral_confidence": {"t": 0.9, "f": 0.1},
              "created_at": 1000.0}
        m2 = {"subject": "GateModule", "version": "v3", "test": "sandbox_run",
              "bilateral_confidence": {"t": 0.1, "f": 0.9},
              "created_at": 1001.0}
        rep = classify_pair(m1, m2)
        assert rep.diagnosis == PairDiagnosis.SAME_OBJECT_CONFLICT
        assert rep.referent_class == IdentityClass.A
        assert is_genuine_conflict(m1, m2)

    def test_bilateralvalue_object_assessments_conflict(self):
        m1 = {"subject": "GateModule", "version": "v3", "test": "sandbox_run",
              "bilateral_assessment": BilateralValue(0.9, 0.1)}
        m2 = {"subject": "GateModule", "version": "v3", "test": "sandbox_run",
              "bilateral_assessment": BilateralValue(0.1, 0.9)}
        rep = classify_pair(m1, m2)
        assert rep.diagnosis == PairDiagnosis.SAME_OBJECT_CONFLICT
        assert is_genuine_conflict(m1, m2)

    def test_different_version_is_different_object(self):
        """Same module name + same test but different versions:
        primary mismatch on version → at least K-class (paradox of
        identification) — and is_genuine_conflict returns False because
        we can't decide if they're the same object."""
        m1 = {"subject": "GateModule", "version": "v3", "test": "sandbox_run",
              "bilateral_confidence": {"t": 0.9, "f": 0.1}}
        m2 = {"subject": "GateModule", "version": "v2", "test": "sandbox_run",
              "bilateral_confidence": {"t": 0.1, "f": 0.9}}
        rep = classify_pair(m1, m2)
        # Same-subject + same-test + different-version → K (ambiguous):
        # supporting evidence for identification (subject, test) AND
        # refuting evidence (version). Not a clean DIFFERENT_OBJECT.
        assert rep.diagnosis in (
            PairDiagnosis.AMBIGUOUS_IDENTIFICATION,
            PairDiagnosis.DIFFERENT_OBJECT,
        )
        # The CRUCIAL property: we don't count this as a genuine conflict.
        # That's the brief's point — version mismatch should NOT count as
        # a contradiction in contradiction-hygiene.
        assert not is_genuine_conflict(m1, m2)

    def test_different_subjects_is_different_object(self):
        m1 = {"subject": "FooMod", "version": "v1",
              "bilateral_confidence": {"t": 0.9, "f": 0.1}}
        m2 = {"subject": "BarMod", "version": "v1",
              "bilateral_confidence": {"t": 0.1, "f": 0.9}}
        rep = classify_pair(m1, m2)
        assert rep.diagnosis == PairDiagnosis.DIFFERENT_OBJECT
        assert not is_genuine_conflict(m1, m2)

    def test_no_subject_info_is_insufficient(self):
        m1 = {"bilateral_confidence": {"t": 0.9, "f": 0.1}}
        m2 = {"bilateral_confidence": {"t": 0.1, "f": 0.9}}
        rep = classify_pair(m1, m2)
        # Both referents are empty MemoryReferent objects → they ARE equal,
        # so identity_path returns refl → A-class.
        # The decision then turns on the bilateral conflict alone.
        # This may be SAME_OBJECT_CONFLICT or AGREE depending on tie-break.
        assert rep.diagnosis in (
            PairDiagnosis.SAME_OBJECT_CONFLICT,
            PairDiagnosis.SAME_OBJECT_AGREE,
            PairDiagnosis.INSUFFICIENT_INFO,
        )

    def test_find_genuine_conflicts_filters_spurious(self):
        memories = [
            {"subject": "M", "version": "v1", "bilateral_confidence": {"t": 0.9, "f": 0.1}},
            {"subject": "M", "version": "v1", "bilateral_confidence": {"t": 0.1, "f": 0.9}},  # conflict with [0]
            {"subject": "M", "version": "v2", "bilateral_confidence": {"t": 0.1, "f": 0.9}},  # different version
        ]
        conflicts = find_genuine_conflicts(memories)
        # Only (0, 1) should be a genuine conflict; (0, 2) is spurious.
        assert len(conflicts) == 1
        i, j, _ = conflicts[0]
        assert (i, j) == (0, 1)


# --- Module equivalence ----------------------------------------------------

class TestModuleEquivalence:
    def _atomic_writes(self, _module): return True
    def _no_eval(self, _module): return False

    def test_identical_contracts_substitute(self):
        atomic = make_probe("atomic_writes", lambda _: True, priority=10)
        no_eval = make_probe("uses_eval", lambda _: False, priority=10)
        A = ModuleContract("DBv1", "v1", {"read", "write"}, [atomic], [no_eval])
        B = ModuleContract("DBv1", "v1", {"read", "write"}, [atomic], [no_eval])
        assert can_substitute(A, B)

    def test_capability_only_identical_contracts_substitute(self):
        A = ModuleContract("CapOnly", "v1", {"read", "write"}, [], [])
        B = ModuleContract("CapOnly", "v1", {"read", "write"}, [], [])
        assert can_substitute(A, B)
        ce = substitution_witness(A, B)
        assert ce.judgment.class_ == IdentityClass.A

    def test_extra_capabilities_dont_block_substitution(self):
        atomic = make_probe("atomic_writes", lambda _: True, priority=10)
        no_eval = make_probe("uses_eval", lambda _: False, priority=10)
        A = ModuleContract("DBv1", "v1", {"read", "write"}, [atomic], [no_eval])
        B = ModuleContract("DBv2", "v2",
                           {"read", "write", "transactions"}, [atomic], [no_eval])
        assert can_substitute(A, B)
        ce = substitution_witness(A, B)
        assert "transactions" in ce.capability_only_b

    def test_missing_capability_blocks_substitution(self):
        atomic = make_probe("atomic_writes", lambda _: True, priority=10)
        A = ModuleContract("DB", "v1", {"read", "write"}, [atomic], [])
        B = ModuleContract("ReadOnly", "v1", {"read"}, [atomic], [])
        assert not can_substitute(A, B)
        ce = substitution_witness(A, B)
        assert "write" in ce.capability_only_a

    def test_dropped_guarantee_blocks_substitution(self):
        atomic = make_probe("atomic_writes", lambda _: True, priority=10)
        A = ModuleContract("DB", "v1", {"read", "write"}, [atomic], [])
        B = ModuleContract("NoAtomic", "v1", {"read", "write"}, [], [])
        assert not can_substitute(A, B)
        ce = substitution_witness(A, B)
        assert "atomic_writes" in ce.guarantees_dropped

    def test_forbid_violation_blocks_substitution(self):
        atomic = make_probe("atomic_writes", lambda _: True, priority=10)
        no_eval = make_probe("uses_eval", lambda _: False, priority=10)
        A = ModuleContract("DB", "v1", {"read"}, [atomic], [no_eval])
        # B GUARANTEES what A FORBIDS.
        B = ModuleContract("EvalDB", "v1", {"read"},
                           [atomic, make_probe("uses_eval", lambda _: True, 10)],
                           [])
        assert not can_substitute(A, B)
        ce = substitution_witness(A, B)
        assert "uses_eval" in ce.forbids_violated


# --- Obstruction classifier ------------------------------------------------

class TestObstruction:
    def _make_fragments(self, names=("U1", "U2", "U3")):
        return [LocalFragment(name=n, domain=frozenset([n])) for n in names]

    def test_coboundary_is_trivializable(self):
        """The coboundary of any 0-cochain is trivializable."""
        frags = self._make_fragments()
        c = coboundary(IntGroup(), frags, {"U1": 1, "U2": 2, "U3": 3})
        ok, assignment = is_trivializable(c, frags)
        assert ok
        # Recovered assignment matches up to pivot translation.
        # Original {U1:1, U2:2, U3:3} → coboundary g_ij = a_i - a_j.
        # Pivot U1 = 0; a_2 = inv(g_12) = inv(1-2) = inv(-1) = 1; a_3 = 2.
        assert assignment["U1"] == 0
        assert assignment["U2"] == 1
        assert assignment["U3"] == 2

    def test_chain_shaped_coboundary_is_trivializable(self):
        frags = self._make_fragments()
        c = Cocycle(group=IntGroup())
        c.transitions[("U1", "U2")] = TransitionSymmetry(
            Overlap("U1", "U2", frozenset()), -1,
        )
        c.transitions[("U2", "U3")] = TransitionSymmetry(
            Overlap("U2", "U3", frozenset()), -1,
        )
        ok, assignment = is_trivializable(c, frags)
        assert ok
        assert assignment == {"U1": 0, "U2": 1, "U3": 2}

    def test_coboundary_globalizes(self):
        frags = self._make_fragments()
        c = coboundary(IntGroup(), frags, {"U1": 0, "U2": 1, "U3": 2})
        result = globalize(frags, c)
        assert result.success
        assert result.global_object is not None

    def test_non_closing_cocycle_blocks_globalization(self):
        """A cocycle that fails the triangle equation cannot globalize."""
        frags = self._make_fragments()
        c = Cocycle(group=IntGroup())
        c.transitions[("U1", "U2")] = TransitionSymmetry(
            Overlap("U1", "U2", frozenset()), 1,
        )
        c.transitions[("U2", "U3")] = TransitionSymmetry(
            Overlap("U2", "U3", frozenset()), 1,
        )
        # g_13 should equal 1+1 = 2, but we set it to 5.
        c.transitions[("U1", "U3")] = TransitionSymmetry(
            Overlap("U1", "U3", frozenset()), 5,
        )
        check = cocycle_check(c, frags)
        assert not check.closes
        result = globalize(frags, c)
        assert not result.success

    def test_mod2_cocycle_z2_check(self):
        frags = self._make_fragments()
        m = ModGroup(2)
        c = Cocycle(group=m)
        # g_12 = 1, g_23 = 1, g_13 = 0 (since 1+1=0 mod 2) → closes.
        c.transitions[("U1", "U2")] = TransitionSymmetry(Overlap("U1", "U2", frozenset()), 1)
        c.transitions[("U2", "U3")] = TransitionSymmetry(Overlap("U2", "U3", frozenset()), 1)
        c.transitions[("U1", "U3")] = TransitionSymmetry(Overlap("U1", "U3", frozenset()), 0)
        check = cocycle_check(c, frags)
        assert check.closes

    def test_lifting_obstruction_trivial(self):
        """When the section IS a homomorphism, the H² class is trivial."""
        frags = self._make_fragments()
        Q = IntGroup()
        A = IntGroup()
        # Trivial central extension: identity section.
        section = lambda q: q
        Ghat_op = lambda x, y: x + y
        Ghat_inv = lambda x: -x
        project_to_A = lambda x: 0  # projection is trivial since A is central
        Q_cycle = coboundary(Q, frags, {"U1": 1, "U2": 2, "U3": 3})
        lift_obs = lifting_obstruction(
            Q_cycle, A, section, Ghat_op, Ghat_inv, project_to_A, frags,
        )
        assert lift_obs.class_ == IdentityClass.A
        assert len(lift_obs.failures) == 0
