"""
Tests for the v14.2.0 HoTT formal layer.

These tests are not just unit tests — they verify the *structural laws*
of HoTT (refl-J reduction, transport along refl, compose associativity,
equiv composition). If any of these fail, the package is no longer
implementing HoTT correctly.
"""
from __future__ import annotations

import pytest

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.hott import (
    Type, Id, Path, refl, compose, inverse,
    transport, J, TransportResult,
    Equiv, is_equiv, equiv_compose,
    Sigma, Pi, check_pi, TruncationLevel,
    IdentityClass, PIdJudgment,
    classify_path, judge_identity, combine_judgments,
    bilateral_J, bilateral_transport,
    check_refl_J_reduction, check_transport_along_refl,
    check_compose_associativity,
    path_bilateral_summary,
)


# --- Core: refl, transport, J ----------------------------------------------

class TestRefl:
    def test_refl_has_perfect_evidence(self):
        A = Type("A", inhabits=lambda _: True)
        r = refl(A, 42)
        assert r.is_refl
        assert r.bilateral.t == 1.0
        assert r.bilateral.f == 0.0
        assert r.source == 42
        assert r.target == 42

    def test_refl_raises_on_non_inhabitant(self):
        A = Type("Bool", inhabits=lambda x: isinstance(x, bool))
        with pytest.raises(ValueError):
            refl(A, "not a bool")

    def test_refl_supports_identification(self):
        A = Type("A")
        r = refl(A, "x")
        assert r.supports_identification


class TestTransport:
    def test_transport_along_refl_is_identity(self):
        A = Type("A", inhabits=lambda _: True)
        P = lambda x: Type(f"P({x})", inhabits=lambda _: True)
        r = refl(A, 7)
        result = transport(P, r, "payload")
        assert result.value == "payload"
        assert result.bilateral.t == 1.0
        assert result.bilateral.f == 0.0

    def test_transport_inherits_path_evidence(self):
        A = Type("A", inhabits=lambda _: True)
        P = lambda x: Type(f"P({x})", inhabits=lambda _: True)
        contested_path = Path(
            id_type=Id(A, "a", "b"),
            source="a", target="b",
            witness="contested",
            bilateral=BilateralValue(0.7, 0.6),  # K-class
        )
        result = transport(P, contested_path, "payload")
        # Bilateral evidence propagates.
        assert result.bilateral.t == 0.7
        assert result.bilateral.f == 0.6
        assert not result.supports_use  # contested

    def test_transport_rejects_target_non_inhabitant(self):
        A = Type("A", inhabits=lambda _: True)
        path = Path(
            id_type=Id(A, "a", "b"),
            source="a", target="b",
            witness="bad_coerce",
            bilateral=BilateralValue(0.9, 0.0),
        )
        def P(endpoint):
            if endpoint == "a":
                return Type("P(a)", inhabits=lambda x: isinstance(x, int))
            return Type("P(b)", inhabits=lambda x: isinstance(x, str))
        with pytest.raises(ValueError, match="does not inhabit target"):
            transport(P, path, 7, coerce=lambda x, _src, _dst: x)


class TestJ:
    def test_J_refl_reduction(self):
        """J(C, d, refl_a) reduces to d(a) — the universal property."""
        A = Type("A")
        C = lambda x, y, p: Type("C", inhabits=lambda _: True)
        d = lambda x: f"d({x})"
        assert check_refl_J_reduction(C, d, "hello", A)

    def test_J_on_non_refl_uses_transport(self):
        A = Type("A", inhabits=lambda _: True)
        # Build a non-refl path.
        p = Path(
            id_type=Id(A, "a", "b"),
            source="a", target="b",
            witness="proof",
            bilateral=BilateralValue(0.9, 0.05),
        )
        C = lambda x, y, p: Type("C", inhabits=lambda _: True)
        d = lambda x: f"d({x})"
        result = J(C, d, p)
        assert isinstance(result, TransportResult)


class TestPathAlgebra:
    def test_compose_endpoints(self):
        A = Type("A", inhabits=lambda _: True)
        p = Path(Id(A, "a", "b"), "a", "b", "p", BilateralValue(0.9, 0.05))
        q = Path(Id(A, "b", "c"), "b", "c", "q", BilateralValue(0.8, 0.1))
        pq = compose(p, q)
        assert pq.source == "a"
        assert pq.target == "c"

    def test_compose_min_max_semantics(self):
        A = Type("A", inhabits=lambda _: True)
        p = Path(Id(A, "a", "b"), "a", "b", "p", BilateralValue(0.9, 0.05))
        q = Path(Id(A, "b", "c"), "b", "c", "q", BilateralValue(0.6, 0.4))
        pq = compose(p, q, merge_bilateral="min")
        assert pq.bilateral.t == pytest.approx(0.6)  # min of (0.9, 0.6)
        assert pq.bilateral.f == pytest.approx(0.4)  # max of (0.05, 0.4)

    def test_compose_associativity_truncated(self):
        A = Type("A", inhabits=lambda _: True)
        p = Path(Id(A, "a", "b"), "a", "b", "p", BilateralValue(0.9, 0.05))
        q = Path(Id(A, "b", "c"), "b", "c", "q", BilateralValue(0.8, 0.1))
        r = Path(Id(A, "c", "d"), "c", "d", "r", BilateralValue(0.7, 0.15))
        assert check_compose_associativity(p, q, r)

    def test_compose_endpoint_mismatch_raises(self):
        A = Type("A", inhabits=lambda _: True)
        p = Path(Id(A, "a", "b"), "a", "b", "p")
        q = Path(Id(A, "x", "c"), "x", "c", "q")  # gap: q.source != p.target
        with pytest.raises(ValueError):
            compose(p, q)

    def test_inverse_preserves_strength(self):
        A = Type("A", inhabits=lambda _: True)
        p = Path(Id(A, "a", "b"), "a", "b", "p", BilateralValue(0.85, 0.1))
        pinv = inverse(p)
        assert pinv.source == "b"
        assert pinv.target == "a"
        assert pinv.bilateral.t == p.bilateral.t
        assert pinv.bilateral.f == p.bilateral.f


# --- Equivalence -----------------------------------------------------------

class TestEquiv:
    def test_identity_equiv(self):
        A = Type("A", inhabits=lambda _: True)
        e = Equiv(
            A=A, B=A,
            f=lambda x: x, g=lambda x: x,
            eta=lambda a: refl(A, a),
            epsilon=lambda b: refl(A, b),
        )
        # Sample check.
        ok, msg = is_equiv(e.f, A, A, e.g, check_samples=[1, "x", (3, 4)])
        assert ok, msg

    def test_equiv_composition(self):
        A = Type("A", inhabits=lambda _: True)
        B = Type("B", inhabits=lambda _: True)
        C = Type("C", inhabits=lambda _: True)
        e1 = Equiv(A=A, B=B,
                   f=lambda x: ("AB", x), g=lambda b: b[1],
                   eta=lambda a: refl(A, a),
                   epsilon=lambda b: refl(B, b))
        e2 = Equiv(A=B, B=C,
                   f=lambda b: ("BC", b), g=lambda c: c[1],
                   eta=lambda a: refl(B, a),
                   epsilon=lambda b: refl(C, b))
        composed = equiv_compose(e1, e2)
        # Sample check that composition works.
        a = 5
        c = composed.f(a)
        a_roundtrip = composed.g(c)
        assert a_roundtrip == a


# --- Sigma / Pi -----------------------------------------------------------

class TestDependentTypes:
    def test_sigma_inhabits(self):
        Nat = Type("Nat", inhabits=lambda n: isinstance(n, int) and n >= 0)
        B = lambda n: Type(f"<{n}", inhabits=lambda k: isinstance(k, int) and k < n)
        SigmaType = Sigma(Nat, B)
        assert SigmaType.inhabits((5, 3))
        assert not SigmaType.inhabits((5, 7))  # 7 not < 5
        assert not SigmaType.inhabits((-1, 0))  # -1 not in Nat

    def test_pi_check(self):
        Nat = Type("Nat", inhabits=lambda n: isinstance(n, int) and n >= 0)
        # B(n) = the int 2n
        B = lambda n: Type(f"{2*n}", inhabits=lambda k: k == 2 * n)
        good = lambda n: 2 * n
        bad = lambda n: n + 1  # wrong
        ok, _ = check_pi(Nat, B, good, [0, 1, 5, 10])
        assert ok
        ok2, msg = check_pi(Nat, B, bad, [0, 1, 5])
        assert not ok2


# --- Paraconsistent layer --------------------------------------------------

class TestClassification:
    def test_refl_is_A_class(self):
        A = Type("A")
        r = refl(A, 1)
        assert classify_path(r) == IdentityClass.A

    def test_contested_path_is_K(self):
        A = Type("A", inhabits=lambda _: True)
        p = Path(Id(A, "x", "y"), "x", "y", "w",
                 BilateralValue(0.8, 0.7))
        assert classify_path(p) == IdentityClass.K

    def test_refuted_path_is_B(self):
        A = Type("A", inhabits=lambda _: True)
        p = Path(Id(A, "x", "y"), "x", "y", "w",
                 BilateralValue(0.2, 0.8))
        assert classify_path(p) == IdentityClass.B

    def test_gap_is_G(self):
        A = Type("A", inhabits=lambda _: True)
        p = Path(Id(A, "x", "y"), "x", "y", "w",
                 BilateralValue(0.3, 0.3))
        assert classify_path(p) == IdentityClass.G


class TestBilateralJTransport:
    def test_bilateral_transport_refuses_on_K(self):
        A = Type("A", inhabits=lambda _: True)
        P = lambda x: Type(f"P({x})", inhabits=lambda _: True)
        contested = Path(Id(A, "a", "b"), "a", "b", "w",
                         BilateralValue(0.8, 0.7))
        j = judge_identity(Id(A, "a", "b"), supporting=[contested], refuting=[contested])
        result, j2 = bilateral_transport(P, j, "x")
        assert result is None
        assert j2.class_ == IdentityClass.K

    def test_bilateral_transport_proceeds_on_A(self):
        A = Type("A", inhabits=lambda _: True)
        P = lambda x: Type(f"P({x})", inhabits=lambda _: True)
        supported = Path(Id(A, "a", "b"), "a", "b", "w",
                         BilateralValue(0.9, 0.05))
        j = judge_identity(Id(A, "a", "b"), supporting=[supported], refuting=[])
        result, _ = bilateral_transport(P, j, "x")
        assert result is not None
        assert result.value == "x"

    def test_bilateral_J_refuses_on_K(self):
        A = Type("A", inhabits=lambda _: True)
        C = lambda x, y, p: Type("C", inhabits=lambda _: True)
        d = lambda x: f"d({x})"
        contested = Path(Id(A, "a", "b"), "a", "b", "w",
                         BilateralValue(0.8, 0.7))
        j = judge_identity(Id(A, "a", "b"), [contested], [contested])
        result, j2 = bilateral_J(C, d, j)
        assert result is None
        assert j2.class_ == IdentityClass.K


class TestJudgmentAggregation:
    def test_combine_judgments_unions_pools(self):
        A = Type("A", inhabits=lambda _: True)
        idt = Id(A, "a", "b")
        p1 = Path(idt, "a", "b", "p1", BilateralValue(0.8, 0.05))
        p2 = Path(idt, "a", "b", "p2", BilateralValue(0.3, 0.9))
        j1 = judge_identity(idt, [p1], [])
        j2 = judge_identity(idt, [], [p2])
        combined = combine_judgments([j1, j2])
        # Combined has both supporting and refuting → K.
        assert combined.class_ == IdentityClass.K
        assert combined.best_t == pytest.approx(0.8)
        assert combined.best_f == pytest.approx(0.9)


class TestPathBilateralSummary:
    def test_summary_counts(self):
        A = Type("A", inhabits=lambda _: True)
        paths = [
            refl(A, 1),
            Path(Id(A, 1, 2), 1, 2, "w", BilateralValue(0.9, 0.05)),
            Path(Id(A, 1, 2), 1, 2, "w", BilateralValue(0.7, 0.7)),  # K
            Path(Id(A, 1, 2), 1, 2, "w", BilateralValue(0.1, 0.9)),  # B
        ]
        s = path_bilateral_summary(paths)
        assert s["n"] == 4
        assert s["n_refl"] == 1
        assert s["n_contested"] == 1
        assert s["n_refuted"] == 1
        assert s["n_supported"] == 2  # refl and the A-class path
