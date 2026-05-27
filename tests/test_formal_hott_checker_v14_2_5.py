"""Regression tests for v14.2.5 formal HoTT checker."""
from tovah_v14.hott.formal import (
    Ann,
    App,
    Context,
    FormalHoTTChecker,
    Fst,
    IdType,
    JElim,
    Lam,
    Pair,
    PiType,
    Refl,
    SigmaType,
    Snd,
    Sort,
    TypeErrorHoTT,
    Var,
    alpha_eq,
    identity_function,
    pretty,
)


def base_checker():
    c = FormalHoTTChecker()
    c.add_axiom("A", Sort(0))
    c.add_axiom("a", Var("A"))
    c.add_axiom("b", Var("A"))
    return c


def test_universe_and_axiom_typing():
    c = FormalHoTTChecker()
    assert c.infer(Sort(0)) == Sort(1)
    c.add_axiom("A", Sort(0))
    assert c.infer(Var("A")) == Sort(0)


def test_pi_lambda_identity_checks_and_beta_reduces():
    c = base_checker()
    A = Var("A")
    idA = identity_function(A)
    id_type = PiType("x", A, A)
    c.check(idA, id_type)
    assert c.defeq(c.infer(idA), id_type)
    assert c.defeq(App(idA, Var("a")), Var("a"))


def test_dependent_pair_checks_and_projects():
    c = base_checker()
    A = Var("A")
    sig = SigmaType("x", A, A)
    pair = Pair(Var("a"), Var("b"), as_type=sig)
    assert c.defeq(c.infer(pair), sig)
    assert c.defeq(Fst(pair), Var("a"))
    assert c.defeq(Snd(pair), Var("b"))
    assert c.defeq(c.infer(Fst(pair)), A)
    assert c.defeq(c.infer(Snd(pair)), A)


def test_identity_type_and_refl_check():
    c = base_checker()
    A = Var("A")
    a = Var("a")
    refl_a = Refl(a)
    assert c.defeq(c.infer(refl_a), IdType(A, a, a))
    c.check(refl_a, IdType(A, a, a))


def test_j_eliminator_computation_rule_on_refl():
    c = base_checker()
    A = Var("A")
    a = Var("a")
    # motive : Π z:A, Π q:Id_A(a,z), Type0
    motive = Lam("z", A, Lam("q", IdType(A, a, Var("z")), Sort(0)))
    base = A  # base : Type0 = motive a refl_a
    term = JElim(motive=motive, base=base, target=a, path=Refl(a))
    assert c.defeq(c.infer(term), Sort(0))
    assert c.defeq(term, base)


def test_j_eliminator_rejects_wrong_endpoint():
    c = base_checker()
    A = Var("A")
    a = Var("a")
    b = Var("b")
    motive = Lam("z", A, Lam("q", IdType(A, a, Var("z")), Sort(0)))
    # path is refl_a, so target must be a, not b.
    term = JElim(motive=motive, base=A, target=b, path=Refl(a))
    try:
        c.infer(term)
        assert False, "expected endpoint mismatch"
    except TypeErrorHoTT:
        pass


def test_global_definition_unfolds_for_definitional_equality():
    c = base_checker()
    A = Var("A")
    idA = identity_function(A)
    id_type = PiType("x", A, A)
    c.add_definition("idA", id_type, idA)
    assert c.defeq(App(Var("idA"), Var("a")), Var("a"))


def test_alpha_equivalence_for_bound_names():
    A = Var("A")
    lhs = PiType("x", A, Var("x"))
    rhs = PiType("y", A, Var("y"))
    assert alpha_eq(lhs, rhs)


def test_rejects_bad_application():
    c = base_checker()
    try:
        c.infer(App(Var("a"), Var("b")))
        assert False, "expected non-function application to fail"
    except TypeErrorHoTT:
        pass
