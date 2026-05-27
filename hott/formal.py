"""
TOVAH v14.2.6 hott/formal.py — bounded formal HoTT checker.

This module adds a real, executable dependent-type-checking kernel to the
existing operational HoTT layer.  It is intentionally small, auditable, and
kernel-like: terms are immutable AST nodes, typing is bidirectional, beta/J
normalization is explicit, and definitional equality is alpha/beta/J-normal
comparison.

Implemented calculus
--------------------
  * predicative universes Type_i : Type_{i+1}
  * variables and transparent global definitions
  * dependent products   Π (x : A), B(x)
  * lambda abstraction and application
  * dependent sums       Σ (x : A), B(x)
  * pairs and projections
  * identity types       Id_A(a,b)
  * refl                 refl_a : Id_A(a,a)
  * J/path induction     J(P, d, y, p)
  * annotations, normalization, alpha-equivalence, substitution

What this does NOT claim
------------------------
This is not Lean/Coq/Agda and not a complete general-purpose proof assistant.
It is a formal HoTT kernel for the fragment TOVAH needs: identity, transport,
J-like elimination, dependent products/sums, and proof-carrying coherence
certificates.  The surrounding HoTT package uses this as the formal substrate
for patch/memory/module/obstruction witnesses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple


class TypeErrorHoTT(Exception):
    """Raised when a term is not well typed in the formal HoTT kernel."""


class DefinitionalEqualityError(TypeErrorHoTT):
    """Raised when two terms are not definitionally equal."""


# ---------------------------------------------------------------------------
# Term language
# ---------------------------------------------------------------------------


class Term:
    """Base class for formal HoTT terms."""


@dataclass(frozen=True)
class Sort(Term):
    """Universe Type_level."""

    level: int = 0

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("universe level must be non-negative")


@dataclass(frozen=True)
class Var(Term):
    name: str


@dataclass(frozen=True)
class PiType(Term):
    var: str
    var_type: Term
    body_type: Term


@dataclass(frozen=True)
class Lam(Term):
    var: str
    var_type: Term
    body: Term


@dataclass(frozen=True)
class App(Term):
    fn: Term
    arg: Term


@dataclass(frozen=True)
class SigmaType(Term):
    var: str
    var_type: Term
    body_type: Term


@dataclass(frozen=True)
class Pair(Term):
    first: Term
    second: Term
    # Optional explicit Σ type; without this, a pair is checked against an
    # expected type but cannot always be inferred.
    as_type: Optional[Term] = None


@dataclass(frozen=True)
class Fst(Term):
    pair: Term


@dataclass(frozen=True)
class Snd(Term):
    pair: Term


@dataclass(frozen=True)
class IdType(Term):
    A: Term
    left: Term
    right: Term


@dataclass(frozen=True)
class Refl(Term):
    value: Term


@dataclass(frozen=True)
class JElim(Term):
    """Identity eliminator/path induction.

    Standard form, anchored at the left endpoint of the supplied path:

      p : Id_A(x, y)
      motive : Π (z : A), Π (q : Id_A(x,z)), Type_i
      base : motive x refl_x
      J(motive, base, y, p) : motive y p

    Computation rule:

      J(motive, base, x, refl_x)  ↦  base
    """

    motive: Term
    base: Term
    target: Term
    path: Term


@dataclass(frozen=True)
class Ann(Term):
    term: Term
    type: Term


# ---------------------------------------------------------------------------
# Context / environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Definition:
    name: str
    type: Term
    term: Optional[Term] = None
    transparent: bool = True


@dataclass(frozen=True)
class Context:
    entries: Tuple[Tuple[str, Term], ...] = ()

    def extend(self, name: str, type_: Term) -> "Context":
        return Context(self.entries + ((name, type_),))

    def lookup(self, name: str) -> Optional[Term]:
        for n, t in reversed(self.entries):
            if n == name:
                return t
        return None

    def names(self) -> Set[str]:
        return {n for n, _ in self.entries}


@dataclass
class Environment:
    definitions: Dict[str, Definition] = field(default_factory=dict)

    def add_axiom(self, name: str, type_: Term) -> Definition:
        if name in self.definitions:
            raise TypeErrorHoTT(f"definition {name!r} already exists")
        d = Definition(name=name, type=type_, term=None, transparent=False)
        self.definitions[name] = d
        return d

    def add_definition(self, name: str, type_: Term, term: Term, *, transparent: bool = True) -> Definition:
        if name in self.definitions:
            raise TypeErrorHoTT(f"definition {name!r} already exists")
        checker = FormalHoTTChecker(self)
        checker.check(term, type_)
        d = Definition(name=name, type=type_, term=term, transparent=transparent)
        self.definitions[name] = d
        return d

    def lookup_type(self, name: str) -> Optional[Term]:
        d = self.definitions.get(name)
        return d.type if d else None

    def lookup_term(self, name: str) -> Optional[Term]:
        d = self.definitions.get(name)
        if d and d.transparent:
            return d.term
        return None


# ---------------------------------------------------------------------------
# Free variables and substitution
# ---------------------------------------------------------------------------


def free_vars(t: Term) -> Set[str]:
    if isinstance(t, Sort):
        return set()
    if isinstance(t, Var):
        return {t.name}
    if isinstance(t, PiType):
        return free_vars(t.var_type) | (free_vars(t.body_type) - {t.var})
    if isinstance(t, Lam):
        return free_vars(t.var_type) | (free_vars(t.body) - {t.var})
    if isinstance(t, App):
        return free_vars(t.fn) | free_vars(t.arg)
    if isinstance(t, SigmaType):
        return free_vars(t.var_type) | (free_vars(t.body_type) - {t.var})
    if isinstance(t, Pair):
        s = free_vars(t.first) | free_vars(t.second)
        if t.as_type is not None:
            s |= free_vars(t.as_type)
        return s
    if isinstance(t, Fst):
        return free_vars(t.pair)
    if isinstance(t, Snd):
        return free_vars(t.pair)
    if isinstance(t, IdType):
        return free_vars(t.A) | free_vars(t.left) | free_vars(t.right)
    if isinstance(t, Refl):
        return free_vars(t.value)
    if isinstance(t, JElim):
        return free_vars(t.motive) | free_vars(t.base) | free_vars(t.target) | free_vars(t.path)
    if isinstance(t, Ann):
        return free_vars(t.term) | free_vars(t.type)
    raise TypeError(f"unknown term in free_vars: {t!r}")


def _fresh(base: str, avoid: Iterable[str]) -> str:
    avoid_set = set(avoid)
    if base not in avoid_set:
        return base
    i = 0
    while True:
        candidate = f"{base}_{i}"
        if candidate not in avoid_set:
            return candidate
        i += 1


def rename_var(t: Term, old: str, new: str) -> Term:
    return subst(t, old, Var(new), _renaming=True)


def subst(t: Term, var: str, replacement: Term, *, _renaming: bool = False) -> Term:
    """Capture-avoiding substitution [replacement/var]t."""

    if isinstance(t, Sort):
        return t
    if isinstance(t, Var):
        return replacement if t.name == var else t
    if isinstance(t, PiType):
        new_var_type = subst(t.var_type, var, replacement, _renaming=_renaming)
        if t.var == var:
            return PiType(t.var, new_var_type, t.body_type)
        body = t.body_type
        if not _renaming and t.var in free_vars(replacement):
            fresh = _fresh(t.var, free_vars(body) | free_vars(replacement) | {var})
            body = rename_var(body, t.var, fresh)
            return PiType(fresh, new_var_type, subst(body, var, replacement))
        return PiType(t.var, new_var_type, subst(body, var, replacement, _renaming=_renaming))
    if isinstance(t, Lam):
        new_var_type = subst(t.var_type, var, replacement, _renaming=_renaming)
        if t.var == var:
            return Lam(t.var, new_var_type, t.body)
        body = t.body
        if not _renaming and t.var in free_vars(replacement):
            fresh = _fresh(t.var, free_vars(body) | free_vars(replacement) | {var})
            body = rename_var(body, t.var, fresh)
            return Lam(fresh, new_var_type, subst(body, var, replacement))
        return Lam(t.var, new_var_type, subst(body, var, replacement, _renaming=_renaming))
    if isinstance(t, App):
        return App(subst(t.fn, var, replacement, _renaming=_renaming), subst(t.arg, var, replacement, _renaming=_renaming))
    if isinstance(t, SigmaType):
        new_var_type = subst(t.var_type, var, replacement, _renaming=_renaming)
        if t.var == var:
            return SigmaType(t.var, new_var_type, t.body_type)
        body = t.body_type
        if not _renaming and t.var in free_vars(replacement):
            fresh = _fresh(t.var, free_vars(body) | free_vars(replacement) | {var})
            body = rename_var(body, t.var, fresh)
            return SigmaType(fresh, new_var_type, subst(body, var, replacement))
        return SigmaType(t.var, new_var_type, subst(body, var, replacement, _renaming=_renaming))
    if isinstance(t, Pair):
        return Pair(
            subst(t.first, var, replacement, _renaming=_renaming),
            subst(t.second, var, replacement, _renaming=_renaming),
            subst(t.as_type, var, replacement, _renaming=_renaming) if t.as_type is not None else None,
        )
    if isinstance(t, Fst):
        return Fst(subst(t.pair, var, replacement, _renaming=_renaming))
    if isinstance(t, Snd):
        return Snd(subst(t.pair, var, replacement, _renaming=_renaming))
    if isinstance(t, IdType):
        return IdType(
            subst(t.A, var, replacement, _renaming=_renaming),
            subst(t.left, var, replacement, _renaming=_renaming),
            subst(t.right, var, replacement, _renaming=_renaming),
        )
    if isinstance(t, Refl):
        return Refl(subst(t.value, var, replacement, _renaming=_renaming))
    if isinstance(t, JElim):
        return JElim(
            subst(t.motive, var, replacement, _renaming=_renaming),
            subst(t.base, var, replacement, _renaming=_renaming),
            subst(t.target, var, replacement, _renaming=_renaming),
            subst(t.path, var, replacement, _renaming=_renaming),
        )
    if isinstance(t, Ann):
        return Ann(
            subst(t.term, var, replacement, _renaming=_renaming),
            subst(t.type, var, replacement, _renaming=_renaming),
        )
    raise TypeError(f"unknown term in subst: {t!r}")


# ---------------------------------------------------------------------------
# Normalization and definitional equality
# ---------------------------------------------------------------------------


class FormalHoTTChecker:
    """Bidirectional checker for the bounded formal HoTT calculus."""

    def __init__(self, env: Optional[Environment] = None):
        self.env = env or Environment()

    # ---- normalization -----------------------------------------------------

    def whnf(self, term: Term, ctx: Context = Context()) -> Term:
        if isinstance(term, Ann):
            return self.whnf(term.term, ctx)
        if isinstance(term, Var):
            unfolded = self.env.lookup_term(term.name)
            if unfolded is not None and ctx.lookup(term.name) is None:
                return self.whnf(unfolded, ctx)
            return term
        if isinstance(term, App):
            fn = self.whnf(term.fn, ctx)
            if isinstance(fn, Lam):
                return self.whnf(subst(fn.body, fn.var, term.arg), ctx)
            return App(fn, term.arg)
        if isinstance(term, Fst):
            pair = self.whnf(term.pair, ctx)
            if isinstance(pair, Pair):
                return self.whnf(pair.first, ctx)
            return Fst(pair)
        if isinstance(term, Snd):
            pair = self.whnf(term.pair, ctx)
            if isinstance(pair, Pair):
                return self.whnf(pair.second, ctx)
            return Snd(pair)
        if isinstance(term, JElim):
            p = self.whnf(term.path, ctx)
            if isinstance(p, Refl):
                return self.whnf(term.base, ctx)
            return JElim(term.motive, term.base, self.whnf(term.target, ctx), p)
        return term

    def normalize(self, term: Term, ctx: Context = Context()) -> Term:
        t = self.whnf(term, ctx)
        if isinstance(t, Sort) or isinstance(t, Var):
            return t
        if isinstance(t, PiType):
            return PiType(t.var, self.normalize(t.var_type, ctx), self.normalize(t.body_type, ctx.extend(t.var, t.var_type)))
        if isinstance(t, Lam):
            return Lam(t.var, self.normalize(t.var_type, ctx), self.normalize(t.body, ctx.extend(t.var, t.var_type)))
        if isinstance(t, App):
            return self.whnf(App(self.normalize(t.fn, ctx), self.normalize(t.arg, ctx)), ctx)
        if isinstance(t, SigmaType):
            return SigmaType(t.var, self.normalize(t.var_type, ctx), self.normalize(t.body_type, ctx.extend(t.var, t.var_type)))
        if isinstance(t, Pair):
            return Pair(
                self.normalize(t.first, ctx),
                self.normalize(t.second, ctx),
                self.normalize(t.as_type, ctx) if t.as_type is not None else None,
            )
        if isinstance(t, Fst):
            return self.whnf(Fst(self.normalize(t.pair, ctx)), ctx)
        if isinstance(t, Snd):
            return self.whnf(Snd(self.normalize(t.pair, ctx)), ctx)
        if isinstance(t, IdType):
            return IdType(self.normalize(t.A, ctx), self.normalize(t.left, ctx), self.normalize(t.right, ctx))
        if isinstance(t, Refl):
            return Refl(self.normalize(t.value, ctx))
        if isinstance(t, JElim):
            return self.whnf(JElim(
                self.normalize(t.motive, ctx),
                self.normalize(t.base, ctx),
                self.normalize(t.target, ctx),
                self.normalize(t.path, ctx),
            ), ctx)
        if isinstance(t, Ann):
            return self.normalize(t.term, ctx)
        raise TypeError(f"unknown term in normalize: {t!r}")

    def defeq(self, a: Term, b: Term, ctx: Context = Context()) -> bool:
        return alpha_eq(self.normalize(a, ctx), self.normalize(b, ctx))

    def require_defeq(self, a: Term, b: Term, ctx: Context = Context(), msg: str = "") -> None:
        if not self.defeq(a, b, ctx):
            suffix = f": {msg}" if msg else ""
            raise DefinitionalEqualityError(f"terms are not definitionally equal{suffix}: {pretty(a)} ≠ {pretty(b)}")

    # ---- typing ------------------------------------------------------------

    def infer(self, term: Term, ctx: Context = Context()) -> Term:
        if isinstance(term, Sort):
            return Sort(term.level + 1)

        if isinstance(term, Var):
            local = ctx.lookup(term.name)
            if local is not None:
                return local
            global_t = self.env.lookup_type(term.name)
            if global_t is not None:
                return global_t
            raise TypeErrorHoTT(f"unbound variable {term.name!r}")

        if isinstance(term, PiType):
            u_dom = self._infer_sort_level(term.var_type, ctx)
            u_body = self._infer_sort_level(term.body_type, ctx.extend(term.var, term.var_type))
            return Sort(max(u_dom, u_body))

        if isinstance(term, SigmaType):
            u_dom = self._infer_sort_level(term.var_type, ctx)
            u_body = self._infer_sort_level(term.body_type, ctx.extend(term.var, term.var_type))
            return Sort(max(u_dom, u_body))

        if isinstance(term, Lam):
            self._infer_sort_level(term.var_type, ctx)
            body_t = self.infer(term.body, ctx.extend(term.var, term.var_type))
            return PiType(term.var, term.var_type, body_t)

        if isinstance(term, App):
            fn_t = self.whnf(self.infer(term.fn, ctx), ctx)
            if not isinstance(fn_t, PiType):
                raise TypeErrorHoTT(f"cannot apply non-function of type {pretty(fn_t)}")
            self.check(term.arg, fn_t.var_type, ctx)
            return subst(fn_t.body_type, fn_t.var, term.arg)

        if isinstance(term, Pair):
            if term.as_type is None:
                raise TypeErrorHoTT("cannot infer an unannotated dependent pair; use Pair(..., as_type=SigmaType(...)) or check against a Sigma type")
            self.check(term, term.as_type, ctx)
            return term.as_type

        if isinstance(term, Fst):
            p_t = self.whnf(self.infer(term.pair, ctx), ctx)
            if not isinstance(p_t, SigmaType):
                raise TypeErrorHoTT(f"fst expected Σ type, got {pretty(p_t)}")
            return p_t.var_type

        if isinstance(term, Snd):
            p_t = self.whnf(self.infer(term.pair, ctx), ctx)
            if not isinstance(p_t, SigmaType):
                raise TypeErrorHoTT(f"snd expected Σ type, got {pretty(p_t)}")
            return subst(p_t.body_type, p_t.var, Fst(term.pair))

        if isinstance(term, IdType):
            self._infer_sort_level(term.A, ctx)
            self.check(term.left, term.A, ctx)
            self.check(term.right, term.A, ctx)
            # Id_A(a,b) lives in the same universe as A for this predicative kernel.
            return self.infer(term.A, ctx)

        if isinstance(term, Refl):
            A = self.infer(term.value, ctx)
            return IdType(A, term.value, term.value)

        if isinstance(term, JElim):
            path_t = self.whnf(self.infer(term.path, ctx), ctx)
            if not isinstance(path_t, IdType):
                raise TypeErrorHoTT(f"J path must have identity type, got {pretty(path_t)}")
            A, x, y = path_t.A, path_t.left, path_t.right
            self.check(term.target, A, ctx)
            self.require_defeq(term.target, y, ctx, "J target must match path endpoint")
            motive_t = self.whnf(self.infer(term.motive, ctx), ctx)
            if not isinstance(motive_t, PiType):
                raise TypeErrorHoTT(f"J motive must be a dependent function over the endpoint, got {pretty(motive_t)}")
            self.require_defeq(motive_t.var_type, A, ctx, "J motive endpoint domain")

            z_var = Var(motive_t.var)
            inner_t = self.whnf(motive_t.body_type, ctx.extend(motive_t.var, A))
            if not isinstance(inner_t, PiType):
                raise TypeErrorHoTT(f"J motive must return a dependent function over paths, got {pretty(inner_t)}")
            expected_path_domain = IdType(A, x, z_var)
            self.require_defeq(inner_t.var_type, expected_path_domain, ctx.extend(motive_t.var, A), "J motive path domain")
            codomain_sort = self.whnf(inner_t.body_type, ctx.extend(motive_t.var, A).extend(inner_t.var, inner_t.var_type))
            if not isinstance(codomain_sort, Sort):
                raise TypeErrorHoTT(f"J motive codomain must be a universe, got {pretty(codomain_sort)}")

            base_type = App(App(term.motive, x), Refl(x))
            self.check(term.base, base_type, ctx)
            return App(App(term.motive, y), term.path)

        if isinstance(term, Ann):
            self._infer_sort_level(term.type, ctx)
            self.check(term.term, term.type, ctx)
            return term.type

        raise TypeErrorHoTT(f"cannot infer type of unknown term {term!r}")

    def check(self, term: Term, expected: Term, ctx: Context = Context()) -> None:
        expected_whnf = self.whnf(expected, ctx)

        if isinstance(term, Lam) and isinstance(expected_whnf, PiType):
            self.require_defeq(term.var_type, expected_whnf.var_type, ctx, "lambda parameter type")
            # Rename the expected codomain binder to the lambda binder.
            expected_body = subst(expected_whnf.body_type, expected_whnf.var, Var(term.var))
            self.check(term.body, expected_body, ctx.extend(term.var, term.var_type))
            return

        if isinstance(term, Pair) and isinstance(expected_whnf, SigmaType):
            self.check(term.first, expected_whnf.var_type, ctx)
            second_type = subst(expected_whnf.body_type, expected_whnf.var, term.first)
            self.check(term.second, second_type, ctx)
            return

        inferred = self.infer(term, ctx)
        self.require_defeq(inferred, expected, ctx, "type check")

    def _infer_sort_level(self, term: Term, ctx: Context) -> int:
        typ = self.whnf(self.infer(term, ctx), ctx)
        if not isinstance(typ, Sort):
            raise TypeErrorHoTT(f"expected a type/universe, got term of type {pretty(typ)}")
        return typ.level

    # ---- convenience -------------------------------------------------------

    def add_axiom(self, name: str, type_: Term) -> Definition:
        self._infer_sort_level(type_, Context())
        return self.env.add_axiom(name, type_)

    def add_definition(self, name: str, type_: Term, term: Term, *, transparent: bool = True) -> Definition:
        self._infer_sort_level(type_, Context())
        return self.env.add_definition(name, type_, term, transparent=transparent)


# ---------------------------------------------------------------------------
# Equality / rendering
# ---------------------------------------------------------------------------


def alpha_eq(a: Term, b: Term, env: Optional[Mapping[str, str]] = None) -> bool:
    env = dict(env or {})
    if type(a) is not type(b):
        return False
    if isinstance(a, Sort) and isinstance(b, Sort):
        return a.level == b.level
    if isinstance(a, Var) and isinstance(b, Var):
        if a.name in env:
            return env[a.name] == b.name
        return a.name == b.name
    if isinstance(a, PiType) and isinstance(b, PiType):
        return alpha_eq(a.var_type, b.var_type, env) and alpha_eq(a.body_type, b.body_type, {**env, a.var: b.var})
    if isinstance(a, Lam) and isinstance(b, Lam):
        return alpha_eq(a.var_type, b.var_type, env) and alpha_eq(a.body, b.body, {**env, a.var: b.var})
    if isinstance(a, App) and isinstance(b, App):
        return alpha_eq(a.fn, b.fn, env) and alpha_eq(a.arg, b.arg, env)
    if isinstance(a, SigmaType) and isinstance(b, SigmaType):
        return alpha_eq(a.var_type, b.var_type, env) and alpha_eq(a.body_type, b.body_type, {**env, a.var: b.var})
    if isinstance(a, Pair) and isinstance(b, Pair):
        return alpha_eq(a.first, b.first, env) and alpha_eq(a.second, b.second, env) and (
            (a.as_type is None and b.as_type is None) or
            (a.as_type is not None and b.as_type is not None and alpha_eq(a.as_type, b.as_type, env))
        )
    if isinstance(a, Fst) and isinstance(b, Fst):
        return alpha_eq(a.pair, b.pair, env)
    if isinstance(a, Snd) and isinstance(b, Snd):
        return alpha_eq(a.pair, b.pair, env)
    if isinstance(a, IdType) and isinstance(b, IdType):
        return alpha_eq(a.A, b.A, env) and alpha_eq(a.left, b.left, env) and alpha_eq(a.right, b.right, env)
    if isinstance(a, Refl) and isinstance(b, Refl):
        return alpha_eq(a.value, b.value, env)
    if isinstance(a, JElim) and isinstance(b, JElim):
        return (
            alpha_eq(a.motive, b.motive, env) and alpha_eq(a.base, b.base, env)
            and alpha_eq(a.target, b.target, env) and alpha_eq(a.path, b.path, env)
        )
    if isinstance(a, Ann) and isinstance(b, Ann):
        return alpha_eq(a.term, b.term, env) and alpha_eq(a.type, b.type, env)
    return False


def pretty(t: Term) -> str:
    if isinstance(t, Sort):
        return f"Type{t.level}"
    if isinstance(t, Var):
        return t.name
    if isinstance(t, PiType):
        return f"(Π ({t.var} : {pretty(t.var_type)}), {pretty(t.body_type)})"
    if isinstance(t, Lam):
        return f"(λ ({t.var} : {pretty(t.var_type)}), {pretty(t.body)})"
    if isinstance(t, App):
        return f"({pretty(t.fn)} {pretty(t.arg)})"
    if isinstance(t, SigmaType):
        return f"(Σ ({t.var} : {pretty(t.var_type)}), {pretty(t.body_type)})"
    if isinstance(t, Pair):
        s = f"({pretty(t.first)}, {pretty(t.second)})"
        if t.as_type is not None:
            s += f" : {pretty(t.as_type)}"
        return s
    if isinstance(t, Fst):
        return f"fst({pretty(t.pair)})"
    if isinstance(t, Snd):
        return f"snd({pretty(t.pair)})"
    if isinstance(t, IdType):
        return f"Id({pretty(t.A)}, {pretty(t.left)}, {pretty(t.right)})"
    if isinstance(t, Refl):
        return f"refl({pretty(t.value)})"
    if isinstance(t, JElim):
        return f"J({pretty(t.motive)}, {pretty(t.base)}, {pretty(t.target)}, {pretty(t.path)})"
    if isinstance(t, Ann):
        return f"({pretty(t.term)} : {pretty(t.type)})"
    return repr(t)


# ---------------------------------------------------------------------------
# Builders useful for tests/docs
# ---------------------------------------------------------------------------


def identity_function(A: Term, x_name: str = "x") -> Lam:
    return Lam(x_name, A, Var(x_name))


def const_function(A: Term, B: Term, value: Term, x_name: str = "_") -> Lam:
    return Lam(x_name, A, value)


__all__ = [
    "TypeErrorHoTT", "DefinitionalEqualityError",
    "Term", "Sort", "Var", "PiType", "Lam", "App", "SigmaType", "Pair", "Fst", "Snd",
    "IdType", "Refl", "JElim", "Ann",
    "Definition", "Context", "Environment", "FormalHoTTChecker",
    "free_vars", "subst", "rename_var", "alpha_eq", "pretty",
    "identity_function", "const_function",
]
