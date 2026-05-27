"""v14.3.5 verifier-grounded FormalHoTT reward and GRPO scaffold.

This module is still intentionally small and dependency-free, but it is no
longer just a smoke-test wrapper.  It provides:

* a tiny S-expression grammar for Π/Σ/Id/Refl/Lam/Pair/App terms;
* deterministic text→AST parsing for constrained model outputs;
* a broader verifier curriculum;
* grouped reward/advantage computation in the GRPO style;
* a policy-gradient loss helper that can consume sampled log-prob sums.

The checker remains the source of truth.  The reward cannot be gamed by matching
strings: candidates must parse and type-check in ``hott.formal``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from tovah_v14.hott.formal import (
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
    Sort,
    Snd,
    Term,
    Var,
    pretty,
)


@dataclass(frozen=True)
class FormalHoTTTask:
    name: str
    expected_type: Term
    context: Context
    positive_term: Term
    description: str = ""
    grammar_hint: str = ""


@dataclass(frozen=True)
class FormalHoTTReward:
    accepted: bool
    reward: float
    error: Optional[str] = None
    parsed: Optional[Term] = None


def default_pi_sigma_id_tasks() -> List[FormalHoTTTask]:
    """Return a small verifier-grounded Π/Σ/Id task curriculum."""
    A = Var("A")
    B = Var("B")
    x = Var("x")
    y = Var("y")
    ctx_A = Context((("A", Sort(0)),))
    ctx_AB = Context((("A", Sort(0)), ("B", Sort(0))))
    ctx_Ax = Context((("A", Sort(0)), ("x", A)))
    ctx_Axy = Context((("A", Sort(0)), ("x", A), ("y", A)))
    sig_AA = SigmaType("z", A, A)
    return [
        FormalHoTTTask("pi_identity", PiType("x", A, A), ctx_A, Lam("x", A, Var("x")), "Produce λx.x."),
        FormalHoTTTask("id_refl", IdType(A, x, x), ctx_Ax, Refl(x), "Produce refl_x."),
        FormalHoTTTask("sigma_duplicate_pair", sig_AA, ctx_Ax, Pair(x, x), "Produce (x,x) : Σ(z:A),A."),
        FormalHoTTTask("fst_duplicate_pair", A, ctx_Ax, Fst(Pair(x, x, as_type=sig_AA)), "Project fst from a dependent pair."),
        FormalHoTTTask("snd_duplicate_pair", A, ctx_Ax, Snd(Pair(x, x, as_type=sig_AA)), "Project snd from a dependent pair."),
        FormalHoTTTask("const_function", PiType("x", A, B), Context((("A", Sort(0)), ("B", Sort(0)), ("b", B))), Lam("x", A, Var("b")), "Produce a constant function."),
        FormalHoTTTask("identity_application", A, ctx_Ax, App(Lam("z", A, Var("z")), x), "Apply identity to x."),
        FormalHoTTTask("nested_identity", PiType("x", A, IdType(A, Var("x"), Var("x"))), ctx_A, Lam("x", A, Refl(Var("x"))), "Produce λx.refl_x."),
        FormalHoTTTask("pair_type_formation", Sort(0), ctx_A, A, "Return an existing Type0 variable."),
    ]


_TOKEN_RE = re.compile(r"\s*(\(|\)|[^\s()]+)")


def _tokenize(src: str) -> List[str]:
    return [m.group(1) for m in _TOKEN_RE.finditer(src) if m.group(1).strip()]


class ParseError(ValueError):
    pass


def parse_term(src: str) -> Term:
    """Parse a tiny S-expression term grammar into ``hott.formal`` AST.

    Grammar examples:
      ``(lam x Type0 x)``
      ``(refl x)``
      ``(pair x x)``
      ``(app (lam z Type0 z) x)``
      ``(pi x Type0 Type0)``
      ``(id A x x)``
    """
    toks = _tokenize(src)
    if not toks:
        raise ParseError("empty candidate")
    pos = 0

    def parse() -> Term:
        nonlocal pos
        if pos >= len(toks):
            raise ParseError("unexpected end of input")
        tok = toks[pos]
        pos += 1
        if tok == "(":
            if pos >= len(toks):
                raise ParseError("missing operator")
            op = toks[pos].lower(); pos += 1
            if op in {"type", "sort"}:
                level_tok = toks[pos]; pos += 1
                term: Term = Sort(int(level_tok.replace("Type", "")))
            elif op == "pi":
                var = toks[pos]; pos += 1
                term = PiType(var, parse(), parse())
            elif op == "sigma":
                var = toks[pos]; pos += 1
                term = SigmaType(var, parse(), parse())
            elif op == "lam":
                var = toks[pos]; pos += 1
                term = Lam(var, parse(), parse())
            elif op == "app":
                term = App(parse(), parse())
            elif op == "pair":
                first = parse(); second = parse()
                # Optional explicit as_type: (pair x x (sigma z A A))
                as_type = None
                if pos < len(toks) and toks[pos] != ")":
                    as_type = parse()
                term = Pair(first, second, as_type=as_type)
            elif op == "fst":
                term = Fst(parse())
            elif op == "snd":
                term = Snd(parse())
            elif op == "id":
                term = IdType(parse(), parse(), parse())
            elif op == "refl":
                term = Refl(parse())
            elif op == "ann":
                # Annotation is represented as identity application in this tiny
                # parser only where callers need it; avoiding extra import keeps
                # the grammar compact.  Use checker.check(term, type) externally.
                term = parse(); _ = parse()
            else:
                raise ParseError(f"unknown operator {op!r}")
            if pos >= len(toks) or toks[pos] != ")":
                raise ParseError(f"missing ')' after {op}")
            pos += 1
            return term
        if tok == ")":
            raise ParseError("unexpected ')'")
        if tok.startswith("Type") and tok[4:].isdigit():
            return Sort(int(tok[4:]))
        return Var(tok)

    term = parse()
    if pos != len(toks):
        raise ParseError(f"trailing tokens: {' '.join(toks[pos:])}")
    return term


def verify_formal_hott_term(term: Term, expected_type: Term, *, context: Optional[Context] = None) -> FormalHoTTReward:
    checker = FormalHoTTChecker()
    ctx = context or Context()
    try:
        checker.check(term, expected_type, ctx)
        return FormalHoTTReward(accepted=True, reward=1.0, parsed=term)
    except Exception as exc:
        return FormalHoTTReward(accepted=False, reward=0.0, error=str(exc), parsed=term)


def reward_task_candidate(task: FormalHoTTTask, candidate: Term | str) -> FormalHoTTReward:
    try:
        term = parse_term(candidate) if isinstance(candidate, str) else candidate
    except Exception as exc:
        return FormalHoTTReward(accepted=False, reward=0.0, error=f"parse: {exc}")
    return verify_formal_hott_term(term, task.expected_type, context=task.context)


def grouped_rewards(tasks: Sequence[FormalHoTTTask], candidates_by_task: Sequence[Sequence[str | Term]]) -> List[List[FormalHoTTReward]]:
    return [[reward_task_candidate(task, cand) for cand in cands] for task, cands in zip(tasks, candidates_by_task)]


def grpo_advantages(reward_groups: Sequence[Sequence[FormalHoTTReward]], eps: float = 1e-6) -> List[torch.Tensor]:
    """Compute per-task normalized advantages A=(r-mean)/std."""
    out: List[torch.Tensor] = []
    for group in reward_groups:
        r = torch.tensor([g.reward for g in group], dtype=torch.float32)
        if r.numel() == 0:
            out.append(r)
            continue
        std = r.std(unbiased=False).clamp_min(eps)
        out.append((r - r.mean()) / std)
    return out


def grpo_policy_loss(logprob_sums_by_task: Sequence[torch.Tensor], advantages: Sequence[torch.Tensor], clip_ratio: float = 0.2, old_logprob_sums_by_task: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
    """PPO/GRPO-style clipped policy loss over grouped candidate log-probs."""
    losses = []
    for i, (lp, adv) in enumerate(zip(logprob_sums_by_task, advantages)):
        if lp.numel() == 0:
            continue
        adv = adv.to(device=lp.device, dtype=lp.dtype).detach()
        if old_logprob_sums_by_task is None:
            losses.append(-(lp * adv).mean())
        else:
            old = old_logprob_sums_by_task[i].to(device=lp.device, dtype=lp.dtype).detach()
            ratio = torch.exp(lp - old)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
            losses.append(-torch.minimum(unclipped, clipped).mean())
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def positive_candidate_strings(tasks: Optional[Iterable[FormalHoTTTask]] = None) -> Dict[str, str]:
    """Human-readable seed candidates for smoke and constrained decoding tests."""
    return {
        "pi_identity": "(lam x A x)",
        "id_refl": "(refl x)",
        "sigma_duplicate_pair": "(pair x x)",
        "fst_duplicate_pair": "(fst (pair x x (sigma z A A)))",
        "snd_duplicate_pair": "(snd (pair x x (sigma z A A)))",
        "const_function": "(lam x A b)",
        "identity_application": "(app (lam z A z) x)",
        "nested_identity": "(lam x A (refl x))",
        "pair_type_formation": "A",
    }


def smoke_score_suite(tasks: Optional[Iterable[FormalHoTTTask]] = None) -> dict:
    ts = list(tasks or default_pi_sigma_id_tasks())
    seeds = positive_candidate_strings()
    results = {t.name: reward_task_candidate(t, seeds.get(t.name, t.positive_term)) for t in ts}
    return {
        "schema_version": "formal-hott-rl-v14.3.5",
        "n": len(ts),
        "accepted": sum(1 for r in results.values() if r.accepted),
        "reward_mean": sum(r.reward for r in results.values()) / max(1, len(ts)),
        "tasks": {k: {"accepted": v.accepted, "reward": v.reward, "error": v.error, "parsed": pretty(v.parsed) if v.parsed else None} for k, v in results.items()},
    }


__all__ = [
    "FormalHoTTTask", "FormalHoTTReward", "ParseError",
    "default_pi_sigma_id_tasks", "parse_term", "verify_formal_hott_term",
    "reward_task_candidate", "grouped_rewards", "grpo_advantages",
    "grpo_policy_loss", "positive_candidate_strings", "smoke_score_suite",
]
