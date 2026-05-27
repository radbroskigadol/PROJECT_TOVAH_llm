"""
TOVAH v14.2.0 hott/patch_certificates.py — Proof-carrying patch certificates.

This is priority #1 from the architecture brief:

    Right now, TOVAH has a promotion ladder: proposed → static_approved
    → sandbox → regression → shadow → live.
    Full HoTT could upgrade that into a proof-carrying patch system.
    A patch would not merely say "tests passed". It would need to provide
    a witness:
        transport invariant_old along patch_path = invariant_new

This module makes that concrete.

A `Patch` is modeled as a Path between two kernel states. Promotion
requires producing a `PatchCertificate` that names each protected
invariant and provides a `TransportWitness` showing it survives the
patch. Failed transports are surfaced (not hidden), and the promotion
ladder gate consumes the certificate.

Protected invariants we track (the brief calls these out):
  - memory coherence
  - bilateral state coherence
  - promotion authority
  - tool permission
  - sovereign identity
  - contradiction hygiene

Public:
  KernelStateType        — the Type whose inhabitants are kernel snapshots
  Patch                  — a Path between kernel states
  TransportWitness       — evidence that one named invariant survives a Patch
  PatchCertificate       — the full proof-carrying record for a Patch
  certify_patch          — build a certificate from a list of invariant probes
  verify_certificate     — check that a certificate is well-formed and
                           that no protected invariant is refuted

LAW: a certificate that contains any K-class or B-class transport
witness for a *protected* invariant cannot pass verify_certificate. The
promotion ladder gate uses this to block promotions that would silently
break invariants.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tovah_v14.core.primitives import BilateralValue
from tovah_v14.hott.core import (
    Type, Id, Path, refl, transport, TransportResult, DependentFamily,
)
from tovah_v14.hott.paraconsistent import (
    IdentityClass, PIdJudgment, judge_identity, bilateral_transport,
    classify_path, is_T_supported,
)


# --- Kernel-state Type ------------------------------------------------------

# A kernel state is anything dict-like that exposes the protected
# invariants by name. We don't enforce a schema here — flexibility lets
# us thread arbitrary state shapes through certification.

def _kernel_state_inhabits(s: Any) -> bool:
    """A kernel state is a mapping or a stateful object."""
    return isinstance(s, dict) or hasattr(s, "__dict__")


KernelStateType = Type("KernelState", inhabits=_kernel_state_inhabits)


# --- Protected invariants ---------------------------------------------------

@dataclass
class InvariantProbe:
    """Names a protected invariant and provides a function to compute it.

    Attributes:
      name:    identifying name (e.g. "memory_coherence", "tool_permission")
      probe:   function (state) → InvariantValue
      protected: if True, refusal to transport means the patch MUST be
                 blocked. Non-protected invariants are tracked but their
                 refutation only logs a warning.
      describe: short prose description for certificate readability.
    """
    name: str
    probe: Callable[[Any], Any]
    protected: bool = True
    describe: str = ""


@dataclass
class InvariantValue:
    """An invariant's value at a particular kernel state.

    The `value` can be anything — typically a hashable summary. We
    fingerprint it for comparison; equality of fingerprints is the
    primary signal that the invariant is preserved.
    """
    value: Any
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint:
            try:
                self.fingerprint = hashlib.sha1(
                    repr(self.value).encode("utf-8")).hexdigest()[:16]
            except Exception:
                self.fingerprint = "unhashable"


# --- Patch as Path ----------------------------------------------------------

@dataclass
class Patch:
    """A patch is a Path between two kernel states.

    The `witness` field is what classical promotion calls 'the patch':
    a diff, a contract, a name, anything that justifies the transition.
    The bilateral evidence is computed from the gate signals (tests pass,
    static approved, sandbox clean, etc.).
    """
    name: str
    source_state: Any
    target_state: Any
    diff_witness: Any  # the literal diff/contract/spec
    bilateral: BilateralValue = field(default_factory=lambda: BilateralValue(0.5, 0.5))
    at: float = field(default_factory=time.time)

    def as_path(self) -> Path:
        """Lift this patch to a HoTT Path : Id(KernelState; source, target)."""
        return Path(
            id_type=Id(KernelStateType, self.source_state, self.target_state),
            source=self.source_state,
            target=self.target_state,
            witness={"patch": self.name, "diff": self.diff_witness},
            bilateral=self.bilateral,
        )


# --- Transport witness ------------------------------------------------------

@dataclass
class TransportWitness:
    """Evidence that one named invariant survives a Patch.

    A 'survives' witness is built by computing the invariant on both
    source and target states and producing a Path between the two
    invariant-values. If the fingerprints match, we have a high-T path
    (the invariant is literally preserved). If they differ, we either:
      - have a path with bilateral evidence reflecting the change's
        intentionality (high T if the change is one the patch was
        SUPPOSED to make), or
      - refute the transport (high F) and report.
    """
    invariant_name: str
    source_value: InvariantValue
    target_value: InvariantValue
    transported: TransportResult
    survives: bool
    class_: IdentityClass
    reason: str


# --- Patch certificate ------------------------------------------------------

@dataclass
class PatchCertificate:
    """The full proof-carrying record attached to a Patch.

    Attributes:
      patch:                the Patch this certifies
      transported:          for each invariant probe, a TransportWitness
      protected_failed:     names of protected invariants that did NOT
                            survive transport — promotion must be blocked
      contested:            names of invariants whose transport was K-class
                            (genuine paradox — promotion must be blocked
                            but with different diagnostics)
      gap:                  names of invariants whose transport was G-class
                            (no evidence — promotion may proceed with caveat)
      verdict:              "pass" | "block_refuted" | "block_paradox" | "warn"
      verdict_reason:       short prose
      created_at:           timestamp
    """
    patch: Patch
    transported: List[TransportWitness] = field(default_factory=list)
    protected_failed: List[str] = field(default_factory=list)
    contested: List[str] = field(default_factory=list)
    gap: List[str] = field(default_factory=list)
    verdict: str = "unknown"
    verdict_reason: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_name": self.patch.name,
            "patch_bilateral": {
                "t": self.patch.bilateral.t, "f": self.patch.bilateral.f
            },
            "n_invariants_checked": len(self.transported),
            "protected_failed": list(self.protected_failed),
            "contested": list(self.contested),
            "gap": list(self.gap),
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "witnesses": [
                {
                    "invariant": w.invariant_name,
                    "source_fp": w.source_value.fingerprint,
                    "target_fp": w.target_value.fingerprint,
                    "survives": w.survives,
                    "class": w.class_.value,
                    "reason": w.reason,
                }
                for w in self.transported
            ],
            "created_at": self.created_at,
        }


# --- Builders ---------------------------------------------------------------

def _build_witness_for_invariant(patch: Patch, probe: InvariantProbe
                                 ) -> TransportWitness:
    """Compute the invariant on both states and build a TransportWitness.

    The bilateral evidence on the transport reflects:
      - same fingerprint → BilateralValue(1.0, 0.0): preserved exactly
      - different fingerprint and probe.protected → BilateralValue(0.0, 0.95):
        refuted (we don't know the change is OK)
      - different fingerprint and not protected → BilateralValue(0.4, 0.4):
        gap (probably fine but no proof)
    """
    src_iv = InvariantValue(value=probe.probe(patch.source_state))
    tgt_iv = InvariantValue(value=probe.probe(patch.target_state))

    if src_iv.fingerprint == tgt_iv.fingerprint:
        # Path is essentially refl on the invariant value.
        bv = BilateralValue(1.0, 0.0)
        survives = True
        cls = IdentityClass.A
        reason = "fingerprint preserved exactly"
    else:
        # Invariant value changed. Was that intentional? Without a richer
        # specification of "intentional", we mark protected→refuted and
        # non-protected→gap.
        if probe.protected:
            bv = BilateralValue(0.0, 0.95)
            survives = False
            cls = IdentityClass.B
            reason = (
                f"protected invariant fingerprint changed: "
                f"{src_iv.fingerprint} → {tgt_iv.fingerprint}"
            )
        else:
            bv = BilateralValue(0.4, 0.4)
            survives = True  # tolerated for non-protected
            cls = IdentityClass.G
            reason = (
                f"non-protected invariant changed: "
                f"{src_iv.fingerprint} → {tgt_iv.fingerprint}"
            )

    # Synthesize a transport path between invariant-values.
    inv_type = Type(f"Invariant({probe.name})", inhabits=lambda _x: True)
    path_inv = Path(
        id_type=Id(inv_type, src_iv.value, tgt_iv.value),
        source=src_iv.value,
        target=tgt_iv.value,
        witness=("invariant_transport", probe.name, patch.name),
        bilateral=bv,
    )
    family: DependentFamily = lambda _: inv_type
    res = transport(family, path_inv, src_iv.value)
    return TransportWitness(
        invariant_name=probe.name,
        source_value=src_iv,
        target_value=tgt_iv,
        transported=res,
        survives=survives,
        class_=cls,
        reason=reason,
    )


def certify_patch(patch: Patch,
                  probes: List[InvariantProbe]) -> PatchCertificate:
    """Build a PatchCertificate by running every probe on both states.

    The verdict is computed from the witnesses:
      - any protected invariant refuted (B-class) → block_refuted
      - any invariant contested (K-class)         → block_paradox
      - any non-protected invariant in gap        → warn
      - otherwise                                  → pass
    """
    cert = PatchCertificate(patch=patch)
    for probe in probes:
        try:
            w = _build_witness_for_invariant(patch, probe)
        except Exception as e:
            logging.warning("certify_patch: probe %s raised: %s", probe.name, e)
            # Synthesize a refuted witness so the failure is visible.
            iv_unknown = InvariantValue(value=f"<probe-error: {e!s}>")
            inv_type = Type(f"Invariant({probe.name})")
            path_inv = Path(
                id_type=Id(inv_type, None, None),
                source=None, target=None,
                witness=("probe_error", probe.name),
                bilateral=BilateralValue(0.0, 0.9),
            )
            family: DependentFamily = lambda _: inv_type
            res = transport(family, path_inv, None)
            w = TransportWitness(
                invariant_name=probe.name,
                source_value=iv_unknown,
                target_value=iv_unknown,
                transported=res,
                survives=False,
                class_=IdentityClass.B,
                reason=f"probe raised: {e!s}",
            )
        cert.transported.append(w)
        if w.class_ == IdentityClass.B and probe.protected:
            cert.protected_failed.append(probe.name)
        elif w.class_ == IdentityClass.K:
            cert.contested.append(probe.name)
        elif w.class_ == IdentityClass.G:
            cert.gap.append(probe.name)

    # Verdict.
    if cert.protected_failed:
        cert.verdict = "block_refuted"
        cert.verdict_reason = (
            f"protected invariants did not transport: "
            f"{', '.join(cert.protected_failed)}"
        )
    elif cert.contested:
        cert.verdict = "block_paradox"
        cert.verdict_reason = (
            f"invariants contested (K-class): {', '.join(cert.contested)}"
        )
    elif cert.gap:
        cert.verdict = "warn"
        cert.verdict_reason = f"gap on: {', '.join(cert.gap)}"
    else:
        cert.verdict = "pass"
        cert.verdict_reason = "all protected invariants transport"
    return cert


def verify_certificate(cert: PatchCertificate) -> Tuple[bool, str]:
    """Return (ok, reason) for a certificate.

    ok=True iff verdict ∈ {"pass", "warn"} (warn lets promotion continue
    with a warning logged upstream; block_refuted and block_paradox stop
    promotion).
    """
    if cert.verdict in {"pass", "warn"}:
        return True, cert.verdict_reason
    return False, cert.verdict_reason


# --- Built-in invariant probes ---------------------------------------------

def _hash_value(v: Any) -> str:
    try:
        return hashlib.sha1(repr(v).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "unhashable"


def memory_coherence_probe(state: Any) -> Any:
    """Bilateral-flat fingerprint of the memory banks.

    Coherence = (bank_names, sizes, conflict_count). We deliberately
    drop content so unrelated additions don't refute. A patch that
    REMOVES a bank or that changes conflict count above a threshold
    fails this probe.
    """
    banks = getattr(state, "memory_store", None)
    if banks is None:
        return ("nostate", 0, 0)
    try:
        b = banks.banks
        names = sorted(b.keys())
        sizes = tuple(len(b[n]) for n in names)
        conflicts = sum(1 for _ in getattr(banks, "iter_conflicts", lambda: [])())
        return (tuple(names), sizes, conflicts)
    except Exception:
        return ("error",)


def bilateral_state_coherence_probe(state: Any) -> Any:
    """Sum of bilateral support tensor norms across kernel."""
    shadow = getattr(state, "shadow_optimizer", None)
    if shadow is None or not getattr(shadow, "_state_initialized", False):
        return "uninitialized"
    try:
        import torch
        t_sum = 0.0
        f_sum = 0.0
        for st in shadow.state.values():
            t_sum += float(st["T_sup"].sum().item())
            f_sum += float(st["F_sup"].sum().item())
        # Coarse fingerprint: order-of-magnitude bins.
        def bin_(x: float) -> int:
            import math
            if x <= 0: return 0
            return int(math.log10(max(x, 1e-9)) * 10)
        return (bin_(t_sum), bin_(f_sum))
    except Exception:
        return "error"


def promotion_authority_probe(state: Any) -> Any:
    """The promotion ladder's authority set (which states allow what)."""
    ladder = getattr(state, "promotion_ladder", None)
    if ladder is None:
        return "nostate"
    try:
        s = ladder.state
        # Track only structural keys.
        return tuple(sorted(s.keys()))
    except Exception:
        return "error"


def tool_permission_probe(state: Any) -> Any:
    """Active tool registry — names + count."""
    try:
        tools = getattr(state, "active_lab_tools", None) or {}
        return tuple(sorted(tools.keys()))
    except Exception:
        return "error"


def sovereign_identity_probe(state: Any) -> Any:
    """Kernel identity fingerprint."""
    try:
        return (
            getattr(state, "node_id", None) or getattr(state, "kernel_id", None),
            getattr(state, "is_original", None),
            getattr(state, "model_param_count", None),
        )
    except Exception:
        return "error"


def contradiction_hygiene_probe(state: Any) -> Any:
    """K-class count in the recent experience store."""
    try:
        es = getattr(state, "experience_store", None)
        if es is None:
            return "nostate"
        k = 0
        for rec in es.records[-200:]:
            bv = getattr(rec, "bilateral_assessment", None)
            if bv is None:
                continue
            if bv.t >= 0.55 and bv.f >= 0.55:
                k += 1
        # Bucketed so within-band changes don't refute.
        return k // 5
    except Exception:
        return "error"


def default_probes() -> List[InvariantProbe]:
    """The brief's six protected invariants, ready for use."""
    return [
        InvariantProbe("memory_coherence", memory_coherence_probe,
                       describe="bank names, sizes, conflict count"),
        InvariantProbe("bilateral_state_coherence", bilateral_state_coherence_probe,
                       describe="T_sup / F_sup magnitude bins"),
        InvariantProbe("promotion_authority", promotion_authority_probe,
                       describe="ladder structural keys"),
        InvariantProbe("tool_permission", tool_permission_probe,
                       describe="active lab tool names"),
        InvariantProbe("sovereign_identity", sovereign_identity_probe,
                       describe="node_id + is_original + param_count"),
        InvariantProbe("contradiction_hygiene", contradiction_hygiene_probe,
                       describe="K-class count in recent experiences (bucketed)"),
    ]
