"""
TOVAH v14 training/corpus_builder.py — Build TrainingExamples from telemetry.

Turns the existing telemetry sources (ExperienceStore, packet log, mutation
log, gate decisions, branch provenance, memory banks, wave histories, etc.)
into a uniform `TrainingExample` schema. Each example carries its own
lineage_id and provenance chain so a downstream trainer can condition on
how the example arose.

Source map (mirrors §4a of the audit):
- ExperienceStore       → kind="experience"
- kernel_packet_log     → kind="packet"
- promotion gate_log    → kind="gate_decision"
- module_proposals      → kind="module_proposal"
- memory banks          → kind="memory"
- mutation log          → kind="mutation"
- wave_resolution_hist  → kind="wave_resolution"
- wave_escalation_hist  → kind="wave_escalation"
- competence_map        → kind="competence_entry"
- traces                → kind="trace"
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --- Stable hashing helpers ------------------------------------------------

def _stable_hash(payload: Any) -> str:
    """Stable short hash for a JSON-serialisable payload.

    Used to mint deterministic lineage_ids: the same source object always
    gets the same id across runs.
    """
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# --- Envelope stripping (P0-4 / P2, v14.1.2) -------------------------------

# Regex matching the structural envelope prefixes we used to embed in the
# training text. The model should learn its own structure from the data,
# not learn to predict our square-bracket bookkeeping.
_ENVELOPE_PREFIX = re.compile(r"^\[[a-z_]+(?:\s+[a-z_]+=[^\]]*)*\]\s*\n?", re.IGNORECASE)


def strip_envelope(text: str) -> str:
    """Remove TOVAH's structural envelope prefix from a training-text body.

    Examples removed:
      "[experience kind=research]\\n..."
      "[packet kind=status from=hub to=main]\\n..."

    The kind/source/etc. metadata is preserved separately in TrainingExample
    fields and the `metadata` dict, where it belongs.
    """
    if not text:
        return text
    stripped = _ENVELOPE_PREFIX.sub("", text, count=1)
    return stripped.strip()


# --- Chunking long texts (P0-4, v14.1.2) -----------------------------------

def _chunk_text(text: str, *, chunk_bytes: int = 1024,
                overlap_bytes: int = 128) -> List[str]:
    """Split a long UTF-8 text into byte-bounded chunks with optional overlap.

    Used to ensure that long research findings, mutation diffs, etc., are
    not silently truncated at max_len. Chunks are produced with a small
    overlap so contiguous semantics survive splits. UTF-8 boundary safety:
    chunks never end mid-multi-byte-codepoint.

    For texts ≤ chunk_bytes, returns [text].
    """
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) <= chunk_bytes:
        return [text]
    chunks: List[str] = []
    step = max(1, chunk_bytes - overlap_bytes)
    i = 0
    while i < len(raw):
        end = min(len(raw), i + chunk_bytes)
        # Back off to a UTF-8 codepoint boundary if needed.
        while end > i and end < len(raw) and (raw[end] & 0xC0) == 0x80:
            end -= 1
        slab = raw[i:end].decode("utf-8", errors="ignore")
        if slab.strip():
            chunks.append(slab)
        if end >= len(raw):
            break
        i += step
    return chunks


# --- TrainingExample schema ------------------------------------------------

@dataclass
class TrainingExample:
    """A single training example with full lineage and quality signals."""

    lineage_id: str  # stable, deterministic across runs
    kind: str        # "experience", "packet", "gate_decision", ...
    text: str        # the actual training content
    time: float = 0.0
    source_kernel_id: str = ""
    mission_context: str = ""
    provenance_chain: List[str] = field(default_factory=list)
    outcome_label: str = ""  # useful/useless/contradictory/improved/regressed/neutral/...
    bilateral_t: float = 0.0  # truth mass (T)
    bilateral_f: float = 0.0  # falsity mass (F)
    quality_score: float = 0.0  # derived combined signal
    paraconsistent_class: str = ""  # filled by quality_filter (A/B/K/G)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --- Source extractors -----------------------------------------------------

def _experience_to_example(rec: Dict[str, Any]) -> TrainingExample:
    """Turn one ExperienceStore record into a TrainingExample.

    Tolerates two equivalent schemas:
      - live: ExperienceRecord-as-dict (record_id, action_type, context,
        outcome, reward_signal, bilateral_assessment, tags, ...)
      - file: state-export form (rec_id, kind, description/topic/findings)
    """
    bv = rec.get("bilateral_assessment", {}) or {}
    t = float(bv.get("t", bv.get("truth", 0.0)) or 0.0)
    f = float(bv.get("f", bv.get("falsity", 0.0)) or 0.0)
    outcome = str(rec.get("outcome", "") or "")
    rec_id = rec.get("rec_id") or rec.get("record_id") or ""
    rec_kind = rec.get("kind") or rec.get("action_type") or ""
    rec_time = rec.get("time") or rec.get("created_at") or 0.0
    reward = rec.get("reward")
    if reward is None:
        reward = rec.get("reward_signal", 0.0)
    # AUDIT FIX (P0-4 / P2, v14.1.2): the training text body is now the
    # natural content only. Structural envelope (kind, source, etc.) lives
    # in the TrainingExample metadata fields, not in the text the model
    # has to predict. This stops the model from wasting capacity learning
    # to predict our square-bracket bookkeeping.
    text_parts: List[str] = []
    desc = str(rec.get("description", "") or rec.get("topic", "") or "")
    if desc:
        text_parts.append(desc)
    if rec.get("findings"):
        text_parts.append("findings: " + str(rec.get("findings"))[:1000])
    ctx = rec.get("context") or {}
    if isinstance(ctx, dict) and ctx:
        for key in ("text", "topic", "description", "tool", "goal"):
            v = ctx.get(key)
            if v:
                text_parts.append(f"{key}: {v}")
        other = {k: v for k, v in ctx.items()
                 if k not in ("text", "topic", "description", "tool", "goal")}
        if other:
            try:
                text_parts.append("context: " + str(other)[:600])
            except Exception:
                pass
    tags = rec.get("tags") or []
    if tags:
        text_parts.append("tags: " + ",".join(str(x) for x in tags[:10]))
    text = "\n".join(p for p in text_parts if p).strip()
    if not text:
        # Last-resort placeholder so we don't yield empty examples.
        text = f"(experience kind={rec_kind} outcome={outcome or 'neutral'})"
    # AUDIT FIX (P0-3, v14.1.2): lineage hash is content-only — no time —
    # so identical records recorded at different times collapse correctly
    # under deduplicate(). rec_id is included only when present and
    # non-empty; otherwise we fingerprint the text body.
    if rec_id:
        lineage_key = f"id={rec_id}|kind={rec_kind}"
    else:
        lineage_key = f"kind={rec_kind}|body={text[:400]}"
    return TrainingExample(
        lineage_id=f"exp::{_stable_hash(lineage_key)}",
        kind="experience",
        text=text,
        time=float(rec_time or 0.0),
        source_kernel_id=str(rec.get("source_kernel_id", "") or ""),
        outcome_label=outcome or "neutral",
        bilateral_t=t,
        bilateral_f=f,
        quality_score=t - f,
        metadata={"rec_kind": rec_kind, "reward": float(reward or 0.0),
                  "envelope": f"experience kind={rec_kind}"},
    )


def _packet_to_example(pkt: Dict[str, Any]) -> Optional[TrainingExample]:
    """Turn a kernel packet log entry into a TrainingExample.

    Skips heartbeat / status packets — they're high-volume low-signal.

    AUDIT FIX (P0-3, v14.1.2): when no packet_id is set, the fallback
    fingerprint is content-only (no handled_at, no ordinal) so the same
    logical packet re-dispatched twice collapses under dedup.
    """
    kind = str(pkt.get("packet_kind", "") or "")
    if kind in {"status", "heartbeat", ""}:
        return None
    payload = pkt.get("payload", {}) or {}
    # AUDIT FIX (P0-4 / P2, v14.1.2): training text body holds the natural
    # content (mission + payload); structural envelope (kind, source, target)
    # is preserved in metadata.
    text = (
        f"mission: {pkt.get('mission_context', '')}\n"
        f"payload: {json.dumps(payload, default=str)[:1500]}"
    ).strip()
    if not text:
        text = f"(packet kind={kind} accepted={bool(pkt.get('accepted_by_main'))})"
    pkt_id = pkt.get("packet_id") or ""
    if pkt_id:
        lineage = f"pkt::{pkt_id}"
    else:
        # Content fingerprint: kind + source + target + payload (no time, no ordinal).
        fp_payload = {
            "kind": kind,
            "from": pkt.get("source_kernel_id", ""),
            "to": pkt.get("target_kernel_id", ""),
            "payload_head": json.dumps(payload, default=str, sort_keys=True)[:600],
        }
        lineage = f"pkt::{_stable_hash(fp_payload)}"
    chain = []
    if pkt.get("parent_goal_id"):
        chain.append(f"goal::{pkt['parent_goal_id']}")
    if pkt.get("provenance"):
        prov = pkt.get("provenance") or {}
        if prov.get("upstream_packet_id"):
            chain.append(f"pkt::{prov['upstream_packet_id']}")
    bv_t = 1.0 if pkt.get("accepted_by_main") else 0.5
    bv_f = 0.0 if pkt.get("accepted_by_main") else 0.3
    return TrainingExample(
        lineage_id=lineage,
        kind="packet",
        text=text,
        time=float(pkt.get("handled_at", 0.0) or 0.0),
        source_kernel_id=str(pkt.get("source_kernel_id", "") or ""),
        mission_context=str(pkt.get("mission_context", "") or ""),
        provenance_chain=chain,
        outcome_label="accepted" if pkt.get("accepted_by_main") else "routed",
        bilateral_t=bv_t,
        bilateral_f=bv_f,
        quality_score=bv_t - bv_f,
        metadata={"packet_kind": kind, "ordinal": pkt.get("ordinal", 0),
                  "envelope": f"packet kind={kind} from={pkt.get('source_kernel_id','')} to={pkt.get('target_kernel_id','')}"},
    )


def _gate_decision_to_example(dec: Dict[str, Any], patch_name: str = "") -> TrainingExample:
    """Gate-log entries from the promotion ladder.

    AUDIT FIX (P0-3, v14.1.2): lineage_id fingerprints decision content
    (patch + from + to + reason), not the whole dict — so the same
    decision recorded twice collapses under dedup. Adaptive metrics
    (evidence count, success rate) intentionally excluded so re-evaluations
    of the same patch state collapse.
    AUDIT FIX (P0-4 / P2, v14.1.2): envelope moved to metadata.
    """
    text = (
        f"transition: {dec.get('from','')} -> {dec.get('to','')}\n"
        f"reason: {dec.get('reason','')}\n"
        f"context: {json.dumps(dec.get('context', {}), default=str)[:1000]}"
    ).strip()
    if not text:
        text = f"(gate decision patch={patch_name})"
    success = "blocked" not in str(dec.get("reason", "")).lower()
    fp = {
        "patch": patch_name,
        "from": dec.get("from", ""),
        "to": dec.get("to", ""),
        "reason": dec.get("reason", ""),
        "allowed": dec.get("allowed", None),
    }
    return TrainingExample(
        lineage_id=f"gate::{_stable_hash(fp)}",
        kind="gate_decision",
        text=text,
        time=float(dec.get("at", dec.get("time", 0.0)) or 0.0),
        outcome_label="advanced" if success else "blocked",
        bilateral_t=0.8 if success else 0.2,
        bilateral_f=0.1 if success else 0.7,
        quality_score=(0.8 - 0.1) if success else (0.2 - 0.7),
        metadata={"patch_name": patch_name, "reason": dec.get("reason", ""),
                  "envelope": f"gate_decision patch={patch_name}"},
        provenance_chain=[f"patch::{patch_name}"] if patch_name else [],
    )


def _module_proposal_to_example(mp: Dict[str, Any]) -> TrainingExample:
    """Module proposal record from kernel ecology.

    AUDIT FIX (P0-3, v14.1.2): when no proposal_id is set, fingerprint
    content (name + kind + target + capabilities), not the full dict.
    AUDIT FIX (P0-4 / P2, v14.1.2): envelope moved to metadata.
    """
    review = mp.get("review_outcome", {}) or {}
    text = (
        f"capabilities: {mp.get('capabilities', [])}\n"
        f"review: status={review.get('status','')} reason={review.get('reason','')}"
    ).strip()
    if not text:
        text = f"(module_proposal name={mp.get('module_name','')})"
    status = str(review.get("status", "") or mp.get("status", ""))
    success = status in {"approved", "promoted"}
    pid = mp.get("proposal_id") or ""
    if pid:
        lineage = f"mod::{pid}"
    else:
        fp = {
            "name": mp.get("module_name", ""),
            "kind": mp.get("module_kind", ""),
            "target": mp.get("promotion_target", ""),
            "caps": sorted(mp.get("capabilities", []) or []),
        }
        lineage = f"mod::{_stable_hash(fp)}"
    return TrainingExample(
        lineage_id=lineage,
        kind="module_proposal",
        text=text,
        time=float(mp.get("time", mp.get("proposed_at", 0.0)) or 0.0),
        source_kernel_id=str(mp.get("proposer_kernel_id", "") or ""),
        outcome_label=status or "pending",
        bilateral_t=0.9 if success else 0.3,
        bilateral_f=0.05 if success else 0.55,
        quality_score=(0.9 - 0.05) if success else (0.3 - 0.55),
        metadata={"module_kind": mp.get("module_kind", ""),
                  "proposal_id": mp.get("proposal_id", ""),
                  "envelope": f"module_proposal name={mp.get('module_name','')} kind={mp.get('module_kind','')} target={mp.get('promotion_target','')}"},
    )


def _memory_to_example(rec: Dict[str, Any], bank: str) -> TrainingExample:
    """Memory bank entry (episodic / semantic / procedural).

    AUDIT FIX (P0-3, v14.1.2): fingerprint by bank + key + content head,
    not the whole record (which carries timestamps and access counts).
    AUDIT FIX (P0-4 / P2, v14.1.2): envelope moved to metadata.
    """
    bv = rec.get("bilateral_confidence", {}) or rec.get("confidence", {}) or {}
    t = float(bv.get("t", bv.get("truth", 0.6)) if isinstance(bv, dict) else 0.6)
    f = float(bv.get("f", bv.get("falsity", 0.1)) if isinstance(bv, dict) else 0.1)
    body = str(rec.get('content', '') or '')[:1500]
    if rec.get("tags"):
        body = (body + f"\ntags: {rec.get('tags', [])}").strip()
    if not body:
        body = f"(memory bank={bank})"
    fp = {
        "bank": bank,
        "key": rec.get("key", ""),
        "content_head": str(rec.get("content", ""))[:300],
    }
    return TrainingExample(
        lineage_id=f"mem::{bank}::{_stable_hash(fp)}",
        kind="memory",
        text=body,
        time=float(rec.get("created_at", rec.get("time", 0.0)) or 0.0),
        outcome_label="recalled",
        bilateral_t=t,
        bilateral_f=f,
        quality_score=t - f,
        metadata={"bank": bank, "tags": rec.get("tags", []),
                  "envelope": f"memory bank={bank}"},
    )


def _mutation_to_example(rec: Dict[str, Any]) -> TrainingExample:
    """Mutation log entry.

    AUDIT FIX (P0-3, v14.1.2): fingerprint by patch_name + status + code head.
    AUDIT FIX (P0-4 / P2, v14.1.2): envelope moved to metadata.
    """
    status = str(rec.get("status", "STAGED") or "STAGED")
    text = (
        f"patch_name: {rec.get('patch_name','')}\n"
        f"rationale: {rec.get('rationale','')}\n"
        f"code: {str(rec.get('code', ''))[:2000]}"
    ).strip()
    if not text:
        text = f"(mutation status={status})"
    success = status in {"APPLIED", "PROMOTED"}
    fp = {
        "patch": rec.get("patch_name", ""),
        "status": status,
        "target": rec.get("target", ""),
        "code_head": str(rec.get("code", ""))[:400],
    }
    return TrainingExample(
        lineage_id=f"mut::{rec.get('patch_name','')}::{status}::{_stable_hash(fp)}",
        kind="mutation",
        text=text,
        time=float(rec.get("time", 0.0) or 0.0),
        outcome_label=status.lower(),
        bilateral_t=0.85 if success else 0.4,
        bilateral_f=0.05 if success else 0.4,
        quality_score=(0.85 - 0.05) if success else (0.4 - 0.4),
        metadata={"target": rec.get("target", ""), "patch_name": rec.get("patch_name", ""),
                  "envelope": f"mutation status={status} target={rec.get('target','')}"},
        provenance_chain=[f"patch::{rec.get('patch_name','')}"],
    )


def _wave_outcome_to_example(rec: Dict[str, Any], outcome_kind: str) -> TrainingExample:
    """Wave resolution / escalation entry — governance signal.

    AUDIT FIX (P0-3, v14.1.2): wave_id alone is the lineage anchor when
    present; fall back to content fingerprint otherwise.
    AUDIT FIX (P0-4 / P2, v14.1.2): envelope moved to metadata.
    """
    text = (
        f"score: {rec.get('score', 0.0)}\n"
        f"targets: {rec.get('targets', [])}\n"
        f"items: {len(rec.get('items', []))} item(s)"
    ).strip()
    if not text:
        text = f"(wave {outcome_kind} outcome={rec.get('outcome','')})"
    closed = outcome_kind == "resolution" and "closed" in str(rec.get("outcome", "")).lower()
    bv_t = 0.8 if closed else 0.4
    bv_f = 0.1 if closed else 0.5
    wid = rec.get("wave_id", "") or ""
    if wid:
        lineage = f"wave::{outcome_kind}::{wid}"
    else:
        fp = {
            "kind": outcome_kind,
            "outcome": rec.get("outcome", ""),
            "artifact_keys": sorted(rec.get("artifact_keys", []) or []),
        }
        lineage = f"wave::{outcome_kind}::{_stable_hash(fp)}"
    return TrainingExample(
        lineage_id=lineage,
        kind=f"wave_{outcome_kind}",
        text=text,
        time=float(rec.get("time", 0.0) or 0.0),
        outcome_label=str(rec.get("outcome", "") or outcome_kind),
        bilateral_t=bv_t,
        bilateral_f=bv_f,
        quality_score=bv_t - bv_f,
        metadata={"wave_id": rec.get("wave_id", ""), "score": rec.get("score", 0.0),
                  "artifact_keys": rec.get("artifact_keys", []),
                  "envelope": f"wave_{outcome_kind} id={rec.get('wave_id','')} outcome={rec.get('outcome','')}"},
        provenance_chain=[f"wave::{rec.get('wave_id','')}"],
    )


def _competence_to_example(domain: str, entry: Any) -> Optional[TrainingExample]:
    """Per-domain competence entry.

    AUDIT FIX (P0-4 / P2, v14.1.2): envelope moved to metadata.
    """
    if entry is None:
        return None
    e = entry if isinstance(entry, dict) else asdict(entry)
    measured = float(e.get("measured_mastery", 0.0) or 0.0)
    claimed = float(e.get("claimed_mastery", 0.0) or 0.0)
    text = (
        f"measured_mastery={measured:.3f} claimed_mastery={claimed:.3f} "
        f"gap={claimed - measured:.3f}"
    )
    return TrainingExample(
        lineage_id=f"comp::{domain}",
        kind="competence_entry",
        text=text,
        outcome_label="strong" if measured > 0.7 else ("weak" if measured < 0.4 else "developing"),
        bilateral_t=measured,
        bilateral_f=max(0.0, claimed - measured),  # overclaim is falsity
        quality_score=measured - max(0.0, claimed - measured),
        metadata={"domain": domain, "measured": measured, "claimed": claimed,
                  "envelope": f"competence domain={domain}"},
    )


# --- Public builders -------------------------------------------------------

def build_corpus(kernel, *, since_cycle: int = 0) -> List[TrainingExample]:
    """Walk a live kernel and produce a list of TrainingExamples.

    Cycle-since filtering: examples produced in `kernel.improvement_count`
    cycles before `since_cycle` are dropped. Set since_cycle=0 to take all.

    Each per-source block logs and continues on failure so a single bad
    record cannot drop the whole corpus. Drops are visible in DEBUG logs.
    """
    import logging
    examples: List[TrainingExample] = []

    # ExperienceStore.
    try:
        for rec in kernel.experience_store.records:
            try:
                if isinstance(rec, dict):
                    rd = rec
                else:
                    rd = asdict(rec) if hasattr(rec, "__dataclass_fields__") else dict(rec.__dict__)
                examples.append(_experience_to_example(rd))
            except Exception as e:
                logging.debug(f"corpus_builder: skip experience record: {e}")
    except Exception as e:
        logging.debug(f"corpus_builder: experience store walk failed: {e}")

    # Kernel packet log.
    try:
        for pkt in getattr(kernel, "kernel_packet_log", []) or []:
            try:
                ex = _packet_to_example(pkt)
                if ex is not None:
                    examples.append(ex)
            except Exception as e:
                logging.debug(f"corpus_builder: skip packet: {e}")
    except Exception as e:
        logging.debug(f"corpus_builder: packet log walk failed: {e}")

    # Promotion ladder gate decisions, per patch.
    try:
        gate_log = getattr(kernel.promotion_ladder, "gate_log", None) or {}
        if isinstance(gate_log, dict):
            for patch_name, decisions in gate_log.items():
                for dec in decisions or []:
                    try:
                        examples.append(_gate_decision_to_example(dec, patch_name))
                    except Exception as e:
                        logging.debug(f"corpus_builder: skip gate decision: {e}")
    except Exception as e:
        logging.debug(f"corpus_builder: gate log walk failed: {e}")

    # Module proposals.
    try:
        for mp in getattr(kernel, "module_proposals", []) or []:
            try:
                examples.append(_module_proposal_to_example(mp))
            except Exception as e:
                logging.debug(f"corpus_builder: skip module proposal: {e}")
    except Exception as e:
        logging.debug(f"corpus_builder: module proposals walk failed: {e}")

    # Memory banks.
    try:
        for bank in ("episodic", "semantic", "procedural"):
            try:
                for rec in kernel.memory_store.get_bank(bank):
                    try:
                        rd = rec if isinstance(rec, dict) else asdict(rec)
                        examples.append(_memory_to_example(rd, bank))
                    except Exception as e:
                        logging.debug(f"corpus_builder: skip memory record in {bank}: {e}")
            except Exception as e:
                logging.debug(f"corpus_builder: skip bank {bank}: {e}")
    except Exception as e:
        logging.debug(f"corpus_builder: memory walk failed: {e}")

    # Wave histories — both resolution and escalation.
    try:
        if kernel.hub_kernel is not None:
            local = kernel.hub_kernel.local_branch_state or {}
            for rec in local.get("wave_resolution_history", []) or []:
                try:
                    examples.append(_wave_outcome_to_example(rec, "resolution"))
                except Exception as e:
                    logging.debug(f"corpus_builder: skip wave resolution: {e}")
            for rec in local.get("wave_escalation_history", []) or []:
                try:
                    examples.append(_wave_outcome_to_example(rec, "escalation"))
                except Exception as e:
                    logging.debug(f"corpus_builder: skip wave escalation: {e}")
    except Exception as e:
        logging.debug(f"corpus_builder: wave history walk failed: {e}")

    # Competence map.
    try:
        cm = getattr(kernel, "competence_map", None)
        if cm is not None and hasattr(cm, "entries"):
            for domain, entry in cm.entries.items():
                try:
                    ex = _competence_to_example(domain, entry)
                    if ex is not None:
                        examples.append(ex)
                except Exception as e:
                    logging.debug(f"corpus_builder: skip competence {domain}: {e}")
    except Exception as e:
        logging.debug(f"corpus_builder: competence walk failed: {e}")

    # Cycle filter.
    if since_cycle > 0:
        improvement = int(getattr(kernel, "improvement_count", 0) or 0)
        # Approximate: keep examples whose time is recent enough that they
        # likely belong to cycles >= since_cycle. Without per-cycle stamping
        # we use a conservative cutoff at the kernel's last_research_time.
        cutoff = float(getattr(kernel, "last_research_time", 0.0) or 0.0)
        if cutoff > 0:
            examples = [e for e in examples if e.time >= cutoff or e.time == 0.0]

    return examples


def build_corpus_from_state_files(state_dir: str | Path,
                                  *, since_cycle: int = 0) -> List[TrainingExample]:
    """Build a corpus from on-disk JSON state files (no live kernel).

    Reads:
      tovah_state.json           (experience records, memory banks, etc.)
      tovah_kernel_ecology.json  (packet log, proposals, hub state)
      tovah_packet_log.json      (separate packet log mirror)
      tovah_mutations.py         (mutation log; lightweight parse)
    """
    state_dir = Path(state_dir)
    examples: List[TrainingExample] = []

    def _safe_load(path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # tovah_state.json — experience, memory, gate state.
    state = _safe_load(state_dir / "tovah_state.json") or {}
    for rec in state.get("experience_records", []) or []:
        examples.append(_experience_to_example(rec))
    for bank in ("episodic", "semantic", "procedural"):
        for rec in state.get(f"memory_{bank}", []) or []:
            examples.append(_memory_to_example(rec, bank))
    # Promotion gate_log inside state's promotion_state if present.
    promo = state.get("promotion_state", {}) or {}
    gate_log = promo.get("gate_log", {}) or {}
    if isinstance(gate_log, dict):
        for patch_name, decisions in gate_log.items():
            for dec in decisions or []:
                examples.append(_gate_decision_to_example(dec, patch_name))

    # tovah_kernel_ecology.json — packets, proposals, wave histories.
    eco = _safe_load(state_dir / "tovah_kernel_ecology.json") or {}
    for pkt in eco.get("packet_log", []) or []:
        ex = _packet_to_example(pkt)
        if ex is not None:
            examples.append(ex)
    for mp in eco.get("module_proposals", []) or []:
        examples.append(_module_proposal_to_example(mp))
    hub_state = eco.get("hub_state", {}) or {}
    local_branch = hub_state.get("local_branch_state", {}) or {}
    for rec in local_branch.get("wave_resolution_history", []) or []:
        examples.append(_wave_outcome_to_example(rec, "resolution"))
    for rec in local_branch.get("wave_escalation_history", []) or []:
        examples.append(_wave_outcome_to_example(rec, "escalation"))

    # tovah_packet_log.json — additional packets.
    pkt_log = _safe_load(state_dir / "tovah_packet_log.json")
    if isinstance(pkt_log, list):
        for pkt in pkt_log:
            ex = _packet_to_example(pkt)
            if ex is not None:
                examples.append(ex)

    return examples
