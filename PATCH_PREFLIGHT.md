# TOVAH v14 — Patch Preflight

Single authoritative validation: validate_patch_preflight().
Both stage_patch() and direct_inject_method() route through it.

## Create-new policy
- Being in EXTENSION_TARGETS does NOT authorize creation.
- create_new=True must be EXPLICITLY passed.
- absent target + create_new=False -> FAIL (even for EXTENSION_TARGETS members)
- absent target + create_new=True + target in EXTENSION_TARGETS -> allowed

## Extension target contracts (Audit S-5)
The four EXTENSION_TARGETS now have minimal MethodContracts in
core/contracts.py::CONTRACT_REGISTRY:
  - _extract_pdf_text_local
  - _summarize_pdf_text_local
  - _tool_use_desire
  - _score_local_results

Their contracts are deliberately light — they only enforce signature shape
and forbid the common kernel-internal misuse patterns. Create-new
injections at these slots are still gated by ALLOWED_PATCH_TARGETS +
analyze_patch_with_contract.

## Deferred commands (Audit S-6)
- INGEST_LEVBEL is registered in COMMAND_REGISTRY with status="deferred".
  The /levbel/ source directory exists but its content isn't migrated yet.
  Until levbel content lands the kernel returns a "deferred — not migrated"
  reply for any INGEST_LEVBEL command. This is intentional, not a bug.
