# EVALS.md — TOVAH v14.2.6

The buyer-facing eval harness lives in `tovah_v14/evals/`.

Run all lightweight evals:

```bash
PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}" python -m tovah_v14.evals.run_all
```

Included evals:

- `smoke_language_modeling`: tiny scalable model forward/loss check.
- `semantic_consistency`: lane routing gradients for A/B/C behavior.
- `high_glut_preservation`: Lane B rewards preserved contradiction.
- `gap_tolerance`: Lane C rewards preserved underdetermination.
- `patch_certification_eval`: invariant-breaking patch blocks.
- `memory_conflict_eval`: contradiction counts only under same referent.

These evals are smoke/semantic checks, not public benchmark claims. A buyer
should add standard LM benchmarks after training a nontrivial checkpoint.
