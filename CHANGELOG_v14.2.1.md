# TOVAH v14.2.1 — HoTT Frontier Fix Pack

This maintenance release applies the requested fixes from the v14.2.0 audit.

## Applied fixes

1. **RoPE axis correctness** — `BilateralGQAttention` now shapes Q/K/V as `(B, heads, L, head_dim)` before applying RoPE, so rotation is by token position rather than head index. Added guardrails for invalid GQA/RoPE configurations and sequence truncation in `ScalableBilateralCore.forward()`.
2. **HoTT promotion fails closed** — HoTT certificate provider/certification exceptions now block `regression_passed → shadow_deployed` instead of proceeding policy-only.
3. **Frontier pretrain wiring** — `training.pretrain()` can now build `ScalableBilateralCore` + `AdamWWrapper` for `frontier_*` profiles, and `run_tovah.py` exposes a `--pretrain` CLI with `--profile`, `--use-fsdp`, `--use-ddp`, and optimizer/scaling flags.
5. **Module substitutability** — capability-only identical contracts now receive aggregate support and substitute correctly when no hard requirements are missing or violated.
6. **Memory identity** — conflict classification now understands live `BilateralValue` instances in addition to dict-shaped bilateral confidence records.
7. **Obstruction trivialization** — `is_trivializable()` now traverses the transition graph, so chain-shaped connected covers are recovered instead of falsely rejected by pivot-star assumptions.
8. **Transport target validation** — `transport()` now rejects coerced values that do not inhabit the target fiber.
9. **Version consistency** — runtime/package/certificate/module/self-model versions now report `14.2.1`.
10. **Regression tests strengthened** — added or tightened tests for RoPE axis semantics, fail-closed HoTT promotion, frontier builder wiring, module substitutability, memory `BilateralValue` conflicts, graph trivialization, target-fiber validation, and version expectations.

## Verification performed in this patch pass

- `python -m compileall -q tovah_v14` passes.
- All non-test modules import successfully.
- Pytest collection finds 464 tests.
- 457 tests were executed and passed in targeted groups.
- Full pytest still exceeds the sandbox time budget because seven long-running research/autonomous tests call kernel research/autonomy paths. Those seven were not changed by this fix pack.
