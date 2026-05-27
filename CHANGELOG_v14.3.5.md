# TOVAH v14.3.5 — Verifier, Locality, and Bilateral-Compute Hardening

This update implements the beneficial pieces from the v14.3.4 audits that stay
faithful to the UAP/ShadowHoTT math.

## Implemented

- **Muon fixed properly**: Nesterov look-ahead, shape-aware matrix LR scaling,
  and default `ns_steps=3`.
- **Metadata-faithful K/G losses**: when per-example bilateral labels exist,
  the global K/G prior is disabled and targets come from metadata. Classical
  records are no longer pushed toward artificial K≈0.12/G≈0.20.
- **Bilateral negation initialization**: `ScalableBilateralCore` initializes
  `embed_F = -embed_T`, giving the F stream a principled apophatic complement
  at step zero.
- **Optional Belnap MoE FFN**: `ffn_kind="belnap_moe"` activates top-k routed
  A/B/K/G-aware bilateral experts.
- **FormalHoTT RL scaffold upgraded**: S-expression grammar, parser, verifier
  rewards, task curriculum, GRPO advantages, and policy loss helper.
- **Differentiable sheaf-obstruction auxiliary**: optional
  `--uap-sheaf-weight` adds local-to-global gluing pressure without collapsing
  genuine K/G metadata.
- **Predictive-coding helpers**: bilateral free-energy objectives for future
  local/asynchronous training.
- **Bilateral DoRA adapter**: pure-PyTorch directional T/F low-rank adapter plus
  PEFT QLoRA/DoRA builder retained.
- **Bilateral SSM prototype**: linear-time T/F state propagation module for
  long-context experiments.
- **Bilateral univalence scaffold** and **non-abelian Čech twist diagnostics**.

## Not overclaimed

- The FormalHoTT loop is now a real verifier-reward scaffold, not a complete
  PPO/GRPO trainer wired to a language model.
- Belnap MoE and BilateralSSM are optional experimental paths, not the default
  production trunk.
- Predictive coding is exposed as a local objective, not a replacement for all
  backpropagation.
