# Commercial Sale Audit — TOVAH v14.3.2a

## Verdict

TOVAH v14.3.2a is a serious technical IP prototype and research-transfer candidate, not yet a turnkey commercial product. The strongest current asset is the distinctive architecture: a ShadowHoTT/UAP training kernel that treats AdamW-style next-token prediction as the classical projection of a richer paraconsistent token ontology.

For a commercial sale, it should be positioned as:

> a differentiated research codebase and mathematical/architectural IP package for paraconsistent token ontology, ShadowOptimizer experiments, UAP token-profile corpus generation, and frontier-model/tool-bridge adaptation.

It should not yet be positioned as:

> a production-proven foundation-model trainer, an externally validated optimizer replacement, or a fully hardened autonomous agent runtime.

## Strong assets

1. **Distinctive ontology**
   - Tokens are treated as local manifestation loci carrying truth-support, falsity-support, glut mass, gap mass, obstruction residue, collapse pressure, and classicalization depth.
   - AdamW is framed correctly as the classicalized floor/projection, not the thing ShadowOptimizer is trying to beat.

2. **Working training path**
   - v14.3.1 repaired the UAP ShadowOptimizer scale problem with size-invariant RMS update control.
   - v14.3.2 added UAP token-profile labels, Shadow-depth metrics, and auxiliary training scaffolding.
   - v14.3.2a adds operational eval CLI hardening.

3. **Evaluation architecture now has provenance controls**
   - v14.3.2a warns when Shadow-depth scores are label-derived from source text.
   - Optional model-generated probing is now available so buyers can separate corpus consistency from actual model behavior.

4. **Commercially legible documentation**
   - The package already includes architecture, buyer summary, safety, runbooks, scale notes, and changelogs.
   - This version adds a sale-specific audit and clearer eval commands.

5. **Offline and license-clean synthetic data path**
   - Corpus generation is synthetic and does not depend on scraping copyrighted corpora.

## Major sale risks

1. **Scientific validation risk**
   - The synthetic paradox corpus is still template-regular.
   - Perfect Shadow-depth metrics on label-derived validation are not proof of learned ShadowHoTT geometry.
   - External validation against adversarial paraphrases, held-out paradox families, and real downstream tasks remains necessary.

2. **Model-head/probe risk**
   - UAP profile targets and auxiliary losses exist, but there is not yet a mature learned UAP profile head with independently audited prediction error.
   - Current calibration between model entropy and truth/falsity support is weak enough that claims should remain conservative.

3. **Security/sandbox risk**
   - The codebase contains dynamic execution paths used for patching, sandboxing, and operator commands.
   - These paths appear intentional, but they require third-party security review before product deployment.
   - Buyer-facing claims should describe these as research-agent capabilities requiring deployment hardening.

4. **Operational maturity risk**
   - The codebase is broad and research-heavy.
   - Some full-suite tests are slow in CPU-only sandbox conditions.
   - The sale package should include a curated smoke-test suite and a separate long-running test matrix.

5. **Integration risk**
   - Frontier-model, Lean, and real production tool integrations are still handoff/adaptation surfaces, not fully packaged SaaS integrations.

## What is commercially saleable now

- Mathematical/architectural IP around UAP/ShadowHoTT token ontology.
- ShadowOptimizer implementation history and scale-fix trajectory.
- UAP token-profile corpus generation and evaluation scaffolding.
- A research kernel for paraconsistent training/evaluation experiments.
- A buyer-ready prototype for an AI lab, proof-tooling group, or frontier-model research team to harden and integrate.

## What is not yet sale-ready

- Claims that ShadowOptimizer generally outperforms AdamW.
- Claims that the model has independently learned robust ShadowHoTT geometry.
- Claims of production-grade sandbox security.
- Claims of completed Lean/proof-assistant integration.
- Claims of broad real-world benchmark superiority.

## Recommended commercial positioning

Use this phrasing:

> TOVAH is a paraconsistent token-ontology training kernel. It treats ordinary next-token prediction as the classicalized projection of richer UAP token profiles carrying bilateral truth/falsity support, gap, glut, obstruction residue, collapse pressure, and classicalization depth. v14.3.2a provides a working ShadowOptimizer path, UAP profile corpus generation, Shadow-depth evaluation, provenance-aware metrics, and optional model-generated probes for measuring whether the model preserves paraconsistent structure rather than merely memorizing paradox-flavored prose.

Avoid this phrasing:

> TOVAH beats AdamW.

Avoid this phrasing:

> The system has proven learned ShadowHoTT geometry.

## Minimum next hardening before buyer demo

1. Run the v14.3.2a CLI eval with `--max-examples-shadow-model 16` or `32` and report model-generated Shadow-depth separately.
2. Generate the dedicated UAP Shadow corpus using `tools/generate_uap_shadow_corpus.py`, not only the broader paradox smoke corpus.
3. Add a small learned UAP-profile probe head and report profile-prediction MSE/MAE.
4. Freeze a curated demo command sequence that runs in under 10 minutes on CPU.
5. Prepare a security appendix that explicitly quarantines dynamic-exec/patch-promotion features as research-only until audited.
6. Add CI tiers:
   - `smoke`: compile + critical unit tests
   - `eval`: corpus/eval CLI tests
   - `slow`: training and autonomy tests
   - `security`: sandbox/patch-policy tests

## Sale readiness grade

- **Research IP sale readiness:** B+
- **Prototype demo readiness:** B
- **Enterprise production readiness:** C+
- **Scientific claim readiness:** C+ until model-generated/adversarial validation improves
- **Security readiness:** C until dynamic execution surfaces receive external review

## Bottom line

The codebase has real differentiated value, but the most credible sale strategy is a technical/IP acquisition or research-license sale, not a claim that this is already a finished commercial training platform. v14.3.2a materially improves buyer trust because it stops overselling perfect label-derived Shadow-depth metrics and makes evaluation reproducible from a real CLI.
