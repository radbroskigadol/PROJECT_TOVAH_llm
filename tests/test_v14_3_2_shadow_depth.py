from pathlib import Path
import json
import tempfile

from tovah_v14.tools.uap_shadow_profiles import label_example, profile_from_text, score_generation_against_profile
from tovah_v14.tools.generate_uap_shadow_corpus import generate
from tovah_v14.training.shadow_depth_eval import evaluate_paths


def test_glut_profile_has_bilateral_mass():
    p = profile_from_text("The claim is supported and refuted; preserve both truth-support and falsity-support without collapse.")
    assert p.t_support > 0
    assert p.f_support > 0
    assert p.glut_mass > 0.1
    assert p.collapse_pressure < 0.6


def test_gap_profile_marks_missing_evidence():
    p = profile_from_text("The claim is underdetermined: neither accepted nor rejected, with missing evidence.")
    assert p.gap_mass > 0.2
    assert p.classicalization_depth < 1.0


def test_generation_score_rewards_residue_preservation():
    rec = label_example("Local charts agree but global gluing fails; preserve obstruction residue.", family="local_global_obstruction")
    score = score_generation_against_profile(rec["uap_profile"], "The local evidence remains, but global gluing is obstructed and residue is retained.")
    assert score.local_global_obstruction_preservation > 0.5
    assert score.shadow_depth_mean > 0.5


def test_corpus_and_eval_smoke():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "corpus"
        manifest = generate(out, n=32, shard_size=16, seed=1432)
        assert manifest["n_train"] == 32
        result = evaluate_paths([out])
        assert result["n_records"] >= 32
        assert "shadow_depth_mean" in result["overall"]
