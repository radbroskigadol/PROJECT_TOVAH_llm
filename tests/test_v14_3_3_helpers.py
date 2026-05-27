from training.loop_stability import score_loop_stability
from training.shadow_profile_objectives import infer_profile_from_record, PROFILE_FIELDS, support_profile_consistency_score


def test_loop_stability_penalizes_degenerate_repetition():
    good = score_loop_stability("The claim is supported under one assumption and refuted under another; keep both supports pending separation.")
    bad = score_loop_stability("falsity support and falsity support and falsity support and falsity support and falsity support")
    assert 0.0 <= good.loop_drift_behavior <= 1.0
    assert 0.0 <= bad.loop_drift_behavior <= 1.0
    assert bad.loop_drift_behavior < good.loop_drift_behavior


def test_profile_inference_from_bilateral_glut():
    profile = infer_profile_from_record({
        "text": "A supports and not-A refutes.",
        "bilateral_t": 0.85,
        "bilateral_f": 0.80,
        "kind": "bilateral_glut",
    })
    assert len(profile.as_vector()) == len(PROFILE_FIELDS)
    assert profile.T_support == 0.85
    assert profile.F_support == 0.80
    assert profile.glut_mass >= 0.75
    assert profile.obstruction_residue > 0.0


def test_profile_consistency_identity_is_one():
    profile = infer_profile_from_record({"bilateral_t": 0.2, "bilateral_f": 0.2, "kind": "bilateral_gap"})
    assert support_profile_consistency_score(profile, profile) == 1.0
