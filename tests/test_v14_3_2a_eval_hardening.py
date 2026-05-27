import json
from pathlib import Path

from tovah_v14.neural.checkpointing import save_training_checkpoint
from tovah_v14.neural.shadow_model import ShadowTokenCore
from tovah_v14.tools.generate_uap_shadow_corpus import generate
import tovah_v14.training.eval as eval_module
from tovah_v14.training.eval import main as eval_main, run_full_eval
from tovah_v14.training.shadow_depth_eval import evaluate_paths
from tovah_v14.training.tokenizer import ByteTokenizer


def test_shadow_depth_reports_label_provenance_warning(tmp_path: Path):
    corpus = tmp_path / "corpus"
    generate(corpus, n=8, shard_size=4, seed=4321)
    result = evaluate_paths([corpus])
    provenance = result["metric_provenance"]
    assert provenance["text_fallback_ratio"] > 0.5
    assert provenance["label_derived_when_text_fallback"] is True
    assert provenance["warnings"]
    assert result["schema_version"] == "tovah-shadow-depth-eval-v14.3.2a"


def test_run_full_eval_includes_disabled_model_shadow_depth(tmp_path: Path):
    corpus = tmp_path / "corpus"
    generate(corpus, n=8, shard_size=4, seed=123)
    tok = ByteTokenizer()
    model = ShadowTokenCore(vocab_size=tok.vocab_size, **{"d_model": 32, "d_hidden": 64, "n_heads": 4, "n_blocks": 1, "max_len": 96})
    result = run_full_eval(
        model,
        corpus,
        tokenizer=tok,
        max_examples_ppl=2,
        max_examples_acc=2,
        max_examples_calib=2,
        n_gen_samples=0,
        max_examples_shadow_model=0,
    )
    assert result["model_shadow_depth"]["enabled"] is False
    assert "metric_provenance" in result["shadow_depth"]


def test_eval_cli_writes_json(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(eval_module.MODEL_PROFILES, "tiny_eval", {"d_model": 32, "d_hidden": 64, "n_heads": 4, "n_blocks": 1, "max_len": 96})
    corpus = tmp_path / "corpus"
    generate(corpus, n=8, shard_size=4, seed=99)
    tok = ByteTokenizer()
    model = ShadowTokenCore(vocab_size=tok.vocab_size, **{"d_model": 32, "d_hidden": 64, "n_heads": 4, "n_blocks": 1, "max_len": 96})
    ckpt = tmp_path / "debug.pt"
    save_training_checkpoint(
        ckpt,
        model,
        step=0,
        epoch=0,
        metadata={"profile_name": "tiny_eval", "tokenizer": tok.info()},
    )
    out = tmp_path / "eval.json"
    rc = eval_main([
        "--checkpoint", str(ckpt),
        "--shard-dir", str(corpus),
        "--tokenizer", "byte",
        "--profile", "tiny_eval",
        "--out", str(out),
        "--max-examples-ppl", "2",
        "--max-examples-acc", "2",
        "--max-examples-calib", "2",
        "--n-gen-samples", "0",
    ])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["eval_schema_version"] == "tovah-full-eval-v14.3.2a"
    assert data["model_shadow_depth"]["enabled"] is False
    assert data["shadow_depth"]["schema_version"] == "tovah-shadow-depth-eval-v14.3.2a"
