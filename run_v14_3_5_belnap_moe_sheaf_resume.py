import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

pt = importlib.import_module("tovah_v14.training.pretrain")

_orig_make_scalable_model = pt.make_scalable_model

def _make_belnap_moe_model(profile_name, **kwargs):
    kwargs["ffn_kind"] = "belnap_moe"
    kwargs["n_experts"] = 4
    kwargs["moe_top_k"] = 2
    model = _orig_make_scalable_model(profile_name, **kwargs)

    moe_keys = [
        k for k in model.state_dict().keys()
        if "expert" in k.lower() or "experts" in k.lower() or "router" in k.lower()
    ]
    print("BELNAP_MOE_KEYS:", len(moe_keys))
    print("PARAM_COUNT:", sum(p.numel() for p in model.parameters()))

    if not moe_keys:
        raise RuntimeError("Belnap MoE did not activate.")

    return model

pt.make_scalable_model = _make_belnap_moe_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--metrics-path", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--profile", default="frontier_dev")
    ap.add_argument("--sheaf-weight", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    args = ap.parse_args()

    summary = pt.pretrain(
        shard_dir=r".\tovah_corpus\paradox_real_mixed_v14_3_2a",
        profile_name=args.profile,
        optimizer_kind="muon",
        tokenizer_spec="auto-bpe",
        train_bpe_if_missing=True,
        bpe_save_path=r".\tokenizer_real_mixed.json",
        epochs=1,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        device=args.device,
        dtype="fp32",
        bilateral_mode="shared",
        gradient_checkpointing=True,
        frontier_semantic_mode="hidden",
        uap_aux_weight=0.05,
        uap_sheaf_weight=args.sheaf_weight,
        resume_from=args.resume_from,
        save_path=args.save_path,
        metrics_path=args.metrics_path,
        log_every=10,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
