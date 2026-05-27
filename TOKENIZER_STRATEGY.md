# TOKENIZER_STRATEGY.md — TOVAH v14.2.6

## Default

The research/debug path can use byte-level tokenization. This is simple and
faithful to arbitrary corpus text.

## Frontier path

For large models, use BPE/Unigram tokenizer infrastructure through the existing
`training/tokenizer.py` abstraction. The model vocabulary must match the
loaded tokenizer's vocabulary size; `pretrain()` enforces this.

## Buyer checklist

- Decide byte vs BPE before frontier-scale allocation.
- Save tokenizer artifact with every checkpoint.
- Record vocab size in checkpoint metadata.
- Preserve `bilateral_t`, `bilateral_f`, and `class` metadata in tokenized shards.
- Avoid changing vocab mid-run without a new checkpoint namespace.

## Memory implication

Large vocabulary increases the mandatory `T_logits` tensor and the embedding
matrices. Hidden semantic heads avoid a second full-vocabulary `F_logits` tensor
for K/G auxiliaries in frontier mode.
