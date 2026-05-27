# v14.3.5 Next Update Notes

The update chooses the mathematically aligned ideas from the audits:

1. Keep UAP/ShadowHoTT semantics at the proposition/position level, not as a
   benchmark against AdamW.
2. Make verifier-grounded formal tasks executable through FormalHoTT rewards.
3. Use local-to-global sheaf obstruction as an optional differentiable signal.
4. Move toward brain-like compute through locality, sparse routing, and
   predictive coding, but keep defaults conservative.

## Suggested command

```powershell
python .\run_tovah.py --pretrain `
  --profile heavy `
  --optimizer shadow `
  --frontier-semantic-mode auto `
  --uap-aux-weight 0.05 `
  --uap-sheaf-weight 0.01
```

For frontier experiments:

```python
from tovah_v14.neural.scaling import make_scalable_model
model = make_scalable_model("frontier_dev", ffn_kind="belnap_moe")
```

For FormalHoTT verifier reward smoke:

```python
from tovah_v14.training.formal_hott_rl import smoke_score_suite
print(smoke_score_suite())
```
