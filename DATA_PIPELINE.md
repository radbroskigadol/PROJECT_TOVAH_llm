# DATA_PIPELINE.md — TOVAH v14.2.6

## Expected shard format

The pretraining path consumes JSONL shards. Each line should include at least:

```json
{"text":"...", "bilateral_t":0.9, "bilateral_f":0.1, "class":"A", "kind":"..."}
```

Metadata is optional for legacy text records, but scale runs should keep it.

## Shard recommendations

- Keep shard files immutable once training begins.
- Store a manifest containing file path, size, checksum, record count, and class mix.
- Deduplicate before tokenization.
- Preserve A/B/K/G class balance reports.
- Record the exact tokenizer artifact used to create each tokenized shard.

## Current status

TOVAH has JSONL corpus loading, metadata-aware collate functions, dedup helpers,
and manifest utilities. v14.2.6 adds scale handoff docs/scripts; an industrial
buyer may still replace the loader with WebDataset, Arrow, Parquet, or a custom
streaming stack.
