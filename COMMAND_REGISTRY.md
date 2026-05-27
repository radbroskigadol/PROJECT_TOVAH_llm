# TOVAH v14 — Command Registry (v16 ecology upgrade path)

67 active + 1 deferred = 68 total.

Key: `INJECT_METHOD` requires explicit `create_new` for absent targets.
`REGRESSION` uses the bounded tier. `SELF_MODEL` returns the richer structured model.
`DISCOVER_SERVICES` remains active with ranked candidates.

| Command | Purpose |
|---|---|
| `DELEGATE_TASK:<goal|specialization|mission>` | delegate governed work to a subkernel |
| `DELEGATION_STATUS` | show active delegation leases |
| `MODULE_PROPOSALS` | inspect governed module proposal packets |
| `MODULE_REGISTRY` | inspect governed module registry summary |
| `MODULE_BUS` | inspect module bus routes and recent traffic |
| `RESOURCE_REQUESTS` | inspect resource request packets |
| `TOOL_REQUESTS` | inspect tool request packets |

| `TOOL_ACCESS_DECISIONS` | inspect worker-role tool access decisions |
| `MODULE_POLICY` | inspect node-aware module promotion policy decisions |
| `PROMOTION_GATES` | inspect trust-weighted promotion gate decisions |
| `WORKER_POLICIES` | inspect worker role policy profiles |
| `WORKER_BUDGETS` | inspect worker-role budget and lease gate decisions |
| `MEMORY_SYNC_REQUESTS` | inspect memory-sync packets |
| `PROMOTION_REQUESTS` | inspect promotion-request packets and evidence requests |
| `HUB_STATUS` | show sovereign + hub + subkernel ecology summary |
| `HUB_REVERT` | revert hub to its last rollback point |
| `SPAWN_SUBKERNEL[:specialization|mission]` | create a governed specialized subkernel |
| `LIST_SUBKERNELS` | list registered subkernels |
| `KERNEL_PACKET_LOG` | inspect recent packet traffic |
| `MEMORY_PROVENANCE` | inspect the branch-memory provenance graph |
| `SAVE_BRANCH_CHECKPOINT` | persist an ecology checkpoint to branch storage |
| `LIST_BRANCH_CHECKPOINTS` | list saved ecology branch checkpoints |
| `CLUSTER_STATUS` | inspect cluster registry and node topology summary |
| `NODE_TRUST` | inspect cluster trust ledger summary |

- `DISTRIBUTED_QUEUE` — cluster-aware delegation queue summary
- `CLUSTER_DELEGATIONS` — delegation lease and routing summary
- `NODE_IDENTITY` — local/distributed node identity summary
- `MODULE_PRIORITIES` — show governed module proposal priorities using maturity/cooldown/reliability.

- `HUB_PROMOTION_PRIORITIES` — show ranked hub promotion queue items using maturity/cooldown/recent operating history.

- `HUB_REVIEW_QUEUE` — show consumed/ranked hub promotion review work items.
- `PROCESS_HUB_PROMOTION_QUEUE[:N]` — rank and consume the top N hub promotion queue items into governed review work.

- GROWTH_PRIORITIES — show self-model growth-priority summary

- WAKE_HUB_DEFERRED
- COMPLETE_EVIDENCE_TASK:<kind|name|proposal_id>
- PROCESS_GROWTH_PRIORITIES[:N]
- REVIEW_WAVES
- COMPLETE_HUB_REVIEW_WAVE:<wave_id>

- WAVE_PRIORITIES — show ranked open review waves awaiting resolution.
- SURFACE_OPEN_REVIEW_WAVES[:N] — surface neglected open review waves into hub work for resolution.
- PROCESS_WAVE_RESOLUTIONS[:N]
- WAVE_RESOLUTION_HISTORY
- PROCESS_WAVE_ESCALATIONS[:N]
- WAVE_ESCALATION_HISTORY

- PROCESS_PROPOSAL_REWORK[:N]
- PROCESS_BLOCKED_GROWTH[:N]
- PROPOSAL_REWORK_HISTORY
- BLOCKED_GROWTH_FOLLOWUP_HISTORY
