"""
TOVAH v14 mutation — Patch analysis, staging, quarantine, promotion ladder.

HARD RULE: The promotion ladder is the ONLY path to live deployment.
No direct apply_staged_patch from autonomous code.

Owns:
- analyze_patch_code (static safety analysis)
- Contract-based patch validation
- Staging
- Quarantine
- Promotion ladder
- Mutation log
- Revert / rollback

Imports from config/, core/, invariants/. MUST NOT import from kernel/.
"""
from tovah_v14.mutation.analysis import analyze_patch_code, PatchDescriptor
from tovah_v14.mutation.staging import stage_patch, StagingResult
from tovah_v14.mutation.quarantine import quarantine_patch, QuarantineRecord
from tovah_v14.mutation.promotion_ladder import PromotionLadder
from tovah_v14.mutation.mutation_log import MutationLogger
