"""
TOVAH v14 tasks/cleanup.py — Task and plan cleanup.

Handles:
- Orphaned tasks (parent completed/failed but subtask still pending)
- Stale tasks past deadline
- Old completed tasks purged from queue
"""
from __future__ import annotations

import time
from typing import Dict, List

from tovah_v14.tasks.queue import TaskQueue


def cleanup_tasks(queue: TaskQueue, max_completed_age: float = 600.0) -> Dict[str, int]:
    """Clean up the task queue.

    - Fail orphaned tasks whose parents are gone
    - Remove old completed/failed/stale tasks

    Returns {action: count}.
    """
    counts = {"orphaned": 0, "purged": 0}
    now = time.time()

    all_ids = {t.task_id for t in queue.tasks}

    for task in queue.tasks:
        # Orphan check: parent exists but is not in queue or completed
        if task.parent_id and task.status in ("pending", "active"):
            parent_exists = task.parent_id in all_ids or task.parent_id in queue.completed_ids
            if not parent_exists:
                task.status = "failed"
                task.result = {"error": "orphaned: parent gone"}
                counts["orphaned"] += 1

    # Purge old terminal tasks
    keep = []
    for task in queue.tasks:
        if task.status in ("completed", "failed", "stale"):
            if now - task.created_at > max_completed_age:
                counts["purged"] += 1
                continue
        keep.append(task)
    queue.tasks = keep

    return counts
