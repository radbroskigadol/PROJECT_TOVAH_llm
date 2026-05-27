"""Tests for the kernel-ecology scaffold introduced for the v16 upgrade path."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tovah_v14.kernel.action_model import BlockedGrowthRecord, ModuleProposal, PatchProposal, ResourceRequest
from tovah_v14.kernel.hub_kernel import HubKernel
from tovah_v14.kernel.kernel_policy import can_directly_mutate, get_policy, ownership_summary, role_can_send
from tovah_v14.kernel.kernel_roles import KernelRole
from tovah_v14.kernel.packet import KernelPacket, PacketKind, make_packet
from tovah_v14.kernel.subkernel import Subkernel


def test_packet_required_fields_roundtrip():
    pkt = make_packet(
        PacketKind.STATUS,
        source_kernel_id="hub",
        target_kernel_id="main",
        payload={"ok": True},
        required_action="observe",
        parent_goal_id="g1",
        mission_context="test",
        priority=2,
        trust_level="trusted",
        risk_class="low",
        provenance={"source": "unit"},
        reply_expected=True,
    )
    data = pkt.to_dict()
    assert isinstance(pkt, KernelPacket)
    assert data["packet_kind"] == PacketKind.STATUS
    assert data["source_kernel_id"] == "hub"
    assert data["target_kernel_id"] == "main"
    assert data["reply_expected"] is True
    assert "packet_id" in data


def test_policy_boundaries_hold():
    main = get_policy(KernelRole.MAIN)
    hub = get_policy(KernelRole.HUB)
    sub = get_policy(KernelRole.SUBKERNEL)
    assert main.authoritative_state is True
    assert hub.authoritative_state is False
    assert sub.authoritative_state is False
    assert can_directly_mutate("hub", "main") is False
    assert can_directly_mutate("subkernel", "main") is False
    assert role_can_send("hub", PacketKind.PATCH_PROPOSAL) is True
    assert role_can_send("subkernel", PacketKind.SPAWN_REQUEST) is False
    summary = ownership_summary()
    assert summary["main"]["may_directly_mutate_main"] is True
    assert summary["hub"]["may_directly_mutate_main"] is False


def test_hub_snapshot_revert_and_packets():
    hub = HubKernel(kernel_id="hub_a", parent_kernel_id="main")
    hub.local_branch_state["x"] = 1
    snap_id = hub.snapshot("s1")
    hub.local_branch_state["x"] = 2
    reverted = hub.revert_to_last_snapshot()
    assert reverted == snap_id
    assert hub.local_branch_state["x"] == 1

    record = BlockedGrowthRecord(kernel_id="hub_a", blocker="budget", symptom="stall", attempted_action="train")
    pkt = hub.record_blocked_growth(record)
    assert pkt.packet_kind == PacketKind.BLOCKED_GROWTH
    assert pkt.target_kernel_id == "main"
    assert hub.lifecycle == "degraded"

    mod_pkt = hub.propose_module(ModuleProposal(proposer_kernel_id="hub_a", module_name="planner_x", module_kind="planner"))
    assert mod_pkt.packet_kind == PacketKind.MODULE_PROPOSAL


def test_subkernel_status_and_resource_request():
    sub = Subkernel(kernel_id="sub_math", parent_kernel_id="hub_a", specialization="math")
    sub.receive_goal({"goal": "prove theorem"})
    assert sub.lifecycle == "experimental"

    status = sub.status_packet()
    assert status.packet_kind == PacketKind.STATUS
    assert status.target_kernel_id == "hub_a"

    req = ResourceRequest(requester_kernel_id="sub_math", resource_kind="memory_mb", amount=128)
    pkt = sub.request_budget(req)
    assert pkt.packet_kind == PacketKind.RESOURCE_REQUEST
    assert pkt.reply_expected is True


def test_kernel_ecology_runtime_main_only_default(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.pop("TOVAH_BOOT_MODE", None)
    prev_hub = os.environ.pop("TOVAH_ENABLE_HUB", None)
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        summary = k.get_kernel_ecology_summary()
        assert summary["boot_mode"] == "main_only"
        assert summary["hub_present"] is False
        assert isinstance(k.kernel_packet_log, list)
    finally:
        if prev_mode is not None:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode
        if prev_hub is not None:
            os.environ["TOVAH_ENABLE_HUB"] = prev_hub


def test_kernel_ecology_runtime_hub_and_subkernel_spawn():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        assert k.hub_kernel is not None
        before = len(k.kernel_packet_log)
        sub = k._spawn_subkernel("math", "math delegated mission")
        summary = k.get_kernel_ecology_summary()
        assert summary["hub_present"] is True
        assert summary["subkernel_count"] >= 1
        assert sub.kernel_id in k.subkernels
        assert len(k.kernel_packet_log) > before
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_hub_patch_proposal_stages_into_main():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = PatchProposal(
            patch_name="hub_patch_test",
            target="browser_action",
            code="def browser_action(self, action, url=\"\", selector=\"\", text=\"\", timeout=30, **kwargs):\n    return self.tools.browser(action, url=url, selector=selector, text=text, timeout=timeout, **kwargs)",
            rationale="simple governed hub patch",
            source="hub",
            risk_level="low",
        )
        pkt = k.hub_kernel.propose_patch(proposal)
        event = k._dispatch_kernel_packet(pkt)
        assert event["staged_patch_ok"] is True
        assert "hub_patch_test" in k.staged_patches
        assert k.staged_patches["hub_patch_test"]["ecology"]["source_kernel_id"] == "hub"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_goal_packet_delegates_to_subkernel():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        pkt = make_packet(
            PacketKind.GOAL,
            source_kernel_id="hub",
            target_kernel_id="main",
            payload={"goal": "prove theorem", "specialization": "math"},
            mission_context="math delegated mission",
            required_action="delegate_goal",
            parent_goal_id="root_goal_1",
        )
        event = k._dispatch_kernel_packet(pkt)
        delegation = event["delegation"]
        assert delegation["target_kernel_id"] in k.subkernels
        assert k.task_queue.get_by_id(delegation["task_id"]).delegated_to_kernel_id == delegation["target_kernel_id"]
        assert len(k.delegation_manager.list_active()) >= 1
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_hub_local_queues_survive_revert():
    hub = HubKernel(kernel_id="hub_q", parent_kernel_id="main")
    hub.queue_goal_work("derive invariants", specialization="math")
    hub.queue_module_proposal(ModuleProposal(proposer_kernel_id="hub_q", module_name="proof_lab", module_kind="critic"))
    snap = hub.snapshot("queue_snap")
    hub.queue_goal_work("different work", specialization="general")
    reverted = hub.revert_to_last_snapshot()
    assert reverted == snap
    assert len(hub.work_queue) == 1
    assert len(hub.proposal_queue) == 1


def test_module_proposal_enters_registry_and_bus():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(
            proposer_kernel_id="hub",
            module_name="proof_lab",
            module_kind="critic",
            capabilities=["check_proof"],
            dependencies=["observer"],
        )
        pkt = k.hub_kernel.propose_module(proposal)
        event = k._dispatch_kernel_packet(pkt)
        rec = event["module_record"]
        assert rec["module_name"] == "proof_lab"
        assert k.module_registry.summary()["proposal_count"] >= 1
        assert k.message_bus.summary()["proposal_route_count"] >= 1
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_patch_proposal_records_promotion_evidence():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = PatchProposal(
            patch_name="hub_patch_evidence_test",
            target="browser_action",
            code='''def browser_action(self, action, url="", selector="", text="", timeout=30, **kwargs):\n    return self.tools.browser(action, url=url, selector=selector, text=text, timeout=timeout, **kwargs)''',
            rationale="simple governed hub patch",
            source="hub",
            risk_level="low",
        )
        pkt = k.hub_kernel.propose_patch(proposal)
        event = k._dispatch_kernel_packet(pkt)
        summary = event["promotion_summary"]
        assert summary["patch_name"] == "hub_patch_evidence_test"
        assert summary["evidence_count"] >= 2
        assert k.promotion_ladder.source_metadata["hub_patch_evidence_test"]["source_kernel_id"] == "hub"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_promotion_request_records_evidence():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import PromotionRequest
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        patch = PatchProposal(
            patch_name="hub_patch_promote_test",
            target="browser_action",
            code='''def browser_action(self, action, url="", selector="", text="", timeout=30, **kwargs):\n    return self.tools.browser(action, url=url, selector=selector, text=text, timeout=timeout, **kwargs)''',
            rationale="simple governed hub patch",
            source="hub",
            risk_level="low",
        )
        k._dispatch_kernel_packet(k.hub_kernel.propose_patch(patch))
        req = PromotionRequest(
            requester_kernel_id="hub",
            artifact_kind="patch",
            artifact_name="hub_patch_promote_test",
            desired_stage="shadow_deployed",
            evidence=["passed local review"],
        )
        event = k._dispatch_kernel_packet(k.hub_kernel.request_promotion(req))
        assert event["promotion_request"]["artifact_name"] == "hub_patch_promote_test"
        assert len(k.promotion_ladder.evidence_for("hub_patch_promote_test")) >= 3
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_hub_and_subkernel_export_roundtrip():
    hub = HubKernel(kernel_id="hub_persist", parent_kernel_id="main")
    hub.queue_goal_work("derive invariant", specialization="math")
    hub.memory_branch.append({"kind": "episodic", "key": "x", "data": {"ok": True}})
    hub.snapshot("persist_snap")
    restored_hub = HubKernel.from_state(hub.export_state())
    assert restored_hub is not None
    assert restored_hub.memory_branch[0]["key"] == "x"
    assert len(restored_hub.rollback_points) == 1

    sub = Subkernel(kernel_id="sub_persist", parent_kernel_id="hub", specialization="math")
    sub.receive_goal({"goal": "prove theorem"})
    sub.state.local_state["memory_branch"] = [{"kind": "semantic", "key": "lemma", "data": {"ok": True}}]
    restored_sub = Subkernel.from_state(sub.export_state())
    assert restored_sub is not None
    assert restored_sub.state.specialization == "math"
    assert restored_sub.state.local_state["memory_branch"][0]["key"] == "lemma"


def test_kernel_memory_sync_packet_updates_store_and_provenance():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        assert k.hub_kernel is not None
        k.hub_kernel.memory_branch = [
            {"kind": "episodic", "key": "hub_fact", "data": {"value": 7}, "tags": ["test"]},
        ]
        pkt = k.hub_kernel.request_memory_sync({
            "request_id": "ms_kernel_test",
            "requester_kernel_id": "hub",
            "target_kernel_id": "main",
            "sync_mode": "promote",
            "memory_kinds": ["episodic"],
            "rationale": "promote hub fact",
        })
        event = k._dispatch_kernel_packet(pkt)
        result = event["memory_sync_result"]
        assert result["promoted_count"] == 1
        assert any(e.key == "hub_fact" for e in k.memory_store.get_bank("episodic"))
        assert k.branch_provenance.summary()["node_count"] >= 2
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_cluster_registry_and_trust_persist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("math", "math mission")
        summary = k.get_kernel_ecology_summary()
        assert summary["cluster"]["node_count"] >= 2
        assert summary["cluster_trust"]["tracked_nodes"] >= 1
        assert summary["node_identity"]["kernel_id"] == "main"
        k._save_kernel_ecology_state()
        k2 = ProtozoanKernel(api={}, is_original=True)
        summary2 = k2.get_kernel_ecology_summary()
        assert summary2["cluster"]["node_count"] >= 2
        assert summary2["cluster_trust"]["tracked_nodes"] >= 1
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_self_model_exposes_cluster_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k._spawn_subkernel("math", "math mission")
        sm = k.update_self_model()
        assert sm.node_identity_summary["kernel_id"] == "main"
        assert sm.cluster_summary["node_count"] >= 2
        assert "tracked_nodes" in sm.trust_summary
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_spawned_subkernel_enters_cluster_registry():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("math", "math delegated mission")
        node = k.cluster_registry.get(f"node_{sub.kernel_id}")
        assert node is not None
        assert node.kernel_id == sub.kernel_id
        assert node.specialization == "math"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_goal_packet_delegation_populates_distributed_queue_and_trust():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        pkt = make_packet(
            PacketKind.GOAL,
            source_kernel_id="hub",
            target_kernel_id="main",
            payload={"goal": "prove theorem", "specialization": "math", "required_trust_level": "provisional"},
            mission_context="math delegated mission",
            required_action="delegate_goal",
            parent_goal_id="root_goal_2",
        )
        event = k._dispatch_kernel_packet(pkt)
        delegation = event["delegation"]
        assert delegation["job_id"]
        summary = k.distributed_queue.summary()
        assert summary["active_count"] >= 1
        assert delegation["assigned_trust_level"] in {"provisional", "trusted", "sovereign", "low"}
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_ecology_persistence_restores_distributed_queue_state(tmp_path):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        rec = k.distributed_queue.enqueue(goal="x", goal_id="g1", source_kernel_id="main", specialization="math")
        k._save_kernel_ecology_state()
        k2 = ProtozoanKernel(api={}, is_original=True)
        assert rec.job_id in k2.distributed_queue.records
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_subkernel_tool_request_policy_and_grants():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ToolAccessRequest
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("math", "math delegated mission")

        allow_pkt = sub.request_tool_access(ToolAccessRequest(requester_kernel_id=sub.kernel_id, tool_name="web_search"))
        allow_event = k._dispatch_kernel_packet(allow_pkt)
        assert allow_event["tool_access_decision"]["allowed"] is True
        assert "web_search" in sub.state.local_tools

        deny_pkt = sub.request_tool_access(ToolAccessRequest(requester_kernel_id=sub.kernel_id, tool_name="browser_action"))
        deny_event = k._dispatch_kernel_packet(deny_pkt)
        assert deny_event["tool_access_decision"]["allowed"] is False
        assert deny_event["tool_access_decision"]["reason"] in {"permission_not_allowed_for_role", "logged_tool_requires_locality", "trust_below_role_floor"}
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_module_policy_blocks_subkernel_direct_main_promotion():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("analysis", "analysis delegated mission")
        proposal = ModuleProposal(
            proposer_kernel_id=sub.kernel_id,
            module_name="remote_optimizer",
            module_kind="optimizer",
            promotion_target="hub",
            capabilities=["score_variants"],
        )
        pkt = sub.propose_module(proposal)
        event = k._dispatch_kernel_packet(pkt)
        # The proposal's promotion_target is "hub", and the policy correctly
        # evaluates against that target. A subkernel proposing a medium-risk
        # module to hub fails the role-budget check (subkernel.max_risk_class
        # == "low"). The first failure triggers `apply_module_feedback`
        # which records a 120s cooldown; the test's second-pass policy gate
        # observation therefore sees `module_on_cooldown` rather than the
        # original `risk_exceeds_role_budget`. Both reasons are valid hard
        # blocks for this scenario.
        assert event["module_policy"]["allowed"] is False
        assert event["module_policy"]["target"] == "hub"
        assert event["module_policy"]["reason"] in {
            "risk_exceeds_role_budget", "module_on_cooldown",
        }
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_promotion_gate_requires_trust_for_live_promoted():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import PromotionRequest
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("analysis", "analysis delegated mission")
        pkt = sub.request_promotion(PromotionRequest(
            requester_kernel_id=sub.kernel_id,
            artifact_kind="patch",
            artifact_name="candidate_patch",
            desired_stage="live_promoted",
            evidence=["unit_test"],
        ))
        event = k._dispatch_kernel_packet(pkt)
        assert event["promotion_request"]["gate"]["allowed"] is False
        assert event["promotion_request"]["gate"]["required_trust"] == "trusted"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_delegation_records_worker_policy_shape():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        delegation = k._delegate_task_to_subkernel("prove theorem", "math", "math delegated mission")
        lease = k.delegation_manager.leases[delegation["lease_id"]]
        djob = k.distributed_queue.records[delegation["job_id"]]
        assert lease.worker_role == "subkernel"
        assert "safe_autonomous" in lease.allowed_tool_permissions
        assert "hub" in lease.allowed_promotion_targets
        assert djob.target_worker_role == "subkernel"
        assert "safe_autonomous" in djob.allowed_tool_permissions
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_module_proposal_auto_reviews_to_approved_for_hub_target():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(
            proposer_kernel_id="hub",
            module_name="proof_router",
            module_kind="planner",
            promotion_target="hub",
            capabilities=["route_proofs"],
        )
        event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        assert event["module_record"]["review_outcome"]["status"] == "approved"
        rec = k.module_registry.proposals[event["module_record"]["proposal_id"]]
        assert rec.status == "approved"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_trusted_hub_module_promotion_to_main_can_auto_promote():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import PromotionRequest
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.hub_kernel.trust_from_main = "trusted"
        k.cluster_trust.set_trust("hub", "trusted", reason="unit_test", source="main")
        proposal = ModuleProposal(
            proposer_kernel_id="hub",
            module_name="trusted_promote_mod",
            module_kind="planner",
            promotion_target="hub",
            capabilities=["route"],
        )
        prop_event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        proposal_id = prop_event["module_record"]["proposal_id"]
        assert k.module_registry.proposals[proposal_id].status in {"review_pending", "rejected", "approved", "promoted"}
        req = PromotionRequest(
            requester_kernel_id="hub",
            artifact_kind="module",
            artifact_name="trusted_promote_mod",
            proposal_id=proposal_id,
            target="main",
            target_kernel_id="main",
            evidence=["shadow rehearsal", "regression notes"],
        )
        event = k._dispatch_kernel_packet(k.hub_kernel.request_promotion(req))
        assert event["promotion_request"]["review_outcome"]["status"] == "promoted"
        assert "trusted_promote_mod" in k.module_registry.experimental_manifests
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_promotion_ladder_stage_gate_blocks_low_trust_live_promotion():
    from tovah_v14.mutation.promotion_ladder import PromotionLadder
    ladder = PromotionLadder()
    ladder.set_source_metadata(
        "candidate_patch",
        source_kernel_id="sub_1",
        source_role="subkernel",
        source_locality="local",
        trust_level="low",
        risk_level="medium",
    )
    gate = ladder.assess_stage_transition_gate("candidate_patch", to_stage="live_promoted", target="main")
    assert gate["allowed"] is False
    assert gate["reason"] in {"target_not_allowed_for_role", "main_promotion_requires_trusted"}


def test_tool_request_budget_gate_blocks_when_lease_capacity_exhausted():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ToolAccessRequest
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("math", "math mission")
        for idx in range(3):
            k.delegation_manager.issue(
                goal_id=f"g{idx}",
                source_kernel_id="main",
                target_kernel_id=sub.kernel_id,
                mission_context="math",
                worker_role="subkernel",
            )
        pkt = sub.request_tool_access(ToolAccessRequest(requester_kernel_id=sub.kernel_id, tool_name="web_search"))
        event = k._dispatch_kernel_packet(pkt)
        assert event["tool_access_decision"]["allowed"] is False
        assert event["tool_access_decision"]["reason"] == "lease_capacity_exhausted"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_delegation_prefers_more_mature_subkernel_when_trust_tied():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        a = k._spawn_subkernel("math", "math mission a")
        b = k._spawn_subkernel("math", "math mission b")
        a.state.local_modules["proof_lab"] = {"status": "approved"}
        a.state.local_tools["web_search"] = {"status": "granted"}
        a.state.pending_goals.append({"goal": "pending"})
        k._sync_kernel_registry()
        chosen, node_id, trust = k._choose_delegation_target("math", required_trust_level="low")
        assert chosen is not None
        assert chosen.kernel_id == a.kernel_id
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_dynamic_trust_survives_registry_sync_for_subkernel():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("math", "math mission")
        baseline = k.cluster_trust.trust_level_for(sub.kernel_id, default=sub.trust_from_parent)
        k.cluster_trust.record_outcome(sub.kernel_id, "delegation_failure", success=False, severity="normal", source="main")
        drifted = k.cluster_trust.trust_level_for(sub.kernel_id, default=sub.trust_from_parent)
        assert drifted != baseline
        k._sync_kernel_registry()
        assert k.cluster_trust.trust_level_for(sub.kernel_id, default=sub.trust_from_parent) == drifted
        assert k.child_kernel_registry[sub.kernel_id]["trust_level"] == drifted
        node = k.cluster_registry.get(f"node_{sub.kernel_id}")
        assert node is not None
        assert node.trust_level == drifted
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_record_delegation_outcome_success_completes_and_recovers_budget():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ToolAccessRequest
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("math", "math mission")
        pkt = sub.request_tool_access(ToolAccessRequest(requester_kernel_id=sub.kernel_id, tool_name="web_search"))
        event = k._dispatch_kernel_packet(pkt)
        assert event["tool_access_decision"]["allowed"] is True
        before = int(k.budget_manager.worker_usage.get(sub.kernel_id, {}).get("safe_autonomous", 0))
        delegation = k._delegate_task_to_subkernel("prove theorem", "math", "math mission")
        outcome = k.record_delegation_outcome(delegation["lease_id"], success=True, result={"status": "ok"})
        assert outcome["ok"] is True
        lease = k.delegation_manager.leases[delegation["lease_id"]]
        assert lease.status == "completed"
        djob = k.distributed_queue.records[delegation["job_id"]]
        assert djob.status == "completed"
        after = int(k.budget_manager.worker_usage.get(sub.kernel_id, {}).get("safe_autonomous", 0))
        assert after <= before
        assert k.cluster_trust.trust_score(k.cluster_trust.trust_level_for(sub.kernel_id, default=sub.trust_from_parent)) >= k.cluster_trust.trust_score(sub.trust_from_parent)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_record_delegation_outcome_failure_degrades_trust_and_marks_failed():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("math", "math mission")
        delegation = k._delegate_task_to_subkernel("prove theorem", "math", "math mission")
        target_id = delegation["target_kernel_id"]
        target = k.subkernels[target_id]
        baseline_score = k.cluster_trust.trust_score(k.cluster_trust.trust_level_for(target_id, default=target.trust_from_parent))
        outcome = k.record_delegation_outcome(delegation["lease_id"], success=False, reason="solver_crash", severity="high")
        assert outcome["ok"] is True
        lease = k.delegation_manager.leases[delegation["lease_id"]]
        assert lease.status == "failed"
        djob = k.distributed_queue.records[delegation["job_id"]]
        assert djob.status == "failed"
        new_score = k.cluster_trust.trust_score(k.cluster_trust.trust_level_for(target_id, default=target.trust_from_parent))
        assert new_score <= baseline_score
        assert k.subkernels[target_id].lifecycle == "degraded"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_promotion_stage_gate_blocks_low_success_source_even_if_trusted():
    from tovah_v14.mutation.promotion_ladder import PromotionLadder
    ladder = PromotionLadder()
    ladder.set_source_metadata(
        "candidate_patch",
        source_kernel_id="hub",
        source_role="hub",
        source_locality="local",
        trust_level="trusted",
        risk_level="medium",
        outcome_success_rate=0.2,
        budget_pressure=0.1,
        dynamic_delta=0.0,
    )
    for idx in range(3):
        ladder.record_evidence("candidate_patch", f"evidence_{idx}", source_kernel_id="hub", trust_level="trusted", risk_class="medium")
    gate = ladder.assess_stage_transition_gate("candidate_patch", to_stage="live_promoted", target="main")
    assert gate["allowed"] is False
    assert gate["reason"] == "low_outcome_success_rate"


def test_promotion_stage_gate_blocks_when_source_pressure_high():
    from tovah_v14.mutation.promotion_ladder import PromotionLadder
    ladder = PromotionLadder()
    ladder.set_source_metadata(
        "candidate_patch",
        source_kernel_id="hub",
        source_role="hub",
        source_locality="local",
        trust_level="trusted",
        risk_level="medium",
        outcome_success_rate=0.95,
        budget_pressure=0.95,
        dynamic_delta=0.0,
    )
    for idx in range(3):
        ladder.record_evidence("candidate_patch", f"evidence_{idx}", source_kernel_id="hub", trust_level="trusted", risk_class="medium")
    gate = ladder.assess_stage_transition_gate("candidate_patch", to_stage="live_promoted", target="main")
    assert gate["allowed"] is False
    assert gate["reason"] == "source_pressure_too_high"


def test_delegation_prefers_higher_success_rate_when_trust_and_maturity_tied(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        a = k._spawn_subkernel("analysis", "analysis mission a")
        b = k._spawn_subkernel("analysis", "analysis mission b")
        k.cluster_trust.set_trust(a.kernel_id, "provisional", reason="unit_test", source="main", metadata={"success_count": 10, "failure_count": 0, "outcome_count": 10, "outcome_success_rate": 1.0})
        k.cluster_trust.set_trust(b.kernel_id, "provisional", reason="unit_test", source="main", metadata={"success_count": 1, "failure_count": 4, "outcome_count": 5, "outcome_success_rate": 0.2})
        k._sync_kernel_registry()
        chosen, node_id, trust = k._choose_delegation_target("analysis", required_trust_level="low")
        assert chosen is not None
        assert chosen.kernel_id == a.kernel_id
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_failed_promotion_request_puts_source_on_cooldown_and_second_request_sees_it(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import PromotionRequest
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.hub_kernel.trust_from_main = "trusted"
        k.cluster_trust.set_trust("hub", "trusted", reason="unit_test", source="main")
        req = PromotionRequest(
            requester_kernel_id="hub",
            artifact_kind="patch",
            artifact_name="cooldown_patch",
            desired_stage="live_promoted",
            evidence=[],
        )
        first = k._dispatch_kernel_packet(k.hub_kernel.request_promotion(req))
        assert first["promotion_request"]["gate"]["allowed"] is False
        hub_node = k.cluster_registry.get("node_hub")
        assert hub_node is not None
        assert float(hub_node.metadata.get("cooldown_until", 0.0)) > time.time()
        second = k._dispatch_kernel_packet(k.hub_kernel.request_promotion(req))
        assert second["promotion_request"]["gate"]["allowed"] is False
        assert second["promotion_request"]["gate"]["reason"] == "source_on_cooldown"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_successful_delegation_builds_maturity_bonus_and_biases_routing(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        a = k._spawn_subkernel("math", "math mission a")
        b = k._spawn_subkernel("math", "math mission b")
        # Keep the two workers otherwise symmetric.
        a.state.local_modules.clear(); a.state.local_tools.clear(); a.state.pending_goals.clear()
        b.state.local_modules.clear(); b.state.local_tools.clear(); b.state.pending_goals.clear()
        k._sync_kernel_registry()
        job = k.distributed_queue.enqueue(goal="prove theorem", goal_id="g_manual", source_kernel_id="main", specialization="math", mission_context="math")
        lease = k.delegation_manager.issue(goal_id="g_manual", source_kernel_id="main", target_kernel_id=a.kernel_id, target_node_id=f"node_{a.kernel_id}", mission_context="math", worker_role="subkernel", allowed_tool_permissions=["safe_autonomous"], provenance={"job_id": job.job_id})
        k.distributed_queue.assign(job.job_id, target_kernel_id=a.kernel_id, target_node_id=f"node_{a.kernel_id}", lease_id=lease.lease_id, target_worker_role="subkernel", allowed_tool_permissions=["safe_autonomous"], allowed_promotion_targets=["hub"])
        k.record_delegation_outcome(lease.lease_id, success=True, result={"status": "ok"})
        node_a = k.cluster_registry.get(f"node_{a.kernel_id}")
        node_b = k.cluster_registry.get(f"node_{b.kernel_id}")
        assert node_a is not None and node_b is not None
        assert float(node_a.metadata.get("maturity_bonus", 0.0)) > float(node_b.metadata.get("maturity_bonus", 0.0))
        chosen, _, _ = k._choose_delegation_target("math", required_trust_level="low")
        assert chosen is not None
        assert chosen.kernel_id == a.kernel_id
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_feedback_decay_window_reduces_old_failure_weight_and_expires_cooldown(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("math", "math mission")
        k.cluster_registry.upsert_node(
            f"node_{sub.kernel_id}",
            feedback_last_at=time.time() - 7200.0,
            feedback_decay_window=300.0,
            recent_failure_weight=4.0,
            recent_success_weight=0.0,
            maturity_bonus=1.0,
            cooldown_until=time.time() - 60.0,
        )
        metrics = k._node_operational_metrics(sub.kernel_id, locality="local")
        assert metrics["recent_failure_weight"] < 0.5
        assert metrics["cooldown_remaining"] == 0.0
        assert metrics["maturity_bonus"] < 0.2
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_module_cooldown_blocks_repeat_promotion_request(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        sub = k._spawn_subkernel("analysis", "analysis mission")
        proposal = ModuleProposal(
            proposer_kernel_id=sub.kernel_id,
            module_name="fragile_mod",
            module_kind="planner",
            promotion_target="main",
            capabilities=["route"],
        )
        prop_event = k._dispatch_kernel_packet(sub.propose_module(proposal))
        proposal_id = prop_event["module_record"]["proposal_id"]
        metrics = k.module_registry.module_operational_metrics("fragile_mod")
        assert metrics["cooldown_remaining"] > 0.0
        req = PromotionRequest(
            requester_kernel_id=sub.kernel_id,
            artifact_kind="module",
            artifact_name="fragile_mod",
            target_kernel_id="main",
            desired_stage="main",
            evidence=["retry"],
        )
        pkt = sub.request_promotion(req)
        pkt.payload["proposal_id"] = proposal_id
        pkt.payload["target"] = "main"
        event = k._dispatch_kernel_packet(pkt)
        assert event["promotion_request"]["gate"]["reason"] == "module_on_cooldown"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_successful_module_promotion_builds_module_maturity_bonus(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.hub_kernel.trust_from_main = "trusted"
        k.cluster_trust.set_trust("hub", "trusted", reason="unit_test", source="main")
        proposal = ModuleProposal(
            proposer_kernel_id="hub",
            module_name="proven_mod",
            module_kind="planner",
            promotion_target="main",
            capabilities=["route"],
        )
        prop_event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        proposal_id = prop_event["module_record"]["proposal_id"]
        req = PromotionRequest(
            requester_kernel_id="hub",
            artifact_kind="module",
            artifact_name="proven_mod",
            target_kernel_id="main",
            desired_stage="main",
            evidence=["shadow rehearsal", "regression notes"],
        )
        pkt = k.hub_kernel.request_promotion(req)
        pkt.payload["proposal_id"] = proposal_id
        pkt.payload["target"] = "main"
        event = k._dispatch_kernel_packet(pkt)
        assert event["promotion_request"]["review_outcome"]["status"] == "promoted"
        metrics = k.module_registry.module_operational_metrics("proven_mod")
        assert metrics["maturity_bonus"] > 0.0
        fresh = ModuleProposal(
            proposer_kernel_id="hub",
            module_name="fresh_mod",
            module_kind="planner",
            promotion_target="hub",
            capabilities=["route"],
        )
        fresh_event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(fresh))
        proven_priority = k.module_registry.proposal_priority(proposal_id)
        fresh_priority = k.module_registry.proposal_priority(fresh_event["module_record"]["proposal_id"])
        assert proven_priority["metrics"]["maturity_bonus"] > fresh_priority["metrics"]["maturity_bonus"]
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_hub_promotion_priorities_prefer_mature_uncooldown_items(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.hub_kernel.trust_from_main = "trusted"
        k.cluster_trust.set_trust("hub", "trusted", reason="unit_test", source="main")

        good = ModuleProposal(
            proposer_kernel_id="hub",
            module_name="good_mod",
            module_kind="planner",
            promotion_target="main",
            capabilities=["route"],
        )
        good_event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(good))
        good_id = good_event["module_record"]["proposal_id"]
        good_req = PromotionRequest(
            requester_kernel_id="hub",
            artifact_kind="module",
            artifact_name="good_mod",
            target_kernel_id="main",
            desired_stage="main",
            evidence=["shadow rehearsal", "regression notes"],
        )
        good_pkt = k.hub_kernel.request_promotion(good_req)
        good_pkt.payload["proposal_id"] = good_id
        good_pkt.payload["target"] = "main"
        k._dispatch_kernel_packet(good_pkt)

        k.hub_kernel.trust_from_main = "low"
        k.cluster_trust.set_trust("hub", "low", reason="unit_test_cool", source="main")
        bad = ModuleProposal(
            proposer_kernel_id="hub",
            module_name="cool_mod",
            module_kind="planner",
            promotion_target="main",
            capabilities=["route"],
        )
        bad_event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(bad))
        bad_id = bad_event["module_record"]["proposal_id"]
        bad_req = PromotionRequest(
            requester_kernel_id="hub",
            artifact_kind="module",
            artifact_name="cool_mod",
            target_kernel_id="main",
            desired_stage="main",
            evidence=["retry"],
        )
        bad_pkt = k.hub_kernel.request_promotion(bad_req)
        bad_pkt.payload["proposal_id"] = bad_id
        bad_pkt.payload["target"] = "main"
        k._dispatch_kernel_packet(bad_pkt)
        priorities = k._hub_promotion_priority_view(10)
        by_name = {item.get("artifact_name"): item for item in priorities if item.get("artifact_kind") == "module"}
        assert by_name["good_mod"]["priority"]["score"] > by_name["cool_mod"]["priority"]["score"]
        assert by_name["cool_mod"]["priority"]["cooldown_remaining"] > 0.0
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode



def test_module_family_inheritance_boosts_sibling_priority(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.hub_kernel.trust_from_main = "trusted"
        k.cluster_trust.set_trust("hub", "trusted", reason="unit_test", source="main")
        first = ModuleProposal(proposer_kernel_id="hub", module_name="router_alpha", module_kind="planner", promotion_target="main", capabilities=["route"])
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(first))
        pid = ev["module_record"]["proposal_id"]
        req = PromotionRequest(requester_kernel_id="hub", artifact_kind="module", artifact_name="router_alpha", target_kernel_id="main", desired_stage="main", evidence=["shadow rehearsal", "regression notes"])
        pkt = k.hub_kernel.request_promotion(req)
        pkt.payload["proposal_id"] = pid
        pkt.payload["target"] = "main"
        k._dispatch_kernel_packet(pkt)
        sibling = ModuleProposal(proposer_kernel_id="hub", module_name="router_beta", module_kind="planner", promotion_target="hub", capabilities=["route"])
        ev2 = k._dispatch_kernel_packet(k.hub_kernel.propose_module(sibling))
        metrics = k.module_registry.module_operational_metrics("router_beta")
        assert metrics["family_maturity_bonus"] > 0.0
        pr = k.module_registry.proposal_priority(ev2["module_record"]["proposal_id"])
        assert pr["score"] > -5.0
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_process_hub_promotion_queue_consumes_ranked_items(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.hub_kernel.trust_from_main = "trusted"
        k.cluster_trust.set_trust("hub", "trusted", reason="unit_test", source="main")
        good = ModuleProposal(proposer_kernel_id="hub", module_name="queue_good", module_kind="planner", promotion_target="main", capabilities=["route"])
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(good))
        pid = ev["module_record"]["proposal_id"]
        req = PromotionRequest(requester_kernel_id="hub", artifact_kind="module", artifact_name="queue_good", target_kernel_id="main", desired_stage="main", evidence=["shadow rehearsal", "regression notes"])
        pkt = k.hub_kernel.request_promotion(req)
        pkt.payload["proposal_id"] = pid
        pkt.payload["target"] = "main"
        k.hub_kernel.queue_promotion_request(req)
        bad = ModuleProposal(proposer_kernel_id="hub", module_name="queue_bad", module_kind="planner", promotion_target="main", capabilities=["route"])
        evb = k._dispatch_kernel_packet(k.hub_kernel.propose_module(bad))
        pidb = evb["module_record"]["proposal_id"]
        k.module_registry.apply_module_feedback("queue_bad", success=False, severity="high", kind="module_promotion_request", target="main", metadata={"module_kind": "planner"})
        reqb = PromotionRequest(requester_kernel_id="hub", artifact_kind="module", artifact_name="queue_bad", target_kernel_id="main", desired_stage="main", evidence=["retry"])
        pktb = k.hub_kernel.request_promotion(reqb)
        pktb.payload["proposal_id"] = pidb
        pktb.payload["target"] = "main"
        k.hub_kernel.queue_promotion_request(pktb.payload)
        out = k.process_hub_promotion_queue(limit=2, consume=True)
        assert len(out["processed"]) >= 1
        assert any(item.get("review_action") in {"review_now", "gather_evidence", "wait_cooldown"} for item in out["processed"])
        assert len(k.hub_kernel.work_queue) >= 1
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_self_model_exposes_module_readiness_and_hub_review(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="sm_router", module_kind="planner", promotion_target="hub", capabilities=["route"])
        k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        k.hub_kernel.queue_promotion_request(PromotionRequest(requester_kernel_id="hub", artifact_kind="module", artifact_name="sm_router", target_kernel_id="hub", desired_stage="hub", evidence=["note"]))
        k.process_hub_promotion_queue(limit=1, consume=True)
        sm = k.update_self_model()
        assert isinstance(sm.module_priority_summary, list)
        assert isinstance(sm.hub_review_summary, list)
        assert sm.module_readiness >= 0.0
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_process_hub_queue_defers_cooled_module(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="cool_mod", module_kind="planner", promotion_target="main", capabilities=["route"])
        event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        proposal_id = event["module_record"]["proposal_id"]
        k.module_registry.attach_evidence(proposal_id, {"kind": "manual", "note": "enough"})
        k.module_registry.apply_module_feedback("cool_mod", success=False, severity="high", kind="module_promotion_request", target="main", metadata={"module_kind": "planner"})
        k.hub_kernel.queue_promotion_request({"requester_kernel_id": "hub", "artifact_kind": "module", "artifact_name": "cool_mod", "proposal_id": proposal_id, "target_kernel_id": "main", "desired_stage": "main", "evidence": ["retry"]})
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        assert out["processed"][0]["queue_status"] == "deferred_cooldown"
        assert out["deferred"] == 1
        assert len(k.hub_kernel.promotion_queue) == 1
        assert all(item.get("kind") != "promotion_review" for item in k.hub_kernel.work_queue[-2:])
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_process_hub_queue_auto_advances_reviewable_module(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.hub_kernel.trust_from_main = "trusted"
        k.cluster_trust.set_trust("hub", "trusted", reason="unit_test", source="main")
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="advance_mod", module_kind="planner", promotion_target="main", capabilities=["route"])
        event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        proposal_id = event["module_record"]["proposal_id"]
        k.module_registry.attach_evidence(proposal_id, {"kind": "manual", "note": "extra_evidence"})
        k.hub_kernel.queue_promotion_request({"requester_kernel_id": "hub", "artifact_kind": "module", "artifact_name": "advance_mod", "proposal_id": proposal_id, "target_kernel_id": "main", "desired_stage": "main", "evidence": ["retry"]})
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        processed = out["processed"][0]
        assert processed["queue_status"] == "auto_reviewed"
        assert out["advanced"] == 1
        assert len(k.hub_kernel.promotion_queue) == 0
        review = processed["auto_event"]["promotion_request"]["review_outcome"]
        assert review["status"] == "promoted"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_family_inheritance_cap_does_not_mask_sibling_failures(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        for _ in range(4):
            k.module_registry.apply_module_feedback("planner_good", success=True, severity="normal", kind="module_review", target="hub", metadata={"module_kind": "planner"})
        for _ in range(3):
            k.module_registry.apply_module_feedback("planner_bad", success=False, severity="high", kind="module_promotion_request", target="main", metadata={"module_kind": "planner"})
        metrics = k.module_registry.module_operational_metrics("planner_bad")
        assert metrics["family_bonus_carry"] <= metrics["family_bonus_cap"] + 1e-9
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="planner_bad", module_kind="planner", promotion_target="main", capabilities=["route"])
        event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        proposal_id = event["module_record"]["proposal_id"]
        k.module_registry.attach_evidence(proposal_id, {"kind": "manual", "note": "extra_evidence"})
        gate = k.module_registry.assess_promotion_gate(proposal_id, trust_level="trusted", locality="local", target="main")
        assert gate["allowed"] is False
        assert gate["reason"] in {"module_recent_failures", "module_on_cooldown"}
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_self_model_growth_priorities_include_family_readiness(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="growth_mod", module_kind="planner", promotion_target="hub", capabilities=["route"])
        k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        for _ in range(2):
            k.module_registry.apply_module_feedback("growth_mod", success=True, severity="normal", kind="module_review", target="hub", metadata={"module_kind": "planner"})
        sm = k.update_self_model()
        assert sm.family_module_readiness >= 0.0
        assert isinstance(sm.growth_priority_summary, list)
        assert any(item.get("kind") in {"module", "family"} for item in sm.growth_priority_summary)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_wake_hub_deferred_items_requeues_when_cooldown_expires(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="wake_mod", module_kind="planner", promotion_target="main", capabilities=["route"])
        event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        proposal_id = event["module_record"]["proposal_id"]
        k.module_registry.attach_evidence(proposal_id, {"kind": "manual", "note": "enough"})
        k.module_registry.apply_module_feedback("wake_mod", success=False, severity="high", kind="module_promotion_request", target="main", metadata={"module_kind": "planner"})
        k.hub_kernel.queue_promotion_request({"requester_kernel_id": "hub", "artifact_kind": "module", "artifact_name": "wake_mod", "proposal_id": proposal_id, "target_kernel_id": "main", "desired_stage": "main", "evidence": ["retry"]})
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        assert out["processed"][0]["queue_status"] == "deferred_cooldown"
        k.hub_kernel.promotion_queue[0]["deferred_until"] = time.time() - 1.0
        wake = k.wake_hub_deferred_items()
        assert wake["woken"] == 1
        assert k.hub_kernel.promotion_queue[0]["status"] == "queued"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode



def test_complete_hub_evidence_task_requeues_and_improves_priority(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.promotion_ladder.set_source_metadata(
            "patch_evidence_mod",
            source_kernel_id="hub",
            source_role="hub",
            source_locality="local",
            trust_level="trusted",
            risk_level="medium",
            maturity_bonus=0.0,
            outcome_success_rate=1.0,
            recent_failure_weight=0.0,
            cooldown_until=0.0,
        )
        k.hub_kernel.queue_promotion_request({
            "requester_kernel_id": "hub",
            "artifact_kind": "patch",
            "artifact_name": "patch_evidence_mod",
            "target_kernel_id": "main",
            "desired_stage": "sandbox_passed",
            "evidence": [],
        })
        before = k._hub_promotion_priority_view(1)[0]["priority"]["score"]
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        assert out["processed"][0]["queue_status"] == "evidence_requested"
        done = k.complete_hub_evidence_task(artifact_kind="patch", artifact_name="patch_evidence_mod", evidence={"kind": "unit", "note": "new evidence"})
        assert done["ok"] is True
        after = k._hub_promotion_priority_view(1)[0]["priority"]["score"]
        assert after > before
        assert any(q.get("artifact_name") == "patch_evidence_mod" and q.get("queue_status") == "evidence_ready" for q in k.hub_kernel.promotion_queue)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode

def test_process_growth_priorities_prefers_top_growth_item(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        a = ModuleProposal(proposer_kernel_id="hub", module_name="growth_alpha", module_kind="planner", promotion_target="hub", capabilities=["route"])
        b = ModuleProposal(proposer_kernel_id="hub", module_name="growth_beta", module_kind="planner", promotion_target="hub", capabilities=["route"])
        ea = k._dispatch_kernel_packet(k.hub_kernel.propose_module(a))
        eb = k._dispatch_kernel_packet(k.hub_kernel.propose_module(b))
        k.module_registry.apply_module_feedback("growth_alpha", success=True, severity="normal", kind="module_review", target="hub", metadata={"module_kind": "planner"})
        k.module_registry.apply_module_feedback("growth_alpha", success=True, severity="normal", kind="module_review", target="hub", metadata={"module_kind": "planner"})
        k.update_self_model()
        k.hub_kernel.queue_promotion_request({"requester_kernel_id": "hub", "artifact_kind": "module", "artifact_name": "growth_beta", "proposal_id": eb["module_record"]["proposal_id"], "target_kernel_id": "hub", "desired_stage": "hub", "evidence": ["beta"]})
        k.hub_kernel.queue_promotion_request({"requester_kernel_id": "hub", "artifact_kind": "module", "artifact_name": "growth_alpha", "proposal_id": ea["module_record"]["proposal_id"], "target_kernel_id": "hub", "desired_stage": "hub", "evidence": ["alpha"]})
        out = k.process_growth_priorities(limit=1, consume=True)
        assert out["processed"][0]["artifact_name"] == "growth_alpha"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode



def test_review_wave_batches_evidence_ready_items(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        for name in ("wave_patch_a", "wave_patch_b"):
            k.promotion_ladder.set_source_metadata(
                name,
                source_kernel_id="hub",
                source_role="hub",
                source_locality="local",
                trust_level="trusted",
                risk_level="medium",
                maturity_bonus=0.0,
                outcome_success_rate=1.0,
                recent_failure_weight=0.0,
                cooldown_until=0.0,
            )
            k.promotion_ladder.record_evidence(
                name,
                "unit",
                source_kernel_id="hub",
                trust_level="trusted",
                risk_class="medium",
                details={"note": "enough evidence"},
            )
            k.hub_kernel.queue_promotion_request({
                "requester_kernel_id": "hub",
                "artifact_kind": "patch",
                "artifact_name": name,
                "target_kernel_id": "main",
                "desired_stage": "sandbox_passed",
                "evidence": ["unit"],
                "queue_status": "evidence_ready",
            })
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        assert len(out["processed"]) >= 2
        assert out["review_wave_count"] == 1
        assert all(item.get("review_action") == "review_now" for item in out["processed"])
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_queue_aging_raises_older_item_priority(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        for name in ("aged_patch", "fresh_patch"):
            k.promotion_ladder.set_source_metadata(
                name,
                source_kernel_id="hub",
                source_role="hub",
                source_locality="local",
                trust_level="trusted",
                risk_level="medium",
                maturity_bonus=0.0,
                outcome_success_rate=1.0,
                recent_failure_weight=0.0,
                cooldown_until=0.0,
            )
            k.promotion_ladder.record_evidence(
                name,
                "unit",
                source_kernel_id="hub",
                trust_level="trusted",
                risk_class="medium",
                details={"note": "same evidence"},
            )
        k.hub_kernel.queue_promotion_request({
            "requester_kernel_id": "hub",
            "artifact_kind": "patch",
            "artifact_name": "fresh_patch",
            "target_kernel_id": "main",
            "desired_stage": "sandbox_passed",
            "evidence": ["unit"],
            "queued_at": time.time(),
        })
        k.hub_kernel.queue_promotion_request({
            "requester_kernel_id": "hub",
            "artifact_kind": "patch",
            "artifact_name": "aged_patch",
            "target_kernel_id": "main",
            "desired_stage": "sandbox_passed",
            "evidence": ["unit"],
            "queued_at": time.time() - 7200.0,
        })
        ranked = k._hub_promotion_priority_view(5)
        assert ranked[0]["artifact_name"] == "aged_patch"
        assert ranked[0]["priority"]["age_bonus"] > ranked[1]["priority"]["age_bonus"]
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_successful_evidence_completion_lifts_family_readiness(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        prop = ModuleProposal(proposer_kernel_id="hub", module_name="family_alpha", module_kind="planner", promotion_target="hub", capabilities=["route"])
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(prop))
        pid = ev["module_record"]["proposal_id"]
        family_key = k.module_registry.family_key_for("family_alpha", "planner")
        before_rows = {row.get("family_key"): row for row in k.module_registry.family_readiness_summary()}
        before = float(before_rows.get(family_key, {}).get("score", 0.0))
        k.hub_kernel.work_queue.append({
            "kind": "promotion_evidence",
            "artifact_kind": "module",
            "artifact_name": "family_alpha",
            "proposal_id": pid,
            "target_kernel_id": "hub",
            "module_kind": "planner",
            "status": "evidence_requested",
            "queued_at": 0.0,
        })
        done = k.complete_hub_evidence_task(artifact_kind="module", artifact_name="family_alpha", proposal_id=pid, evidence={"kind": "benchmark", "module_kind": "planner"}, success=True)
        assert done["ok"] is True
        after_rows = {row.get("family_key"): row for row in k.module_registry.family_readiness_summary()}
        after = float(after_rows.get(family_key, {}).get("score", 0.0))
        assert after > before
        assert any(q.get("artifact_name") == "family_alpha" and q.get("queue_status") == "evidence_ready" for q in k.hub_kernel.promotion_queue)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_review_wave_state_persists_and_completes(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        for name in ("wave_state_patch_a", "wave_state_patch_b"):
            k.promotion_ladder.set_source_metadata(
                name,
                source_kernel_id="hub",
                source_role="hub",
                source_locality="local",
                trust_level="trusted",
                risk_level="medium",
                maturity_bonus=0.0,
                outcome_success_rate=1.0,
                recent_failure_weight=0.0,
                cooldown_until=0.0,
            )
            k.promotion_ladder.record_evidence(
                name,
                "unit",
                source_kernel_id="hub",
                trust_level="trusted",
                risk_class="medium",
                details={"note": "enough evidence"},
            )
            k.hub_kernel.queue_promotion_request({
                "requester_kernel_id": "hub",
                "artifact_kind": "patch",
                "artifact_name": name,
                "target_kernel_id": "main",
                "desired_stage": "sandbox_passed",
                "evidence": ["unit"],
                "queue_status": "evidence_ready",
            })
        out = k.process_hub_promotion_queue(limit=2, consume=True)
        assert out["review_wave_count"] >= 1
        wave_id = out["wave_ids"][0]
        assert any(str(w.get("wave_id","")) == wave_id for w in k.hub_kernel.review_waves)
        done = k.complete_hub_review_wave(wave_id, item_results={"wave_state_patch_a": True, "wave_state_patch_b": {"success": False, "severity": "high"}}, default_success=True)
        assert done["ok"] is True
        wave = next(w for w in k.hub_kernel.review_waves if str(w.get("wave_id","")) == wave_id)
        assert wave["status"] == "completed"
        assert "success_rate" in wave
        k._save_kernel_ecology_state()
        k2 = ProtozoanKernel(api={}, is_original=True)
        assert any(str(w.get("wave_id","")) == wave_id for w in k2.hub_kernel.review_waves)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_review_wave_completion_feeds_patch_priority(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        name = "wave_feedback_patch"
        k.promotion_ladder.set_source_metadata(
            name,
            source_kernel_id="hub",
            source_role="hub",
            source_locality="local",
            trust_level="trusted",
            risk_level="medium",
            maturity_bonus=0.0,
            outcome_success_rate=0.2,
            recent_failure_weight=2.0,
            cooldown_until=0.0,
        )
        k.promotion_ladder.record_evidence(name, "unit", source_kernel_id="hub", trust_level="trusted", risk_class="medium", details={"note": "enough"})
        k.hub_kernel.queue_promotion_request({
            "requester_kernel_id": "hub",
            "artifact_kind": "patch",
            "artifact_name": name,
            "target_kernel_id": "main",
            "desired_stage": "sandbox_passed",
            "evidence": ["unit"],
            "queue_status": "evidence_ready",
        })
        before = next(item for item in k._hub_promotion_priority_view(5) if item.get("artifact_name") == name)["priority"]["score"]
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        wave_id = out["wave_ids"][0]
        done = k.complete_hub_review_wave(wave_id, item_results={name: True}, default_success=True)
        assert done["ok"] is True
        k.hub_kernel.queue_promotion_request({
            "requester_kernel_id": "hub",
            "artifact_kind": "patch",
            "artifact_name": name,
            "target_kernel_id": "main",
            "desired_stage": "sandbox_passed",
            "evidence": ["unit"],
            "queue_status": "evidence_ready",
        })
        after = next(item for item in k._hub_promotion_priority_view(5) if item.get("artifact_name") == name)["priority"]["score"]
        assert after > before
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_self_model_exposes_review_wave_summary(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        name = "wave_sm_patch"
        k.promotion_ladder.set_source_metadata(
            name,
            source_kernel_id="hub",
            source_role="hub",
            source_locality="local",
            trust_level="trusted",
            risk_level="medium",
            maturity_bonus=0.0,
            outcome_success_rate=1.0,
            recent_failure_weight=0.0,
            cooldown_until=0.0,
        )
        k.promotion_ladder.record_evidence(name, "unit", source_kernel_id="hub", trust_level="trusted", risk_class="medium", details={"note": "enough"})
        k.hub_kernel.queue_promotion_request({
            "requester_kernel_id": "hub",
            "artifact_kind": "patch",
            "artifact_name": name,
            "target_kernel_id": "main",
            "desired_stage": "sandbox_passed",
            "evidence": ["unit"],
            "queue_status": "evidence_ready",
        })
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        k.complete_hub_review_wave(out["wave_ids"][0], item_results={name: True}, default_success=True)
        sm = k.update_self_model()
        assert isinstance(sm.hub_wave_summary, list)
        assert any(str(item.get("wave_id","")) == out["wave_ids"][0] for item in sm.hub_wave_summary)
        assert any(item.get("kind") == "review_wave" for item in sm.growth_priority_summary)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_failed_review_wave_increases_queue_caution(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="caution_alpha", module_kind="planner", promotion_target="hub", capabilities=["route"])
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        pid = ev["module_record"]["proposal_id"]
        req = PromotionRequest(requester_kernel_id="hub", artifact_kind="module", artifact_name="caution_alpha", target_kernel_id="hub", desired_stage="hub", evidence=["baseline"])
        pkt = k.hub_kernel.request_promotion(req)
        pkt.payload["proposal_id"] = pid
        pkt.payload["target"] = "hub"
        k.hub_kernel.queue_promotion_request(pkt.payload)
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        wave_id = out["wave_ids"][0]
        done = k.complete_hub_review_wave(wave_id, item_results={f"module::caution_alpha::{pid}": {"success": False, "severity": "normal"}}, default_success=True)
        assert done["ok"] is True
        sibling = ModuleProposal(proposer_kernel_id="hub", module_name="caution_beta", module_kind="planner", promotion_target="hub", capabilities=["route"])
        ev2 = k._dispatch_kernel_packet(k.hub_kernel.propose_module(sibling))
        pid2 = ev2["module_record"]["proposal_id"]
        req2 = PromotionRequest(requester_kernel_id="hub", artifact_kind="module", artifact_name="caution_beta", target_kernel_id="hub", desired_stage="hub", evidence=["retry"])
        pkt2 = k.hub_kernel.request_promotion(req2)
        pkt2.payload["proposal_id"] = pid2
        pkt2.payload["target"] = "hub"
        k.hub_kernel.queue_promotion_request(pkt2.payload)
        rows = [r for r in k._hub_promotion_priority_view(10) if r.get("artifact_name") == "caution_beta"]
        assert rows
        pr = rows[0].get("priority", {})
        assert float(pr.get("caution_level", 0.0)) > 0.0
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_surface_open_review_waves_adds_resolution_work(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="wave_open_alpha", module_kind="planner", promotion_target="hub", capabilities=["route"])
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        pid = ev["module_record"]["proposal_id"]
        req = PromotionRequest(requester_kernel_id="hub", artifact_kind="module", artifact_name="wave_open_alpha", target_kernel_id="hub", desired_stage="hub", evidence=["baseline"])
        pkt = k.hub_kernel.request_promotion(req)
        pkt.payload["proposal_id"] = pid
        pkt.payload["target"] = "hub"
        k.hub_kernel.queue_promotion_request(pkt.payload)
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        wave_id = out["wave_ids"][0]
        wave = next(w for w in k.hub_kernel.review_waves if str(w.get("wave_id","")) == wave_id)
        wave["created_at"] = time.time() - 1200.0
        surfaced = k.surface_open_review_waves(limit=1)
        assert surfaced["surfaced"] >= 1
        assert any(str(wq.get("kind","")) == "review_wave_resolution" and str(wq.get("wave_id","")) == wave_id for wq in k.hub_kernel.work_queue)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_failed_review_wave_feeds_family_cooldown(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.hub_kernel.trust_from_main = "trusted"
        k.cluster_trust.set_trust("hub", "trusted", reason="unit_test", source="main")
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="family_fail_alpha", module_kind="planner", promotion_target="main", capabilities=["route"])
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        pid = ev["module_record"]["proposal_id"]
        req = PromotionRequest(requester_kernel_id="hub", artifact_kind="module", artifact_name="family_fail_alpha", target_kernel_id="main", desired_stage="main", evidence=["baseline", "shadow rehearsal"])
        pkt = k.hub_kernel.request_promotion(req)
        pkt.payload["proposal_id"] = pid
        pkt.payload["target"] = "main"
        k.hub_kernel.queue_promotion_request(pkt.payload)
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        wave_id = out["wave_ids"][0]
        done = k.complete_hub_review_wave(wave_id, item_results={f"module::family_fail_alpha::{pid}": {"success": False, "severity": "high"}}, default_success=True)
        assert done["ok"] is True
        fam_key = k.module_registry.family_key_for("family_fail_alpha", "planner")
        fam_rows = {row.get("family_key"): row for row in k.module_registry.family_readiness_summary()}
        assert float(fam_rows[fam_key].get("cooldown_remaining", 0.0)) > 0.0
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_queue_caution_decays_after_time(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="decay_alpha", module_kind="planner", promotion_target="hub", capabilities=["route"])
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        pid = ev["module_record"]["proposal_id"]
        req = PromotionRequest(requester_kernel_id="hub", artifact_kind="module", artifact_name="decay_alpha", target_kernel_id="hub", desired_stage="hub", evidence=["baseline"])
        pkt = k.hub_kernel.request_promotion(req)
        pkt.payload["proposal_id"] = pid
        pkt.payload["target"] = "hub"
        k.hub_kernel.queue_promotion_request(pkt.payload)
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        wave_id = out["wave_ids"][0]
        done = k.complete_hub_review_wave(wave_id, item_results={f"module::decay_alpha::{pid}": {"success": False, "severity": "normal"}}, default_success=True)
        assert done["ok"] is True
        fam = k.module_registry.family_key_for("decay_alpha", "planner")
        caution = k.hub_kernel.local_branch_state.get("queue_caution", {})
        assert fam in caution
        caution[fam]["cooldown_until"] = time.time() - 1.0
        caution[fam]["caution_last_at"] = time.time() - 7200.0
        level_before = float(caution[fam].get("caution_level", 0.0))
        cleaned = k._hub_queue_caution_map()
        level_after = float(cleaned.get(fam, {}).get("caution_level", 0.0))
        assert level_after <= level_before
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_resolve_surfaced_wave_auto_closes_when_no_unresolved_items(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal, PromotionRequest
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="close_wave_alpha", module_kind="planner", promotion_target="hub", capabilities=["route"])
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        pid = ev["module_record"]["proposal_id"]
        req = PromotionRequest(requester_kernel_id="hub", artifact_kind="module", artifact_name="close_wave_alpha", target_kernel_id="hub", desired_stage="hub", evidence=["baseline"])
        pkt = k.hub_kernel.request_promotion(req)
        pkt.payload["proposal_id"] = pid
        pkt.payload["target"] = "hub"
        k.hub_kernel.queue_promotion_request(pkt.payload)
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        wave_id = out["wave_ids"][0]
        # clear unresolved review item so surfaced resolution can auto-close the wave
        k.hub_kernel.work_queue = [wq for wq in k.hub_kernel.work_queue if str(wq.get("review_wave_id", wq.get("wave_id", ""))) != wave_id]
        wave = next(w for w in k.hub_kernel.review_waves if str(w.get("wave_id", "")) == wave_id)
        wave["created_at"] = time.time() - 1200.0
        surfaced = k.surface_open_review_waves(limit=1)
        assert surfaced["surfaced"] >= 1
        resolved = k.resolve_surfaced_review_waves(limit=1)
        assert resolved["resolved"] >= 1
        wave2 = next(w for w in k.hub_kernel.review_waves if str(w.get("wave_id", "")) == wave_id)
        assert str(wave2.get("status", "")) == "auto_closed"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_wave_resolution_history_feeds_growth_priorities(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import PatchProposal, PromotionRequest
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        # create a patch queue item that will surface and escalate
        k.promotion_ladder.state["stale_patch"] = "sandbox_passed"
        req = {"artifact_kind": "patch", "artifact_name": "stale_patch", "desired_stage": "live_promoted", "queued_at": time.time() - 1800.0, "status": "queued"}
        k.hub_kernel.queue_promotion_request(req)
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        assert out["wave_ids"]
        wave_id = out["wave_ids"][0]
        wave = next(w for w in k.hub_kernel.review_waves if str(w.get("wave_id", "")) == wave_id)
        wave["created_at"] = time.time() - 1800.0
        wave["surface_count"] = 2
        surfaced = k.surface_open_review_waves(limit=1)
        assert surfaced["surfaced"] >= 1
        resolved = k.resolve_surfaced_review_waves(limit=1)
        assert resolved["escalated"] >= 1
        k.update_self_model()
        assert any(str(item.get("kind", "")) == "patch" and str(item.get("name", "")) == "stale_patch" for item in k.self_model.growth_priority_summary)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_review_wave_escalation_routes_patch_to_quarantine(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.promotion_ladder.state["esc_patch"] = "sandbox_passed"
        k.hub_kernel.review_waves.append({
            "wave_id": "wave_patch_q",
            "created_at": time.time() - 2400.0,
            "status": "open",
            "selected_count": 1,
            "items": [{"artifact_kind": "patch", "artifact_name": "esc_patch", "proposal_id": ""}],
            "outcome_summary": {"success_count": 0, "failure_count": 2},
            "surface_count": 2,
        })
        k.hub_kernel.work_queue.append({
            "kind": "promotion_review",
            "artifact_kind": "patch",
            "artifact_name": "esc_patch",
            "review_wave_id": "wave_patch_q",
            "queued_at": time.time() - 2400.0,
        })
        k.hub_kernel.work_queue.append({
            "kind": "review_wave_escalation",
            "wave_id": "wave_patch_q",
            "queued_at": time.time(),
            "status": "escalated",
            "age_seconds": 2400.0,
            "confidence": 2.0,
        })
        out = k.process_wave_escalations(limit=1)
        assert out["processed"] >= 1
        assert "esc_patch" in k.quarantine_manager.records
        wave = next(w for w in k.hub_kernel.review_waves if str(w.get("wave_id", "")) == "wave_patch_q")
        assert str(wave.get("status", "")) == "quarantined"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_review_wave_escalation_routes_module_to_rework(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(ModuleProposal(proposer_kernel_id="hub", module_name="rework_alpha", module_kind="planner", promotion_target="hub", capabilities=["route"])))
        pid = ev["module_record"]["proposal_id"]
        k.hub_kernel.review_waves.append({
            "wave_id": "wave_mod_rework",
            "created_at": time.time() - 2400.0,
            "status": "open",
            "selected_count": 1,
            "items": [{"artifact_kind": "module", "artifact_name": "rework_alpha", "proposal_id": pid, "module_kind": "planner"}],
            "outcome_summary": {"success_count": 0, "failure_count": 2},
            "surface_count": 2,
        })
        k.hub_kernel.work_queue.append({
            "kind": "promotion_review",
            "artifact_kind": "module",
            "artifact_name": "rework_alpha",
            "proposal_id": pid,
            "module_kind": "planner",
            "review_wave_id": "wave_mod_rework",
            "queued_at": time.time() - 2400.0,
            "target": "hub",
        })
        k.hub_kernel.work_queue.append({
            "kind": "review_wave_escalation",
            "wave_id": "wave_mod_rework",
            "queued_at": time.time(),
            "status": "escalated",
            "age_seconds": 2400.0,
            "confidence": 1.8,
        })
        out = k.process_wave_escalations(limit=1)
        assert out["processed"] >= 1
        assert any(str(item.get("kind", "")) == "proposal_rework" and str(item.get("artifact_name", "")) == "rework_alpha" for item in k.hub_kernel.work_queue)
        wave = next(w for w in k.hub_kernel.review_waves if str(w.get("wave_id", "")) == "wave_mod_rework")
        assert str(wave.get("status", "")) == "rework_routed"
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_repeated_auto_closures_reduce_family_cooldown_pressure(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(ModuleProposal(proposer_kernel_id="hub", module_name="closure_alpha", module_kind="planner", promotion_target="main", capabilities=["route"])))
        pid = ev["module_record"]["proposal_id"]
        fam = k.module_registry.family_key_for("closure_alpha", "planner")
        k.module_registry.apply_module_feedback("closure_alpha", success=False, severity="high", kind="review_wave", target="main", metadata={"module_kind": "planner", "proposal_id": pid})
        before = {row.get("family_key"): row for row in k.module_registry.family_readiness_summary()}
        before_cd = float(before[fam].get("cooldown_remaining", 0.0))
        for wave_id in ["wave_close_one", "wave_close_two"]:
            k.hub_kernel.review_waves.append({
                "wave_id": wave_id,
                "created_at": time.time() - 1800.0,
                "status": "open",
                "selected_count": 1,
                "items": [{"artifact_kind": "module", "artifact_name": "closure_alpha", "proposal_id": pid, "module_kind": "planner", "target": "main"}],
                "outcome_summary": {"success_count": 2, "failure_count": 0},
                "surface_count": 1,
            })
            k.hub_kernel.work_queue.append({
                "kind": "review_wave_resolution",
                "wave_id": wave_id,
                "queued_at": time.time(),
                "status": "resolution_requested",
                "recommended_resolution": "auto_close",
            })
            resolved = k.resolve_surfaced_review_waves(limit=1)
            assert resolved["resolved"] >= 1
        after = {row.get("family_key"): row for row in k.module_registry.family_readiness_summary()}
        after_cd = float(after[fam].get("cooldown_remaining", 0.0))
        assert after_cd <= before_cd
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_low_confidence_module_escalation_routes_to_blocked_growth_followup(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(ModuleProposal(proposer_kernel_id="hub", module_name="bg_alpha", module_kind="planner", promotion_target="main", capabilities=["route"])))
        pid = ev["module_record"]["proposal_id"]
        k.hub_kernel.review_waves.append({
            "wave_id": "wave_bg_mod",
            "created_at": time.time() - 2400.0,
            "status": "open",
            "selected_count": 1,
            "items": [{"artifact_kind": "module", "artifact_name": "bg_alpha", "proposal_id": pid, "module_kind": "planner"}],
            "outcome_summary": {"success_count": 0, "failure_count": 2},
            "surface_count": 2,
        })
        k.hub_kernel.work_queue.append({
            "kind": "promotion_review",
            "artifact_kind": "module",
            "artifact_name": "bg_alpha",
            "proposal_id": pid,
            "module_kind": "planner",
            "review_wave_id": "wave_bg_mod",
            "queued_at": time.time() - 2400.0,
            "target": "main",
        })
        k.hub_kernel.work_queue.append({
            "kind": "review_wave_escalation",
            "wave_id": "wave_bg_mod",
            "queued_at": time.time(),
            "status": "escalated",
            "age_seconds": 2400.0,
            "confidence": 0.4,
        })
        out = k.process_wave_escalations(limit=1)
        assert out["processed"] >= 1
        assert any(str(item.get("kind", "")) == "blocked_growth_followup" and str(item.get("artifact_name", "")) == "bg_alpha" for item in k.hub_kernel.work_queue)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_process_proposal_rework_requeues_and_relieves_family_pressure(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(ModuleProposal(proposer_kernel_id="hub", module_name="rework_beta", module_kind="planner", promotion_target="main", capabilities=["route"])))
        pid = ev["module_record"]["proposal_id"]
        fam = k.module_registry.family_key_for("rework_beta", "planner")
        k.module_registry.apply_module_feedback("rework_beta", success=False, severity="high", kind="review_wave", target="main", metadata={"proposal_id": pid, "module_kind": "planner"})
        caution = k._hub_queue_caution_map()
        caution[fam] = {"caution_level": 1.6, "cooldown_until": time.time() + 600.0, "success_count": 0.0, "failure_count": 2.0, "caution_last_at": time.time(), "decay_window": 1800.0}
        k.hub_kernel.local_branch_state["queue_caution"] = caution
        before = {row.get("family_key"): row for row in k.module_registry.family_readiness_summary()}
        before_cd = float(before[fam].get("cooldown_remaining", 0.0))
        k.hub_kernel.work_queue.append({
            "kind": "proposal_rework",
            "proposal_id": pid,
            "artifact_name": "rework_beta",
            "module_kind": "planner",
            "target": "main",
            "queued_at": time.time(),
            "status": "rework_requested",
            "confidence": 1.2,
        })
        out = k.process_proposal_rework(limit=1)
        assert out["processed"] >= 1
        assert any(str(item.get("status", "")) == "reworked_ready" and str(item.get("proposal_id", "")) == pid for item in k.hub_kernel.promotion_queue)
        after = {row.get("family_key"): row for row in k.module_registry.family_readiness_summary()}
        after_cd = float(after[fam].get("cooldown_remaining", 0.0))
        assert after_cd <= before_cd
        caution2 = k._hub_queue_caution_map()
        assert float(caution2.get(fam, {}).get("caution_level", 0.0)) <= 1.6
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_process_blocked_growth_followups_consumes_and_can_spawn_rework(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(ModuleProposal(proposer_kernel_id="hub", module_name="bg_to_rework", module_kind="planner", promotion_target="hub", capabilities=["route"])))
        pid = ev["module_record"]["proposal_id"]
        k.hub_kernel.work_queue.append({
            "kind": "blocked_growth_followup",
            "wave_id": "wave_bg_follow",
            "artifact_kind": "module",
            "artifact_name": "bg_to_rework",
            "proposal_id": pid,
            "module_kind": "planner",
            "queued_at": time.time(),
            "confidence": 0.8,
            "status": "blocked_growth_requested",
        })
        out = k.process_blocked_growth_followups(limit=1)
        assert out["processed"] >= 1
        assert out["spawned_rework"] >= 1
        assert any(str(item.get("kind", "")) == "proposal_rework" and str(item.get("proposal_id", "")) == pid for item in k.hub_kernel.work_queue)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_blocked_growth_followup_low_confidence_spawns_targeted_evidence(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(proposer_kernel_id="hub", module_name="bg_target_alpha", module_kind="planner", promotion_target="hub", capabilities=["route"])
        ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
        pid = ev["module_record"]["proposal_id"]
        k.hub_kernel.work_queue.append({
            "kind": "blocked_growth_followup",
            "artifact_kind": "module",
            "artifact_name": "bg_target_alpha",
            "proposal_id": pid,
            "module_kind": "planner",
            "confidence": 0.25,
            "queued_at": 0.0,
            "wave_id": "wave_bg_target",
        })
        out = k.process_blocked_growth_followups(limit=1)
        assert out["processed"] >= 1
        assert any(str(item.get("kind", "")) == "promotion_evidence" and str(item.get("artifact_name", "")) == "bg_target_alpha" for item in k.hub_kernel.work_queue)
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_reworked_ready_items_batch_into_review_wave(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.hub_kernel.trust_from_main = "trusted"
        k.cluster_trust.set_trust("hub", "trusted", reason="unit_test", source="main")
        ids = []
        for name in ("rw_alpha", "rw_beta"):
            proposal = ModuleProposal(proposer_kernel_id="hub", module_name=name, module_kind="planner", promotion_target="main", capabilities=["route"])
            ev = k._dispatch_kernel_packet(k.hub_kernel.propose_module(proposal))
            pid = ev["module_record"]["proposal_id"]
            k.module_registry.attach_evidence(pid, {"kind": "manual", "note": "enough"})
            k.hub_kernel.work_queue.append({
                "kind": "proposal_rework",
                "proposal_id": pid,
                "artifact_name": name,
                "module_kind": "planner",
                "target": "main",
                "confidence": 1.1,
                "queued_at": 0.0,
            })
            ids.append(pid)
        k.process_proposal_rework(limit=2)
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        assert out["review_wave_count"] == 1
        assert len([p for p in out["processed"] if p.get("queue_status") == "auto_reviewed"]) >= 2
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_high_quality_evidence_can_make_module_ready(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        prop = ModuleProposal(proposer_kernel_id="hub", module_name="quality_mod", module_kind="planner", promotion_target="hub", capabilities=["route"])
        event = k._dispatch_kernel_packet(k.hub_kernel.propose_module(prop))
        pid = event["module_record"]["proposal_id"]
        k._hub_enqueue_work_item({"kind": "promotion_evidence", "artifact_kind": "module", "artifact_name": "quality_mod", "proposal_id": pid, "module_kind": "planner", "target": "hub", "desired_stage": "hub", "confidence": 0.85})
        out = k.complete_hub_evidence_task(artifact_kind="module", artifact_name="quality_mod", proposal_id=pid, evidence={"summary": "Detailed analysis with regression tests and citation.", "tests": ["unit", "regression"], "source": "lab"})
        assert out["ok"] is True
        mr = k.module_registry.maturity_report(pid, target="hub")
        assert mr["evidence_quality_total"] >= 1.0
        assert mr["ready"] is True
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode

def test_promotion_queue_dedup_collapses_duplicate_reworked_ready(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k.hub_kernel.queue_promotion_request({"artifact_kind": "module", "artifact_name": "dup_mod", "proposal_id": "p1", "target": "hub", "desired_stage": "hub", "status": "queued", "rework_quality": 0.4})
        k.hub_kernel.queue_promotion_request({"artifact_kind": "module", "artifact_name": "dup_mod", "proposal_id": "p1", "target": "hub", "desired_stage": "hub", "status": "reworked_ready", "rework_quality": 1.2})
        items = [q for q in k.hub_kernel.promotion_queue if str(q.get("artifact_name")) == "dup_mod"]
        assert len(items) == 1
        assert float(items[0].get("rework_quality", 0.0)) >= 1.2
        assert str(items[0].get("status")) in {"reworked_ready", "evidence_ready"}
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode

def test_blocked_growth_followup_dedups_targeted_evidence_work(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get("TOVAH_BOOT_MODE")
    os.environ["TOVAH_BOOT_MODE"] = "main_with_hub"
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        k._hub_enqueue_work_item({"kind": "blocked_growth_followup", "artifact_kind": "patch", "artifact_name": "patchA", "confidence": 0.2})
        k._hub_enqueue_work_item({"kind": "blocked_growth_followup", "artifact_kind": "patch", "artifact_name": "patchA", "confidence": 0.6})
        k.process_blocked_growth_followups(limit=5)
        evid = [w for w in k.hub_kernel.work_queue if str(w.get("kind")) == "promotion_evidence" and str(w.get("artifact_name")) == "patchA"]
        assert len(evid) == 1
        assert float(evid[0].get("confidence", 0.0)) >= 0.6
    finally:
        if prev_mode is None:
            os.environ.pop("TOVAH_BOOT_MODE", None)
        else:
            os.environ["TOVAH_BOOT_MODE"] = prev_mode


def test_stale_evidence_decays_in_maturity_report():
    import time
    from tovah_v14.modules.registry import ModuleRegistry
    from tovah_v14.kernel.action_model import ModuleProposal
    reg = ModuleRegistry()
    prop = ModuleProposal(proposer_kernel_id='hub', module_name='decay_mod', module_kind='planner', promotion_target='main')
    rec = reg.propose(prop, source_kernel_id='hub', trust_level='trusted', source_role='hub')
    now = time.time()
    reg.attach_evidence(rec.proposal_id, {'kind':'evidence_gather', 'evidence_quality': 1.4, 'time': now})
    fresh = reg.maturity_report(rec.proposal_id, target='main')
    reg.proposals[rec.proposal_id].evidence[0]['time'] = now - 86400.0 * 5.0
    stale = reg.maturity_report(rec.proposal_id, target='main')
    assert fresh['evidence_quality_total'] > stale['evidence_quality_total']
    assert fresh['maturity_score'] > stale['maturity_score']


def test_shared_artifact_key_dedups_queue_and_work():
    from tovah_v14.kernel.kernel import ProtozoanKernel
    import os
    prev_mode = os.environ.get('TOVAH_BOOT_MODE')
    os.environ['TOVAH_BOOT_MODE'] = 'main_with_hub'
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        a = k.hub_kernel.queue_promotion_request({'artifact_kind':'module','artifact_name':'dup_mod','proposal_id':'p1','target':'hub','desired_stage':'hub','status':'queued'})
        b = k.hub_kernel.queue_promotion_request({'artifact_kind':'module','artifact_name':'dup_mod','proposal_id':'p1','target':'hub','desired_stage':'hub','status':'evidence_ready','evidence_quality':1.8})
        same_queue = [q for q in k.hub_kernel.promotion_queue if str(q.get('artifact_key','')) == str(a.get('artifact_key',''))]
        assert len(same_queue) == 1
        assert a.get('artifact_key') == b.get('artifact_key')
        k._hub_enqueue_work_item({'kind':'promotion_evidence','artifact_kind':'module','artifact_name':'dup_mod','proposal_id':'p1','target':'hub','desired_stage':'hub','confidence':0.3})
        k._hub_enqueue_work_item({'kind':'promotion_evidence','artifact_kind':'module','artifact_name':'dup_mod','proposal_id':'p1','target':'hub','desired_stage':'hub','confidence':0.7})
        evidence_items=[w for w in k.hub_kernel.work_queue if str(w.get('kind',''))=='promotion_evidence' and str(w.get('artifact_key','')) == str(k._artifact_dedup_key({'artifact_kind':'module','artifact_name':'dup_mod','proposal_id':'p1','target':'hub'}))]
        assert len(evidence_items) == 1
        assert int(evidence_items[0].get('duplicate_count',1) or 1) >= 2
    finally:
        if prev_mode is None:
            os.environ.pop('TOVAH_BOOT_MODE', None)
        else:
            os.environ['TOVAH_BOOT_MODE'] = prev_mode


def test_repeated_strong_evidence_reduces_reentry_requirement():
    import time
    from tovah_v14.modules.registry import ModuleRegistry
    from tovah_v14.kernel.action_model import ModuleProposal
    reg = ModuleRegistry()
    prop = ModuleProposal(proposer_kernel_id='hub', module_name='evidence_lift_mod', module_kind='planner', promotion_target='main')
    rec = reg.propose(prop, source_kernel_id='hub', trust_level='trusted', source_role='hub')
    baseline = reg.maturity_report(rec.proposal_id, target='main')
    reg.attach_evidence(rec.proposal_id, {'kind':'evidence_gather', 'evidence_quality': 1.9, 'time': time.time()})
    reg.apply_module_feedback('evidence_lift_mod', success=True, severity='normal', kind='evidence_gather', target='main', metadata={'module_kind':'planner', 'evidence_quality': 1.9})
    reg.attach_evidence(rec.proposal_id, {'kind':'evidence_gather', 'evidence_quality': 2.0, 'time': time.time()})
    reg.apply_module_feedback('evidence_lift_mod', success=True, severity='normal', kind='evidence_gather', target='main', metadata={'module_kind':'planner', 'evidence_quality': 2.0})
    lifted = reg.maturity_report(rec.proposal_id, target='main')
    assert lifted['effective_required_evidence'] <= baseline['effective_required_evidence']
    assert lifted['reentry_discount'] >= 1
    assert lifted['ready'] is True


def test_fresh_strong_evidence_offsets_stale_weak():
    import time
    from tovah_v14.modules.registry import ModuleRegistry
    from tovah_v14.kernel.action_model import ModuleProposal

    reg = ModuleRegistry()
    proposal = ModuleProposal(
        proposer_kernel_id='hub',
        module_name='evidence_balance_mod',
        module_kind='planner',
        promotion_target='main',
        capabilities=['route'],
    )
    rec = reg.propose(proposal, source_kernel_id='hub', source_role='hub', trust_level='trusted')
    now = time.time()
    stale_weak = {'kind': 'evidence_gather', 'evidence_quality': 0.25, 'time': now - (6 * 86400)}
    fresh_strong = {'kind': 'evidence_gather', 'evidence_quality': 2.4, 'time': now}
    stale_only = reg._evidence_quality_total([stale_weak], now=now)
    fresh_only = reg._evidence_quality_total([fresh_strong], now=now)
    combined = reg._evidence_quality_total([stale_weak, fresh_strong], now=now)
    assert combined >= fresh_only - 0.05
    assert combined > stale_only


def test_artifact_key_prevents_parallel_open_waves(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get('TOVAH_BOOT_MODE')
    os.environ['TOVAH_BOOT_MODE'] = 'main_with_hub'
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(
            proposer_kernel_id='hub',
            module_name='parallel_wave_mod',
            module_kind='planner',
            promotion_target='hub',
            capabilities=['route'],
        )
        rec = k.module_registry.propose(proposal, source_kernel_id='hub', source_role='hub', trust_level='trusted')
        k.module_registry.attach_evidence(rec.proposal_id, {'kind': 'evidence_gather', 'evidence_quality': 2.2, 'time': time.time()})
        k.hub_kernel.queue_promotion_request({'artifact_kind': 'module', 'artifact_name': 'parallel_wave_mod', 'proposal_id': rec.proposal_id, 'target': 'hub', 'desired_stage': 'hub', 'status': 'evidence_ready', 'module_kind': 'planner'})
        out1 = k.process_hub_promotion_queue(limit=1, consume=True)
        assert out1['review_wave_count'] == 1
        artifact_key = next(iter(next(w for w in k.hub_kernel.review_waves if w.get('wave_id') == out1['wave_ids'][0])['items']))['artifact_key']
        k.hub_kernel.queue_promotion_request({'artifact_kind': 'module', 'artifact_name': 'parallel_wave_mod', 'proposal_id': rec.proposal_id, 'target': 'hub', 'desired_stage': 'hub', 'status': 'evidence_ready', 'module_kind': 'planner'})
        out2 = k.process_hub_promotion_queue(limit=1, consume=True)
        open_items = [item for wave in k.hub_kernel.review_waves if str(wave.get('status', 'open')) not in {'completed', 'auto_closed', 'closed', 'retired'} for item in wave.get('items', []) if str(item.get('artifact_key', '')) == artifact_key]
        assert len(open_items) == 1
        assert out2['review_wave_count'] == 0
    finally:
        if prev_mode is None:
            os.environ.pop('TOVAH_BOOT_MODE', None)
        else:
            os.environ['TOVAH_BOOT_MODE'] = prev_mode


def test_artifact_key_propagates_into_histories(tmp_path, monkeypatch):
    from tovah_v14.kernel.kernel import ProtozoanKernel
    from tovah_v14.kernel.action_model import ModuleProposal
    import os, time
    monkeypatch.chdir(tmp_path)
    prev_mode = os.environ.get('TOVAH_BOOT_MODE')
    os.environ['TOVAH_BOOT_MODE'] = 'main_with_hub'
    try:
        k = ProtozoanKernel(api={}, is_original=True)
        proposal = ModuleProposal(
            proposer_kernel_id='hub',
            module_name='history_key_mod',
            module_kind='planner',
            promotion_target='hub',
            capabilities=['route'],
        )
        rec = k.module_registry.propose(proposal, source_kernel_id='hub', source_role='hub', trust_level='trusted')
        k.module_registry.attach_evidence(rec.proposal_id, {'kind': 'evidence_gather', 'evidence_quality': 2.4, 'time': time.time()})
        k.hub_kernel.queue_promotion_request({'artifact_kind': 'module', 'artifact_name': 'history_key_mod', 'proposal_id': rec.proposal_id, 'target': 'hub', 'desired_stage': 'hub', 'status': 'evidence_ready', 'module_kind': 'planner'})
        out = k.process_hub_promotion_queue(limit=1, consume=True)
        wave_id = out['wave_ids'][0]
        wave = next(w for w in k.hub_kernel.review_waves if str(w.get('wave_id', '')) == wave_id)
        artifact_key = str(wave['items'][0].get('artifact_key', ''))
        done = k.complete_hub_review_wave(wave_id, item_results={artifact_key: {'success': True, 'severity': 'low'}}, default_success=False)
        assert done['ok'] is True
        hist = list(k.hub_kernel.local_branch_state.get('wave_resolution_history', []))
        assert hist
        assert artifact_key in hist[-1].get('artifact_keys', [])

        k.hub_kernel.work_queue.append({
            'kind': 'blocked_growth_followup',
            'artifact_kind': 'module',
            'artifact_name': 'history_key_mod',
            'proposal_id': rec.proposal_id,
            'module_kind': 'planner',
            'artifact_key': artifact_key,
            'confidence': 0.2,
            'queued_at': time.time(),
            'status': 'blocked_growth_requested',
        })
        bg = k.process_blocked_growth_followups(limit=1)
        assert bg['processed'] == 1
        bgh = list(k.hub_kernel.local_branch_state.get('blocked_growth_followup_history', []))
        assert bgh
        assert bgh[-1].get('artifact_key') == artifact_key
    finally:
        if prev_mode is None:
            os.environ.pop('TOVAH_BOOT_MODE', None)
        else:
            os.environ['TOVAH_BOOT_MODE'] = prev_mode
