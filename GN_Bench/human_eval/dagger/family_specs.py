"""Mission-family DAgger contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .schema import JsonDict


@dataclass(frozen=True)
class MissionFamilyDaggerSpec:
    """Inspectable contract for one human-centric DAgger mission family."""

    mission_type: str
    handler_key: str
    primary_faults: tuple[str, ...]
    metric_signals: tuple[str, ...]
    oracle_actions: tuple[str, ...]
    recovery_counts: tuple[str, ...] = ()
    remaining_work: tuple[str, ...] = field(default_factory=tuple)

    def to_json_dict(self) -> JsonDict:
        return asdict(self)


def build_default_family_specs() -> dict[str, MissionFamilyDaggerSpec]:
    specs = [
        MissionFamilyDaggerSpec(
            mission_type="deliver_to_human",
            handler_key="deliver_to_human",
            primary_faults=(
                "missing_human_target",
                "wrong_human_target",
                "target_ambiguity_unhandled",
                "unsafe_human_approach",
                "early_or_late_stop",
            ),
            metric_signals=(
                "wrong_human_contact_count",
                "non_target_physical_clearance_respected",
                "personal_space_respected",
                "deadline_success",
                "human_identification_*",
            ),
            oracle_actions=("assign_mission",),
            remaining_work=(
                "Replace metadata-only target similarity with rendered-image similarity when available.",
                "Attach low-level path actions after native simulator oracle is wired.",
            ),
        ),
        MissionFamilyDaggerSpec(
            mission_type="navigate_with_social_constraints",
            handler_key="navigate_with_social_constraints",
            primary_faults=(
                "goal_not_reached",
                "personal_space_violation",
                "pedestrian_yield_failure",
                "group_integrity_violation",
                "queue_order_violation",
                "collision_or_near_miss",
            ),
            metric_signals=(
                "goal_reached",
                "personal_space_violation_count",
                "pedestrian_yield_violation_count",
                "group_region_violation_count",
                "queue_order_violation_count",
                "collision_count",
            ),
            oracle_actions=("set_subgoal",),
            remaining_work=(
                "Generate law-specific subgoals for queue-tail, yield, and group-region repair.",
            ),
        ),
        MissionFamilyDaggerSpec(
            mission_type="human_guided_uncertain_region",
            handler_key="human_guided_uncertain_region",
            primary_faults=(
                "guidance_not_requested",
                "wrong_informant",
                "guidance_wait_violation",
                "resolved_target_ignored",
            ),
            metric_signals=(
                "guidance_requested",
                "guidance_stop_required",
                "guidance_stop_verified",
                "uncertainty_resolved",
                "resolved_target_reached",
            ),
            oracle_actions=("interact", "no_op", "assign_mission"),
            remaining_work=(
                "Attach post-response path oracle to the resolved target after simulator oracle is available.",
            ),
        ),
        MissionFamilyDaggerSpec(
            mission_type="serve_queue",
            handler_key="serve_queue",
            primary_faults=(
                "missing_queue_member",
                "wrong_queue_member",
                "queue_order_violation",
                "queue_cutting",
                "wait_required",
            ),
            metric_signals=(
                "previous_queue_missions_completed",
                "queue_order_preserved",
                "queue_position",
                "target_human_id",
                "serve_queue_target_identification_difficulty",
            ),
            oracle_actions=("no_op", "assign_mission"),
            remaining_work=(
                "Add queue-tail navigation repair for social-law queue cases.",
            ),
        ),
        MissionFamilyDaggerSpec(
            mission_type="mission_stream",
            handler_key="mission_stream",
            primary_faults=(
                "priority_inversion",
                "missed_release",
                "wrong_child_assignment",
                "missing_eos",
                "terminal_goal_missed",
            ),
            metric_signals=(
                "released_child_mission_count",
                "assigned_child_mission_count",
                "completed_child_mission_count",
                "eos_child_mission_count",
                "priority_order_respected",
                "stream_terminal_goals_reached",
            ),
            oracle_actions=("assign_mission", "robot_eos"),
            remaining_work=(
                "Sequence child-level DAgger samples into one stream-level training record.",
            ),
        ),
        MissionFamilyDaggerSpec(
            mission_type="dense_dynamic_humans",
            handler_key="dense_dynamic_humans",
            primary_faults=(
                "active_robot_goal_missed",
                "robot_human_collision",
                "unsafe_robot_human_clearance",
                "human_motion_stalled",
                "stuck_robot",
                "unnecessary_freeze",
                "recovery_required",
            ),
            metric_signals=(
                "all_active_robot_goal_regions_reached",
                "collision_count",
                "dense_robot_human_clearance_policy_respected",
                "humans_keep_moving_until_robot_completion",
                "robot_wait_violation_count",
                "corner_case_recovery_count",
            ),
            oracle_actions=("set_subgoal",),
            recovery_counts=(
                "stuck_recovery_count",
                "safe_reposition_count",
                "teleport_recovery_count",
                "collision_recovery_count",
            ),
            remaining_work=(
                "Wire live stuck/collision detection to simulator recovery events.",
            ),
        ),
        MissionFamilyDaggerSpec(
            mission_type="dense_multi_robot",
            handler_key="dense_multi_robot",
            primary_faults=(
                "active_robot_goal_missed",
                "robot_robot_collision",
                "unsafe_robot_robot_clearance",
                "deadlock",
                "starvation",
                "stuck_robot",
                "unnecessary_freeze",
                "recovery_required",
            ),
            metric_signals=(
                "all_active_robot_goal_regions_reached",
                "no_robot_robot_collision",
                "robot_robot_clearance_policy_respected",
                "robot_robot_deadlock_recovery_count",
                "robot_wait_violation_count",
                "starvation_count",
            ),
            oracle_actions=("set_subgoal",),
            recovery_counts=(
                "stuck_recovery_count",
                "safe_reposition_count",
                "teleport_recovery_count",
                "collision_recovery_count",
            ),
            remaining_work=(
                "Promote per-robot priority/starvation events from diagnostics into simulator rollout state.",
            ),
        ),
        MissionFamilyDaggerSpec(
            mission_type="dense_dynamic_combined",
            handler_key="dense_dynamic_combined",
            primary_faults=(
                "active_robot_goal_missed",
                "robot_human_collision",
                "robot_robot_collision",
                "combined_clearance_violation",
                "human_motion_stalled",
                "deadlock",
                "starvation",
                "stuck_robot",
                "unnecessary_freeze",
                "recovery_required",
            ),
            metric_signals=(
                "all_active_robot_goal_regions_reached",
                "collision_count",
                "dense_robot_human_clearance_policy_respected",
                "no_robot_robot_collision",
                "robot_robot_clearance_policy_respected",
                "humans_keep_moving_until_robot_completion",
                "robot_robot_deadlock_recovery_count",
            ),
            oracle_actions=("set_subgoal",),
            recovery_counts=(
                "stuck_recovery_count",
                "safe_reposition_count",
                "teleport_recovery_count",
                "collision_recovery_count",
            ),
            remaining_work=(
                "Split combined failures into per-robot, per-human, and per-pair records for dense rollout debugging.",
            ),
        ),
    ]
    return {spec.mission_type: spec for spec in specs}
