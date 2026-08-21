"""Mission-aware DAgger validation and oracle-correction hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .recovery import build_recovery_strategy, recovery_counts_from_metrics
from .schema import DaggerFault, DaggerOracleCorrection, DaggerValidationResult, JsonDict


@dataclass(frozen=True)
class FaultCatalogEntry:
    """Documented fault type for one mission family."""

    fault_type: str
    description: str
    default_recovery: str = ""
    severity: str = "diagnostic"


@dataclass(frozen=True)
class MissionDaggerContext:
    """Inputs available to a mission-specific DAgger handler."""

    episode_id: str
    scenario_id: str
    payload: JsonDict
    mission: JsonDict
    observation: JsonDict
    model_action: JsonDict
    metrics_snapshot: JsonDict = field(default_factory=dict)

    @property
    def mission_id(self) -> str:
        return str(self.mission.get("mission_id", ""))

    @property
    def mission_type(self) -> str:
        return str(self.mission.get("mission_type", ""))


class MissionDaggerHandler:
    """Base mission handler.

    Subclasses should keep their outputs in the shared DAgger schema and only
    add mission-specific validation/oracle details where the family needs them.
    """

    mission_type = "default"
    fault_catalog: tuple[FaultCatalogEntry, ...] = ()

    def validate(self, context: MissionDaggerContext) -> DaggerValidationResult:
        faults = _generic_action_faults(context)
        faults.extend(self._mission_faults(context))
        return DaggerValidationResult(
            is_valid=not faults,
            faults=faults,
            details={
                "handler": type(self).__name__,
                "catalog_fault_types": [entry.fault_type for entry in self.fault_catalog],
            },
        )

    def oracle_correction(
        self,
        context: MissionDaggerContext,
        validation: DaggerValidationResult,
    ) -> DaggerOracleCorrection:
        return DaggerOracleCorrection(
            oracle_intent=context.mission_type or "human_mission",
            action_type="assign_mission",
            payload=_oracle_assignment_payload(context),
            recovery=_oracle_recovery(context, validation),
            source=f"{type(self).__name__}.oracle_correction",
            trainable=bool(validation.faults),
        )

    def mission_context(self, context: MissionDaggerContext) -> JsonDict:
        metadata = _dict_value(context.mission.get("metadata"))
        return {
            "mission_id": context.mission_id,
            "mission_type": context.mission_type,
            "assigned_robot_id": str(context.mission.get("assigned_robot_id", "")),
            "target_human_id": str(context.mission.get("target_human_id", "")),
            "target_region_id": str(context.mission.get("target_region_id", "")),
            "release_time": _optional_number(context.mission.get("release_time")),
            "deadline": _optional_number(context.mission.get("deadline")),
            "priority": _optional_number(context.mission.get("priority")),
            "planned_goal_world": metadata.get("planned_goal_world"),
            "queue_id": str(metadata.get("queue_id", "")),
            "queue_index": _optional_number(metadata.get("queue_index")),
            "queue_position": _optional_number(metadata.get("queue_position")),
            "previous_queue_human_ids": _string_list(metadata.get("previous_queue_human_ids")),
            "queue_order": _string_list(metadata.get("queue_order")),
            "informant_human_id": _human_guided_informant_id(context.mission),
            "mission_stream_parent_id": str(metadata.get("mission_stream_parent_id", "")),
            "child_mission_ids": _string_list(metadata.get("child_mission_ids")),
            "mission_stack_index": _optional_number(metadata.get("mission_stack_index")),
            "fault_catalog": [entry.fault_type for entry in self.fault_catalog],
        }

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        return []


class DeliverToHumanDaggerHandler(MissionDaggerHandler):
    mission_type = "deliver_to_human"
    fault_catalog = (
        FaultCatalogEntry("missing_human_target", "Model action omitted the human target.", "assign_target_human", "failure"),
        FaultCatalogEntry("wrong_human_target", "Model selected a non-target human.", "assign_target_human", "failure"),
        FaultCatalogEntry("target_ambiguity_unhandled", "Model failed to disambiguate similar humans.", "increase_identity_evidence", "diagnostic"),
        FaultCatalogEntry("unsafe_human_approach", "Model entered unsafe personal/contact space.", "safe_reposition", "failure"),
        FaultCatalogEntry("early_or_late_stop", "Model stopped before contact or after deadline.", "resume_or_stop_at_target", "failure"),
    )

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        faults = _target_human_faults(
            context,
            wrong_fault_type="wrong_human_target",
            missing_fault_type="missing_human_target",
        )
        metrics = context.metrics_snapshot
        if _metric_positive(metrics, "wrong_human_contact_count"):
            faults.append(
                DaggerFault(
                    fault_type="unsafe_human_approach",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "wrong_human_contact_count": int(
                            _number(metrics.get("wrong_human_contact_count"), 0.0)
                        )
                    },
                )
            )
        if (
            metrics.get("non_target_physical_clearance_respected") is False
            or metrics.get("personal_space_respected") is False
            or _metric_positive(metrics, "personal_space_violation_count")
        ):
            faults.append(
                DaggerFault(
                    fault_type="unsafe_human_approach",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "non_target_physical_clearance_respected": metrics.get(
                            "non_target_physical_clearance_respected"
                        ),
                        "personal_space_respected": metrics.get("personal_space_respected"),
                        "personal_space_violation_count": int(
                            _number(metrics.get("personal_space_violation_count"), 0.0)
                        ),
                    },
                )
            )
        if metrics.get("deadline_success") is False:
            faults.append(
                DaggerFault(
                    fault_type="early_or_late_stop",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "deadline_success": False,
                        "deadline_s": metrics.get("deadline_s"),
                        "completion_time_s": metrics.get("completion_time_s"),
                    },
                )
            )
        if _high_ambiguity_without_target(context):
            faults.append(
                DaggerFault(
                    fault_type="target_ambiguity_unhandled",
                    mission_id=context.mission_id,
                    severity="diagnostic",
                    details=_difficulty_details(metrics),
                )
            )
        return faults


class ServeQueueDaggerHandler(MissionDaggerHandler):
    mission_type = "serve_queue"
    fault_catalog = (
        FaultCatalogEntry("missing_queue_member", "Model action omitted the queue participant.", "assign_queue_member", "failure"),
        FaultCatalogEntry("wrong_queue_member", "Model selected the wrong queue participant.", "assign_queue_member", "failure"),
        FaultCatalogEntry("queue_order_violation", "Model attempted service before previous members.", "wait_for_previous_member", "failure"),
        FaultCatalogEntry("queue_cutting", "Model approached an unlawful service point.", "move_to_queue_tail", "failure"),
        FaultCatalogEntry("wait_required", "Model should wait for prior queue completion.", "no_op_until_previous_complete", "failure"),
    )

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        faults: list[DaggerFault] = []
        attempts_service = _action_attempts_service(context.model_action)
        if attempts_service:
            faults.extend(
                _target_human_faults(
                    context,
                    wrong_fault_type="wrong_queue_member",
                    missing_fault_type="missing_queue_member",
                )
            )
        metrics = context.metrics_snapshot
        if metrics.get("previous_queue_missions_completed") is False and attempts_service:
            faults.append(
                DaggerFault(
                    fault_type="wait_required",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={"previous_queue_missions_completed": False},
                )
            )
        if metrics.get("queue_order_preserved") is False and attempts_service:
            faults.append(
                DaggerFault(
                    fault_type="queue_order_violation",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={"queue_order_preserved": False},
                )
            )
        if _previous_queue_missions_pending(context) and attempts_service:
            faults.append(
                DaggerFault(
                    fault_type="wait_required",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "pending_previous_mission_ids": _pending_previous_queue_mission_ids(
                            context
                        )
                    },
                )
            )
        queue_cutting_details = _queue_cutting_details(context)
        if queue_cutting_details and attempts_service:
            faults.append(
                DaggerFault(
                    fault_type="queue_cutting",
                    mission_id=context.mission_id,
                    severity="failure",
                    details=queue_cutting_details,
                )
            )
        return faults

    def oracle_correction(
        self,
        context: MissionDaggerContext,
        validation: DaggerValidationResult,
    ) -> DaggerOracleCorrection:
        if any(fault.fault_type == "wait_required" for fault in validation.faults):
            return DaggerOracleCorrection(
                oracle_intent="serve_queue_wait",
                action_type="no_op",
                payload={
                    "mission_id": context.mission_id,
                    "mission_type": context.mission_type,
                    "wait_for_mission_ids": _pending_previous_queue_mission_ids(context),
                    "route_mode": "oracle_human_dagger",
                },
                recovery=_oracle_recovery(context, validation),
                source=f"{type(self).__name__}.oracle_correction",
                trainable=True,
            )
        return super().oracle_correction(context, validation)


class NavigateWithSocialConstraintsDaggerHandler(MissionDaggerHandler):
    mission_type = "navigate_with_social_constraints"
    fault_catalog = (
        FaultCatalogEntry("goal_not_reached", "Model did not reach the social navigation goal.", "set_socially_valid_subgoal", "failure"),
        FaultCatalogEntry("personal_space_violation", "Model violates L1 personal-space rules.", "safe_reposition", "failure"),
        FaultCatalogEntry("pedestrian_yield_failure", "Model violates L2 yield timing.", "yield_or_retime_path", "failure"),
        FaultCatalogEntry("group_integrity_violation", "Model crosses an L3 protected group region.", "route_around_group", "failure"),
        FaultCatalogEntry("queue_order_violation", "Model violates L4 queue-order constraints.", "move_to_queue_tail", "failure"),
        FaultCatalogEntry("collision_or_near_miss", "Model collides or enters unsafe clearance.", "collision_recovery", "failure"),
    )

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        metrics = context.metrics_snapshot
        faults: list[DaggerFault] = []
        if metrics.get("goal_reached") is False:
            faults.append(
                DaggerFault(
                    fault_type="goal_not_reached",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "minimum_goal_distance_m": metrics.get("minimum_goal_distance_m"),
                        "goal_threshold_m": metrics.get("goal_threshold_m"),
                    },
                )
            )
        if (
            metrics.get("personal_space_respected") is False
            or _metric_positive(metrics, "personal_space_violation_count")
        ):
            faults.append(
                DaggerFault(
                    fault_type="personal_space_violation",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "personal_space_violation_count": int(
                            _number(metrics.get("personal_space_violation_count"), 0.0)
                        ),
                        "personal_space_violation_human_ids": metrics.get(
                            "personal_space_violation_human_ids",
                            [],
                        ),
                    },
                )
            )
        if (
            metrics.get("pedestrian_yield_respected") is False
            or _metric_positive(metrics, "pedestrian_yield_violation_count")
        ):
            faults.append(
                DaggerFault(
                    fault_type="pedestrian_yield_failure",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "pedestrian_yield_violation_count": int(
                            _number(metrics.get("pedestrian_yield_violation_count"), 0.0)
                        ),
                        "pedestrian_yield_min_time_gap_s": metrics.get(
                            "pedestrian_yield_min_time_gap_s"
                        ),
                        "pedestrian_yield_min_distance_m": metrics.get(
                            "pedestrian_yield_min_distance_m"
                        ),
                    },
                )
            )
        if (
            metrics.get("group_integrity_respected") is False
            or _metric_positive(metrics, "group_region_violation_count")
        ):
            faults.append(
                DaggerFault(
                    fault_type="group_integrity_violation",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "group_region_violation_count": int(
                            _number(metrics.get("group_region_violation_count"), 0.0)
                        ),
                        "group_region_min_clearance_m": metrics.get(
                            "group_region_min_clearance_m"
                        ),
                    },
                )
            )
        if (
            metrics.get("queue_order_respected") is False
            or _metric_positive(metrics, "queue_order_violation_count")
        ):
            faults.append(
                DaggerFault(
                    fault_type="queue_order_violation",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "queue_order_violation_count": int(
                            _number(metrics.get("queue_order_violation_count"), 0.0)
                        ),
                        "queue_terminal_tail_distance_m": metrics.get(
                            "queue_terminal_tail_distance_m"
                        ),
                        "queue_terminal_service_distance_m": metrics.get(
                            "queue_terminal_service_distance_m"
                        ),
                    },
                )
            )
        if _metric_positive(metrics, "collision_count") or metrics.get("collision_free") is False:
            faults.append(
                DaggerFault(
                    fault_type="collision_or_near_miss",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "collision_count": int(_number(metrics.get("collision_count"), 0.0)),
                        "min_clearance_m": metrics.get("min_clearance_m"),
                    },
                )
            )
        return faults

    def oracle_correction(
        self,
        context: MissionDaggerContext,
        validation: DaggerValidationResult,
    ) -> DaggerOracleCorrection:
        payload = _oracle_assignment_payload(context)
        payload["active_social_law_ids"] = _string_list(
            context.metrics_snapshot.get("active_social_law_ids")
        )
        payload["social_repair_faults"] = [
            fault.fault_type for fault in validation.faults
        ]
        return DaggerOracleCorrection(
            oracle_intent="social_navigation_repair",
            action_type="set_subgoal",
            payload=payload,
            recovery=_oracle_recovery(context, validation),
            source=f"{type(self).__name__}.oracle_correction",
            trainable=bool(validation.faults),
        )


class HumanGuidedUncertainRegionDaggerHandler(MissionDaggerHandler):
    mission_type = "human_guided_uncertain_region"
    fault_catalog = (
        FaultCatalogEntry("guidance_not_requested", "Model moves through uncertainty without asking.", "request_human_guidance", "failure"),
        FaultCatalogEntry("wrong_informant", "Model asks the wrong human for guidance.", "request_correct_informant", "failure"),
        FaultCatalogEntry("guidance_wait_violation", "Model does not stop while waiting.", "no_op_until_guidance_response", "failure"),
        FaultCatalogEntry("resolved_target_ignored", "Model ignores the clarified target.", "set_resolved_target_subgoal", "failure"),
    )

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        metrics = context.metrics_snapshot
        faults: list[DaggerFault] = []
        informant_id = _human_guided_informant_id(context.mission)
        action_target = _action_human_target(context.model_action)
        if metrics.get("guidance_requested") is False and not _action_requests_guidance(
            context.model_action
        ):
            faults.append(
                DaggerFault(
                    fault_type="guidance_not_requested",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={"informant_human_id": informant_id},
                )
            )
        if action_target and informant_id and action_target != informant_id:
            faults.append(
                DaggerFault(
                    fault_type="wrong_informant",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "expected_informant_human_id": informant_id,
                        "model_target_human_id": action_target,
                    },
                )
            )
        if (
            metrics.get("guidance_stop_required") is True
            and metrics.get("guidance_stop_verified") is False
        ):
            faults.append(
                DaggerFault(
                    fault_type="guidance_wait_violation",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "guidance_stop_interval_s": metrics.get("guidance_stop_interval_s"),
                        "guidance_stop_max_displacement_m": metrics.get(
                            "guidance_stop_max_displacement_m"
                        ),
                    },
                )
            )
        if (
            metrics.get("uncertainty_resolved") is True
            and metrics.get("resolved_target_reached") is False
        ) or metrics.get("resolved_target_ignored") is True:
            faults.append(
                DaggerFault(
                    fault_type="resolved_target_ignored",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "resolved_target_xy": metrics.get("resolved_target_xy"),
                        "minimum_goal_distance_m": metrics.get("minimum_goal_distance_m"),
                    },
                )
            )
        return faults

    def oracle_correction(
        self,
        context: MissionDaggerContext,
        validation: DaggerValidationResult,
    ) -> DaggerOracleCorrection:
        fault_types = {fault.fault_type for fault in validation.faults}
        if "guidance_not_requested" in fault_types or "wrong_informant" in fault_types:
            informant_id = _human_guided_informant_id(context.mission)
            return DaggerOracleCorrection(
                oracle_intent="request_human_guidance",
                action_type="interact",
                payload={
                    "mission_id": context.mission_id,
                    "mission_type": context.mission_type,
                    "interaction": "request_guidance",
                    "target_human_id": informant_id,
                    "route_mode": "oracle_human_dagger",
                },
                recovery=_oracle_recovery(context, validation),
                source=f"{type(self).__name__}.oracle_correction",
                trainable=True,
            )
        if "guidance_wait_violation" in fault_types:
            return DaggerOracleCorrection(
                oracle_intent="wait_for_human_guidance",
                action_type="no_op",
                payload={
                    "mission_id": context.mission_id,
                    "mission_type": context.mission_type,
                    "route_mode": "oracle_human_dagger",
                },
                recovery=_oracle_recovery(context, validation),
                source=f"{type(self).__name__}.oracle_correction",
                trainable=True,
            )
        return super().oracle_correction(context, validation)


class MissionStreamDaggerHandler(MissionDaggerHandler):
    mission_type = "mission_stream"
    fault_catalog = (
        FaultCatalogEntry("priority_inversion", "Model dispatches child missions out of priority order.", "dispatch_highest_priority_child", "failure"),
        FaultCatalogEntry("missed_release", "Model ignores a released child mission.", "assign_released_child", "failure"),
        FaultCatalogEntry("wrong_child_assignment", "Model assigns the wrong robot or child mission.", "assign_correct_child", "failure"),
        FaultCatalogEntry("missing_eos", "Model completes a child mission without end-of-sequence closeout.", "emit_robot_eos", "failure"),
        FaultCatalogEntry("terminal_goal_missed", "Mission stream terminal goal was not reached.", "set_terminal_goal_subgoal", "failure"),
    )

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        metrics = context.metrics_snapshot
        if _dict_value(context.mission.get("metadata")).get("mission_stream_parent_id"):
            return _mission_stream_child_faults(context)

        faults: list[DaggerFault] = []
        child_count = int(_number(metrics.get("child_mission_count"), 0.0))
        expected_child_count = int(_number(metrics.get("expected_child_mission_count"), 0.0))
        if metrics.get("priority_order_respected") is False:
            faults.append(
                DaggerFault(
                    fault_type="priority_inversion",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "child_timing_checks": metrics.get("child_timing_checks", []),
                        "dispatch_record_count_matches": metrics.get(
                            "dispatch_record_count_matches"
                        ),
                    },
                )
            )
        if (
            metrics.get("released_missions_completed") is False
            or (
                child_count > 0
                and int(_number(metrics.get("released_child_mission_count"), 0.0))
                < child_count
            )
        ):
            faults.append(
                DaggerFault(
                    fault_type="missed_release",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "released_child_mission_count": metrics.get(
                            "released_child_mission_count"
                        ),
                        "child_mission_count": child_count,
                    },
                )
            )
        if (
            metrics.get("child_ids_present") is False
            or metrics.get("child_count_matches_config") is False
            or metrics.get("dispatch_record_count_matches") is False
            or (
                child_count > 0
                and int(_number(metrics.get("assigned_child_mission_count"), 0.0))
                < child_count
            )
        ):
            faults.append(
                DaggerFault(
                    fault_type="wrong_child_assignment",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "child_ids_present": metrics.get("child_ids_present"),
                        "child_count_matches_config": metrics.get(
                            "child_count_matches_config"
                        ),
                        "assigned_child_mission_count": metrics.get(
                            "assigned_child_mission_count"
                        ),
                        "expected_child_mission_count": expected_child_count,
                    },
                )
            )
        if child_count > 0 and int(_number(metrics.get("eos_child_mission_count"), 0.0)) < child_count:
            faults.append(
                DaggerFault(
                    fault_type="missing_eos",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "eos_child_mission_count": metrics.get("eos_child_mission_count"),
                        "child_mission_count": child_count,
                    },
                )
            )
        if metrics.get("stream_terminal_goals_reached") is False:
            faults.append(
                DaggerFault(
                    fault_type="terminal_goal_missed",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "stream_terminal_goal_max_distance_m": metrics.get(
                            "stream_terminal_goal_max_distance_m"
                        ),
                        "stream_terminal_goal_checks": metrics.get(
                            "stream_terminal_goal_checks",
                            [],
                        ),
                    },
                )
            )
        return faults

    def oracle_correction(
        self,
        context: MissionDaggerContext,
        validation: DaggerValidationResult,
    ) -> DaggerOracleCorrection:
        if any(fault.fault_type == "missing_eos" for fault in validation.faults):
            return DaggerOracleCorrection(
                oracle_intent="mission_stream_eos",
                action_type="robot_eos",
                payload={
                    "mission_id": context.mission_id,
                    "mission_type": context.mission_type,
                    "route_mode": "oracle_human_dagger",
                },
                recovery=_oracle_recovery(context, validation),
                source=f"{type(self).__name__}.oracle_correction",
                trainable=True,
            )
        return super().oracle_correction(context, validation)


class DenseDynamicHumansDaggerHandler(MissionDaggerHandler):
    mission_type = "dense_dynamic_humans"
    fault_catalog = (
        FaultCatalogEntry("active_robot_goal_missed", "One or more active robots did not reach their goal.", "set_active_robot_subgoals", "failure"),
        FaultCatalogEntry("robot_human_collision", "Model collides with a moving human.", "collision_recovery", "failure"),
        FaultCatalogEntry("unsafe_robot_human_clearance", "Model violates required robot-human clearance.", "safe_reposition", "failure"),
        FaultCatalogEntry("human_motion_stalled", "Moving humans stop before robot completion.", "repair_dynamic_human_schedule", "diagnostic"),
        FaultCatalogEntry("stuck_robot", "Robot makes no progress while a path remains available.", "stuck_recovery", "failure"),
        FaultCatalogEntry("unnecessary_freeze", "Robot waits without a blocking human.", "resume_motion_or_replan", "failure"),
        FaultCatalogEntry("recovery_required", "Rollout requires local replan or safe reposition.", "record_recovery_counts", "diagnostic"),
    )

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        faults = _dense_common_faults(context)
        metrics = context.metrics_snapshot
        if _metric_positive(metrics, "collision_count"):
            faults.append(
                DaggerFault(
                    fault_type="robot_human_collision",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={"collision_count": int(_number(metrics.get("collision_count"), 0.0))},
                )
            )
        if (
            metrics.get("dense_robot_human_clearance_policy_respected") is False
            or _metric_positive(metrics, "dense_nominal_robot_human_conflict_count")
        ):
            faults.append(
                DaggerFault(
                    fault_type="unsafe_robot_human_clearance",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "dense_robot_human_clearance_policy_respected": metrics.get(
                            "dense_robot_human_clearance_policy_respected"
                        ),
                        "dense_nominal_robot_human_conflict_count": int(
                            _number(
                                metrics.get("dense_nominal_robot_human_conflict_count"),
                                0.0,
                            )
                        ),
                    },
                )
            )
        return faults

    def oracle_correction(
        self,
        context: MissionDaggerContext,
        validation: DaggerValidationResult,
    ) -> DaggerOracleCorrection:
        return _dense_oracle_correction(self, context, validation)


class DenseMultiRobotDaggerHandler(MissionDaggerHandler):
    mission_type = "dense_multi_robot"
    fault_catalog = (
        FaultCatalogEntry("active_robot_goal_missed", "One or more active robots did not reach their goal.", "set_active_robot_subgoals", "failure"),
        FaultCatalogEntry("robot_robot_collision", "Model collides with another robot.", "collision_recovery", "failure"),
        FaultCatalogEntry("unsafe_robot_robot_clearance", "Model violates robot-robot spacing.", "safe_reposition", "failure"),
        FaultCatalogEntry("deadlock", "Robots block each other without progress.", "stuck_recovery", "failure"),
        FaultCatalogEntry("starvation", "One robot or mission is repeatedly deferred.", "rebalance_robot_priorities", "failure"),
        FaultCatalogEntry("stuck_robot", "Robot makes no progress while a path remains available.", "stuck_recovery", "failure"),
        FaultCatalogEntry("unnecessary_freeze", "Robot waits without a blocking robot.", "resume_motion_or_replan", "failure"),
        FaultCatalogEntry("recovery_required", "Rollout requires deadlock or safe-reposition recovery.", "record_recovery_counts", "diagnostic"),
    )

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        faults = _dense_common_faults(context)
        metrics = context.metrics_snapshot
        if (
            metrics.get("no_robot_robot_collision") is False
            or _metric_positive(metrics, "robot_robot_collision_count")
        ):
            faults.append(
                DaggerFault(
                    fault_type="robot_robot_collision",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "no_robot_robot_collision": metrics.get("no_robot_robot_collision"),
                        "robot_robot_collision_count": int(
                            _number(metrics.get("robot_robot_collision_count"), 0.0)
                        ),
                    },
                )
            )
        if metrics.get("robot_robot_clearance_policy_respected") is False:
            faults.append(
                DaggerFault(
                    fault_type="unsafe_robot_robot_clearance",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={"robot_robot_clearance_policy_respected": False},
                )
            )
        if _metric_positive(metrics, "robot_robot_deadlock_recovery_count"):
            faults.append(
                DaggerFault(
                    fault_type="deadlock",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "robot_robot_deadlock_recovery_count": int(
                            _number(metrics.get("robot_robot_deadlock_recovery_count"), 0.0)
                        )
                    },
                )
            )
        if _metric_positive(metrics, "starvation_count") or _metric_positive(
            metrics,
            "mission_starvation_count",
        ):
            faults.append(
                DaggerFault(
                    fault_type="starvation",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "starvation_count": int(
                            _number(metrics.get("starvation_count"), 0.0)
                        ),
                        "mission_starvation_count": int(
                            _number(metrics.get("mission_starvation_count"), 0.0)
                        ),
                    },
                )
            )
        return faults

    def oracle_correction(
        self,
        context: MissionDaggerContext,
        validation: DaggerValidationResult,
    ) -> DaggerOracleCorrection:
        return _dense_oracle_correction(self, context, validation)


class DenseDynamicCombinedDaggerHandler(MissionDaggerHandler):
    mission_type = "dense_dynamic_combined"
    fault_catalog = (
        FaultCatalogEntry("active_robot_goal_missed", "One or more active robots did not reach their goal.", "set_active_robot_subgoals", "failure"),
        FaultCatalogEntry("robot_human_collision", "Model collides with a moving human.", "collision_recovery", "failure"),
        FaultCatalogEntry("robot_robot_collision", "Model collides with another robot.", "collision_recovery", "failure"),
        FaultCatalogEntry("combined_clearance_violation", "Model violates human or robot clearance.", "safe_reposition", "failure"),
        FaultCatalogEntry("human_motion_stalled", "Moving humans stop before robot completion.", "repair_dynamic_human_schedule", "diagnostic"),
        FaultCatalogEntry("deadlock", "Robots block each other in a dense human scene.", "stuck_recovery", "failure"),
        FaultCatalogEntry("starvation", "One robot or mission is repeatedly deferred.", "rebalance_robot_priorities", "failure"),
        FaultCatalogEntry("stuck_robot", "Robot makes no progress while a path remains available.", "stuck_recovery", "failure"),
        FaultCatalogEntry("unnecessary_freeze", "Robot waits without a blocking human or robot.", "resume_motion_or_replan", "failure"),
        FaultCatalogEntry("recovery_required", "Rollout requires human or robot recovery intervention.", "record_recovery_counts", "diagnostic"),
    )

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        faults = _dense_common_faults(context)
        metrics = context.metrics_snapshot
        if _metric_positive(metrics, "collision_count"):
            faults.append(
                DaggerFault(
                    fault_type="robot_human_collision",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={"collision_count": int(_number(metrics.get("collision_count"), 0.0))},
                )
            )
        if (
            metrics.get("no_robot_robot_collision") is False
            or _metric_positive(metrics, "robot_robot_collision_count")
        ):
            faults.append(
                DaggerFault(
                    fault_type="robot_robot_collision",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "no_robot_robot_collision": metrics.get("no_robot_robot_collision"),
                        "robot_robot_collision_count": int(
                            _number(metrics.get("robot_robot_collision_count"), 0.0)
                        ),
                    },
                )
            )
        if (
            metrics.get("dense_robot_human_clearance_policy_respected") is False
            or metrics.get("robot_robot_clearance_policy_respected") is False
            or _metric_positive(metrics, "dense_nominal_robot_human_conflict_count")
        ):
            faults.append(
                DaggerFault(
                    fault_type="combined_clearance_violation",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "dense_robot_human_clearance_policy_respected": metrics.get(
                            "dense_robot_human_clearance_policy_respected"
                        ),
                        "robot_robot_clearance_policy_respected": metrics.get(
                            "robot_robot_clearance_policy_respected"
                        ),
                        "dense_nominal_robot_human_conflict_count": int(
                            _number(
                                metrics.get("dense_nominal_robot_human_conflict_count"),
                                0.0,
                            )
                        ),
                    },
                )
            )
        if _metric_positive(metrics, "robot_robot_deadlock_recovery_count"):
            faults.append(
                DaggerFault(
                    fault_type="deadlock",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "robot_robot_deadlock_recovery_count": int(
                            _number(metrics.get("robot_robot_deadlock_recovery_count"), 0.0)
                        )
                    },
                )
            )
        if _metric_positive(metrics, "starvation_count") or _metric_positive(
            metrics,
            "mission_starvation_count",
        ):
            faults.append(
                DaggerFault(
                    fault_type="starvation",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "starvation_count": int(
                            _number(metrics.get("starvation_count"), 0.0)
                        ),
                        "mission_starvation_count": int(
                            _number(metrics.get("mission_starvation_count"), 0.0)
                        ),
                    },
                )
            )
        return faults

    def oracle_correction(
        self,
        context: MissionDaggerContext,
        validation: DaggerValidationResult,
    ) -> DaggerOracleCorrection:
        return _dense_oracle_correction(self, context, validation)


def build_default_mission_registry() -> dict[str, MissionDaggerHandler]:
    handlers: list[MissionDaggerHandler] = [
        DeliverToHumanDaggerHandler(),
        NavigateWithSocialConstraintsDaggerHandler(),
        HumanGuidedUncertainRegionDaggerHandler(),
        ServeQueueDaggerHandler(),
        MissionStreamDaggerHandler(),
        DenseDynamicHumansDaggerHandler(),
        DenseMultiRobotDaggerHandler(),
        DenseDynamicCombinedDaggerHandler(),
    ]
    return {handler.mission_type: handler for handler in handlers}


def _generic_action_faults(context: MissionDaggerContext) -> list[DaggerFault]:
    faults: list[DaggerFault] = []
    model_action = context.model_action
    action_type = str(model_action.get("action_type", ""))
    if not action_type:
        faults.append(
            DaggerFault(
                fault_type="missing_action_type",
                mission_id=context.mission_id,
                severity="failure",
                details={"model_action": model_action},
            )
        )
    payload = _action_payload(model_action)
    action_mission_id = str(payload.get("mission_id", ""))
    if action_mission_id and action_mission_id != context.mission_id:
        faults.append(
            DaggerFault(
                fault_type="wrong_mission",
                mission_id=context.mission_id,
                severity="failure",
                details={
                    "expected_mission_id": context.mission_id,
                    "model_mission_id": action_mission_id,
                },
            )
        )
    return faults


def _target_human_faults(
    context: MissionDaggerContext,
    *,
    wrong_fault_type: str,
    missing_fault_type: str,
) -> list[DaggerFault]:
    expected_target = str(context.mission.get("target_human_id", ""))
    payload = _action_payload(context.model_action)
    actual_target = payload.get("target_human_id")
    if not actual_target:
        return [
            DaggerFault(
                fault_type=missing_fault_type,
                mission_id=context.mission_id,
                severity="failure",
                details={"expected_target_human_id": expected_target},
            )
        ]
    if str(actual_target) == expected_target:
        return []
    return [
        DaggerFault(
            fault_type=wrong_fault_type,
            mission_id=context.mission_id,
            severity="failure",
            details={
                "expected_target_human_id": expected_target,
                "model_target_human_id": str(actual_target),
            },
        )
    ]


def _high_ambiguity_without_target(context: MissionDaggerContext) -> bool:
    payload = _action_payload(context.model_action)
    if payload.get("target_human_id"):
        return False
    difficulty = _number(context.metrics_snapshot.get("human_identification_difficulty"), 0.0)
    if difficulty <= 0.0:
        difficulty = _number(
            context.metrics_snapshot.get(
                f"{context.mission_type}_target_identification_difficulty"
            ),
            0.0,
        )
    return difficulty >= 0.70 or _metric_positive(
        context.metrics_snapshot,
        "human_identification_confuser_count",
    )


def _difficulty_details(metrics: JsonDict) -> JsonDict:
    return {
        "human_identification_difficulty": metrics.get("human_identification_difficulty"),
        "human_identification_confuser_count": metrics.get(
            "human_identification_confuser_count"
        ),
        "human_identification_best_confuser_id": metrics.get(
            "human_identification_best_confuser_id"
        ),
    }


def _previous_queue_missions_pending(context: MissionDaggerContext) -> bool:
    return bool(_pending_previous_queue_mission_ids(context))


def _pending_previous_queue_mission_ids(context: MissionDaggerContext) -> list[str]:
    previous_ids = _previous_queue_mission_ids(context)
    completed_ids = {
        str(mission_id)
        for mission_id in context.observation.get("completed_mission_ids", [])
        if str(mission_id)
    }
    return [mission_id for mission_id in previous_ids if mission_id not in completed_ids]


def _previous_queue_mission_ids(context: MissionDaggerContext) -> list[str]:
    metadata = _dict_value(context.mission.get("metadata"))
    queue_id = str(metadata.get("queue_id", ""))
    queue_index = _optional_number(metadata.get("queue_index"))
    previous_human_ids = set(_string_list(metadata.get("previous_queue_human_ids")))
    previous_ids: list[str] = []
    for mission in _dict_list(context.payload.get("missions")):
        if str(mission.get("mission_id", "")) == context.mission_id:
            continue
        mission_metadata = _dict_value(mission.get("metadata"))
        same_queue = queue_id and str(mission_metadata.get("queue_id", "")) == queue_id
        mission_index = _optional_number(mission_metadata.get("queue_index"))
        lower_index = queue_index is not None and mission_index is not None and mission_index < queue_index
        previous_human = str(mission.get("target_human_id", "")) in previous_human_ids
        if same_queue and (lower_index or previous_human):
            mission_id = str(mission.get("mission_id", ""))
            if mission_id:
                previous_ids.append(mission_id)
    return sorted(set(previous_ids))


def _queue_cutting_details(context: MissionDaggerContext) -> JsonDict:
    payload = _action_payload(context.model_action)
    details: JsonDict = {}
    metadata = _dict_value(context.mission.get("metadata"))
    expected_queue_id = str(metadata.get("queue_id", ""))
    model_queue_id = str(payload.get("queue_id", ""))
    if model_queue_id and expected_queue_id and model_queue_id != expected_queue_id:
        details["expected_queue_id"] = expected_queue_id
        details["model_queue_id"] = model_queue_id

    expected_position = _optional_number(metadata.get("queue_position"))
    model_position = _optional_number(payload.get("queue_position"))
    if (
        expected_position is not None
        and model_position is not None
        and int(model_position) != int(expected_position)
    ):
        details["expected_queue_position"] = expected_position
        details["model_queue_position"] = model_position
    return details


def _action_attempts_service(action: JsonDict) -> bool:
    action_type = str(action.get("action_type", ""))
    if action_type in {"no_op", "wait", "hold"}:
        return False
    payload = _action_payload(action)
    if payload.get("target_human_id") or payload.get("target_region_id"):
        return True
    return action_type in {"assign_mission", "reassign_mission", "set_subgoal", "interact"}


def _human_guided_informant_id(mission: JsonDict) -> str:
    metadata = _dict_value(mission.get("metadata"))
    human_guidance = _dict_value(metadata.get("human_guidance"))
    informant_id = human_guidance.get("informant_human_id") or metadata.get(
        "informant_human_id"
    )
    if informant_id:
        return str(informant_id)
    active_human_ids = _string_list(metadata.get("active_human_ids"))
    return active_human_ids[0] if active_human_ids else ""


def _action_human_target(action: JsonDict) -> str:
    payload = _action_payload(action)
    return str(
        payload.get("target_human_id")
        or payload.get("informant_human_id")
        or payload.get("human_id")
        or ""
    )


def _action_requests_guidance(action: JsonDict) -> bool:
    payload = _action_payload(action)
    action_type = str(action.get("action_type", ""))
    interaction = str(
        payload.get("interaction")
        or payload.get("intent")
        or payload.get("request_type")
        or ""
    ).lower()
    return action_type == "interact" and any(
        token in interaction for token in ("guide", "guidance", "clarify", "ask")
    )


def _mission_stream_child_faults(context: MissionDaggerContext) -> list[DaggerFault]:
    metrics = context.metrics_snapshot
    faults: list[DaggerFault] = []
    if metrics.get("release_event_present") is False:
        faults.append(
            DaggerFault(
                fault_type="missed_release",
                mission_id=context.mission_id,
                severity="failure",
                details={
                    "release_event_time_s": metrics.get("release_event_time_s"),
                    "expected_release_time_s": metrics.get("expected_release_time_s"),
                },
            )
        )
    if (
        metrics.get("assignment_event_present") is False
        or metrics.get("assignment_time_matches_expected") is False
    ):
        faults.append(
            DaggerFault(
                fault_type="wrong_child_assignment",
                mission_id=context.mission_id,
                severity="failure",
                details={
                    "assignment_event_time_s": metrics.get("assignment_event_time_s"),
                    "expected_assignment_time_s": metrics.get("expected_assignment_time_s"),
                },
            )
        )
    if (
        metrics.get("timing_respected") is False
        or metrics.get("stream_event_order_respected") is False
    ):
        faults.append(
            DaggerFault(
                fault_type="priority_inversion",
                mission_id=context.mission_id,
                severity="failure",
                details={
                    "stream_event_order_respected": metrics.get(
                        "stream_event_order_respected"
                    ),
                    "timing_respected": metrics.get("timing_respected"),
                },
            )
        )
    if metrics.get("eos_event_present") is False or metrics.get("eos_time_matches_expected") is False:
        faults.append(
            DaggerFault(
                fault_type="missing_eos",
                mission_id=context.mission_id,
                severity="failure",
                details={
                    "eos_time_s": metrics.get("eos_time_s"),
                    "expected_eos_time_s": metrics.get("expected_eos_time_s"),
                },
            )
        )
    if metrics.get("goal_reached") is False:
        faults.append(
            DaggerFault(
                fault_type="terminal_goal_missed",
                mission_id=context.mission_id,
                severity="failure",
                details={
                    "minimum_goal_distance_m": metrics.get("minimum_goal_distance_m"),
                    "goal_threshold_m": metrics.get("goal_threshold_m"),
                },
            )
        )
    return faults


def _dense_common_faults(context: MissionDaggerContext) -> list[DaggerFault]:
    metrics = context.metrics_snapshot
    faults: list[DaggerFault] = []
    if metrics.get("all_active_robot_goal_regions_reached") is False:
        faults.append(
            DaggerFault(
                fault_type="active_robot_goal_missed",
                mission_id=context.mission_id,
                severity="failure",
                details={
                    "active_robot_goal_checks": metrics.get(
                        "active_robot_goal_checks",
                        [],
                    )
                },
            )
        )
    if metrics.get("humans_keep_moving_until_robot_completion") is False:
        faults.append(
            DaggerFault(
                fault_type="human_motion_stalled",
                mission_id=context.mission_id,
                severity="diagnostic",
                details={
                    "dense_human_activity_checks": metrics.get(
                        "dense_human_activity_checks",
                        [],
                    )
                },
            )
        )
    if _metric_positive(metrics, "robot_stalled_recovery_count") or _metric_positive(
        metrics,
        "stuck_recovery_count",
    ):
        faults.append(
            DaggerFault(
                fault_type="stuck_robot",
                mission_id=context.mission_id,
                severity="failure",
                details={
                    "robot_stalled_recovery_count": int(
                        _number(metrics.get("robot_stalled_recovery_count"), 0.0)
                    ),
                    "stuck_recovery_count": int(
                        _number(metrics.get("stuck_recovery_count"), 0.0)
                    ),
                },
            )
        )
    if _metric_positive(metrics, "robot_wait_violation_count"):
        faults.append(
            DaggerFault(
                fault_type="unnecessary_freeze",
                mission_id=context.mission_id,
                severity="failure",
                details={
                    "robot_wait_violation_count": int(
                        _number(metrics.get("robot_wait_violation_count"), 0.0)
                    )
                },
            )
        )
    recovery_counts = recovery_counts_from_metrics(metrics)
    if sum(recovery_counts.values()) > 0 or _metric_positive(metrics, "corner_case_recovery_count"):
        faults.append(
            DaggerFault(
                fault_type="recovery_required",
                mission_id=context.mission_id,
                severity="diagnostic",
                details={
                    "recovery_counts": recovery_counts,
                    "corner_case_recovery_count": int(
                        _number(metrics.get("corner_case_recovery_count"), 0.0)
                    ),
                },
            )
        )
    return faults


def _dense_oracle_correction(
    handler: MissionDaggerHandler,
    context: MissionDaggerContext,
    validation: DaggerValidationResult,
) -> DaggerOracleCorrection:
    payload = _oracle_assignment_payload(context)
    metadata = _dict_value(context.mission.get("metadata"))
    payload["active_robot_ids"] = _string_list(metadata.get("active_robot_ids"))
    payload["subgoals_by_robot"] = _dict_value(metadata.get("planned_goal_world_by_robot"))
    if not payload["subgoals_by_robot"] and "subgoal" in payload:
        assigned_robot_id = str(context.mission.get("assigned_robot_id", ""))
        if assigned_robot_id:
            payload["subgoals_by_robot"] = {assigned_robot_id: payload["subgoal"]}
    payload["dense_repair_faults"] = [fault.fault_type for fault in validation.faults]
    return DaggerOracleCorrection(
        oracle_intent=f"{context.mission_type}_repair",
        action_type="set_subgoal",
        payload=payload,
        recovery=_oracle_recovery(context, validation),
        source=f"{type(handler).__name__}.oracle_correction",
        trainable=bool(validation.faults),
    )


def _oracle_assignment_payload(context: MissionDaggerContext) -> JsonDict:
    metadata = _dict_value(context.mission.get("metadata"))
    payload: JsonDict = {
        "mission_id": context.mission_id,
        "mission_type": context.mission_type,
        "target_human_id": str(context.mission.get("target_human_id", "")),
        "target_region_id": str(context.mission.get("target_region_id", "")),
        "route_mode": "oracle_human_dagger",
    }
    if "planned_goal_world" in metadata:
        payload["subgoal"] = metadata["planned_goal_world"]
    return payload


def _oracle_recovery(
    context: MissionDaggerContext,
    validation: DaggerValidationResult,
) -> JsonDict | None:
    strategy = build_recovery_strategy(validation.faults, context.metrics_snapshot)
    return strategy.to_json_dict() if strategy is not None else None


def _action_payload(action: JsonDict) -> JsonDict:
    return _dict_value(action.get("payload"))


def _dict_list(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_value(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _metric_positive(metrics: JsonDict, key: str) -> bool:
    return _number(metrics.get(key), 0.0) > 0.0
