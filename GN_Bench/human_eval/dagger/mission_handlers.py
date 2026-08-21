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
            "fault_catalog": [entry.fault_type for entry in self.fault_catalog],
        }

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        return []


class DeliverToHumanDaggerHandler(MissionDaggerHandler):
    mission_type = "deliver_to_human"
    fault_catalog = (
        FaultCatalogEntry("missing_human_target", "Model action omitted the human target."),
        FaultCatalogEntry("wrong_human_target", "Model selected a non-target human."),
        FaultCatalogEntry("target_ambiguity_unhandled", "Model failed to disambiguate similar humans."),
        FaultCatalogEntry("unsafe_human_approach", "Model entered unsafe personal/contact space."),
        FaultCatalogEntry("early_or_late_stop", "Model stopped before contact or after deadline."),
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
        FaultCatalogEntry("missing_queue_member", "Model action omitted the queue participant."),
        FaultCatalogEntry("wrong_queue_member", "Model selected the wrong queue participant."),
        FaultCatalogEntry("queue_order_violation", "Model attempted service before previous members."),
        FaultCatalogEntry("queue_cutting", "Model approached an unlawful service point."),
        FaultCatalogEntry("wait_required", "Model should wait for prior queue completion."),
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
        FaultCatalogEntry("personal_space_violation", "Model violates L1 personal-space rules."),
        FaultCatalogEntry("pedestrian_yield_failure", "Model violates L2 yield timing."),
        FaultCatalogEntry("group_integrity_violation", "Model crosses an L3 protected group region."),
        FaultCatalogEntry("queue_order_violation", "Model violates L4 queue-order constraints."),
        FaultCatalogEntry("collision_or_near_miss", "Model collides or enters unsafe clearance."),
    )


class HumanGuidedUncertainRegionDaggerHandler(MissionDaggerHandler):
    mission_type = "human_guided_uncertain_region"
    fault_catalog = (
        FaultCatalogEntry("guidance_not_requested", "Model moves through uncertainty without asking."),
        FaultCatalogEntry("wrong_informant", "Model asks the wrong human for guidance."),
        FaultCatalogEntry("guidance_wait_violation", "Model does not stop while waiting."),
        FaultCatalogEntry("resolved_target_ignored", "Model ignores the clarified target."),
    )


class MissionStreamDaggerHandler(MissionDaggerHandler):
    mission_type = "mission_stream"
    fault_catalog = (
        FaultCatalogEntry("priority_inversion", "Model dispatches child missions out of priority order."),
        FaultCatalogEntry("missed_release", "Model ignores a released child mission."),
        FaultCatalogEntry("wrong_child_assignment", "Model assigns the wrong robot or child mission."),
        FaultCatalogEntry("missing_eos", "Model completes a child mission without end-of-sequence closeout."),
    )


class DenseDynamicHumansDaggerHandler(MissionDaggerHandler):
    mission_type = "dense_dynamic_humans"
    fault_catalog = (
        FaultCatalogEntry("robot_human_collision", "Model collides with a moving human."),
        FaultCatalogEntry("unsafe_robot_human_clearance", "Model violates required robot-human clearance."),
        FaultCatalogEntry("stuck_robot", "Robot makes no progress while a path remains available."),
        FaultCatalogEntry("unnecessary_freeze", "Robot waits without a blocking human."),
        FaultCatalogEntry("recovery_required", "Rollout requires local replan or safe reposition."),
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


class DenseMultiRobotDaggerHandler(MissionDaggerHandler):
    mission_type = "dense_multi_robot"
    fault_catalog = (
        FaultCatalogEntry("robot_robot_collision", "Model collides with another robot."),
        FaultCatalogEntry("unsafe_robot_robot_clearance", "Model violates robot-robot spacing."),
        FaultCatalogEntry("deadlock", "Robots block each other without progress."),
        FaultCatalogEntry("starvation", "One robot or mission is repeatedly deferred."),
        FaultCatalogEntry("recovery_required", "Rollout requires deadlock or safe-reposition recovery."),
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
        return faults


class DenseDynamicCombinedDaggerHandler(MissionDaggerHandler):
    mission_type = "dense_dynamic_combined"
    fault_catalog = (
        FaultCatalogEntry("robot_human_collision", "Model collides with a moving human."),
        FaultCatalogEntry("robot_robot_collision", "Model collides with another robot."),
        FaultCatalogEntry("combined_clearance_violation", "Model violates human or robot clearance."),
        FaultCatalogEntry("deadlock", "Robots block each other in a dense human scene."),
        FaultCatalogEntry("recovery_required", "Rollout requires human or robot recovery intervention."),
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
        return faults


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


def _dense_common_faults(context: MissionDaggerContext) -> list[DaggerFault]:
    metrics = context.metrics_snapshot
    faults: list[DaggerFault] = []
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
