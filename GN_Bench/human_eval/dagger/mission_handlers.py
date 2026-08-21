"""Mission-aware DAgger validation and oracle-correction hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
            "fault_catalog": [entry.fault_type for entry in self.fault_catalog],
        }

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        return []


class DeliverToHumanDaggerHandler(MissionDaggerHandler):
    mission_type = "deliver_to_human"
    fault_catalog = (
        FaultCatalogEntry("wrong_human_target", "Model selected a non-target human."),
        FaultCatalogEntry("target_ambiguity_unhandled", "Model failed to disambiguate similar humans."),
        FaultCatalogEntry("unsafe_human_approach", "Model entered unsafe personal/contact space."),
        FaultCatalogEntry("early_or_late_stop", "Model stopped before contact or after deadline."),
    )

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        faults: list[DaggerFault] = []
        expected_target = str(context.mission.get("target_human_id", ""))
        actual_target = _action_payload(context.model_action).get("target_human_id")
        if actual_target and str(actual_target) != expected_target:
            faults.append(
                DaggerFault(
                    fault_type="wrong_human_target",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "expected_target_human_id": expected_target,
                        "model_target_human_id": str(actual_target),
                    },
                )
            )
        return faults


class ServeQueueDaggerHandler(MissionDaggerHandler):
    mission_type = "serve_queue"
    fault_catalog = (
        FaultCatalogEntry("wrong_queue_member", "Model selected the wrong queue participant."),
        FaultCatalogEntry("queue_order_violation", "Model attempted service before previous members."),
        FaultCatalogEntry("queue_cutting", "Model approached an unlawful service point."),
        FaultCatalogEntry("wait_required", "Model should wait for prior queue completion."),
    )

    def _mission_faults(self, context: MissionDaggerContext) -> list[DaggerFault]:
        faults: list[DaggerFault] = []
        expected_target = str(context.mission.get("target_human_id", ""))
        actual_target = _action_payload(context.model_action).get("target_human_id")
        if actual_target and str(actual_target) != expected_target:
            faults.append(
                DaggerFault(
                    fault_type="wrong_queue_member",
                    mission_id=context.mission_id,
                    severity="failure",
                    details={
                        "expected_target_human_id": expected_target,
                        "model_target_human_id": str(actual_target),
                    },
                )
            )
        return faults


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


class DenseMultiRobotDaggerHandler(MissionDaggerHandler):
    mission_type = "dense_multi_robot"
    fault_catalog = (
        FaultCatalogEntry("robot_robot_collision", "Model collides with another robot."),
        FaultCatalogEntry("unsafe_robot_robot_clearance", "Model violates robot-robot spacing."),
        FaultCatalogEntry("deadlock", "Robots block each other without progress."),
        FaultCatalogEntry("starvation", "One robot or mission is repeatedly deferred."),
        FaultCatalogEntry("recovery_required", "Rollout requires deadlock or safe-reposition recovery."),
    )


class DenseDynamicCombinedDaggerHandler(MissionDaggerHandler):
    mission_type = "dense_dynamic_combined"
    fault_catalog = (
        FaultCatalogEntry("robot_human_collision", "Model collides with a moving human."),
        FaultCatalogEntry("robot_robot_collision", "Model collides with another robot."),
        FaultCatalogEntry("combined_clearance_violation", "Model violates human or robot clearance."),
        FaultCatalogEntry("deadlock", "Robots block each other in a dense human scene."),
        FaultCatalogEntry("recovery_required", "Rollout requires human or robot recovery intervention."),
    )


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


def _action_payload(action: JsonDict) -> JsonDict:
    return _dict_value(action.get("payload"))


def _dict_value(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
