"""Deterministic evaluator skeleton for human-centric GN-Bench episodes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .scenario_adapter import HumanCentricEpisode


JsonDict = dict[str, Any]
SUPPORTED_SOCIAL_NAVIGATION_LAWS = {
    "L1_personal_space",
    "L2_non_obstruction_yield",
    "L3_group_integrity",
    "L4_queue_order",
}
GROUP_REGION_CLEARANCE_TOLERANCE_M = 0.05
STREAM_TIMING_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class ReplayResult:
    """Output record for one deterministic sync replay."""

    episode_id: str
    metrics: JsonDict = field(default_factory=dict)
    events: list[JsonDict] = field(default_factory=list)
    mission_results: list[JsonDict] = field(default_factory=list)
    success: bool | None = None
    # TODO: Add violation traces for paper figures.


class HumanCentricEvaluator:
    """Evaluates NavDP-generated scenarios inside GN-Bench.

    Purpose: this is the authoritative evaluation path for benchmark reporting
    and RL reward computation. It should remain deterministic for sync-mode
    replay.
    """

    def replay(self, episode: HumanCentricEpisode) -> ReplayResult:
        """Run deterministic sync replay for one episode.

        This first pass evaluates the frozen NavDP artifact itself. It consumes
        the producer's event log and metadata to provide non-empty, deterministic
        smoke metrics before full simulator replay is wired in.
        """

        payload = episode.payload
        missions = _dict_list(payload.get("missions"))
        robots = _dict_list(payload.get("robots"))
        humans = _dict_list(payload.get("humans"))
        events = sorted(
            _event_entries(payload.get("event_log")),
            key=lambda event: (_number(event.get("t"), 0.0), str(event.get("event_id", ""))),
        )

        event_counts = Counter(str(event.get("event_type", "")) for event in events)
        mission_ids = {str(mission.get("mission_id", "")) for mission in missions}
        mission_ids.discard("")
        released_ids = _event_mission_ids(events, {"mission_release"})
        assigned_ids = _event_mission_ids(events, {"robot_assignment", "assign_mission"})
        completed_ids = _event_mission_ids(events, {"completion", "mission_completion"})
        expected_result = payload.get("expected_result", {})
        if not isinstance(expected_result, dict):
            expected_result = {}
        expected_metrics = expected_result.get("metrics", {})
        if not isinstance(expected_metrics, dict):
            expected_metrics = {}

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        collision_check = metadata.get("collision_check", {})
        if not isinstance(collision_check, dict):
            collision_check = {}

        mission_count = len(mission_ids)
        completion_replay_available = bool(completed_ids)
        completion_rate = (
            len(completed_ids & mission_ids) / mission_count
            if mission_count and completion_replay_available
            else None
        )
        mission_results = _evaluate_missions(payload, missions, robots, humans, events)
        mission_success_values = [
            result.get("success")
            for result in mission_results
            if result.get("success") is not None
        ]
        mission_success_rate = (
            sum(1 for value in mission_success_values if value is True)
            / len(mission_success_values)
            if mission_success_values
            else None
        )
        expected_passed = expected_result.get("passed")
        if mission_success_rate is not None:
            success = mission_success_rate == 1.0 and expected_passed is not False
        elif completion_rate is None:
            success = expected_passed if isinstance(expected_passed, bool) else None
        else:
            success = completion_rate == 1.0 and expected_passed is not False

        family_metrics = _summarize_mission_results(mission_results)
        metrics: JsonDict = {
            "schema_version": payload.get("schema_version", ""),
            "mission_count": mission_count,
            "robot_count": len(robots),
            "human_count": len(humans),
            "event_count": len(events),
            "release_count": event_counts.get("mission_release", 0),
            "assignment_count": event_counts.get("robot_assignment", 0)
            + event_counts.get("assign_mission", 0),
            "completion_count": event_counts.get("completion", 0)
            + event_counts.get("mission_completion", 0),
            "robot_eos_count": event_counts.get("robot_eos", 0),
            "released_mission_count": len(released_ids & mission_ids),
            "assigned_mission_count": len(assigned_ids & mission_ids),
            "completed_mission_count": len(completed_ids & mission_ids),
            "completion_replay_available": completion_replay_available,
            "completion_rate": completion_rate,
            "mission_success_rate": mission_success_rate,
            "all_missions_completed": completion_rate == 1.0
            if completion_rate is not None
            else None,
            "expected_result_passed": expected_passed,
            "expected_fixture_valid": expected_metrics.get("fixture_expected_valid"),
            "duration_s": _duration_seconds(events, robots, humans),
            "collision_free": collision_check.get("collision_free"),
            "collision_count": collision_check.get("collision_count"),
            "min_clearance_m": collision_check.get("min_clearance_m"),
        }
        metrics.update(family_metrics)

        return ReplayResult(
            episode_id=episode.episode_id,
            metrics=metrics,
            events=events,
            mission_results=mission_results,
            success=success,
        )

    def evaluate_split(self, episodes: list[HumanCentricEpisode]) -> list[ReplayResult]:
        """Evaluate a list of episodes with deterministic ordering."""

        return [self.replay(episode) for episode in episodes]


def _dict_list(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _event_entries(event_log: Any) -> list[JsonDict]:
    if isinstance(event_log, dict):
        return _dict_list(event_log.get("events"))
    return _dict_list(event_log)


def _event_mission_ids(events: list[JsonDict], event_types: set[str]) -> set[str]:
    return {
        str(event.get("mission_id", ""))
        for event in events
        if str(event.get("event_type", "")) in event_types and event.get("mission_id")
    }


def _duration_seconds(
    events: list[JsonDict],
    robots: list[JsonDict],
    humans: list[JsonDict],
) -> float:
    timestamps = [_number(event.get("t"), 0.0) for event in events]
    for actor in [*robots, *humans]:
        for point in _dict_list(actor.get("trajectory")):
            timestamps.append(_number(point.get("t"), 0.0))
    return max(timestamps) if timestamps else 0.0


def _evaluate_missions(
    payload: JsonDict,
    missions: list[JsonDict],
    robots: list[JsonDict],
    humans: list[JsonDict],
    events: list[JsonDict],
) -> list[JsonDict]:
    return [
        _evaluate_deliver_to_human(payload, mission, robots, humans, events)
        if mission.get("mission_type") == "deliver_to_human"
        and _uses_delivery_contact_contract(mission)
        else _evaluate_social_navigation(payload, mission, robots, humans, events)
        if mission.get("mission_type") == "navigate_with_social_constraints"
        and _uses_single_mission_contract(mission)
        and _supports_social_navigation_trajectory_contract(mission)
        else _evaluate_human_guided_uncertain_region(payload, mission, robots, events)
        if mission.get("mission_type") == "human_guided_uncertain_region"
        and _uses_single_mission_contract(mission)
        else _evaluate_serve_queue(payload, mission, missions, robots, humans, events)
        if mission.get("mission_type") == "serve_queue"
        else _evaluate_mission_stream_parent(payload, mission, missions, robots, events)
        if mission.get("mission_type") == "mission_stream"
        else _evaluate_mission_stream_child(payload, mission, robots, events)
        if _is_mission_stream_child(mission)
        else _evaluate_dense_dynamic_humans(payload, mission, robots, humans, events)
        if mission.get("mission_type") == "dense_dynamic_humans"
        else _evaluate_dense_multi_robot(payload, mission, robots, events)
        if mission.get("mission_type") == "dense_multi_robot"
        else _evaluate_dense_dynamic_combined(payload, mission, robots, humans, events)
        if mission.get("mission_type") == "dense_dynamic_combined"
        else _evaluate_event_log_mission(mission, events)
        for mission in missions
    ]


def _evaluate_event_log_mission(mission: JsonDict, events: list[JsonDict]) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    completion_time = _mission_event_time(events, mission_id, {"completion", "mission_completion"})
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "success": completion_time is not None,
        "completion_time_s": completion_time,
        "evidence": "event_log_completion" if completion_time is not None else "event_log_incomplete",
    }


def _evaluate_deliver_to_human(
    payload: JsonDict,
    mission: JsonDict,
    robots: list[JsonDict],
    humans: list[JsonDict],
    events: list[JsonDict],
) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    robot_id = _assigned_robot_id(mission, events)
    target_human_id = str(mission.get("target_human_id", ""))
    robot = _find_by_id(robots, "robot_id", robot_id)
    target_human = _find_by_id(humans, "human_id", target_human_id)
    if robot is None or target_human is None:
        return {
            "mission_id": mission_id,
            "mission_type": mission.get("mission_type", ""),
            "success": False,
            "failure_reason": "missing_robot_or_target_human",
            "assigned_robot_id": robot_id,
            "target_human_id": target_human_id,
        }

    contact_threshold_m = _contact_threshold_m(payload, mission)
    contact_time, target_min_distance = _first_contact_time(robot, target_human, contact_threshold_m)
    deadline = _deadline_s(mission)
    completion_time = _mission_event_time(events, mission_id, {"completion", "mission_completion"})
    terminal_time = contact_time if contact_time is not None else completion_time
    correct_human_reached = contact_time is not None
    deadline_success = (
        terminal_time is not None and (deadline is None or terminal_time <= deadline + 1e-6)
    )
    non_target_contact_threshold_m = _delivery_physical_contact_threshold_m(payload, mission)
    wrong_human_contact_count = _wrong_human_contact_count(
        robot,
        humans,
        target_human_id,
        non_target_contact_threshold_m,
    )
    personal_space = _non_target_personal_space_metrics(payload, robot, humans, target_human_id)
    object_delivered = correct_human_reached and deadline_success
    non_target_physical_clearance_respected = wrong_human_contact_count == 0
    success = (
        correct_human_reached
        and object_delivered
        and deadline_success
        and non_target_physical_clearance_respected
    )
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "success": success,
        "assigned_robot_id": robot_id,
        "target_human_id": target_human_id,
        "contact_threshold_m": contact_threshold_m,
        "target_contact_time_s": contact_time,
        "completion_time_s": completion_time,
        "deadline_s": deadline,
        "target_min_distance_m": target_min_distance,
        "correct_human_reached": correct_human_reached,
        "object_delivered": object_delivered,
        "deadline_success": deadline_success,
        "non_target_contact_threshold_m": non_target_contact_threshold_m,
        "wrong_human_contact_count": wrong_human_contact_count,
        "non_target_physical_clearance_respected": non_target_physical_clearance_respected,
        "personal_space_respected": non_target_physical_clearance_respected,
        **personal_space,
        "evidence": "trajectory_contact",
    }


def _uses_delivery_contact_contract(mission: JsonDict) -> bool:
    return _uses_single_mission_contract(mission)


def _uses_single_mission_contract(mission: JsonDict) -> bool:
    metadata = mission.get("metadata", {})
    if not isinstance(metadata, dict):
        return True
    return "mission_stream_parent_id" not in metadata


def _supports_social_navigation_trajectory_contract(mission: JsonDict) -> bool:
    active_laws = set(_active_social_law_ids(mission))
    return not active_laws or active_laws.issubset(SUPPORTED_SOCIAL_NAVIGATION_LAWS)


def _evaluate_social_navigation(
    payload: JsonDict,
    mission: JsonDict,
    robots: list[JsonDict],
    humans: list[JsonDict],
    events: list[JsonDict],
) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    robot_id = _assigned_robot_id(mission, events)
    robot = _find_by_id(robots, "robot_id", robot_id)
    goal_xy = _mission_goal_xy(mission)
    if robot is None or goal_xy is None:
        return {
            "mission_id": mission_id,
            "mission_type": mission.get("mission_type", ""),
            "success": False,
            "failure_reason": "missing_robot_or_goal",
            "assigned_robot_id": robot_id,
            "evidence": "trajectory_social_navigation",
        }

    goal_threshold_m = _goal_threshold_m(mission)
    goal_reach_time, min_goal_distance = _first_goal_reach_time(
        robot,
        goal_xy,
        goal_threshold_m,
    )
    deadline = _deadline_s(mission)
    completion_time = _mission_event_time(events, mission_id, {"completion", "mission_completion"})
    terminal_time = goal_reach_time if goal_reach_time is not None else completion_time
    goal_reached = goal_reach_time is not None
    deadline_success = (
        terminal_time is not None and (deadline is None or terminal_time <= deadline + 1e-6)
    )
    collision_count = _robot_human_collision_count(payload, robot, humans)
    active_social_law_ids = _active_social_law_ids(mission)
    law_metrics = _social_navigation_law_metrics(
        payload,
        mission,
        robot,
        humans,
        active_social_law_ids,
    )
    social_law_success = law_metrics.get("social_law_success")
    success = (
        goal_reached
        and deadline_success
        and collision_count == 0
        and social_law_success is not False
    )
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "success": success,
        "assigned_robot_id": robot_id,
        "active_social_law_ids": active_social_law_ids,
        "goal_xy": [goal_xy[0], goal_xy[1]],
        "goal_threshold_m": goal_threshold_m,
        "goal_reach_time_s": goal_reach_time,
        "completion_time_s": completion_time,
        "deadline_s": deadline,
        "minimum_goal_distance_m": min_goal_distance,
        "goal_reached": goal_reached,
        "deadline_success": deadline_success,
        "collision_count": collision_count,
        **law_metrics,
        "evidence": "trajectory_social_navigation",
    }


def _evaluate_human_guided_uncertain_region(
    payload: JsonDict,
    mission: JsonDict,
    robots: list[JsonDict],
    events: list[JsonDict],
) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    robot_id = _assigned_robot_id(mission, events)
    robot = _find_by_id(robots, "robot_id", robot_id)
    resolved_target_xy = _human_guided_resolved_target_xy(mission)
    if robot is None or resolved_target_xy is None:
        return {
            "mission_id": mission_id,
            "mission_type": mission.get("mission_type", ""),
            "success": False,
            "failure_reason": "missing_robot_or_resolved_target",
            "assigned_robot_id": robot_id,
            "evidence": "trajectory_human_guided_uncertain_region",
        }

    request_time = _mission_event_time(events, mission_id, {"human_guidance_request"})
    response_time = _mission_event_time(events, mission_id, {"human_guidance_response"})
    uncertainty_resolved_time = _mission_event_time(events, mission_id, {"uncertainty_resolved"})
    completion_time = _mission_event_time(events, mission_id, {"completion", "mission_completion"})
    goal_threshold_m = _goal_threshold_m(mission)
    goal_reach_time, min_goal_distance = _first_goal_reach_time(
        robot,
        resolved_target_xy,
        goal_threshold_m,
    )
    deadline = _deadline_s(mission)
    terminal_time = goal_reach_time if goal_reach_time is not None else completion_time
    deadline_success = (
        terminal_time is not None and (deadline is None or terminal_time <= deadline + 1e-6)
    )
    stop_metrics = _human_guidance_stop_metrics(mission, robot, request_time, response_time)
    collision_count = _checked_collision_count(payload)
    if collision_count is None:
        collision_count = 0

    guidance_requested = request_time is not None
    guidance_received = response_time is not None
    uncertainty_resolved = uncertainty_resolved_time is not None
    resolved_target_reached = goal_reach_time is not None
    stop_required = stop_metrics["guidance_stop_required"]
    stop_verified = stop_metrics["guidance_stop_verified"]
    success = (
        guidance_requested
        and guidance_received
        and uncertainty_resolved
        and resolved_target_reached
        and deadline_success
        and collision_count == 0
        and (not stop_required or stop_verified is True)
    )
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "success": success,
        "assigned_robot_id": robot_id,
        "informant_human_id": _human_guided_informant_id(mission),
        "resolved_target_xy": [resolved_target_xy[0], resolved_target_xy[1]],
        "goal_threshold_m": goal_threshold_m,
        "guidance_request_time_s": request_time,
        "guidance_response_time_s": response_time,
        "uncertainty_resolved_time_s": uncertainty_resolved_time,
        "completion_time_s": completion_time,
        "goal_reach_time_s": goal_reach_time,
        "deadline_s": deadline,
        "minimum_goal_distance_m": min_goal_distance,
        "guidance_requested": guidance_requested,
        "human_guidance_received": guidance_received,
        "uncertainty_resolved": uncertainty_resolved,
        "resolved_target_reached": resolved_target_reached,
        "deadline_success": deadline_success,
        "collision_count": collision_count,
        **stop_metrics,
        "evidence": "trajectory_human_guided_uncertain_region",
    }


def _evaluate_serve_queue(
    payload: JsonDict,
    mission: JsonDict,
    all_missions: list[JsonDict],
    robots: list[JsonDict],
    humans: list[JsonDict],
    events: list[JsonDict],
) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    robot_id = _assigned_robot_id(mission, events)
    target_human_id = str(mission.get("target_human_id", ""))
    robot = _find_by_id(robots, "robot_id", robot_id)
    target_human = _find_by_id(humans, "human_id", target_human_id)
    if robot is None or target_human is None:
        return {
            "mission_id": mission_id,
            "mission_type": mission.get("mission_type", ""),
            "success": False,
            "failure_reason": "missing_robot_or_target_human",
            "assigned_robot_id": robot_id,
            "target_human_id": target_human_id,
            "evidence": "trajectory_serve_queue",
        }

    release_time = _number(mission.get("release_time"), 0.0)
    deadline = _deadline_s(mission)
    contact_threshold_m = _contact_threshold_m(payload, mission)
    contact_time, target_min_distance = _first_contact_time(
        robot,
        target_human,
        contact_threshold_m,
        start_time=release_time,
        end_time=deadline,
    )
    completion_time = _mission_event_time(events, mission_id, {"completion", "mission_completion"})
    terminal_time = contact_time if contact_time is not None else completion_time
    deadline_success = (
        terminal_time is not None and (deadline is None or terminal_time <= deadline + 1e-6)
    )
    previous_complete = _previous_queue_missions_completed(mission, all_missions, events)
    queue_order_preserved = _serve_queue_order_preserved(mission, all_missions, events)
    collision_count = _checked_collision_count(payload)
    if collision_count is None:
        collision_count = 0

    correct_human_reached = contact_time is not None
    nearest_queue_contact_reached = correct_human_reached
    success = (
        correct_human_reached
        and nearest_queue_contact_reached
        and previous_complete
        and queue_order_preserved
        and deadline_success
        and collision_count == 0
    )
    metadata = _dict_value(mission.get("metadata"))
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "success": success,
        "assigned_robot_id": robot_id,
        "target_human_id": target_human_id,
        "queue_id": metadata.get("queue_id", ""),
        "queue_index": metadata.get("queue_index"),
        "queue_position": metadata.get("queue_position"),
        "queue_order": _string_list(metadata.get("queue_order")),
        "previous_queue_human_ids": _string_list(metadata.get("previous_queue_human_ids")),
        "contact_threshold_m": contact_threshold_m,
        "target_contact_time_s": contact_time,
        "completion_time_s": completion_time,
        "release_time_s": release_time,
        "deadline_s": deadline,
        "target_min_distance_m": target_min_distance,
        "correct_human_reached": correct_human_reached,
        "nearest_queue_contact_reached": nearest_queue_contact_reached,
        "previous_queue_missions_completed": previous_complete,
        "queue_order_preserved": queue_order_preserved,
        "deadline_success": deadline_success,
        "collision_count": collision_count,
        "evidence": "trajectory_serve_queue",
    }


def _evaluate_mission_stream_parent(
    payload: JsonDict,
    mission: JsonDict,
    all_missions: list[JsonDict],
    robots: list[JsonDict],
    events: list[JsonDict],
) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    metadata = _dict_value(mission.get("metadata"))
    declared_child_ids = _string_list(metadata.get("child_mission_ids"))
    child_missions = [
        child
        for child in all_missions
        if _dict_value(child.get("metadata")).get("mission_stream_parent_id") == mission_id
    ]
    child_ids = [str(child.get("mission_id", "")) for child in child_missions]
    child_ids_present = set(declared_child_ids) == set(child_ids) if declared_child_ids else bool(child_ids)
    expected_stack_size = _number(metadata.get("configured_mission_stream_stack_size"), 0.0)
    expected_child_count = int(expected_stack_size - 1) if expected_stack_size > 0 else len(child_ids)
    child_count_matches = len(child_ids) == expected_child_count

    released_child_ids = _event_mission_ids(events, {"mission_release"}) & set(child_ids)
    assigned_child_ids = _event_mission_ids(events, {"robot_assignment", "assign_mission"}) & set(child_ids)
    completed_child_ids = _event_mission_ids(events, {"completion", "mission_completion"}) & set(
        child_ids
    )
    eos_child_ids = _event_mission_ids(events, {"robot_eos"}) & set(child_ids)
    child_timing_checks = [
        _mission_stream_child_timing_check(child, events) for child in child_missions
    ]
    child_timing_respected = bool(child_timing_checks) and all(
        check["timing_respected"] for check in child_timing_checks
    )
    terminal_goal_metrics = _stream_parent_terminal_goal_metrics(metadata, robots)
    terminal_goals_reached = terminal_goal_metrics["stream_terminal_goals_reached"]
    parent_completion_time = _mission_event_time(
        events,
        mission_id,
        {"completion", "mission_completion"},
    )
    expected_completion_time = _optional_number(metadata.get("expected_completion_t"))
    parent_completion_matches_expected = _time_matches(
        parent_completion_time,
        expected_completion_time,
    )
    dispatch_record_count = _optional_int(metadata.get("dispatch_record_count"))
    dispatch_record_count_matches = (
        dispatch_record_count is None or dispatch_record_count == len(child_ids)
    )
    deadline = _deadline_s(mission)
    deadline_success = (
        parent_completion_time is not None
        and (deadline is None or parent_completion_time <= deadline + 1e-6)
    )
    collision_count = _checked_collision_count(payload)
    if collision_count is None:
        collision_count = 0

    released_missions_completed = (
        bool(child_ids)
        and len(released_child_ids) == len(child_ids)
        and len(completed_child_ids) == len(child_ids)
    )
    priority_order_respected = child_timing_respected and dispatch_record_count_matches
    success = (
        child_ids_present
        and child_count_matches
        and released_missions_completed
        and len(assigned_child_ids) == len(child_ids)
        and len(eos_child_ids) == len(child_ids)
        and parent_completion_time is not None
        and parent_completion_matches_expected
        and priority_order_respected
        and terminal_goals_reached
        and deadline_success
        and collision_count == 0
    )
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "success": success,
        "assigned_robot_id": _assigned_robot_id(mission, events),
        "child_mission_ids": child_ids,
        "declared_child_mission_ids": declared_child_ids,
        "child_mission_count": len(child_ids),
        "expected_child_mission_count": expected_child_count,
        "child_ids_present": child_ids_present,
        "child_count_matches_config": child_count_matches,
        "released_child_mission_count": len(released_child_ids),
        "assigned_child_mission_count": len(assigned_child_ids),
        "completed_child_mission_count": len(completed_child_ids),
        "eos_child_mission_count": len(eos_child_ids),
        "dispatch_record_count": dispatch_record_count,
        "dispatch_record_count_matches": dispatch_record_count_matches,
        "parent_completion_time_s": parent_completion_time,
        "expected_completion_time_s": expected_completion_time,
        "parent_completion_matches_expected": parent_completion_matches_expected,
        "deadline_s": deadline,
        "deadline_success": deadline_success,
        "released_missions_completed": released_missions_completed,
        "priority_order_respected": priority_order_respected,
        "child_timing_checks": child_timing_checks,
        "collision_count": collision_count,
        **terminal_goal_metrics,
        "evidence": "trajectory_mission_stream_parent",
    }


def _evaluate_mission_stream_child(
    payload: JsonDict,
    mission: JsonDict,
    robots: list[JsonDict],
    events: list[JsonDict],
) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    metadata = _dict_value(mission.get("metadata"))
    robot_id = _assigned_robot_id(mission, events)
    robot = _find_by_id(robots, "robot_id", robot_id)
    goal_xy = _mission_goal_xy(mission)
    if robot is None or goal_xy is None:
        return {
            "mission_id": mission_id,
            "mission_type": mission.get("mission_type", ""),
            "mission_stream_child": True,
            "mission_stream_parent_id": str(metadata.get("mission_stream_parent_id", "")),
            "success": False,
            "failure_reason": "missing_robot_or_goal",
            "assigned_robot_id": robot_id,
            "evidence": "trajectory_mission_stream_child",
        }

    release_time = _number(mission.get("release_time"), 0.0)
    deadline = _deadline_s(mission)
    goal_threshold_m = _goal_threshold_m(mission)
    goal_reach_time, min_goal_distance = _first_goal_reach_time(
        robot,
        goal_xy,
        goal_threshold_m,
        start_time=release_time,
        end_time=deadline,
    )
    timing_check = _mission_stream_child_timing_check(mission, events)
    completion_time = timing_check["completion_time_s"]
    terminal_time = completion_time if completion_time is not None else goal_reach_time
    deadline_success = (
        terminal_time is not None and (deadline is None or terminal_time <= deadline + 1e-6)
    )
    collision_count = _checked_collision_count(payload)
    if collision_count is None:
        collision_count = 0

    goal_reached = min_goal_distance is not None and min_goal_distance <= goal_threshold_m + 1e-6
    success = (
        timing_check["release_event_present"]
        and timing_check["assignment_event_present"]
        and timing_check["completion_event_present"]
        and timing_check["eos_event_present"]
        and timing_check["timing_respected"]
        and goal_reached
        and deadline_success
        and collision_count == 0
    )
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "mission_stream_child": True,
        "mission_stream_parent_id": str(metadata.get("mission_stream_parent_id", "")),
        "success": success,
        "assigned_robot_id": robot_id,
        "mission_stack_index": metadata.get("mission_stack_index"),
        "mission_stack_size": metadata.get("mission_stack_size"),
        "goal_xy": [goal_xy[0], goal_xy[1]],
        "goal_threshold_m": goal_threshold_m,
        "goal_reach_time_s": goal_reach_time,
        "minimum_goal_distance_m": min_goal_distance,
        "goal_reached": goal_reached,
        "release_time_s": release_time,
        "deadline_s": deadline,
        "deadline_success": deadline_success,
        "collision_count": collision_count,
        "priority_order_respected": timing_check["timing_respected"],
        **timing_check,
        "evidence": "trajectory_mission_stream_child",
    }


def _evaluate_dense_dynamic_humans(
    payload: JsonDict,
    mission: JsonDict,
    robots: list[JsonDict],
    humans: list[JsonDict],
    events: list[JsonDict],
) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    metadata = _dict_value(mission.get("metadata"))
    dense_metadata = _dense_dynamic_metadata(payload, mission)
    active_robot_ids = _string_list(metadata.get("active_robot_ids"))
    if not active_robot_ids:
        active_robot_ids = [_assigned_robot_id(mission, events)]
    active_robot_ids = [robot_id for robot_id in active_robot_ids if robot_id]

    goal_checks = _dense_robot_goal_checks(mission, robots, active_robot_ids)
    all_goals_reached = bool(goal_checks) and all(check["goal_reached"] for check in goal_checks)
    collision_count = _checked_collision_count(payload)
    if collision_count is None:
        collision_count = 0
    collision_free = collision_count == 0
    nominal_conflict_count = _dense_nominal_conflict_count(dense_metadata)
    dense_clearance_respected = collision_free and nominal_conflict_count == 0

    completion_time = _max_optional(check.get("goal_reach_time_s") for check in goal_checks)
    if completion_time is None:
        completion_time = _optional_number(metadata.get("mission_end_time_s"))
    deadline = _deadline_s(mission)
    deadline_success = (
        completion_time is not None and (deadline is None or completion_time <= deadline + 1e-6)
    )
    human_activity = _dense_human_activity_metrics(humans, completion_time)
    human_human = _dense_human_human_clearance_metrics(
        humans,
        _number(metadata.get("minimum_human_human_distance_m"), 0.05),
    )
    robot_motion = _dense_robot_motion_metrics(dense_metadata, active_robot_ids)
    success = (
        all_goals_reached
        and dense_clearance_respected
        and human_human["human_human_collision_free"]
        and human_activity["humans_keep_moving_until_robot_completion"]
        and robot_motion["robots_keep_moving_when_passable"]
        and robot_motion["wait_only_when_immediately_blocked"]
        and deadline_success
    )
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "success": success,
        "assigned_robot_id": _assigned_robot_id(mission, events),
        "active_robot_ids": active_robot_ids,
        "active_robot_goal_checks": goal_checks,
        "all_active_robot_goal_regions_reached": all_goals_reached,
        "completion_time_s": completion_time,
        "deadline_s": deadline,
        "deadline_success": deadline_success,
        "collision_count": collision_count,
        "dense_nominal_robot_human_conflict_count": nominal_conflict_count,
        "dense_robot_human_clearance_policy_respected": dense_clearance_respected,
        **human_activity,
        **human_human,
        **robot_motion,
        "evidence": "trajectory_dense_dynamic_humans",
    }


def _evaluate_dense_multi_robot(
    payload: JsonDict,
    mission: JsonDict,
    robots: list[JsonDict],
    events: list[JsonDict],
) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    metadata = _dict_value(mission.get("metadata"))
    dense_metadata = _dense_multi_robot_metadata(payload, mission)
    active_robot_ids = _string_list(metadata.get("active_robot_ids"))
    if not active_robot_ids:
        active_robot_ids = [str(robot.get("robot_id", "")) for robot in robots]
    active_robot_ids = [robot_id for robot_id in active_robot_ids if robot_id]

    goal_checks = _dense_robot_goal_checks(mission, robots, active_robot_ids)
    all_goals_reached = bool(goal_checks) and all(check["goal_reached"] for check in goal_checks)
    completion_time = _max_optional(check.get("goal_reach_time_s") for check in goal_checks)
    if completion_time is None:
        completion_time = _optional_number(metadata.get("mission_end_time_s"))
    deadline = _deadline_s(mission)
    deadline_success = (
        completion_time is not None and (deadline is None or completion_time <= deadline + 1e-6)
    )

    collision_count = _checked_collision_count(payload)
    if collision_count is None:
        collision_count = 0
    robot_robot_clearance = _dense_robot_robot_clearance_metrics(
        payload,
        mission,
        robots,
        active_robot_ids,
    )
    robot_motion = _dense_multi_robot_motion_metrics(
        dense_metadata,
        metadata,
        active_robot_ids,
    )
    success = (
        all_goals_reached
        and robot_robot_clearance["no_robot_robot_collision"]
        and robot_robot_clearance["robot_robot_clearance_policy_respected"]
        and robot_motion["robots_keep_moving_when_passable"]
        and robot_motion["wait_only_when_immediately_blocked"]
        and deadline_success
    )
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "success": success,
        "assigned_robot_id": _assigned_robot_id(mission, events),
        "active_robot_ids": active_robot_ids,
        "active_robot_goal_checks": goal_checks,
        "all_active_robot_goal_regions_reached": all_goals_reached,
        "completion_time_s": completion_time,
        "deadline_s": deadline,
        "deadline_success": deadline_success,
        "collision_count": collision_count,
        **robot_robot_clearance,
        **robot_motion,
        "evidence": "trajectory_dense_multi_robot",
    }


def _evaluate_dense_dynamic_combined(
    payload: JsonDict,
    mission: JsonDict,
    robots: list[JsonDict],
    humans: list[JsonDict],
    events: list[JsonDict],
) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    metadata = _dict_value(mission.get("metadata"))
    dense_metadata = _dense_dynamic_combined_metadata(payload, mission)
    active_robot_ids = _string_list(metadata.get("active_robot_ids"))
    if not active_robot_ids:
        active_robot_ids = [str(robot.get("robot_id", "")) for robot in robots]
    active_robot_ids = [robot_id for robot_id in active_robot_ids if robot_id]

    goal_checks = _dense_robot_goal_checks(mission, robots, active_robot_ids)
    all_goals_reached = bool(goal_checks) and all(check["goal_reached"] for check in goal_checks)
    completion_time = _max_optional(check.get("goal_reach_time_s") for check in goal_checks)
    if completion_time is None:
        completion_time = _optional_number(metadata.get("mission_end_time_s"))
    deadline = _deadline_s(mission)
    deadline_success = (
        completion_time is not None and (deadline is None or completion_time <= deadline + 1e-6)
    )

    collision_count = _checked_collision_count(payload)
    if collision_count is None:
        collision_count = 0
    nominal_conflict_count = _dense_nominal_conflict_count(dense_metadata)
    dense_clearance_respected = collision_count == 0 and nominal_conflict_count == 0
    human_activity = _dense_human_activity_metrics(humans, completion_time)
    human_human = _dense_human_human_clearance_metrics(
        humans,
        _number(metadata.get("minimum_human_human_distance_m"), 0.05),
    )
    robot_robot_clearance = _dense_robot_robot_clearance_metrics(
        payload,
        mission,
        robots,
        active_robot_ids,
    )
    robot_motion = _dense_multi_robot_motion_metrics(
        dense_metadata,
        metadata,
        active_robot_ids,
    )
    success = (
        all_goals_reached
        and dense_clearance_respected
        and human_human["human_human_collision_free"]
        and human_activity["humans_keep_moving_until_robot_completion"]
        and robot_robot_clearance["no_robot_robot_collision"]
        and robot_robot_clearance["robot_robot_clearance_policy_respected"]
        and robot_motion["robots_keep_moving_when_passable"]
        and robot_motion["wait_only_when_immediately_blocked"]
        and deadline_success
    )
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "success": success,
        "assigned_robot_id": _assigned_robot_id(mission, events),
        "active_robot_ids": active_robot_ids,
        "active_robot_goal_checks": goal_checks,
        "all_active_robot_goal_regions_reached": all_goals_reached,
        "completion_time_s": completion_time,
        "deadline_s": deadline,
        "deadline_success": deadline_success,
        "collision_count": collision_count,
        "dense_nominal_robot_human_conflict_count": nominal_conflict_count,
        "dense_robot_human_clearance_policy_respected": dense_clearance_respected,
        **human_activity,
        **human_human,
        **robot_robot_clearance,
        **robot_motion,
        "evidence": "trajectory_dense_dynamic_combined",
    }


def _summarize_mission_results(mission_results: list[JsonDict]) -> JsonDict:
    deliver_results = [
        result
        for result in mission_results
        if result.get("mission_type") == "deliver_to_human"
        and not result.get("mission_stream_child")
    ]
    social_nav_results = [
        result
        for result in mission_results
        if result.get("mission_type") == "navigate_with_social_constraints"
        and not result.get("mission_stream_child")
    ]
    human_guided_results = [
        result
        for result in mission_results
        if result.get("mission_type") == "human_guided_uncertain_region"
    ]
    serve_queue_results = [
        result
        for result in mission_results
        if result.get("mission_type") == "serve_queue"
    ]
    mission_stream_parent_results = [
        result
        for result in mission_results
        if result.get("mission_type") == "mission_stream"
    ]
    mission_stream_child_results = [
        result
        for result in mission_results
        if result.get("mission_stream_child") is True
    ]
    dense_dynamic_human_results = [
        result
        for result in mission_results
        if result.get("mission_type") == "dense_dynamic_humans"
    ]
    dense_multi_robot_results = [
        result
        for result in mission_results
        if result.get("mission_type") == "dense_multi_robot"
    ]
    dense_dynamic_combined_results = [
        result
        for result in mission_results
        if result.get("mission_type") == "dense_dynamic_combined"
    ]
    success_values = [
        result.get("success")
        for result in mission_results
        if result.get("success") is not None
    ]
    metrics: JsonDict = {
        "mission_metric_count": len(success_values),
        "mission_metric_success_count": sum(1 for value in success_values if value is True),
    }
    if deliver_results:
        metrics.update(
            {
                "deliver_to_human_count": len(deliver_results),
                "deliver_to_human_success_count": sum(
                    1 for result in deliver_results if result.get("success") is True
                ),
                "correct_human_reached_count": sum(
                    1 for result in deliver_results if result.get("correct_human_reached") is True
                ),
                "object_delivered_count": sum(
                    1 for result in deliver_results if result.get("object_delivered") is True
                ),
                "deadline_success_count": sum(
                    1 for result in deliver_results if result.get("deadline_success") is True
                ),
                "wrong_human_contact_count": sum(
                    int(result.get("wrong_human_contact_count", 0))
                    for result in deliver_results
                ),
                "personal_space_violation_count": sum(
                    int(result.get("personal_space_violation_count", 0))
                    for result in deliver_results
                ),
                "personal_space_violation_duration_s": sum(
                    float(result.get("personal_space_violation_duration_s", 0.0))
                    for result in deliver_results
                ),
            }
        )
    if social_nav_results:
        metrics.update(
            {
                "navigate_with_social_constraints_count": len(social_nav_results),
                "navigate_with_social_constraints_success_count": sum(
                    1 for result in social_nav_results if result.get("success") is True
                ),
                "social_navigation_goal_reached_count": sum(
                    1 for result in social_nav_results if result.get("goal_reached") is True
                ),
                "social_navigation_deadline_success_count": sum(
                    1 for result in social_nav_results if result.get("deadline_success") is True
                ),
                "social_navigation_personal_space_violation_count": sum(
                    int(result.get("personal_space_violation_count", 0))
                    for result in social_nav_results
                ),
                "social_navigation_personal_space_violation_duration_s": sum(
                    float(result.get("personal_space_violation_duration_s", 0.0))
                    for result in social_nav_results
                ),
                "social_navigation_collision_count": sum(
                    int(result.get("collision_count", 0))
                    for result in social_nav_results
                ),
                "social_navigation_social_law_success_count": sum(
                    1 for result in social_nav_results if result.get("social_law_success") is True
                ),
                "pedestrian_yield_success_count": sum(
                    1
                    for result in social_nav_results
                    if result.get("pedestrian_yield_respected") is True
                ),
                "pedestrian_yield_violation_count": sum(
                    int(result.get("pedestrian_yield_violation_count", 0))
                    for result in social_nav_results
                ),
                "group_integrity_success_count": sum(
                    1
                    for result in social_nav_results
                    if result.get("group_integrity_respected") is True
                ),
                "group_region_violation_count": sum(
                    int(result.get("group_region_violation_count", 0))
                    for result in social_nav_results
                ),
                "group_region_violation_duration_s": sum(
                    float(result.get("group_region_violation_duration_s", 0.0))
                    for result in social_nav_results
                ),
                "queue_order_success_count": sum(
                    1
                    for result in social_nav_results
                    if result.get("queue_order_respected") is True
                ),
                "queue_order_violation_count": sum(
                    int(result.get("queue_order_violation_count", 0))
                    for result in social_nav_results
                ),
            }
        )
    if human_guided_results:
        metrics.update(
            {
                "human_guided_uncertain_region_count": len(human_guided_results),
                "human_guided_uncertain_region_success_count": sum(
                    1 for result in human_guided_results if result.get("success") is True
                ),
                "guidance_requested_count": sum(
                    1 for result in human_guided_results if result.get("guidance_requested") is True
                ),
                "human_guidance_received_count": sum(
                    1
                    for result in human_guided_results
                    if result.get("human_guidance_received") is True
                ),
                "uncertainty_resolved_count": sum(
                    1
                    for result in human_guided_results
                    if result.get("uncertainty_resolved") is True
                ),
                "resolved_target_reached_count": sum(
                    1
                    for result in human_guided_results
                    if result.get("resolved_target_reached") is True
                ),
                "guidance_stop_verified_count": sum(
                    1
                    for result in human_guided_results
                    if result.get("guidance_stop_verified") is True
                ),
            }
        )
    if serve_queue_results:
        metrics.update(
            {
                "serve_queue_count": len(serve_queue_results),
                "serve_queue_success_count": sum(
                    1 for result in serve_queue_results if result.get("success") is True
                ),
                "serve_queue_correct_human_reached_count": sum(
                    1
                    for result in serve_queue_results
                    if result.get("correct_human_reached") is True
                ),
                "serve_queue_nearest_contact_reached_count": sum(
                    1
                    for result in serve_queue_results
                    if result.get("nearest_queue_contact_reached") is True
                ),
                "serve_queue_previous_complete_count": sum(
                    1
                    for result in serve_queue_results
                    if result.get("previous_queue_missions_completed") is True
                ),
                "serve_queue_order_preserved_count": sum(
                    1
                    for result in serve_queue_results
                    if result.get("queue_order_preserved") is True
                ),
            }
        )
    if mission_stream_parent_results or mission_stream_child_results:
        metrics.update(
            {
                "mission_stream_parent_count": len(mission_stream_parent_results),
                "mission_stream_parent_success_count": sum(
                    1
                    for result in mission_stream_parent_results
                    if result.get("success") is True
                ),
                "mission_stream_child_count": len(mission_stream_child_results),
                "mission_stream_child_success_count": sum(
                    1
                    for result in mission_stream_child_results
                    if result.get("success") is True
                ),
                "mission_stream_child_goal_reached_count": sum(
                    1
                    for result in mission_stream_child_results
                    if result.get("goal_reached") is True
                ),
                "mission_stream_child_timing_success_count": sum(
                    1
                    for result in mission_stream_child_results
                    if result.get("timing_respected") is True
                ),
                "mission_stream_child_eos_count": sum(
                    1
                    for result in mission_stream_child_results
                    if result.get("eos_event_present") is True
                ),
                "mission_stream_parent_terminal_goal_success_count": sum(
                    1
                    for result in mission_stream_parent_results
                    if result.get("stream_terminal_goals_reached") is True
                ),
                "mission_stream_priority_order_success_count": sum(
                    1
                    for result in mission_stream_parent_results
                    if result.get("priority_order_respected") is True
                ),
                "mission_stream_released_children_completed_count": sum(
                    int(result.get("completed_child_mission_count", 0))
                    for result in mission_stream_parent_results
                ),
            }
        )
    if dense_dynamic_human_results:
        metrics.update(
            {
                "dense_dynamic_humans_count": len(dense_dynamic_human_results),
                "dense_dynamic_humans_success_count": sum(
                    1
                    for result in dense_dynamic_human_results
                    if result.get("success") is True
                ),
                "dense_active_robot_goal_success_count": sum(
                    1
                    for result in dense_dynamic_human_results
                    if result.get("all_active_robot_goal_regions_reached") is True
                ),
                "dense_robot_human_clearance_success_count": sum(
                    1
                    for result in dense_dynamic_human_results
                    if result.get("dense_robot_human_clearance_policy_respected") is True
                ),
                "dense_humans_keep_moving_success_count": sum(
                    1
                    for result in dense_dynamic_human_results
                    if result.get("humans_keep_moving_until_robot_completion") is True
                ),
                "dense_human_human_collision_free_count": sum(
                    1
                    for result in dense_dynamic_human_results
                    if result.get("human_human_collision_free") is True
                ),
                "dense_robot_wait_violation_count": sum(
                    int(result.get("robot_wait_violation_count", 0))
                    for result in dense_dynamic_human_results
                ),
                "dense_nominal_robot_human_conflict_count": sum(
                    int(result.get("dense_nominal_robot_human_conflict_count", 0))
                    for result in dense_dynamic_human_results
                ),
                "dense_corner_case_recovery_count": sum(
                    int(result.get("corner_case_recovery_count", 0))
                    for result in dense_dynamic_human_results
                ),
            }
        )
    if dense_multi_robot_results:
        metrics.update(
            {
                "dense_multi_robot_count": len(dense_multi_robot_results),
                "dense_multi_robot_success_count": sum(
                    1
                    for result in dense_multi_robot_results
                    if result.get("success") is True
                ),
                "dense_multi_active_robot_goal_success_count": sum(
                    1
                    for result in dense_multi_robot_results
                    if result.get("all_active_robot_goal_regions_reached") is True
                ),
                "dense_multi_robot_robot_collision_free_count": sum(
                    1
                    for result in dense_multi_robot_results
                    if result.get("no_robot_robot_collision") is True
                ),
                "dense_multi_robot_clearance_success_count": sum(
                    1
                    for result in dense_multi_robot_results
                    if result.get("robot_robot_clearance_policy_respected") is True
                ),
                "dense_multi_robot_wait_count": sum(
                    int(result.get("robot_wait_count", 0))
                    for result in dense_multi_robot_results
                ),
                "dense_multi_robot_wait_violation_count": sum(
                    int(result.get("robot_wait_violation_count", 0))
                    for result in dense_multi_robot_results
                ),
                "dense_multi_corner_case_recovery_count": sum(
                    int(result.get("corner_case_recovery_count", 0))
                    for result in dense_multi_robot_results
                ),
                "dense_multi_robot_robot_deadlock_recovery_count": sum(
                    int(result.get("robot_robot_deadlock_recovery_count", 0))
                    for result in dense_multi_robot_results
                ),
            }
        )
    if dense_dynamic_combined_results:
        metrics.update(
            {
                "dense_dynamic_combined_count": len(dense_dynamic_combined_results),
                "dense_dynamic_combined_success_count": sum(
                    1
                    for result in dense_dynamic_combined_results
                    if result.get("success") is True
                ),
                "dense_combined_active_robot_goal_success_count": sum(
                    1
                    for result in dense_dynamic_combined_results
                    if result.get("all_active_robot_goal_regions_reached") is True
                ),
                "dense_combined_robot_human_clearance_success_count": sum(
                    1
                    for result in dense_dynamic_combined_results
                    if result.get("dense_robot_human_clearance_policy_respected") is True
                ),
                "dense_combined_robot_robot_collision_free_count": sum(
                    1
                    for result in dense_dynamic_combined_results
                    if result.get("no_robot_robot_collision") is True
                ),
                "dense_combined_robot_robot_clearance_success_count": sum(
                    1
                    for result in dense_dynamic_combined_results
                    if result.get("robot_robot_clearance_policy_respected") is True
                ),
                "dense_combined_humans_keep_moving_success_count": sum(
                    1
                    for result in dense_dynamic_combined_results
                    if result.get("humans_keep_moving_until_robot_completion") is True
                ),
                "dense_combined_human_human_collision_free_count": sum(
                    1
                    for result in dense_dynamic_combined_results
                    if result.get("human_human_collision_free") is True
                ),
                "dense_combined_robot_wait_violation_count": sum(
                    int(result.get("robot_wait_violation_count", 0))
                    for result in dense_dynamic_combined_results
                ),
                "dense_combined_nominal_robot_human_conflict_count": sum(
                    int(result.get("dense_nominal_robot_human_conflict_count", 0))
                    for result in dense_dynamic_combined_results
                ),
                "dense_combined_corner_case_recovery_count": sum(
                    int(result.get("corner_case_recovery_count", 0))
                    for result in dense_dynamic_combined_results
                ),
                "dense_combined_robot_robot_deadlock_recovery_count": sum(
                    int(result.get("robot_robot_deadlock_recovery_count", 0))
                    for result in dense_dynamic_combined_results
                ),
            }
        )
    return metrics


def _assigned_robot_id(mission: JsonDict, events: list[JsonDict]) -> str:
    assigned_robot_id = mission.get("assigned_robot_id")
    if assigned_robot_id:
        return str(assigned_robot_id)
    mission_id = str(mission.get("mission_id", ""))
    for event in events:
        if str(event.get("mission_id", "")) != mission_id:
            continue
        if str(event.get("event_type", "")) not in {"robot_assignment", "assign_mission"}:
            continue
        actor_id = event.get("actor_id")
        if actor_id:
            return str(actor_id)
    return ""


def _find_by_id(items: list[JsonDict], key: str, value: str) -> JsonDict | None:
    for item in items:
        if str(item.get(key, "")) == value:
            return item
    return None


def _contact_threshold_m(payload: JsonDict, mission: JsonDict) -> float:
    metadata = mission.get("metadata", {})
    if isinstance(metadata, dict):
        contact_distance = metadata.get("contact_distance_m")
        if isinstance(contact_distance, (int, float)) and not isinstance(contact_distance, bool):
            return float(contact_distance) + 0.05
    defaults = _collision_defaults(payload)
    return (
        _number(defaults.get("robot_cylinder_radius_m"), 0.3)
        + _number(defaults.get("human_cylinder_radius_m"), 0.4)
        + 0.05
    )


def _delivery_physical_contact_threshold_m(payload: JsonDict, mission: JsonDict) -> float:
    metadata = mission.get("metadata", {})
    if isinstance(metadata, dict):
        contact_distance = metadata.get("contact_distance_m")
        if isinstance(contact_distance, (int, float)) and not isinstance(contact_distance, bool):
            return float(contact_distance)
    defaults = _collision_defaults(payload)
    return _number(defaults.get("robot_cylinder_radius_m"), 0.3) + _number(
        defaults.get("human_cylinder_radius_m"),
        0.4,
    )


def _deadline_s(mission: JsonDict) -> float | None:
    deadline = mission.get("deadline")
    if isinstance(deadline, bool):
        return None
    if isinstance(deadline, (int, float)):
        return float(deadline)
    metadata = mission.get("metadata", {})
    if isinstance(metadata, dict):
        mission_end_time = metadata.get("mission_end_time_s")
        if isinstance(mission_end_time, (int, float)) and not isinstance(mission_end_time, bool):
            return float(mission_end_time)
    return None


def _mission_event_time(
    events: list[JsonDict],
    mission_id: str,
    event_types: set[str],
) -> float | None:
    times = [
        _number(event.get("t"), 0.0)
        for event in events
        if str(event.get("mission_id", "")) == mission_id
        and str(event.get("event_type", "")) in event_types
    ]
    return min(times) if times else None


def _first_contact_time(
    robot: JsonDict,
    human: JsonDict,
    threshold_m: float,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[float | None, float | None]:
    min_distance: float | None = None
    contact_time: float | None = None
    for point in _trajectory_points(robot):
        t = _number(point.get("t"), 0.0)
        if start_time is not None and t + 1e-6 < start_time:
            continue
        if end_time is not None and t - 1e-6 > end_time:
            continue
        robot_xy = _pose_xy(point.get("map_pose"))
        human_xy = _actor_xy_at_time(human, t)
        if robot_xy is None or human_xy is None:
            continue
        distance = _distance(robot_xy, human_xy)
        min_distance = distance if min_distance is None else min(min_distance, distance)
        if contact_time is None and distance <= threshold_m:
            contact_time = t
    return contact_time, min_distance


def _wrong_human_contact_count(
    robot: JsonDict,
    humans: list[JsonDict],
    target_human_id: str,
    threshold_m: float,
) -> int:
    count = 0
    for human in humans:
        if str(human.get("human_id", "")) == target_human_id:
            continue
        contact_time, _ = _first_contact_time(robot, human, threshold_m)
        if contact_time is not None:
            count += 1
    return count


def _non_target_personal_space_metrics(
    payload: JsonDict,
    robot: JsonDict,
    humans: list[JsonDict],
    target_human_id: str,
) -> JsonDict:
    return _personal_space_metrics(
        payload,
        robot,
        humans,
        excluded_human_ids={target_human_id},
    )


def _personal_space_metrics(
    payload: JsonDict,
    robot: JsonDict,
    humans: list[JsonDict],
    *,
    excluded_human_ids: set[str] | None = None,
) -> JsonDict:
    excluded_human_ids = excluded_human_ids or set()
    violation_count = 0
    violation_duration = 0.0
    violation_human_ids: set[str] = set()
    min_distance: float | None = None
    for human in humans:
        human_id = str(human.get("human_id", ""))
        if human_id in excluded_human_ids:
            continue
        threshold = _personal_space_radius_m(payload, human)
        points = _trajectory_points(robot)
        in_violation = False
        for index, point in enumerate(points):
            t = _number(point.get("t"), 0.0)
            robot_xy = _pose_xy(point.get("map_pose"))
            human_xy = _actor_xy_at_time(human, t)
            if robot_xy is None or human_xy is None:
                continue
            distance = _distance(robot_xy, human_xy)
            min_distance = distance if min_distance is None else min(min_distance, distance)
            if distance <= threshold:
                if not in_violation:
                    violation_count += 1
                    if human_id:
                        violation_human_ids.add(human_id)
                    in_violation = True
                violation_duration += _sample_dt(points, index)
            else:
                in_violation = False
    return {
        "minimum_non_target_human_distance_m": min_distance,
        "personal_space_violation_count": violation_count,
        "personal_space_violation_duration_s": violation_duration,
        "personal_space_violation_human_ids": sorted(violation_human_ids),
    }


def _social_navigation_law_metrics(
    payload: JsonDict,
    mission: JsonDict,
    robot: JsonDict,
    humans: list[JsonDict],
    active_social_law_ids: list[str],
) -> JsonDict:
    active_laws = set(active_social_law_ids)
    law_success_values: list[bool] = []
    metrics: JsonDict = {}

    if not active_laws or "L1_personal_space" in active_laws:
        task_relevant_human_ids = _task_relevant_human_ids(mission, payload)
        personal_space = _personal_space_metrics(
            payload,
            robot,
            humans,
            excluded_human_ids=task_relevant_human_ids,
        )
        personal_space_respected = personal_space["personal_space_violation_count"] == 0
        law_success_values.append(personal_space_respected)
        metrics.update(
            {
                "personal_space_respected": personal_space_respected,
                **personal_space,
            }
        )

    if "L2_non_obstruction_yield" in active_laws:
        yield_metrics = _pedestrian_yield_metrics(payload, robot, humans)
        law_success_values.append(bool(yield_metrics["pedestrian_yield_respected"]))
        metrics.update(yield_metrics)

    if "L3_group_integrity" in active_laws:
        group_metrics = _group_integrity_metrics(payload, robot)
        law_success_values.append(bool(group_metrics["group_integrity_respected"]))
        metrics.update(group_metrics)

    if "L4_queue_order" in active_laws:
        queue_metrics = _queue_order_metrics(payload, mission, robot)
        law_success_values.append(bool(queue_metrics["queue_order_respected"]))
        metrics.update(queue_metrics)

    metrics["social_law_success"] = all(law_success_values) if law_success_values else True
    return metrics


def _pedestrian_yield_metrics(
    payload: JsonDict,
    robot: JsonDict,
    humans: list[JsonDict],
) -> JsonDict:
    checks: list[JsonDict] = []
    for structure in _social_structures_for_law(
        payload,
        law_id="L2_non_obstruction_yield",
        structure_type="pedestrian_flow",
    ):
        rules = _dict_value(structure.get("rules"))
        geometry = _dict_value(structure.get("geometry"))
        conflict_point = _xy_from_value(geometry.get("conflict_point"))
        if conflict_point is None:
            conflict_point = _xy_from_value(rules.get("conflict_point"))
        human_id = str(rules.get("human_id") or _first_string(structure.get("human_ids")) or "")
        human = _find_by_id(humans, "human_id", human_id)
        conflict_radius_m = _number(
            rules.get("conflict_radius_m"),
            _number(geometry.get("conflict_radius_m"), 0.75),
        )
        yield_margin_s = _number(rules.get("yield_margin_s"), 0.0)
        minimum_distance_m = (
            _number(rules.get("minimum_moving_robot_human_distance_m"), 0.0)
            or (_physical_contact_threshold_m(human) if human is not None else 0.7)
        )

        if conflict_point is None or human is None:
            checks.append(
                {
                    "structure_id": structure.get("structure_id", ""),
                    "human_id": human_id,
                    "pedestrian_yield_respected": False,
                    "failure_reason": "missing_conflict_point_or_human",
                }
            )
            continue

        robot_conflict_time, robot_conflict_distance = _closest_time_to_point(
            robot,
            conflict_point,
        )
        human_conflict_time, human_conflict_distance = _closest_time_to_point(
            human,
            conflict_point,
        )
        min_actor_distance = _min_actor_distance_at_robot_samples(robot, human)
        time_gap = (
            robot_conflict_time - human_conflict_time
            if robot_conflict_time is not None and human_conflict_time is not None
            else None
        )
        human_right_of_way = rules.get("human_right_of_way") is not False
        if human_right_of_way:
            order_respected = time_gap is not None and time_gap >= yield_margin_s
        else:
            order_respected = time_gap is not None and abs(time_gap) >= yield_margin_s
        conflict_reached = (
            robot_conflict_distance is not None
            and robot_conflict_distance <= conflict_radius_m + 1e-6
            and human_conflict_distance is not None
            and human_conflict_distance <= conflict_radius_m + 1e-6
        )
        distance_respected = (
            min_actor_distance is None or min_actor_distance > minimum_distance_m + 1e-6
        )
        respected = conflict_reached and order_respected and distance_respected
        checks.append(
            {
                "structure_id": structure.get("structure_id", ""),
                "human_id": human_id,
                "conflict_point": [conflict_point[0], conflict_point[1]],
                "conflict_radius_m": conflict_radius_m,
                "yield_margin_s": yield_margin_s,
                "robot_conflict_time_s": robot_conflict_time,
                "human_conflict_time_s": human_conflict_time,
                "yield_time_gap_s": time_gap,
                "minimum_moving_robot_human_distance_m": min_actor_distance,
                "required_minimum_moving_distance_m": minimum_distance_m,
                "conflict_reached": conflict_reached,
                "yield_order_respected": order_respected,
                "moving_distance_respected": distance_respected,
                "pedestrian_yield_respected": respected,
            }
        )

    success = bool(checks) and all(check["pedestrian_yield_respected"] for check in checks)
    return {
        "pedestrian_yield_respected": success,
        "pedestrian_yield_check_count": len(checks),
        "pedestrian_yield_violation_count": sum(
            1 for check in checks if not check["pedestrian_yield_respected"]
        ),
        "pedestrian_yield_min_time_gap_s": _min_optional(
            check.get("yield_time_gap_s") for check in checks
        ),
        "pedestrian_yield_min_distance_m": _min_optional(
            check.get("minimum_moving_robot_human_distance_m") for check in checks
        ),
        "pedestrian_yield_checks": checks,
    }


def _group_integrity_metrics(payload: JsonDict, robot: JsonDict) -> JsonDict:
    checks: list[JsonDict] = []
    for structure in _social_structures_for_law(
        payload,
        law_id="L3_group_integrity",
        structure_type="f_formation",
    ):
        region = _group_law_region(structure)
        if region is None:
            checks.append(
                {
                    "structure_id": structure.get("structure_id", ""),
                    "group_integrity_respected": False,
                    "failure_reason": "missing_group_law_region",
                }
            )
            continue
        clearance = _trajectory_region_clearance(robot, region)
        min_clearance = clearance["region_min_clearance_m"]
        violation_count = int(clearance["region_violation_count"])
        if min_clearance is not None and min_clearance >= -GROUP_REGION_CLEARANCE_TOLERANCE_M:
            violation_count = 0
        elif min_clearance is not None and violation_count == 0:
            violation_count = 1
        respected = (
            min_clearance is not None
            and min_clearance >= -GROUP_REGION_CLEARANCE_TOLERANCE_M
        )
        checks.append(
            {
                "structure_id": structure.get("structure_id", ""),
                "group_id": _dict_value(structure.get("rules")).get(
                    "group_id",
                    structure.get("structure_id", ""),
                ),
                "law_region_type": region["type"],
                "law_region_radius_m": region["radius"],
                "group_region_clearance_tolerance_m": GROUP_REGION_CLEARANCE_TOLERANCE_M,
                "group_region_min_clearance_m": min_clearance,
                "group_region_violation_count": violation_count,
                "group_region_violation_duration_s": clearance[
                    "region_violation_duration_s"
                ],
                "group_integrity_respected": respected,
            }
        )

    success = bool(checks) and all(check["group_integrity_respected"] for check in checks)
    return {
        "group_integrity_respected": success,
        "group_integrity_check_count": len(checks),
        "group_region_violation_count": sum(
            int(check.get("group_region_violation_count", 0)) for check in checks
        ),
        "group_region_violation_duration_s": sum(
            float(check.get("group_region_violation_duration_s", 0.0))
            for check in checks
        ),
        "group_region_min_clearance_m": _min_optional(
            check.get("group_region_min_clearance_m") for check in checks
        ),
        "group_integrity_checks": checks,
    }


def _queue_order_metrics(
    payload: JsonDict,
    mission: JsonDict,
    robot: JsonDict,
) -> JsonDict:
    checks: list[JsonDict] = []
    terminal_xy = _terminal_actor_xy(robot)
    for structure in _social_structures_for_law(
        payload,
        law_id="L4_queue_order",
        structure_type="queue",
    ):
        rules = _dict_value(structure.get("rules"))
        geometry = _dict_value(structure.get("geometry"))
        tail_goal = _xy_from_value(geometry.get("queue_tail_goal"))
        if tail_goal is None and rules.get("lawful_endpoint") == "queue_tail":
            tail_goal = _mission_goal_xy(mission)
        service_point = _xy_from_value(geometry.get("service_point"))
        if service_point is None:
            service_point = _xy_from_value(rules.get("service_point"))
        threshold_m = _goal_threshold_m(mission)
        terminal_tail_distance = (
            _distance(terminal_xy, tail_goal)
            if terminal_xy is not None and tail_goal is not None
            else None
        )
        terminal_service_distance = (
            _distance(terminal_xy, service_point)
            if terminal_xy is not None and service_point is not None
            else None
        )
        tail_reached = (
            terminal_tail_distance is not None
            and terminal_tail_distance <= threshold_m + 1e-6
        )
        service_cutting = (
            terminal_service_distance is not None
            and terminal_service_distance <= threshold_m + 1e-6
            and (terminal_tail_distance is None or terminal_service_distance < terminal_tail_distance)
        )
        respected = tail_reached and not service_cutting
        checks.append(
            {
                "structure_id": structure.get("structure_id", ""),
                "queue_count": rules.get("queue_count"),
                "queue_order": _string_list(rules.get("queue_order")),
                "queue_tail_goal": [tail_goal[0], tail_goal[1]] if tail_goal else None,
                "service_point": [service_point[0], service_point[1]]
                if service_point
                else None,
                "queue_tail_threshold_m": threshold_m,
                "queue_terminal_tail_distance_m": terminal_tail_distance,
                "queue_terminal_service_distance_m": terminal_service_distance,
                "queue_tail_reached": tail_reached,
                "queue_cutting_detected": service_cutting,
                "queue_order_respected": respected,
            }
        )

    success = bool(checks) and all(check["queue_order_respected"] for check in checks)
    return {
        "queue_order_respected": success,
        "queue_order_check_count": len(checks),
        "queue_order_violation_count": sum(
            1 for check in checks if not check["queue_order_respected"]
        ),
        "queue_terminal_tail_distance_m": _min_optional(
            check.get("queue_terminal_tail_distance_m") for check in checks
        ),
        "queue_terminal_service_distance_m": _min_optional(
            check.get("queue_terminal_service_distance_m") for check in checks
        ),
        "queue_order_checks": checks,
    }


def _human_guided_resolved_target_xy(mission: JsonDict) -> tuple[float, float] | None:
    metadata = _dict_value(mission.get("metadata"))
    human_guidance = _dict_value(metadata.get("human_guidance"))
    resolved_target = _dict_value(human_guidance.get("resolved_target"))
    for value in (
        resolved_target.get("target_world"),
        _dict_value(human_guidance.get("actual_poi")).get("target_world"),
        _dict_value(_dict_value(metadata.get("hidden_ground_truth")).get("actual_poi")).get(
            "target_world"
        ),
        _dict_value(metadata.get("hidden_ground_truth")).get("exact_target_world"),
    ):
        xy = _xy_from_value(value)
        if xy is not None:
            return xy
    return _mission_goal_xy(mission)


def _previous_queue_missions_completed(
    mission: JsonDict,
    all_missions: list[JsonDict],
    events: list[JsonDict],
) -> bool:
    previous_human_ids = _string_list(
        _dict_value(mission.get("metadata")).get("previous_queue_human_ids")
    )
    if not previous_human_ids:
        return True
    release_time = _number(mission.get("release_time"), 0.0)
    for human_id in previous_human_ids:
        previous_mission = _queue_mission_for_human(all_missions, human_id)
        if previous_mission is None:
            return False
        completion_time = _mission_event_time(
            events,
            str(previous_mission.get("mission_id", "")),
            {"completion", "mission_completion"},
        )
        if completion_time is None or completion_time > release_time + 1e-6:
            return False
    return True


def _serve_queue_order_preserved(
    mission: JsonDict,
    all_missions: list[JsonDict],
    events: list[JsonDict],
) -> bool:
    metadata = _dict_value(mission.get("metadata"))
    queue_order = _string_list(metadata.get("queue_order"))
    target_human_id = str(mission.get("target_human_id", ""))
    if target_human_id not in queue_order:
        return _previous_queue_missions_completed(mission, all_missions, events)

    completion_time = _mission_event_time(
        events,
        str(mission.get("mission_id", "")),
        {"completion", "mission_completion"},
    )
    if completion_time is None:
        return False

    target_index = queue_order.index(target_human_id)
    for index, human_id in enumerate(queue_order):
        other_mission = _queue_mission_for_human(all_missions, human_id)
        if other_mission is None:
            return False
        other_completion = _mission_event_time(
            events,
            str(other_mission.get("mission_id", "")),
            {"completion", "mission_completion"},
        )
        if other_completion is None:
            return False
        if index < target_index and other_completion > completion_time + 1e-6:
            return False
        if index > target_index and other_completion < completion_time - 1e-6:
            return False
    return True


def _queue_mission_for_human(
    missions: list[JsonDict],
    target_human_id: str,
) -> JsonDict | None:
    for mission in missions:
        if (
            mission.get("mission_type") == "serve_queue"
            and str(mission.get("target_human_id", "")) == target_human_id
        ):
            return mission
    return None


def _is_mission_stream_child(mission: JsonDict) -> bool:
    metadata = mission.get("metadata", {})
    return isinstance(metadata, dict) and bool(metadata.get("mission_stream_parent_id"))


def _mission_stream_child_timing_check(mission: JsonDict, events: list[JsonDict]) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    metadata = _dict_value(mission.get("metadata"))
    release_time = _mission_event_time(events, mission_id, {"mission_release"})
    assignment_time = _mission_event_time(events, mission_id, {"robot_assignment", "assign_mission"})
    completion_time = _mission_event_time(events, mission_id, {"completion", "mission_completion"})
    eos_time = _mission_event_time(events, mission_id, {"robot_eos"})
    expected_release_time = _optional_number(mission.get("release_time"))
    expected_assignment_time = _optional_number(metadata.get("assignment_time_s"))
    expected_completion_time = _optional_number(metadata.get("expected_completion_t"))
    expected_eos_time = _optional_number(metadata.get("robot_eos_t"))

    release_matches = _time_matches(release_time, expected_release_time)
    assignment_matches = _time_matches(assignment_time, expected_assignment_time)
    completion_matches = _time_matches(completion_time, expected_completion_time)
    eos_matches = _time_matches(eos_time, expected_eos_time)
    event_order_respected = (
        release_time is not None
        and assignment_time is not None
        and completion_time is not None
        and eos_time is not None
        and release_time <= assignment_time + 1e-6
        and assignment_time <= completion_time + 1e-6
        and completion_time <= eos_time + STREAM_TIMING_TOLERANCE_S
    )
    timing_respected = (
        release_matches
        and assignment_matches
        and completion_matches
        and eos_matches
        and event_order_respected
    )
    return {
        "release_event_present": release_time is not None,
        "assignment_event_present": assignment_time is not None,
        "completion_event_present": completion_time is not None,
        "eos_event_present": eos_time is not None,
        "release_event_time_s": release_time,
        "assignment_event_time_s": assignment_time,
        "completion_time_s": completion_time,
        "eos_time_s": eos_time,
        "expected_release_time_s": expected_release_time,
        "expected_assignment_time_s": expected_assignment_time,
        "expected_completion_time_s": expected_completion_time,
        "expected_eos_time_s": expected_eos_time,
        "release_time_matches_expected": release_matches,
        "assignment_time_matches_expected": assignment_matches,
        "completion_time_matches_expected": completion_matches,
        "eos_time_matches_expected": eos_matches,
        "stream_event_order_respected": event_order_respected,
        "timing_respected": timing_respected,
    }


def _stream_parent_terminal_goal_metrics(metadata: JsonDict, robots: list[JsonDict]) -> JsonDict:
    planned_goals = _dict_value(metadata.get("planned_goal_world_by_robot"))
    checks: list[JsonDict] = []
    for robot_id, raw_goal in sorted(planned_goals.items()):
        goal_xy = _xy_from_value(raw_goal)
        robot = _find_by_id(robots, "robot_id", str(robot_id))
        terminal_xy = _terminal_actor_xy(robot) if robot is not None else None
        distance = (
            _distance(terminal_xy, goal_xy)
            if terminal_xy is not None and goal_xy is not None
            else None
        )
        reached = distance is not None and distance <= _number(metadata.get("goal_tolerance_m"), 0.5)
        checks.append(
            {
                "robot_id": str(robot_id),
                "planned_goal_xy": [goal_xy[0], goal_xy[1]] if goal_xy else None,
                "terminal_xy": [terminal_xy[0], terminal_xy[1]] if terminal_xy else None,
                "terminal_goal_distance_m": distance,
                "terminal_goal_reached": reached,
            }
        )
    return {
        "stream_terminal_goal_check_count": len(checks),
        "stream_terminal_goals_reached": bool(checks)
        and all(check["terminal_goal_reached"] for check in checks),
        "stream_terminal_goal_max_distance_m": _max_optional(
            check.get("terminal_goal_distance_m") for check in checks
        ),
        "stream_terminal_goal_checks": checks,
    }


def _time_matches(actual: float | None, expected: float | None) -> bool:
    if expected is None:
        return actual is not None
    return actual is not None and abs(actual - expected) <= STREAM_TIMING_TOLERANCE_S


def _dense_dynamic_metadata(payload: JsonDict, mission: JsonDict) -> JsonDict:
    payload_metadata = _dict_value(payload.get("metadata"))
    dense_metadata = _dict_value(payload_metadata.get("dense_dynamic_humans"))
    if dense_metadata:
        return dense_metadata
    trajectory_planner = _dict_value(payload_metadata.get("trajectory_planner"))
    planner_metadata = _dict_value(trajectory_planner.get("planner_metadata"))
    if planner_metadata:
        return planner_metadata
    return _dict_value(_dict_value(mission.get("metadata")).get("dense_dynamic_humans"))


def _dense_robot_goal_checks(
    mission: JsonDict,
    robots: list[JsonDict],
    active_robot_ids: list[str],
) -> list[JsonDict]:
    metadata = _dict_value(mission.get("metadata"))
    planned_goals = _dict_value(metadata.get("planned_goal_world_by_robot"))
    if not planned_goals and active_robot_ids:
        mission_goal = _mission_goal_xy(mission)
        if mission_goal is not None:
            planned_goals = {active_robot_ids[0]: [mission_goal[0], mission_goal[1]]}
    deadline = _deadline_s(mission)
    release_time = _number(mission.get("release_time"), 0.0)
    goal_threshold_m = _goal_threshold_m(mission)
    checks: list[JsonDict] = []
    for robot_id in active_robot_ids:
        goal_xy = _xy_from_value(planned_goals.get(robot_id))
        robot = _find_by_id(robots, "robot_id", robot_id)
        if robot is None or goal_xy is None:
            checks.append(
                {
                    "robot_id": robot_id,
                    "goal_reached": False,
                    "failure_reason": "missing_robot_or_goal",
                }
            )
            continue
        goal_reach_time, min_goal_distance = _first_goal_reach_time(
            robot,
            goal_xy,
            goal_threshold_m,
            start_time=release_time,
            end_time=deadline,
        )
        checks.append(
            {
                "robot_id": robot_id,
                "goal_xy": [goal_xy[0], goal_xy[1]],
                "goal_threshold_m": goal_threshold_m,
                "goal_reach_time_s": goal_reach_time,
                "minimum_goal_distance_m": min_goal_distance,
                "goal_reached": min_goal_distance is not None
                and min_goal_distance <= goal_threshold_m + 1e-6,
            }
        )
    return checks


def _dense_human_activity_metrics(
    humans: list[JsonDict],
    completion_time: float | None,
) -> JsonDict:
    checks: list[JsonDict] = []
    for human in humans:
        points = _trajectory_points(human)
        path_distance = _actor_path_distance(human)
        terminal_time = _number(points[-1].get("t"), 0.0) if points else None
        active_until_completion = (
            completion_time is not None
            and terminal_time is not None
            and terminal_time + STREAM_TIMING_TOLERANCE_S >= completion_time
        )
        moved = path_distance > 0.05
        checks.append(
            {
                "human_id": str(human.get("human_id", "")),
                "trajectory_point_count": len(points),
                "path_distance_m": path_distance,
                "terminal_time_s": terminal_time,
                "moved": moved,
                "active_until_completion": active_until_completion,
                "keep_moving_success": moved and active_until_completion,
            }
        )
    return {
        "moving_human_count": len(humans),
        "humans_keep_moving_until_robot_completion": bool(checks)
        and all(check["keep_moving_success"] for check in checks),
        "dense_human_activity_checks": checks,
    }


def _dense_human_human_clearance_metrics(
    humans: list[JsonDict],
    threshold_m: float,
) -> JsonDict:
    sample_times = sorted(
        {
            _number(point.get("t"), 0.0)
            for human in humans
            for point in _trajectory_points(human)
        }
    )
    min_distance: float | None = None
    violation_count = 0
    for index, first in enumerate(humans):
        for second in humans[index + 1 :]:
            for t in sample_times:
                first_xy = _actor_xy_at_time(first, t)
                second_xy = _actor_xy_at_time(second, t)
                if first_xy is None or second_xy is None:
                    continue
                distance = _distance(first_xy, second_xy)
                min_distance = distance if min_distance is None else min(min_distance, distance)
                if distance + 1e-9 < threshold_m:
                    violation_count += 1
    return {
        "minimum_human_human_distance_m": min_distance,
        "minimum_required_human_human_distance_m": threshold_m,
        "human_human_collision_violation_count": violation_count,
        "human_human_collision_free": violation_count == 0,
    }


def _dense_robot_motion_metrics(dense_metadata: JsonDict, robot_ids: list[str]) -> JsonDict:
    adjustments = _dict_value(dense_metadata.get("robot_adjustments"))
    if not adjustments:
        adjustments = _dict_value(dense_metadata.get("agent_adjustments"))
    total_wait_count = 0
    total_inserted_waits = 0
    total_stand_ground = 0
    total_yield_count = 0
    total_teleport_count = 0
    wait_violation_count = 0
    checks: list[JsonDict] = []
    for robot_id in robot_ids:
        adjustment = _dict_value(adjustments.get(robot_id))
        wait_count = int(_number(adjustment.get("wait_count"), 0.0))
        inserted_waits = int(_number(adjustment.get("inserted_waits"), 0.0))
        stand_ground_count = int(_number(adjustment.get("stand_ground_count"), 0.0))
        yield_count = int(_number(adjustment.get("yield_count"), 0.0))
        teleport_count = int(_number(adjustment.get("teleport_count"), 0.0))
        yield_blockers = _dict_value(adjustment.get("yield_blockers"))
        wait_violation = (wait_count + inserted_waits) > 0 and not yield_blockers
        total_wait_count += wait_count
        total_inserted_waits += inserted_waits
        total_stand_ground += stand_ground_count
        total_yield_count += yield_count
        total_teleport_count += teleport_count
        wait_violation_count += int(wait_violation)
        checks.append(
            {
                "robot_id": robot_id,
                "wait_count": wait_count,
                "inserted_waits": inserted_waits,
                "stand_ground_count": stand_ground_count,
                "yield_count": yield_count,
                "teleport_count": teleport_count,
                "yield_blockers": yield_blockers,
                "wait_violation": wait_violation,
            }
        )
    return {
        "robot_wait_count": total_wait_count,
        "robot_inserted_wait_count": total_inserted_waits,
        "robot_stand_ground_count": total_stand_ground,
        "robot_yield_count": total_yield_count,
        "robot_teleport_count": total_teleport_count,
        "robot_wait_violation_count": wait_violation_count,
        "robots_keep_moving_when_passable": wait_violation_count == 0,
        "wait_only_when_immediately_blocked": wait_violation_count == 0,
        "corner_case_recovery_count": int(
            _number(
                _dict_value(_dict_value(dense_metadata.get("corner_case_recovery")).get("summary")).get(
                    "event_count"
                ),
                0.0,
            )
        ),
        "dense_robot_motion_checks": checks,
    }


def _dense_nominal_conflict_count(dense_metadata: JsonDict) -> int:
    conflicts = _dict_value(dense_metadata.get("nominal_robot_human_conflict_samples"))
    return sum(int(_number(value, 0.0)) for value in conflicts.values())


def _dense_multi_robot_metadata(payload: JsonDict, mission: JsonDict) -> JsonDict:
    payload_metadata = _dict_value(payload.get("metadata"))
    dense_metadata = _dict_value(payload_metadata.get("dense_multi_robot"))
    if dense_metadata:
        return dense_metadata
    trajectory_planner = _dict_value(payload_metadata.get("trajectory_planner"))
    planner_metadata = _dict_value(trajectory_planner.get("planner_metadata"))
    if planner_metadata:
        return planner_metadata
    return _dict_value(_dict_value(mission.get("metadata")).get("dense_multi_robot"))


def _dense_dynamic_combined_metadata(payload: JsonDict, mission: JsonDict) -> JsonDict:
    payload_metadata = _dict_value(payload.get("metadata"))
    dense_metadata = _dict_value(payload_metadata.get("dense_dynamic_combined"))
    if dense_metadata:
        return dense_metadata
    trajectory_planner = _dict_value(payload_metadata.get("trajectory_planner"))
    planner_metadata = _dict_value(trajectory_planner.get("planner_metadata"))
    if planner_metadata:
        return planner_metadata
    return _dict_value(_dict_value(mission.get("metadata")).get("dense_dynamic_combined"))


def _dense_robot_robot_clearance_metrics(
    payload: JsonDict,
    mission: JsonDict,
    robots: list[JsonDict],
    active_robot_ids: list[str],
) -> JsonDict:
    metadata = _dict_value(mission.get("metadata"))
    required_distance = _optional_number(metadata.get("minimum_robot_robot_distance_m"))
    collision_check = _dict_value(_dict_value(payload.get("metadata")).get("collision_check"))
    closest_pair = _dict_value(collision_check.get("closest_pair"))
    active_robot_id_set = set(active_robot_ids)
    closest_pair_ids = [
        str(value)
        for value in (closest_pair.get("actor_a"), closest_pair.get("actor_b"))
        if value
    ]
    closest_pair_is_robot_robot = (
        len(closest_pair_ids) == 2
        and all(actor_id in active_robot_id_set for actor_id in closest_pair_ids)
    )
    minimum_distance = (
        _optional_number(closest_pair.get("distance_m"))
        if closest_pair_is_robot_robot
        else None
    )
    closest_pair_time = (
        _optional_number(closest_pair.get("t")) if closest_pair_is_robot_robot else None
    )
    if minimum_distance is None:
        sampled = _sampled_robot_robot_min_distance(robots, active_robot_ids)
        minimum_distance = sampled["minimum_robot_robot_distance_m"]
        closest_pair_ids = sampled["closest_robot_pair"]
        closest_pair_time = sampled["closest_robot_pair_time_s"]

    minimum_clearance = (
        _optional_number(closest_pair.get("clearance_m"))
        if closest_pair_is_robot_robot
        else None
    )
    if minimum_clearance is None:
        checked_min_clearance = _optional_number(collision_check.get("min_clearance_m"))
        if closest_pair_is_robot_robot:
            minimum_clearance = checked_min_clearance
        elif minimum_distance is not None and required_distance is not None:
            minimum_clearance = minimum_distance - required_distance
    required_clearance = (
        _optional_number(closest_pair.get("required_clearance_m"))
        if closest_pair_is_robot_robot
        else required_distance
    )
    robot_robot_collision_count = _robot_robot_collision_count(
        collision_check,
        active_robot_id_set,
    )
    no_collision = robot_robot_collision_count == 0
    distance_respected = (
        required_distance is None
        or minimum_distance is None
        or minimum_distance + 1e-6 >= required_distance
    )
    clearance_respected = no_collision and distance_respected
    return {
        "no_robot_robot_collision": no_collision,
        "robot_robot_clearance_policy_respected": clearance_respected,
        "robot_robot_collision_count": robot_robot_collision_count,
        "minimum_robot_robot_distance_m": minimum_distance,
        "minimum_required_robot_robot_distance_m": required_distance,
        "minimum_robot_robot_clearance_m": minimum_clearance,
        "required_robot_robot_clearance_m": required_clearance,
        "closest_robot_pair": closest_pair_ids,
        "closest_robot_pair_time_s": closest_pair_time,
    }


def _robot_robot_collision_count(collision_check: JsonDict, active_robot_ids: set[str]) -> int:
    collisions = _dict_list(collision_check.get("collisions"))
    if collisions:
        count = 0
        for collision in collisions:
            actor_ids = {
                str(value)
                for value in (collision.get("actor_a"), collision.get("actor_b"))
                if value
            }
            if len(actor_ids) == 2 and actor_ids <= active_robot_ids:
                count += 1
        return count

    collision_count = int(_number(collision_check.get("collision_count"), 0.0))
    closest_pair = _dict_value(collision_check.get("closest_pair"))
    closest_actor_ids = {
        str(value)
        for value in (closest_pair.get("actor_a"), closest_pair.get("actor_b"))
        if value
    }
    if len(closest_actor_ids) == 2 and closest_actor_ids <= active_robot_ids:
        return collision_count
    if collision_count == 0:
        return 0
    return 0


def _sampled_robot_robot_min_distance(
    robots: list[JsonDict],
    active_robot_ids: list[str],
) -> JsonDict:
    active_robots = [
        robot
        for robot in robots
        if str(robot.get("robot_id", "")) in set(active_robot_ids)
    ]
    sample_times = sorted(
        {
            _number(point.get("t"), 0.0)
            for robot in active_robots
            for point in _trajectory_points(robot)
        }
    )
    min_distance: float | None = None
    closest_pair: list[str] = []
    closest_time: float | None = None
    for index, first in enumerate(active_robots):
        first_id = str(first.get("robot_id", ""))
        for second in active_robots[index + 1 :]:
            second_id = str(second.get("robot_id", ""))
            for t in sample_times:
                first_xy = _actor_xy_at_time(first, t)
                second_xy = _actor_xy_at_time(second, t)
                if first_xy is None or second_xy is None:
                    continue
                distance = _distance(first_xy, second_xy)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    closest_pair = [first_id, second_id]
                    closest_time = t
    return {
        "minimum_robot_robot_distance_m": min_distance,
        "closest_robot_pair": closest_pair,
        "closest_robot_pair_time_s": closest_time,
    }


def _dense_multi_robot_motion_metrics(
    dense_metadata: JsonDict,
    mission_metadata: JsonDict,
    robot_ids: list[str],
) -> JsonDict:
    adjustments = _dict_value(dense_metadata.get("robot_adjustments"))
    if not adjustments:
        adjustments = _dict_value(dense_metadata.get("agent_adjustments"))
    expected_motion = _dict_value(mission_metadata.get("expected_robot_motion"))
    requires_stops = expected_motion.get("requires_stops") is True
    total_wait_count = 0
    total_inserted_waits = 0
    total_glide_count = 0
    total_inserted_turns = 0
    total_teleport_count = 0
    wait_violation_count = 0
    checks: list[JsonDict] = []
    for robot_id in robot_ids:
        adjustment = _dict_value(adjustments.get(robot_id))
        wait_count = int(_number(adjustment.get("wait_count"), 0.0))
        inserted_waits = int(_number(adjustment.get("inserted_waits"), 0.0))
        glide_count = int(_number(adjustment.get("glide_count"), 0.0))
        inserted_turns = int(_number(adjustment.get("inserted_turns"), 0.0))
        teleport_count = int(_number(adjustment.get("teleport_count"), 0.0))
        yield_blockers = _dict_value(adjustment.get("yield_blockers"))
        unblocked_wait = (
            (wait_count + inserted_waits) > 0
            and not yield_blockers
            and not requires_stops
        )
        total_wait_count += wait_count
        total_inserted_waits += inserted_waits
        total_glide_count += glide_count
        total_inserted_turns += inserted_turns
        total_teleport_count += teleport_count
        wait_violation_count += int(unblocked_wait)
        checks.append(
            {
                "robot_id": robot_id,
                "wait_count": wait_count,
                "inserted_waits": inserted_waits,
                "glide_count": glide_count,
                "inserted_turns": inserted_turns,
                "teleport_count": teleport_count,
                "yield_blockers": yield_blockers,
                "wait_violation": unblocked_wait,
            }
        )

    recovery_summary = _dict_value(
        _dict_value(dense_metadata.get("corner_case_recovery")).get("summary")
    )
    recovery_by_issue = _dict_value(recovery_summary.get("by_issue_type"))
    return {
        "dense_multi_requires_stops": requires_stops,
        "robot_wait_count": total_wait_count,
        "robot_inserted_wait_count": total_inserted_waits,
        "robot_glide_count": total_glide_count,
        "robot_inserted_turn_count": total_inserted_turns,
        "robot_teleport_count": total_teleport_count,
        "robot_wait_violation_count": wait_violation_count,
        "robots_keep_moving_when_passable": wait_violation_count == 0,
        "wait_only_when_immediately_blocked": wait_violation_count == 0,
        "corner_case_recovery_count": int(_number(recovery_summary.get("event_count"), 0.0)),
        "robot_robot_deadlock_recovery_count": int(
            _number(recovery_by_issue.get("robot_robot_deadlock"), 0.0)
        ),
        "robot_stalled_recovery_count": int(
            _number(recovery_by_issue.get("robot_stalled_without_goal_progress"), 0.0)
        ),
        "dense_robot_motion_checks": checks,
    }


def _human_guided_informant_id(mission: JsonDict) -> str:
    metadata = _dict_value(mission.get("metadata"))
    human_guidance = _dict_value(metadata.get("human_guidance"))
    informant_id = human_guidance.get("informant_human_id") or metadata.get(
        "informant_human_id"
    )
    if informant_id:
        return str(informant_id)
    return _first_string(metadata.get("active_human_ids"))


def _human_guidance_stop_metrics(
    mission: JsonDict,
    robot: JsonDict,
    request_time: float | None,
    response_time: float | None,
) -> JsonDict:
    expected_actions = _dict_value(
        _dict_value(_dict_value(mission.get("metadata")).get("human_guidance")).get(
            "expected_robot_actions"
        )
    )
    stop_required = expected_actions.get("must_stop_for_guidance") is True
    interval = expected_actions.get("stop_interval_s")
    stop_interval: list[float] | None = None
    if isinstance(interval, list) and len(interval) >= 2:
        stop_interval = [_number(interval[0], 0.0), _number(interval[1], 0.0)]
    elif request_time is not None and response_time is not None:
        stop_interval = [request_time, response_time]

    stop_verified: bool | None = None
    stopped_distance_m: float | None = None
    motion_states: list[str] = []
    if stop_interval is not None:
        stop_verified, stopped_distance_m, motion_states = _robot_stopped_in_interval(
            robot,
            stop_interval[0],
            stop_interval[1],
        )
        if request_time is not None and response_time is not None:
            stop_verified = (
                stop_verified
                and stop_interval[0] - 1e-6 <= request_time <= stop_interval[1] + 1e-6
                and stop_interval[0] - 1e-6 <= response_time <= stop_interval[1] + 1e-6
            )

    return {
        "guidance_stop_required": stop_required,
        "guidance_stop_interval_s": stop_interval,
        "guidance_stop_verified": stop_verified,
        "guidance_stop_motion_states": motion_states,
        "guidance_stop_max_displacement_m": stopped_distance_m,
    }


def _mission_goal_xy(mission: JsonDict) -> tuple[float, float] | None:
    metadata = mission.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("planned_goal_world", "goal_world", "target_world"):
        value = metadata.get(key)
        xy = _xy_from_value(value)
        if xy is not None:
            return xy
    for key in ("target_pose", "goal_pose", "target_region_pose"):
        value = metadata.get(key)
        xy = _xy_from_value(value)
        if xy is not None:
            return xy
    return None


def _xy_from_value(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        return _pose_xy(value)
    if isinstance(value, list) and len(value) >= 2:
        return (_number(value[0], 0.0), _number(value[1], 0.0))
    return None


def _goal_threshold_m(mission: JsonDict) -> float:
    metadata = mission.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("goal_tolerance_m", "success_distance_m", "target_radius_m"):
            value = metadata.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return 0.5


def _first_goal_reach_time(
    robot: JsonDict,
    goal_xy: tuple[float, float],
    threshold_m: float,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[float | None, float | None]:
    min_distance: float | None = None
    reach_time: float | None = None
    for point in _trajectory_points(robot):
        t = _number(point.get("t"), 0.0)
        if start_time is not None and t + 1e-6 < start_time:
            continue
        if end_time is not None and t - 1e-6 > end_time:
            continue
        robot_xy = _pose_xy(point.get("map_pose"))
        if robot_xy is None:
            continue
        distance = _distance(robot_xy, goal_xy)
        min_distance = distance if min_distance is None else min(min_distance, distance)
        if reach_time is None and distance <= threshold_m:
            reach_time = t
    return reach_time, min_distance


def _task_relevant_human_ids(mission: JsonDict, payload: JsonDict) -> set[str]:
    ids: set[str] = set()
    target_human_id = mission.get("target_human_id")
    if target_human_id:
        ids.add(str(target_human_id))
    metadata = mission.get("metadata", {})
    if isinstance(metadata, dict):
        ids.update(_string_list(metadata.get("task_relevant_human_ids")))
    for structure in _dict_list(payload.get("social_structures")):
        rules = structure.get("rules", {})
        if isinstance(rules, dict):
            ids.update(_string_list(rules.get("task_relevant_human_ids")))
    ids.discard("")
    return ids


def _social_structures_for_law(
    payload: JsonDict,
    *,
    law_id: str,
    structure_type: str,
) -> list[JsonDict]:
    matches: list[JsonDict] = []
    for structure in _dict_list(payload.get("social_structures")):
        law_ids = set(_string_list(structure.get("law_ids")))
        rules = _dict_value(structure.get("rules"))
        law_ids.update(_string_list(rules.get("law_ids")))
        if law_id not in law_ids and str(structure.get("structure_type", "")) != structure_type:
            continue
        matches.append(structure)
    return matches


def _active_social_law_ids(mission: JsonDict) -> list[str]:
    ids = set(_string_list(mission.get("social_law_ids")))
    metadata = mission.get("metadata", {})
    if isinstance(metadata, dict):
        ids.update(_string_list(metadata.get("active_social_law_ids")))
    ids.discard("")
    return sorted(ids)


def _dict_value(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _first_string(value: Any) -> str:
    values = _string_list(value)
    return values[0] if values else ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _robot_human_collision_count(
    payload: JsonDict,
    robot: JsonDict,
    humans: list[JsonDict],
) -> int:
    collision_count = _checked_collision_count(payload)
    if collision_count is not None:
        return collision_count

    count = 0
    for human in humans:
        threshold = _physical_contact_threshold_m(human)
        contact_time, _ = _first_contact_time(robot, human, threshold)
        if contact_time is not None:
            count += 1
    return count


def _checked_collision_count(payload: JsonDict) -> int | None:
    collision_check = _dict_value(_dict_value(payload.get("metadata")).get("collision_check"))
    if collision_check.get("checked") is not True:
        return None
    collision_count = collision_check.get("collision_count")
    if isinstance(collision_count, (int, float)) and not isinstance(collision_count, bool):
        return int(collision_count)
    return None


def _physical_contact_threshold_m(human: JsonDict) -> float:
    social_defaults = human.get("social_defaults", {})
    if isinstance(social_defaults, dict):
        metadata = social_defaults.get("metadata", {})
        if isinstance(metadata, dict):
            collision_geometry = metadata.get("collision_geometry", {})
            if isinstance(collision_geometry, dict):
                human_radius = collision_geometry.get("radius_m")
                if isinstance(human_radius, (int, float)) and not isinstance(human_radius, bool):
                    return 0.3 + float(human_radius)
    return 0.7


def _personal_space_radius_m(payload: JsonDict, human: JsonDict) -> float:
    social_defaults = human.get("social_defaults", {})
    if isinstance(social_defaults, dict):
        radius = social_defaults.get("personal_space_radius")
        if isinstance(radius, (int, float)) and not isinstance(radius, bool):
            return float(radius)
    return _number(_collision_defaults(payload).get("human_personal_space_radius_m"), 0.8)


def _collision_defaults(payload: JsonDict) -> JsonDict:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    defaults = metadata.get("collision_defaults", {})
    return defaults if isinstance(defaults, dict) else {}


def _trajectory_points(actor: JsonDict) -> list[JsonDict]:
    return sorted(
        _dict_list(actor.get("trajectory")),
        key=lambda point: _number(point.get("t"), 0.0),
    )


def _actor_path_distance(actor: JsonDict) -> float:
    points = _trajectory_points(actor)
    distance = 0.0
    previous_xy: tuple[float, float] | None = None
    for point in points:
        xy = _pose_xy(point.get("map_pose"))
        if xy is None:
            continue
        if previous_xy is not None:
            distance += _distance(previous_xy, xy)
        previous_xy = xy
    return distance


def _actor_xy_at_time(actor: JsonDict, t: float) -> tuple[float, float] | None:
    points = _trajectory_points(actor)
    if not points:
        return _pose_xy(actor.get("start_map_pose"))
    previous = points[0]
    for point in points:
        point_t = _number(point.get("t"), 0.0)
        if point_t > t:
            break
        previous = point
    return _pose_xy(previous.get("map_pose"))


def _terminal_actor_xy(actor: JsonDict) -> tuple[float, float] | None:
    points = _trajectory_points(actor)
    if points:
        return _pose_xy(points[-1].get("map_pose"))
    return _pose_xy(actor.get("start_map_pose"))


def _robot_stopped_in_interval(
    robot: JsonDict,
    start_time: float,
    end_time: float,
) -> tuple[bool, float | None, list[str]]:
    points = [
        point
        for point in _trajectory_points(robot)
        if start_time - 1e-6 <= _number(point.get("t"), 0.0) <= end_time + 1e-6
    ]
    if not points:
        return False, None, []
    reference_xy = _pose_xy(points[0].get("map_pose"))
    max_displacement = 0.0
    motion_states: list[str] = []
    for point in points:
        state = str(point.get("motion_state", ""))
        if state:
            motion_states.append(state)
        xy = _pose_xy(point.get("map_pose"))
        if reference_xy is None or xy is None:
            continue
        max_displacement = max(max_displacement, _distance(reference_xy, xy))
    allowed_states = {"idle", "waiting_for_guidance", "turn_to_face_human", "stopped"}
    state_ok = not motion_states or all(state in allowed_states for state in motion_states)
    return max_displacement <= 0.05 and state_ok, max_displacement, motion_states


def _closest_time_to_point(
    actor: JsonDict,
    target_xy: tuple[float, float],
) -> tuple[float | None, float | None]:
    closest_time: float | None = None
    closest_distance: float | None = None
    for point in _trajectory_points(actor):
        xy = _pose_xy(point.get("map_pose"))
        if xy is None:
            continue
        distance = _distance(xy, target_xy)
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_time = _number(point.get("t"), 0.0)
    return closest_time, closest_distance


def _min_actor_distance_at_robot_samples(
    robot: JsonDict,
    other_actor: JsonDict,
) -> float | None:
    min_distance: float | None = None
    for point in _trajectory_points(robot):
        t = _number(point.get("t"), 0.0)
        robot_xy = _pose_xy(point.get("map_pose"))
        other_xy = _actor_xy_at_time(other_actor, t)
        if robot_xy is None or other_xy is None:
            continue
        distance = _distance(robot_xy, other_xy)
        min_distance = distance if min_distance is None else min(min_distance, distance)
    return min_distance


def _pose_xy(pose: Any) -> tuple[float, float] | None:
    if not isinstance(pose, dict):
        return None
    return (_number(pose.get("x"), 0.0), _number(pose.get("y"), 0.0))


def _xy_list(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        xy = _xy_from_value(item)
        if xy is not None:
            points.append(xy)
    return points


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _group_law_region(structure: JsonDict) -> JsonDict | None:
    rules = _dict_value(structure.get("rules"))
    geometry = _dict_value(structure.get("geometry"))
    hull = _xy_list(geometry.get("law_region_hull_world"))
    if not hull:
        hull = _xy_list(geometry.get("hull_world"))
    radius = _number(
        rules.get("law_region_inflation_radius_m"),
        _number(geometry.get("law_region_inflation_radius_m"), 0.0),
    )
    if len(hull) >= 2 and radius > 0.0:
        return {
            "type": "capsule",
            "points": hull,
            "radius": radius,
        }

    center = _xy_from_value(geometry.get("o_space_center"))
    circle_radius = _number(
        rules.get("o_space_radius_m"),
        _number(geometry.get("o_space_radius"), _number(geometry.get("radius"), 0.0)),
    )
    if center is not None and circle_radius > 0.0:
        return {
            "type": "circle",
            "center": center,
            "radius": circle_radius,
        }
    return None


def _trajectory_region_clearance(robot: JsonDict, region: JsonDict) -> JsonDict:
    points = _trajectory_points(robot)
    min_clearance: float | None = None
    violation_count = 0
    violation_duration = 0.0
    in_violation = False
    for index, point in enumerate(points):
        xy = _pose_xy(point.get("map_pose"))
        if xy is None:
            continue
        clearance = _region_clearance(xy, region)
        min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
        if clearance <= 0.0:
            if not in_violation:
                violation_count += 1
                in_violation = True
            violation_duration += _sample_dt(points, index)
        else:
            in_violation = False

    for first, second in zip(points, points[1:]):
        start = _pose_xy(first.get("map_pose"))
        end = _pose_xy(second.get("map_pose"))
        if start is None or end is None:
            continue
        clearance = _region_segment_clearance(start, end, region)
        min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)

    return {
        "region_min_clearance_m": min_clearance,
        "region_violation_count": violation_count,
        "region_violation_duration_s": violation_duration,
    }


def _region_clearance(point: tuple[float, float], region: JsonDict) -> float:
    radius = _number(region.get("radius"), 0.0)
    if region.get("type") == "capsule":
        points = region.get("points", [])
        if isinstance(points, list) and len(points) >= 2:
            return _polyline_distance(point, points) - radius
    center = region.get("center")
    if isinstance(center, tuple):
        return _distance(point, center) - radius
    return float("inf")


def _region_segment_clearance(
    start: tuple[float, float],
    end: tuple[float, float],
    region: JsonDict,
) -> float:
    radius = _number(region.get("radius"), 0.0)
    if region.get("type") == "capsule":
        points = region.get("points", [])
        if isinstance(points, list) and len(points) >= 2:
            return _segment_polyline_distance(start, end, points) - radius
    center = region.get("center")
    if isinstance(center, tuple):
        return _point_to_segment_distance(center, start, end) - radius
    return float("inf")


def _polyline_distance(point: tuple[float, float], points: list[tuple[float, float]]) -> float:
    return min(
        _point_to_segment_distance(point, start, end)
        for start, end in zip(points, points[1:])
    )


def _segment_polyline_distance(
    start: tuple[float, float],
    end: tuple[float, float],
    points: list[tuple[float, float]],
) -> float:
    return min(
        _segment_distance(start, end, first, second)
        for first, second in zip(points, points[1:])
    )


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    wx = point[0] - start[0]
    wy = point[1] - start[1]
    denominator = vx * vx + vy * vy
    if denominator == 0.0:
        return _distance(point, start)
    ratio = max(0.0, min(1.0, (wx * vx + wy * vy) / denominator))
    closest = (start[0] + ratio * vx, start[1] + ratio * vy)
    return _distance(point, closest)


def _segment_distance(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> float:
    if _segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        _point_to_segment_distance(first_start, second_start, second_end),
        _point_to_segment_distance(first_end, second_start, second_end),
        _point_to_segment_distance(second_start, first_start, first_end),
        _point_to_segment_distance(second_end, first_start, first_end),
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    def orientation(
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
    ) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
    ) -> bool:
        return (
            min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
            and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
        )

    o1 = orientation(first_start, first_end, second_start)
    o2 = orientation(first_start, first_end, second_end)
    o3 = orientation(second_start, second_end, first_start)
    o4 = orientation(second_start, second_end, first_end)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    if abs(o1) <= 1e-9 and on_segment(first_start, second_start, first_end):
        return True
    if abs(o2) <= 1e-9 and on_segment(first_start, second_end, first_end):
        return True
    if abs(o3) <= 1e-9 and on_segment(second_start, first_start, second_end):
        return True
    if abs(o4) <= 1e-9 and on_segment(second_start, first_end, second_end):
        return True
    return False


def _min_optional(values: Any) -> float | None:
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return min(numeric_values) if numeric_values else None


def _max_optional(values: Any) -> float | None:
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return max(numeric_values) if numeric_values else None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_int(value: Any) -> int | None:
    number = _optional_number(value)
    return int(number) if number is not None else None


def _sample_dt(points: list[JsonDict], index: int) -> float:
    if len(points) < 2:
        return 0.0
    current_t = _number(points[index].get("t"), 0.0)
    if index + 1 < len(points):
        return max(0.0, _number(points[index + 1].get("t"), current_t) - current_t)
    return max(0.0, current_t - _number(points[index - 1].get("t"), current_t))


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default
