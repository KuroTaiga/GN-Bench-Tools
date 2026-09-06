"""JSON-level baseline runner for NavDP human-centric episodes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from GN_Bench_baselines.human_eval.policies import (
    GreedyNearestPolicy,
    HumanAwareGreedyPolicy,
    NoHumanAwarenessPolicy,
    OracleHumanCentricPolicy,
    PriorityGreedyPolicy,
    SingleRobotSerialPolicy,
)

from .scenario_adapter import HumanCentricEpisode


JsonDict = dict[str, Any]


POLICY_CLASSES = {
    "oracle_route_follower": OracleHumanCentricPolicy,
    "shortest_path_no_human": NoHumanAwarenessPolicy,
    "human_aware_greedy": HumanAwareGreedyPolicy,
    "priority_deadline_greedy": PriorityGreedyPolicy,
    "single_robot_serial": SingleRobotSerialPolicy,
    # Backward-compatible names used by the initial GN0 bridge.
    "oracle_human_centric": OracleHumanCentricPolicy,
    "greedy_nearest": GreedyNearestPolicy,
    "priority_greedy": PriorityGreedyPolicy,
    "no_human_awareness": NoHumanAwarenessPolicy,
}
DEFAULT_POLICY_NAMES = (
    "oracle_route_follower",
    "shortest_path_no_human",
    "human_aware_greedy",
    "priority_deadline_greedy",
    "single_robot_serial",
)


def run_policy_assignment_sweep(
    policy_name: str,
    episode: HumanCentricEpisode,
    *,
    max_steps: int | None = None,
) -> JsonDict:
    """Run one deterministic assignment sweep for a baseline policy.

    The sweep exercises policy decision logic over released missions without
    claiming to be a full simulator rollout. Mission completion is still
    measured by ``HumanCentricEvaluator`` against the frozen NavDP event log.
    """

    policy_cls = POLICY_CLASSES[policy_name]
    policy = policy_cls()
    policy.reset(episode.payload)

    missions = _missions(episode.payload)
    mission_count = len(missions)
    release_times = _release_times(missions)
    current_time = release_times[0] if release_times else 0.0
    assigned_mission_ids: set[str] = set()
    actions: list[JsonDict] = []
    max_steps = max_steps or max(1, mission_count + len(release_times) + 2)

    for step_index in range(max_steps):
        action = policy.act(
            {
                "time": current_time,
                "assigned_mission_ids": sorted(assigned_mission_ids),
                "completed_mission_ids": [],
            }
        )
        record = asdict(action)
        record["step_index"] = step_index
        record["time"] = current_time
        actions.append(record)

        if action.action_type == "assign_mission":
            mission_id = str(action.payload.get("mission_id", ""))
            if mission_id:
                assigned_mission_ids.add(mission_id)
            if len(assigned_mission_ids) >= mission_count:
                break
            continue

        next_time = _next_release_time(release_times, current_time, missions, assigned_mission_ids)
        if next_time is None:
            break
        current_time = next_time

    assignment_actions = [action for action in actions if action["action_type"] == "assign_mission"]
    no_op_actions = [action for action in actions if action["action_type"] == "no_op"]
    oracle_match_count, oracle_match_rate = _oracle_assignment_match(missions, assignment_actions)
    return {
        "policy_name": policy_name,
        "episode_id": episode.episode_id,
        "metrics": {
            "mission_count": mission_count,
            "action_count": len(actions),
            "assignment_count": len(assignment_actions),
            "no_op_count": len(no_op_actions),
            "assigned_mission_count": len(assigned_mission_ids),
            "assignment_coverage": len(assigned_mission_ids) / mission_count
            if mission_count
            else 0.0,
            "oracle_assignment_match_count": oracle_match_count,
            "oracle_assignment_match_rate": oracle_match_rate,
        },
        "actions": actions,
    }


def run_policies_on_episode(
    episode: HumanCentricEpisode,
    policy_names: list[str] | None = None,
) -> dict[str, JsonDict]:
    names = policy_names or list(DEFAULT_POLICY_NAMES)
    return {
        policy_name: run_policy_assignment_sweep(policy_name, episode)
        for policy_name in names
    }


def summarize_policy_results(results: list[JsonDict]) -> JsonDict:
    summary: JsonDict = {"episode_count": len(results), "policies": {}}
    for policy_name in POLICY_CLASSES:
        policy_rows = [
            row["policies"][policy_name]["metrics"]
            for row in results
            if policy_name in row.get("policies", {})
        ]
        if not policy_rows:
            continue
        summary["policies"][policy_name] = {
            "mean_assignment_coverage": _mean(row["assignment_coverage"] for row in policy_rows),
            "mean_oracle_assignment_match_rate": _mean(
                row["oracle_assignment_match_rate"] for row in policy_rows
            ),
            "total_assignments": sum(int(row["assignment_count"]) for row in policy_rows),
        }
    return summary


def _missions(payload: JsonDict) -> list[JsonDict]:
    missions = payload.get("missions", [])
    if not isinstance(missions, list):
        return []
    return [mission for mission in missions if isinstance(mission, dict)]


def _release_times(missions: list[JsonDict]) -> list[float]:
    return sorted({_number(mission.get("release_time"), 0.0) for mission in missions})


def _next_release_time(
    release_times: list[float],
    current_time: float,
    missions: list[JsonDict],
    assigned_mission_ids: set[str],
) -> float | None:
    for release_time in release_times:
        if release_time <= current_time:
            continue
        for mission in missions:
            mission_id = str(mission.get("mission_id", ""))
            if mission_id in assigned_mission_ids:
                continue
            if _number(mission.get("release_time"), 0.0) <= release_time:
                return release_time
    return None


def _oracle_assignment_match(
    missions: list[JsonDict],
    assignment_actions: list[JsonDict],
) -> tuple[int, float | None]:
    oracle_robot_by_mission = {
        str(mission.get("mission_id", "")): str(mission.get("assigned_robot_id", ""))
        for mission in missions
        if mission.get("mission_id") and mission.get("assigned_robot_id")
    }
    comparable = 0
    matches = 0
    for action in assignment_actions:
        mission_id = str(action.get("payload", {}).get("mission_id", ""))
        oracle_robot_id = oracle_robot_by_mission.get(mission_id)
        if not oracle_robot_id:
            continue
        comparable += 1
        if str(action.get("robot_id", "")) == oracle_robot_id:
            matches += 1
    return matches, matches / comparable if comparable else None


def _mean(values: Any) -> float | None:
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default
