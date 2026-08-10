"""RL task skeleton for human-centric GN-Bench evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .evaluator import HumanCentricEvaluator, ReplayResult
from .scenario_adapter import HumanCentricEpisode


JsonDict = dict[str, Any]
SUPPORTED_ACTION_TYPES = {
    "assign_mission",
    "reassign_mission",
    "set_subgoal",
    "interact",
    "no_op",
    "evaluate_replay",
}


@dataclass
class HumanCentricRLState:
    """Mutable RL episode state extracted from a human-centric scenario."""

    episode: HumanCentricEpisode
    step_index: int = 0
    current_time_s: float = 0.0
    assigned_mission_ids: set[str] = field(default_factory=set)
    completed_mission_ids: set[str] = field(default_factory=set)
    action_log: list[JsonDict] = field(default_factory=list)
    last_observation: JsonDict = field(default_factory=dict)
    last_result: ReplayResult | None = None
    done: bool = False


class HumanCentricRLTask:
    """Replay-backed task wrapper for training/evaluating human-centric agents."""

    def __init__(self, evaluator: HumanCentricEvaluator | None = None) -> None:
        self.evaluator = evaluator or HumanCentricEvaluator()
        self.state: HumanCentricRLState | None = None

    def reset(self, episode: HumanCentricEpisode) -> JsonDict:
        """Initialize an RL episode from a NavDP-generated scenario."""

        release_times = _release_times(_missions(episode.payload))
        self.state = HumanCentricRLState(
            episode=episode,
            current_time_s=release_times[0] if release_times else 0.0,
        )
        observation = _observation_for_state(self.state)
        self.state.last_observation = observation
        return observation

    def step(self, action: JsonDict) -> tuple[JsonDict, float, bool, JsonDict]:
        """Apply one JSON action and return a deterministic replay reward if done."""

        if self.state is None:
            raise RuntimeError("HumanCentricRLTask.reset must be called before step")
        if self.state.done:
            return self.state.last_observation, 0.0, True, {"already_done": True}

        action_record = _normalize_action(action, self.state.step_index, self.state.current_time_s)
        self._apply_action(action_record)
        self.state.step_index += 1
        self.state.current_time_s = _next_task_time_s(self.state)

        done = self._should_finish(action_record)
        reward = 0.0
        info: JsonDict = {"accepted_action": action_record}
        if done:
            replay = self.evaluator.replay(self.state.episode)
            self.state.last_result = replay
            self.state.completed_mission_ids = {
                str(result.get("mission_id", ""))
                for result in replay.mission_results
                if result.get("success") is True and result.get("mission_id")
            }
            reward, reward_components = replay_reward(replay)
            info.update(
                {
                    "replay": asdict(replay),
                    "reward_components": reward_components,
                    "action_log": list(self.state.action_log),
                }
            )
            self.state.done = True

        observation = _observation_for_state(self.state)
        self.state.last_observation = observation
        return observation, reward, self.state.done, info

    def _apply_action(self, action: JsonDict) -> None:
        if self.state is None:
            raise RuntimeError("HumanCentricRLTask.reset must be called before step")
        action_type = str(action.get("action_type", ""))
        payload = _dict_value(action.get("payload"))
        mission_id = str(payload.get("mission_id", ""))
        if action_type in {"assign_mission", "reassign_mission"} and mission_id:
            self.state.assigned_mission_ids.add(mission_id)
        self.state.action_log.append(action)

    def _should_finish(self, action: JsonDict) -> bool:
        if self.state is None:
            return True
        if action.get("action_type") == "evaluate_replay":
            return True
        mission_ids = {str(mission.get("mission_id", "")) for mission in _missions(self.state.episode.payload)}
        mission_ids.discard("")
        if not mission_ids:
            return True
        if mission_ids <= self.state.assigned_mission_ids:
            return True
        return _next_unassigned_release_time(self.state, mission_ids) is None


def replay_reward(replay: ReplayResult) -> tuple[float, JsonDict]:
    """Map deterministic replay metrics into a bounded scalar reward."""

    mission_success = _optional_float(replay.metrics.get("mission_success_rate"))
    if mission_success is None:
        mission_success = 1.0 if replay.success is True else 0.0
    completion = _optional_float(replay.metrics.get("completion_rate"))
    if completion is None:
        completion = mission_success
    collision_count = _float_value(replay.metrics.get("collision_count"), 0.0)
    personal_space_violations = _sum_metric_suffix(replay.metrics, "personal_space_violation_count")
    robot_wait_violations = _sum_metric_suffix(replay.metrics, "robot_wait_violation_count")
    reward = (
        mission_success
        + 0.25 * completion
        - 0.05 * collision_count
        - 0.01 * personal_space_violations
        - 0.02 * robot_wait_violations
    )
    reward = max(-1.0, min(1.0, reward))
    return reward, {
        "mission_success": mission_success,
        "completion": completion,
        "collision_penalty": -0.05 * collision_count,
        "personal_space_penalty": -0.01 * personal_space_violations,
        "robot_wait_penalty": -0.02 * robot_wait_violations,
        "bounded_reward": reward,
    }


def _observation_for_state(state: HumanCentricRLState) -> JsonDict:
    payload = state.episode.payload
    missions = _missions(payload)
    mission_summaries = [
        _mission_observation(mission, state)
        for mission in sorted(
            missions,
            key=lambda item: (_float_value(item.get("release_time"), 0.0), str(item.get("mission_id", ""))),
        )
    ]
    released_missions = [
        summary
        for summary in mission_summaries
        if summary["status"] in {"released", "assigned", "completed"}
    ]
    return {
        "episode_id": state.episode.episode_id,
        "scenario_id": payload.get("scenario_id", state.episode.episode_id),
        "mission_type": state.episode.mission_type,
        "time": state.current_time_s,
        "step_index": state.step_index,
        "done": state.done,
        "mission_count": len(missions),
        "robot_count": len(_dict_list(payload.get("robots"))),
        "human_count": len(_dict_list(payload.get("humans"))),
        "assigned_mission_ids": sorted(state.assigned_mission_ids),
        "completed_mission_ids": sorted(state.completed_mission_ids),
        "busy_robot_ids": _busy_robot_ids(state),
        "robots": [_robot_observation(robot) for robot in _dict_list(payload.get("robots"))],
        "humans": [_human_observation(human) for human in _dict_list(payload.get("humans"))],
        "missions": mission_summaries,
        "released_missions": released_missions,
        "last_replay_metrics": state.last_result.metrics if state.last_result else {},
    }


def _mission_observation(mission: JsonDict, state: HumanCentricRLState) -> JsonDict:
    mission_id = str(mission.get("mission_id", ""))
    release_time = _float_value(mission.get("release_time"), 0.0)
    status = "pending"
    if mission_id in state.completed_mission_ids:
        status = "completed"
    elif mission_id in state.assigned_mission_ids:
        status = "assigned"
    elif release_time <= state.current_time_s + 1e-6:
        status = "released"
    return {
        "mission_id": mission_id,
        "mission_type": mission.get("mission_type", ""),
        "status": status,
        "assigned_robot_id": mission.get("assigned_robot_id", ""),
        "release_time": release_time,
        "deadline": _optional_float(mission.get("deadline")),
        "priority": _float_value(mission.get("priority"), 0.0),
        "target_human_id": mission.get("target_human_id", ""),
        "target_region_id": mission.get("target_region_id", ""),
        "success_conditions": list(mission.get("success_conditions", []))
        if isinstance(mission.get("success_conditions"), list)
        else [],
    }


def _robot_observation(robot: JsonDict) -> JsonDict:
    return {
        "robot_id": str(robot.get("robot_id", "")),
        "capabilities": list(robot.get("capabilities", []))
        if isinstance(robot.get("capabilities"), list)
        else [],
        "start_map_pose": _dict_value(robot.get("start_map_pose")),
        "trajectory_point_count": len(_dict_list(robot.get("trajectory"))),
    }


def _human_observation(human: JsonDict) -> JsonDict:
    return {
        "human_id": str(human.get("human_id", "")),
        "role": str(human.get("role", "")),
        "start_map_pose": _dict_value(human.get("start_map_pose")),
        "trajectory_point_count": len(_dict_list(human.get("trajectory"))),
    }


def _normalize_action(action: JsonDict, step_index: int, current_time_s: float) -> JsonDict:
    if not isinstance(action, dict):
        raise TypeError("HumanCentricRLTask actions must be JSON dictionaries")
    action_type = str(action.get("action_type", ""))
    if action_type not in SUPPORTED_ACTION_TYPES:
        raise ValueError(f"Unsupported human-centric action_type: {action_type}")
    payload = _dict_value(action.get("payload"))
    return {
        "robot_id": str(action.get("robot_id", "")),
        "action_type": action_type,
        "payload": payload,
        "step_index": int(_float_value(action.get("step_index"), float(step_index))),
        "time": _float_value(action.get("time"), current_time_s),
    }


def _busy_robot_ids(state: HumanCentricRLState) -> list[str]:
    busy: set[str] = set()
    for action in state.action_log:
        if action.get("action_type") not in {"assign_mission", "reassign_mission"}:
            continue
        robot_id = str(action.get("robot_id", ""))
        if robot_id:
            busy.add(robot_id)
    return sorted(busy)


def _next_task_time_s(state: HumanCentricRLState) -> float:
    mission_ids = {str(mission.get("mission_id", "")) for mission in _missions(state.episode.payload)}
    mission_ids.discard("")
    next_release = _next_unassigned_release_time(state, mission_ids)
    if next_release is None:
        return state.current_time_s
    return max(state.current_time_s, next_release)


def _next_unassigned_release_time(
    state: HumanCentricRLState,
    mission_ids: set[str],
) -> float | None:
    candidates = []
    for mission in _missions(state.episode.payload):
        mission_id = str(mission.get("mission_id", ""))
        if mission_id not in mission_ids or mission_id in state.assigned_mission_ids:
            continue
        release_time = _float_value(mission.get("release_time"), 0.0)
        if release_time > state.current_time_s + 1e-6:
            candidates.append(release_time)
    return min(candidates) if candidates else None


def _release_times(missions: list[JsonDict]) -> list[float]:
    return sorted({_float_value(mission.get("release_time"), 0.0) for mission in missions})


def _missions(payload: JsonDict) -> list[JsonDict]:
    return _dict_list(payload.get("missions"))


def _dict_list(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_value(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _float_value(value: Any, default: float) -> float:
    number = _optional_float(value)
    return default if number is None else number


def _sum_metric_suffix(metrics: JsonDict, suffix: str) -> float:
    return sum(
        _float_value(value, 0.0)
        for key, value in metrics.items()
        if str(key).endswith(suffix)
    )
