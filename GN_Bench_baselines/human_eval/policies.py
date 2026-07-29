"""Deterministic baseline policies for human-centric evaluation and RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


JsonDict = dict[str, Any]


@dataclass
class BaselineAction:
    """Baseline action record emitted to the GN-Bench task wrapper."""

    robot_id: str
    action_type: str
    payload: JsonDict


@dataclass(frozen=True)
class _MissionChoice:
    mission: JsonDict
    robot: JsonDict
    distance: float


class BaseHumanCentricPolicy:
    """Common interface for deterministic benchmark baselines.

    The policy contract is intentionally plain JSON for the first evaluator
    pass. ``reset`` receives a full NavDP-produced scenario payload. ``act`` may
    receive either a partial observation or an empty dict; missing values fall
    back to scenario-level defaults.
    """

    policy_name = "base"

    def __init__(self) -> None:
        self.scenario: JsonDict = {}
        self.missions: list[JsonDict] = []
        self.robots: list[JsonDict] = []
        self.social_structures: list[JsonDict] = []
        self._assigned_mission_ids: set[str] = set()
        self._completed_mission_ids: set[str] = set()

    def reset(self, scenario: JsonDict) -> None:
        """Reset internal policy state for one scenario."""

        self.scenario = scenario
        self.missions = sorted(
            list(scenario.get("missions", [])),
            key=lambda mission: (
                _number(mission.get("release_time"), 0.0),
                str(mission.get("mission_id", "")),
            ),
        )
        self.robots = sorted(
            list(scenario.get("robots", [])),
            key=lambda robot: str(robot.get("robot_id", "")),
        )
        self.social_structures = list(scenario.get("social_structures", []))
        self._assigned_mission_ids = set()
        self._completed_mission_ids = set()

    def act(self, observation: JsonDict) -> BaselineAction:
        """Return the next action for the current observation."""

        raise NotImplementedError

    def _base_observation_state(self, observation: JsonDict) -> tuple[float, set[str], set[str]]:
        current_time = _number(
            observation.get("time", observation.get("t", observation.get("step_time"))),
            0.0,
        )
        completed = set(map(str, observation.get("completed_mission_ids", [])))
        assigned = set(map(str, observation.get("assigned_mission_ids", [])))
        self._completed_mission_ids.update(completed)
        self._assigned_mission_ids.update(assigned)
        return current_time, self._completed_mission_ids, self._assigned_mission_ids

    def _available_missions(self, observation: JsonDict) -> list[JsonDict]:
        current_time, completed, assigned = self._base_observation_state(observation)
        override = observation.get("released_missions")
        missions = override if isinstance(override, list) else self.missions
        available: list[JsonDict] = []
        for mission in missions:
            mission_id = str(mission.get("mission_id", ""))
            if not mission_id:
                continue
            if mission_id in completed or mission_id in assigned:
                continue
            if _number(mission.get("release_time"), 0.0) > current_time:
                continue
            available.append(mission)
        return available

    def _available_robots(self, observation: JsonDict, mission: JsonDict | None = None) -> list[JsonDict]:
        busy_robot_ids = set(map(str, observation.get("busy_robot_ids", [])))
        robots = observation.get("robots")
        candidates = robots if isinstance(robots, list) else self.robots
        available = [
            robot
            for robot in candidates
            if str(robot.get("robot_id", "")) and str(robot.get("robot_id", "")) not in busy_robot_ids
        ]
        if mission is not None:
            available = [robot for robot in available if _robot_can_handle(robot, mission)]
        return sorted(available, key=lambda robot: str(robot.get("robot_id", "")))

    def _choose_nearest_pair(self, missions: list[JsonDict], robots: list[JsonDict]) -> _MissionChoice | None:
        choices: list[_MissionChoice] = []
        for mission in missions:
            target = _mission_target_pose(self.scenario, mission)
            capable_robots = [robot for robot in robots if _robot_can_handle(robot, mission)]
            for robot in capable_robots:
                distance = _distance(_robot_pose(robot), target)
                choices.append(_MissionChoice(mission=mission, robot=robot, distance=distance))
        if not choices:
            return None
        return min(
            choices,
            key=lambda choice: (
                choice.distance,
                _number(choice.mission.get("release_time"), 0.0),
                str(choice.mission.get("mission_id", "")),
                str(choice.robot.get("robot_id", "")),
            ),
        )

    def _assignment_action(self, robot: JsonDict, mission: JsonDict, **extra_payload: Any) -> BaselineAction:
        mission_id = str(mission.get("mission_id", ""))
        robot_id = str(robot.get("robot_id", ""))
        self._assigned_mission_ids.add(mission_id)
        payload = {
            "mission_id": mission_id,
            "mission_type": mission.get("mission_type", ""),
            "policy_name": self.policy_name,
        }
        payload.update(extra_payload)
        return BaselineAction(robot_id=robot_id, action_type="assign_mission", payload=payload)

    def _no_op(self, reason: str = "no_available_mission") -> BaselineAction:
        return BaselineAction(
            robot_id="",
            action_type="no_op",
            payload={"policy_name": self.policy_name, "reason": reason},
        )


class OracleHumanCentricPolicy(BaseHumanCentricPolicy):
    """Oracle assignment and socially safe route baseline."""

    policy_name = "oracle_human_centric"

    def act(self, observation: JsonDict) -> BaselineAction:
        missions = _priority_order(self._available_missions(observation))
        if not missions:
            return self._no_op()

        for mission in missions:
            assigned_robot_id = mission.get("assigned_robot_id")
            if assigned_robot_id:
                robot = _find_robot(self._available_robots(observation, mission), str(assigned_robot_id))
                if robot is not None:
                    return self._assignment_action(
                        robot,
                        mission,
                        route_mode="oracle_socially_safe",
                        use_oracle_assignment=True,
                        use_oracle_trajectory=True,
                    )

        robots = self._available_robots(observation)
        choice = self._choose_nearest_pair(missions, robots)
        if choice is None:
            return self._no_op("no_capable_robot")
        return self._assignment_action(
            choice.robot,
            choice.mission,
            route_mode="oracle_socially_safe",
            use_oracle_assignment=False,
            use_oracle_trajectory=True,
            estimated_distance=choice.distance,
        )


class GreedyNearestPolicy(BaseHumanCentricPolicy):
    """Assign released missions to the nearest available robot."""

    policy_name = "greedy_nearest"

    def act(self, observation: JsonDict) -> BaselineAction:
        missions = self._available_missions(observation)
        robots = self._available_robots(observation)
        choice = self._choose_nearest_pair(missions, robots)
        if choice is None:
            return self._no_op("no_reachable_pair")
        return self._assignment_action(
            choice.robot,
            choice.mission,
            route_mode="nearest_target",
            estimated_distance=choice.distance,
        )


class PriorityGreedyPolicy(BaseHumanCentricPolicy):
    """Sort missions by priority/deadline, then pick nearest feasible robot."""

    policy_name = "priority_greedy"

    def act(self, observation: JsonDict) -> BaselineAction:
        missions = _priority_order(self._available_missions(observation))
        for mission in missions:
            robots = self._available_robots(observation, mission)
            if not robots:
                continue
            target = _mission_target_pose(self.scenario, mission)
            robot = min(
                robots,
                key=lambda candidate: (
                    _distance(_robot_pose(candidate), target),
                    str(candidate.get("robot_id", "")),
                ),
            )
            return self._assignment_action(
                robot,
                mission,
                route_mode="priority_then_nearest",
                estimated_distance=_distance(_robot_pose(robot), target),
            )
        return self._no_op("no_capable_robot")


class NoHumanAwarenessPolicy(BaseHumanCentricPolicy):
    """Plan shortest static-occupancy paths while ignoring humans."""

    policy_name = "no_human_awareness"

    def act(self, observation: JsonDict) -> BaselineAction:
        missions = self._available_missions(observation)
        robots = self._available_robots(observation)
        choice = self._choose_nearest_pair(missions, robots)
        if choice is None:
            return self._no_op("no_reachable_pair")
        return self._assignment_action(
            choice.robot,
            choice.mission,
            route_mode="static_shortest_path",
            ignore_humans=True,
            ignore_social_structures=True,
            estimated_distance=choice.distance,
        )


class SingleRobotSerialPolicy(BaseHumanCentricPolicy):
    """Use one robot to execute all missions in release order."""

    policy_name = "single_robot_serial"

    def __init__(self) -> None:
        super().__init__()
        self._selected_robot_id: str | None = None

    def reset(self, scenario: JsonDict) -> None:
        super().reset(scenario)
        self._selected_robot_id = str(self.robots[0].get("robot_id", "")) if self.robots else None

    def act(self, observation: JsonDict) -> BaselineAction:
        if not self._selected_robot_id:
            return self._no_op("no_robot")
        robot = _find_robot(self._available_robots(observation), self._selected_robot_id)
        if robot is None:
            return self._no_op("selected_robot_busy")
        missions = self._available_missions(observation)
        if not missions:
            return self._no_op()
        mission = min(
            missions,
            key=lambda item: (
                _number(item.get("release_time"), 0.0),
                str(item.get("mission_id", "")),
            ),
        )
        return self._assignment_action(
            robot,
            mission,
            route_mode="serial_release_order",
            selected_robot_id=self._selected_robot_id,
        )


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _pose_from_mapping(payload: Any) -> tuple[float, float]:
    if not isinstance(payload, dict):
        return (0.0, 0.0)
    return (_number(payload.get("x"), 0.0), _number(payload.get("y"), 0.0))


def _robot_pose(robot: JsonDict) -> tuple[float, float]:
    trajectory = robot.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        last_point = max(
            (point for point in trajectory if isinstance(point, dict)),
            key=lambda point: _number(point.get("t"), 0.0),
            default=None,
        )
        if last_point is not None:
            return _pose_from_mapping(last_point.get("map_pose"))
    return _pose_from_mapping(robot.get("start_map_pose"))


def _mission_target_pose(scenario: JsonDict, mission: JsonDict) -> tuple[float, float]:
    target_human_id = mission.get("target_human_id")
    if target_human_id:
        for human in scenario.get("humans", []):
            if str(human.get("human_id", "")) != str(target_human_id):
                continue
            trajectory = human.get("trajectory")
            if isinstance(trajectory, list) and trajectory:
                last_point = max(
                    (point for point in trajectory if isinstance(point, dict)),
                    key=lambda point: _number(point.get("t"), 0.0),
                    default=None,
                )
                if last_point is not None:
                    return _pose_from_mapping(last_point.get("map_pose"))
            return _pose_from_mapping(human.get("start_map_pose"))

    metadata = mission.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("target_pose", "goal_pose", "target_region_pose"):
            pose = metadata.get(key)
            if isinstance(pose, dict):
                return _pose_from_mapping(pose)
    return (0.0, 0.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _find_robot(robots: list[JsonDict], robot_id: str) -> JsonDict | None:
    for robot in robots:
        if str(robot.get("robot_id", "")) == robot_id:
            return robot
    return None


def _robot_can_handle(robot: JsonDict, mission: JsonDict) -> bool:
    metadata = mission.get("metadata", {})
    required = mission.get("required_capabilities")
    if required is None and isinstance(metadata, dict):
        required = metadata.get("required_capabilities")
    if not required:
        return True
    if not isinstance(required, list):
        return True
    capabilities = set(map(str, robot.get("capabilities", [])))
    return all(str(capability) in capabilities for capability in required)


def _priority_order(missions: list[JsonDict]) -> list[JsonDict]:
    return sorted(
        missions,
        key=lambda mission: (
            -int(_number(mission.get("priority"), 0.0)),
            _number(mission.get("deadline"), float("inf")),
            _number(mission.get("release_time"), 0.0),
            str(mission.get("mission_id", "")),
        ),
    )
