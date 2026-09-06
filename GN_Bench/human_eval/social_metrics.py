"""Reusable social metrics for human-centric replay state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class MetricValue:
    """Single metric output with optional debug context."""

    name: str
    value: float
    details: JsonDict | None = None


class HumanDistanceMetrics:
    """Computes robot-human distance, collision, and personal-space metrics."""

    def compute(self, replay_state: JsonDict) -> list[MetricValue]:
        robot = _first_robot(replay_state)
        humans = _humans(replay_state)
        if not robot:
            return []
        threshold = _number(replay_state.get("collision_threshold_m"), 0.7)
        personal_space_default = _number(replay_state.get("personal_space_radius_m"), 0.8)
        min_distance: float | None = None
        collision_count = 0
        violation_count = 0
        violation_duration = 0.0
        for human in humans:
            human_min = _min_actor_distance_at_robot_samples(robot, human)
            if human_min is not None:
                min_distance = human_min if min_distance is None else min(min_distance, human_min)
                if human_min <= threshold:
                    collision_count += 1
            radius = _human_personal_space_radius(human, personal_space_default)
            duration, count = _personal_space_duration(robot, human, radius)
            violation_duration += duration
            violation_count += count
        return [
            MetricValue("minimum_human_distance_m", min_distance or 0.0),
            MetricValue("collision_count", float(collision_count)),
            MetricValue("collision_rate", 1.0 if collision_count else 0.0),
            MetricValue("personal_space_violation_count", float(violation_count)),
            MetricValue("personal_space_violation_s", violation_duration),
        ]


class FFormationMetrics:
    """Computes conversation-group social violations."""

    def compute(self, replay_state: JsonDict) -> list[MetricValue]:
        mission_result = _dict_value(replay_state.get("mission_result"))
        if "group_region_violation_count" in mission_result:
            violation_count = _number(mission_result.get("group_region_violation_count"), 0.0)
            return [
                MetricValue("group_integrity_violation_count", violation_count),
                MetricValue(
                    "group_integrity_violation_rate",
                    1.0 if violation_count else 0.0,
                    {"source": "mission_result"},
                ),
            ]
        robot = _first_robot(replay_state)
        if not robot:
            return []
        violation_count = 0
        min_clearance: float | None = None
        for structure in _social_structures(replay_state, "f_formation"):
            center = _xy_from_value(_dict_value(structure.get("geometry")).get("o_space_center"))
            radius = _number(_dict_value(structure.get("geometry")).get("o_space_radius"), 0.0)
            if center is None or radius <= 0.0:
                continue
            for point in _trajectory_points(robot):
                xy = _pose_xy(point.get("map_pose"))
                if xy is None:
                    continue
                clearance = _distance(xy, center) - radius
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                if clearance <= 0.0:
                    violation_count += 1
        return [
            MetricValue("group_region_min_clearance_m", min_clearance or 0.0),
            MetricValue("group_integrity_violation_count", float(violation_count)),
            MetricValue("group_integrity_violation_rate", 1.0 if violation_count else 0.0),
        ]


class QueueMetrics:
    """Computes queue order and segmentation violations."""

    def compute(self, replay_state: JsonDict) -> list[MetricValue]:
        mission_result = _dict_value(replay_state.get("mission_result"))
        if "queue_order_violation_count" in mission_result:
            violation_count = _number(mission_result.get("queue_order_violation_count"), 0.0)
            check_count = _number(mission_result.get("queue_order_check_count"), 1.0)
            return [
                MetricValue("queue_order_violation_count", violation_count),
                MetricValue(
                    "queue_order_violation_rate",
                    violation_count / check_count if check_count else 0.0,
                    {"source": "mission_result"},
                ),
            ]
        events = _events(replay_state)
        queue_order = _string_list(replay_state.get("queue_order"))
        completion_times = {
            str(event.get("target_human_id") or event.get("human_id") or ""): _number(
                event.get("t"),
                0.0,
            )
            for event in events
            if str(event.get("event_type", "")) in {"completion", "mission_completion"}
        }
        violations = 0
        comparable = 0
        for before, after in zip(queue_order, queue_order[1:]):
            if before not in completion_times or after not in completion_times:
                continue
            comparable += 1
            violations += int(completion_times[before] > completion_times[after])
        return [
            MetricValue("queue_order_violation_count", float(violations)),
            MetricValue(
                "queue_order_violation_rate",
                violations / comparable if comparable else 0.0,
            ),
        ]


class PedestrianFlowMetrics:
    """Computes flow-following and right-side-yield violations."""

    def compute(self, replay_state: JsonDict) -> list[MetricValue]:
        mission_result = _dict_value(replay_state.get("mission_result"))
        if "pedestrian_yield_violation_count" in mission_result:
            violation_count = _number(mission_result.get("pedestrian_yield_violation_count"), 0.0)
            check_count = _number(mission_result.get("pedestrian_yield_check_count"), 1.0)
            return [
                MetricValue("pedestrian_yield_violation_count", violation_count),
                MetricValue(
                    "pedestrian_yield_violation_rate",
                    violation_count / check_count if check_count else 0.0,
                    {"source": "mission_result"},
                ),
            ]
        robot = _first_robot(replay_state)
        human = _first_human(replay_state)
        conflict_point = _xy_from_value(replay_state.get("conflict_point"))
        if not robot or not human or conflict_point is None:
            return []
        robot_time, _ = _closest_time_to_point(robot, conflict_point)
        human_time, _ = _closest_time_to_point(human, conflict_point)
        margin = _number(replay_state.get("yield_margin_s"), 0.0)
        respected = (
            robot_time is not None
            and human_time is not None
            and robot_time - human_time >= margin
        )
        return [
            MetricValue("pedestrian_yield_time_gap_s", (robot_time or 0.0) - (human_time or 0.0)),
            MetricValue("pedestrian_yield_violation_count", 0.0 if respected else 1.0),
            MetricValue("pedestrian_yield_violation_rate", 0.0 if respected else 1.0),
        ]


class VulnerableHumanMetrics:
    """Computes larger-buffer penalties for vulnerable humans."""

    def compute(self, replay_state: JsonDict) -> list[MetricValue]:
        robot = _first_robot(replay_state)
        if not robot:
            return []
        vulnerable = [
            human
            for human in _humans(replay_state)
            if _dict_value(human.get("social_defaults")).get("vulnerable_buffer_radius")
            is not None
            or human.get("vulnerability_tag")
        ]
        violation_count = 0
        violation_duration = 0.0
        min_clearance: float | None = None
        for human in vulnerable:
            radius = _number(
                _dict_value(human.get("social_defaults")).get("vulnerable_buffer_radius"),
                _human_personal_space_radius(human, 1.2),
            )
            duration, count = _personal_space_duration(robot, human, radius)
            violation_duration += duration
            violation_count += count
            human_min = _min_actor_distance_at_robot_samples(robot, human)
            if human_min is not None:
                clearance = human_min - radius
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
        return [
            MetricValue("vulnerable_human_count", float(len(vulnerable))),
            MetricValue("vulnerable_buffer_violation_count", float(violation_count)),
            MetricValue("vulnerable_buffer_violation_s", violation_duration),
            MetricValue("vulnerable_buffer_min_clearance_m", min_clearance or 0.0),
        ]


def _first_robot(replay_state: JsonDict) -> JsonDict:
    robot = replay_state.get("robot")
    if isinstance(robot, dict):
        return robot
    robots = replay_state.get("robots")
    if isinstance(robots, list):
        return next((item for item in robots if isinstance(item, dict)), {})
    payload = _dict_value(replay_state.get("payload"))
    robots = payload.get("robots")
    if isinstance(robots, list):
        return next((item for item in robots if isinstance(item, dict)), {})
    return {}


def _first_human(replay_state: JsonDict) -> JsonDict:
    return next(iter(_humans(replay_state)), {})


def _humans(replay_state: JsonDict) -> list[JsonDict]:
    humans = replay_state.get("humans")
    if isinstance(humans, list):
        return [item for item in humans if isinstance(item, dict)]
    payload = _dict_value(replay_state.get("payload"))
    humans = payload.get("humans")
    if isinstance(humans, list):
        return [item for item in humans if isinstance(item, dict)]
    return []


def _events(replay_state: JsonDict) -> list[JsonDict]:
    event_log = replay_state.get("event_log")
    if event_log is None:
        event_log = _dict_value(replay_state.get("payload")).get("event_log")
    if isinstance(event_log, dict):
        event_log = event_log.get("events", [])
    if isinstance(event_log, list):
        return [item for item in event_log if isinstance(item, dict)]
    return []


def _social_structures(replay_state: JsonDict, structure_type: str) -> list[JsonDict]:
    payload = _dict_value(replay_state.get("payload"))
    structures = replay_state.get("social_structures", payload.get("social_structures", []))
    if not isinstance(structures, list):
        return []
    return [
        item
        for item in structures
        if isinstance(item, dict) and str(item.get("structure_type", "")) == structure_type
    ]


def _trajectory_points(actor: JsonDict) -> list[JsonDict]:
    trajectory = actor.get("trajectory", [])
    if not isinstance(trajectory, list):
        return []
    return sorted(
        [point for point in trajectory if isinstance(point, dict)],
        key=lambda point: _number(point.get("t"), 0.0),
    )


def _actor_xy_at_time(actor: JsonDict, t: float) -> tuple[float, float] | None:
    points = _trajectory_points(actor)
    if not points:
        return _pose_xy(actor.get("start_map_pose"))
    previous = points[0]
    for point in points:
        if _number(point.get("t"), 0.0) > t:
            break
        previous = point
    return _pose_xy(previous.get("map_pose"))


def _min_actor_distance_at_robot_samples(
    robot: JsonDict,
    human: JsonDict,
) -> float | None:
    min_distance: float | None = None
    for point in _trajectory_points(robot):
        t = _number(point.get("t"), 0.0)
        robot_xy = _pose_xy(point.get("map_pose"))
        human_xy = _actor_xy_at_time(human, t)
        if robot_xy is None or human_xy is None:
            continue
        distance = _distance(robot_xy, human_xy)
        min_distance = distance if min_distance is None else min(min_distance, distance)
    return min_distance


def _personal_space_duration(
    robot: JsonDict,
    human: JsonDict,
    radius: float,
) -> tuple[float, int]:
    points = _trajectory_points(robot)
    violation_duration = 0.0
    violation_count = 0
    in_violation = False
    for index, point in enumerate(points):
        t = _number(point.get("t"), 0.0)
        robot_xy = _pose_xy(point.get("map_pose"))
        human_xy = _actor_xy_at_time(human, t)
        if robot_xy is None or human_xy is None:
            continue
        if _distance(robot_xy, human_xy) <= radius:
            if not in_violation:
                violation_count += 1
                in_violation = True
            violation_duration += _sample_dt(points, index)
        else:
            in_violation = False
    return violation_duration, violation_count


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


def _human_personal_space_radius(human: JsonDict, default: float) -> float:
    social_defaults = human.get("social_defaults", {})
    if isinstance(social_defaults, dict):
        value = social_defaults.get("personal_space_radius")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return default


def _pose_xy(pose: Any) -> tuple[float, float] | None:
    if not isinstance(pose, dict):
        return None
    return (_number(pose.get("x"), 0.0), _number(pose.get("y"), 0.0))


def _xy_from_value(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        return _pose_xy(value)
    if isinstance(value, list) and len(value) >= 2:
        return (_number(value[0], 0.0), _number(value[1], 0.0))
    return None


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _sample_dt(points: list[JsonDict], index: int) -> float:
    if len(points) < 2:
        return 0.0
    current_t = _number(points[index].get("t"), 0.0)
    if index + 1 < len(points):
        return max(0.0, _number(points[index + 1].get("t"), current_t) - current_t)
    return max(0.0, current_t - _number(points[index - 1].get("t"), current_t))


def _dict_value(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default
