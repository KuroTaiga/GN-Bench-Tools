"""Scenario-level target identification difficulty metrics."""

from __future__ import annotations

from math import exp, log
from typing import Any


JsonDict = dict[str, Any]
XY = tuple[float, float]

CONFUSER_THRESHOLD = 0.70
SIGMOID_MIDPOINT = 0.55
SIGMOID_STEEPNESS = 8.0

_DESCRIPTOR_CONTAINER_KEYS = {
    "appearance",
    "asset",
    "assets",
    "attributes",
    "avatar",
    "body",
    "clothing",
    "description",
    "metadata",
    "persona",
    "profile",
    "visual",
}
_DESCRIPTOR_KEY_FRAGMENTS = {
    "accessory",
    "action",
    "age",
    "animation",
    "appearance",
    "asset",
    "avatar",
    "body",
    "bottom",
    "carried",
    "clothing",
    "color",
    "dress",
    "gender",
    "hair",
    "height",
    "mesh",
    "model",
    "object",
    "outfit",
    "pants",
    "persona",
    "profile",
    "role",
    "sequence",
    "shape",
    "shirt",
    "smpl",
    "texture",
    "top",
    "uniform",
    "visual",
}
_IGNORED_KEY_FRAGMENTS = {
    "collision",
    "deadline",
    "event",
    "human_id",
    "map_pose",
    "mission_id",
    "personal_space",
    "priority",
    "radius",
    "release",
    "robot_id",
    "social_defaults",
    "start_map_pose",
    "trajectory",
    "yaw",
}


def compute_human_identification_difficulty(
    humans: list[JsonDict],
    target_human_id: str,
    *,
    robot: JsonDict | None = None,
    threshold: float = CONFUSER_THRESHOLD,
) -> JsonDict:
    """Compute group-level human target confusability from scenario metadata."""

    target = _find_human(humans, target_human_id)
    distractors = [
        human
        for human in humans
        if str(human.get("human_id", "")) and str(human.get("human_id", "")) != target_human_id
    ]
    if target is None or not distractors:
        return _empty_metrics(distractor_count=len(distractors))

    target_tokens = _descriptor_tokens(target)
    scored: list[JsonDict] = []
    for human in distractors:
        identity_similarity = _weighted_jaccard(target_tokens, _descriptor_tokens(human))
        exposure = _spatial_exposure(target, human, robot)
        confusability = identity_similarity * exposure
        scored.append(
            {
                "human_id": str(human.get("human_id", "")),
                "identity_similarity": identity_similarity,
                "spatial_exposure": exposure,
                "confusability": confusability,
            }
        )

    scored.sort(key=lambda item: (-float(item["confusability"]), str(item["human_id"])))
    top_scores = [float(item["confusability"]) for item in scored[:3]]
    best = scored[0] if scored else {}
    s_max = float(best.get("confusability", 0.0))
    s_top3 = sum(top_scores) / len(top_scores) if top_scores else 0.0
    confuser_count = sum(1 for item in scored if float(item["confusability"]) >= threshold)
    confuser_mass = (
        log(1.0 + confuser_count) / log(1.0 + len(scored)) if scored else 0.0
    )
    raw = 0.55 * s_max + 0.30 * s_top3 + 0.15 * confuser_mass
    difficulty = _normalized_sigmoid(raw)
    similarity_source = "asset_descriptor_v0" if target_tokens else "none"

    return {
        "human_identification_difficulty": difficulty,
        "human_identification_best_confuser_id": str(best.get("human_id", "")),
        "human_identification_best_confuser_similarity": s_max,
        "human_identification_top3_confuser_similarity": s_top3,
        "human_identification_confuser_count": confuser_count,
        "human_identification_similarity_source": similarity_source,
        "human_identification_distractor_count": len(scored),
        "human_identification_target_descriptor_count": len(target_tokens),
    }


def _empty_metrics(*, distractor_count: int) -> JsonDict:
    return {
        "human_identification_difficulty": 0.0,
        "human_identification_best_confuser_id": "",
        "human_identification_best_confuser_similarity": 0.0,
        "human_identification_top3_confuser_similarity": 0.0,
        "human_identification_confuser_count": 0,
        "human_identification_similarity_source": "none",
        "human_identification_distractor_count": distractor_count,
        "human_identification_target_descriptor_count": 0,
    }


def _find_human(humans: list[JsonDict], human_id: str) -> JsonDict | None:
    for human in humans:
        if str(human.get("human_id", "")) == human_id:
            return human
    return None


def _descriptor_tokens(human: JsonDict) -> dict[str, float]:
    tokens: dict[str, float] = {}
    for key, value in human.items():
        if _ignored_key(key):
            continue
        if key == "role" or _descriptor_key(key) or _descriptor_container(key):
            _collect_descriptor_tokens(tokens, (str(key),), value)
    return tokens


def _collect_descriptor_tokens(
    tokens: dict[str, float],
    path: tuple[str, ...],
    value: Any,
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if _ignored_key(key):
                continue
            if _descriptor_key(key) or _descriptor_container(key) or _descriptor_container(path[-1]):
                _collect_descriptor_tokens(tokens, (*path, str(key)), nested)
        return
    if isinstance(value, list):
        for item in value:
            _collect_descriptor_tokens(tokens, path, item)
        return
    if not isinstance(value, (str, int, float, bool)):
        return
    normalized = _normalize_descriptor_value(value)
    if not normalized:
        return
    category = _descriptor_category(path)
    token = f"{category}:{normalized}"
    tokens[token] = max(tokens.get(token, 0.0), _descriptor_weight(category))


def _descriptor_category(path: tuple[str, ...]) -> str:
    key = ".".join(path).lower()
    if any(fragment in key for fragment in ("avatar", "mesh", "model", "smpl")):
        return "avatar"
    if any(fragment in key for fragment in ("shirt", "pants", "dress", "top", "bottom", "outfit", "uniform", "clothing")):
        return "clothing"
    if "texture" in key:
        return "texture"
    if "color" in key:
        return "color"
    if "hair" in key:
        return "hair"
    if any(fragment in key for fragment in ("body", "shape", "height", "age", "gender")):
        return "body"
    if "accessory" in key:
        return "accessory"
    if any(fragment in key for fragment in ("action", "animation", "sequence")):
        return "action"
    if "role" in key:
        return "role"
    if any(fragment in key for fragment in ("object", "carried")):
        return "object"
    if "asset" in key:
        return "asset"
    return "descriptor"


def _descriptor_weight(category: str) -> float:
    return {
        "avatar": 1.0,
        "clothing": 0.9,
        "texture": 0.9,
        "color": 0.75,
        "hair": 0.75,
        "body": 0.65,
        "asset": 0.9,
        "object": 0.7,
        "accessory": 0.65,
        "action": 0.45,
        "role": 0.35,
        "descriptor": 0.5,
    }.get(category, 0.5)


def _normalize_descriptor_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    normalized = str(value).strip().lower()
    if not normalized or normalized in {"none", "null", "unknown", "n/a"}:
        return ""
    return normalized


def _descriptor_key(key: str) -> bool:
    normalized = key.lower()
    return any(fragment in normalized for fragment in _DESCRIPTOR_KEY_FRAGMENTS)


def _descriptor_container(key: str) -> bool:
    return key.lower() in _DESCRIPTOR_CONTAINER_KEYS


def _ignored_key(key: str) -> bool:
    normalized = key.lower()
    return any(fragment in normalized for fragment in _IGNORED_KEY_FRAGMENTS)


def _weighted_jaccard(first: dict[str, float], second: dict[str, float]) -> float:
    if not first or not second:
        return 0.0
    keys = set(first) | set(second)
    numerator = sum(min(first.get(key, 0.0), second.get(key, 0.0)) for key in keys)
    denominator = sum(max(first.get(key, 0.0), second.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


def _spatial_exposure(
    target: JsonDict,
    distractor: JsonDict,
    robot: JsonDict | None,
) -> float:
    values: list[float] = []
    target_xy = _actor_xy(target)
    distractor_xy = _actor_xy(distractor)
    if target_xy is not None and distractor_xy is not None:
        values.append(exp(-_distance(target_xy, distractor_xy) / 6.0))
    if robot is not None and distractor_xy is not None:
        path_distance = _min_actor_path_distance(robot, distractor_xy)
        if path_distance is not None:
            values.append(exp(-path_distance / 6.0))
    if not values:
        return 1.0
    return min(1.0, max(0.25, max(values)))


def _actor_xy(actor: JsonDict) -> XY | None:
    trajectory = actor.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        first = trajectory[0]
        if isinstance(first, dict):
            xy = _pose_xy(first.get("map_pose"))
            if xy is not None:
                return xy
    return _pose_xy(actor.get("start_map_pose"))


def _min_actor_path_distance(actor: JsonDict, xy: XY) -> float | None:
    points = actor.get("trajectory")
    distances: list[float] = []
    if isinstance(points, list):
        for point in points:
            if not isinstance(point, dict):
                continue
            point_xy = _pose_xy(point.get("map_pose"))
            if point_xy is not None:
                distances.append(_distance(point_xy, xy))
    start_xy = _pose_xy(actor.get("start_map_pose"))
    if start_xy is not None:
        distances.append(_distance(start_xy, xy))
    return min(distances) if distances else None


def _pose_xy(value: Any) -> XY | None:
    if isinstance(value, dict):
        x = _number(value.get("x"))
        y = _number(value.get("y"))
        if x is not None and y is not None:
            return (x, y)
    if isinstance(value, list) and len(value) >= 2:
        x = _number(value[0])
        y = _number(value[1])
        if x is not None and y is not None:
            return (x, y)
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _distance(first: XY, second: XY) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _normalized_sigmoid(raw: float) -> float:
    floor = _sigmoid(SIGMOID_STEEPNESS * (0.0 - SIGMOID_MIDPOINT))
    ceiling = _sigmoid(SIGMOID_STEEPNESS * (1.0 - SIGMOID_MIDPOINT))
    if ceiling <= floor:
        return 0.0
    value = (_sigmoid(SIGMOID_STEEPNESS * (raw - SIGMOID_MIDPOINT)) - floor) / (
        ceiling - floor
    )
    return min(1.0, max(0.0, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))
