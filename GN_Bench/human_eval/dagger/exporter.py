"""Publication DAgger iteration export for NavDP human mission scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..baseline_runner import POLICY_CLASSES
from ..evaluator import HumanCentricEvaluator
from ..scenario_adapter import HumanCentricEpisode


JsonDict = dict[str, Any]
DAGGER_ITERATION_SCHEMA_VERSION = "human_dagger_iteration_v0.1"
DEFAULT_ACTION_TYPES = (
    "assign_mission",
    "set_subgoal",
    "interact",
    "no_op",
    "robot_eos",
)


def export_iteration_zero(
    episodes_by_split: dict[str, list[HumanCentricEpisode]],
    *,
    output_root: str | Path,
    model_input_root: str | Path | None = None,
    hidden_gt_root: str | Path | None = None,
    policy_name: str = "oracle_route_follower",
    episodes_per_split: int = 0,
    model_family: str = "json_policy",
    random_seed: int = 0,
    policy_version: str | None = None,
) -> JsonDict:
    """Write ``dagger/iteration_000`` records and return the metrics summary."""

    if policy_name not in POLICY_CLASSES:
        raise ValueError(f"Unknown DAgger teacher policy: {policy_name}")

    iteration_dir = Path(output_root) / "dagger" / "iteration_000"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    model_packets = _load_packet_records(model_input_root)
    hidden_packets = _load_packet_records(hidden_gt_root)
    evaluator = HumanCentricEvaluator()

    rollouts: list[JsonDict] = []
    teacher_actions: list[JsonDict] = []
    failures: list[JsonDict] = []

    for split_name in sorted(episodes_by_split):
        split_episodes = sorted(episodes_by_split[split_name], key=lambda item: item.episode_id)
        if episodes_per_split > 0:
            split_episodes = split_episodes[:episodes_per_split]
        for episode in split_episodes:
            replay = evaluator.replay(episode)
            mission_result_by_id = {
                str(result.get("mission_id", "")): result
                for result in replay.mission_results
                if result.get("mission_id")
            }
            for step_index, mission in enumerate(_missions(episode.payload)):
                mission_id = str(mission.get("mission_id", ""))
                mission_result = mission_result_by_id.get(mission_id, {})
                packet = _matching_packet(
                    model_packets,
                    split_name=split_name,
                    episode=episode,
                    mission_id=mission_id,
                )
                hidden_gt = _matching_packet(
                    hidden_packets,
                    split_name=split_name,
                    episode=episode,
                    mission_id=mission_id,
                )
                observation_packet_id = _packet_id(
                    packet,
                    split_name=split_name,
                    episode=episode,
                    mission_id=mission_id,
                    suffix="obs",
                )
                teacher_action_id = (
                    f"{split_name}:{episode.episode_id}:{mission_id}:teacher:{step_index:06d}"
                )
                teacher_action = _teacher_action(
                    episode=episode,
                    mission=mission,
                    policy_name=policy_name,
                    hidden_gt=hidden_gt,
                )
                teacher_record = {
                    "schema_version": DAGGER_ITERATION_SCHEMA_VERSION,
                    "teacher_action_id": teacher_action_id,
                    "split": split_name,
                    "scenario_id": str(episode.payload.get("scenario_id", episode.episode_id)),
                    "episode_id": episode.episode_id,
                    "mission_id": mission_id,
                    "step_index": step_index,
                    "time_s": _number(mission.get("release_time"), 0.0),
                    "policy_name": policy_name,
                    "policy_version": policy_version or f"{policy_name}_replay_v0",
                    "teacher_action": teacher_action,
                    "hidden_gt_packet_id": _packet_id(
                        hidden_gt,
                        split_name=split_name,
                        episode=episode,
                        mission_id=mission_id,
                        suffix="hidden_gt",
                    ),
                    "hidden_gt": hidden_gt or _synthetic_hidden_gt(episode, mission),
                }
                teacher_actions.append(teacher_record)

                if packet is None:
                    failures.append(
                        _warning_record(
                            "missing_model_input_packet",
                            split_name,
                            episode,
                            mission_id,
                            "Synthetic public observation packet was generated.",
                        )
                    )
                if hidden_gt is None:
                    failures.append(
                        _warning_record(
                            "missing_hidden_gt_packet",
                            split_name,
                            episode,
                            mission_id,
                            "Teacher action used scenario payload as local fallback.",
                        )
                    )

                rollouts.append(
                    {
                        "schema_version": DAGGER_ITERATION_SCHEMA_VERSION,
                        "iteration": "iteration_000",
                        "split": split_name,
                        "scenario_id": str(
                            episode.payload.get("scenario_id", episode.episode_id)
                        ),
                        "episode_id": episode.episode_id,
                        "mission_id": mission_id,
                        "mission_type": str(mission.get("mission_type", "")),
                        "timestep": step_index,
                        "step_index": step_index,
                        "time_s": _number(mission.get("release_time"), 0.0),
                        "observation_packet_id": observation_packet_id,
                        "public_observation": packet
                        if packet is not None
                        else _synthetic_public_observation(
                            episode,
                            mission,
                            split_name=split_name,
                            observation_packet_id=observation_packet_id,
                        ),
                        "teacher_action_id": teacher_action_id,
                        "teacher_action": teacher_action,
                        "student_action": None,
                        "action_mask": _action_mask(episode, mission),
                        "reward_terms": _reward_terms(mission_result),
                        "collision_flags": _collision_flags(mission_result),
                        "social_law_flags": _social_law_flags(mission_result),
                        "assignment_state": _assignment_state(episode.payload, mission),
                        "random_seed": int(random_seed),
                        "policy_version": policy_version or f"{policy_name}_replay_v0",
                        "model_family": model_family,
                        "metrics_snapshot": mission_result,
                    }
                )

    paths = {
        "rollouts_jsonl": str(_write_jsonl(iteration_dir / "rollouts.jsonl", rollouts)),
        "teacher_actions_jsonl": str(
            _write_jsonl(iteration_dir / "teacher_actions.jsonl", teacher_actions)
        ),
        "failures_jsonl": str(_write_jsonl(iteration_dir / "failures.jsonl", failures)),
        "metrics_json": str(iteration_dir / "metrics.json"),
    }
    metrics = {
        "schema_version": DAGGER_ITERATION_SCHEMA_VERSION,
        "iteration": "iteration_000",
        "output_dir": str(iteration_dir),
        "policy_name": policy_name,
        "policy_version": policy_version or f"{policy_name}_replay_v0",
        "model_family": model_family,
        "split_count": len(episodes_by_split),
        "episode_count": len({row["episode_id"] for row in rollouts}),
        "rollout_record_count": len(rollouts),
        "teacher_action_count": len(teacher_actions),
        "failure_record_count": len(failures),
        "missing_model_input_packet_count": sum(
            1 for row in failures if row["failure_type"] == "missing_model_input_packet"
        ),
        "missing_hidden_gt_packet_count": sum(
            1 for row in failures if row["failure_type"] == "missing_hidden_gt_packet"
        ),
        "records_by_split": _count_by_key(rollouts, "split"),
        "records_by_mission_type": _count_by_key(rollouts, "mission_type"),
        "paths": paths,
    }
    (iteration_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _missions(payload: JsonDict) -> list[JsonDict]:
    missions = payload.get("missions", [])
    if not isinstance(missions, list):
        return []
    return sorted(
        [mission for mission in missions if isinstance(mission, dict)],
        key=lambda mission: (
            _number(mission.get("release_time"), 0.0),
            str(mission.get("mission_id", "")),
        ),
    )


def _teacher_action(
    *,
    episode: HumanCentricEpisode,
    mission: JsonDict,
    policy_name: str,
    hidden_gt: JsonDict | None,
) -> JsonDict:
    policy = POLICY_CLASSES[policy_name]()
    policy.reset(episode.payload)
    mission_id = str(mission.get("mission_id", ""))
    release_time = _number(mission.get("release_time"), 0.0)
    action = policy.act(
        {
            "time": release_time,
            "released_missions": [mission],
            "assigned_mission_ids": [],
            "completed_mission_ids": [],
        }
    )
    payload = dict(action.payload)
    payload.setdefault("mission_id", mission_id)
    payload.setdefault("mission_type", mission.get("mission_type", ""))
    payload.setdefault("target_human_id", mission.get("target_human_id") or "")
    metadata = _dict_value(mission.get("metadata"))
    if "planned_goal_world" in metadata:
        payload.setdefault("planned_goal_world", metadata["planned_goal_world"])
    if hidden_gt:
        for key in ("oracle_path", "oracle_actions", "teacher_actions", "hidden_targets"):
            if key in hidden_gt:
                payload.setdefault(key, hidden_gt[key])
    return {
        "robot_id": action.robot_id or str(mission.get("assigned_robot_id", "")),
        "action_type": action.action_type,
        "payload": payload,
    }


def _synthetic_public_observation(
    episode: HumanCentricEpisode,
    mission: JsonDict,
    *,
    split_name: str,
    observation_packet_id: str,
) -> JsonDict:
    metadata = _dict_value(mission.get("metadata"))
    return {
        "schema_version": "navdp_model_input_fallback_v0.1",
        "observation_packet_id": observation_packet_id,
        "split": split_name,
        "scenario_id": str(episode.payload.get("scenario_id", episode.episode_id)),
        "episode_id": episode.episode_id,
        "scene_id": episode.raw_scene_id,
        "dataset": episode.dataset,
        "mission": {
            "mission_id": str(mission.get("mission_id", "")),
            "mission_type": str(mission.get("mission_type", "")),
            "release_time": _optional_number(mission.get("release_time")),
            "deadline": _optional_number(mission.get("deadline")),
            "priority": _optional_number(mission.get("priority")),
            "instruction": _public_instruction(mission),
            "social_law_ids": _string_list(mission.get("social_law_ids"))
            or _string_list(metadata.get("active_social_law_ids")),
        },
        "robots": [
            {
                "robot_id": str(robot.get("robot_id", "")),
                "capabilities": _string_list(robot.get("capabilities")),
                "start_map_pose": _dict_value(robot.get("start_map_pose")),
            }
            for robot in _dict_list(episode.payload.get("robots"))
        ],
        "public_humans": [
            {
                "public_human_id": f"human_{index:03d}",
                "appearance": _dict_value(human.get("appearance")),
                "tags": _string_list(human.get("tags")),
            }
            for index, human in enumerate(_dict_list(episode.payload.get("humans")))
        ],
        "scene_assets": _public_scene_assets(episode.scene_assets),
        "packet_source": "synthetic_from_scenario_public_allowlist",
    }


def _synthetic_hidden_gt(episode: HumanCentricEpisode, mission: JsonDict) -> JsonDict:
    metadata = _dict_value(mission.get("metadata"))
    return {
        "schema_version": "navdp_hidden_gt_fallback_v0.1",
        "scenario_id": str(episode.payload.get("scenario_id", episode.episode_id)),
        "episode_id": episode.episode_id,
        "mission_id": str(mission.get("mission_id", "")),
        "target_human_id": str(mission.get("target_human_id") or ""),
        "target_region_id": str(mission.get("target_region_id") or ""),
        "target_object_id": str(mission.get("target_object_id") or ""),
        "assigned_robot_id": str(mission.get("assigned_robot_id") or ""),
        "planned_goal_world": metadata.get("planned_goal_world"),
        "planned_goal_world_by_robot": metadata.get("planned_goal_world_by_robot"),
        "active_social_law_ids": _string_list(metadata.get("active_social_law_ids"))
        or _string_list(mission.get("social_law_ids")),
        "success_conditions": _string_list(mission.get("success_conditions")),
        "event_log": episode.payload.get("event_log", {}),
    }


def _load_packet_records(root: str | Path | None) -> dict[tuple[str, str, str], JsonDict]:
    records: dict[tuple[str, str, str], JsonDict] = {}
    if not root:
        return records
    root_path = Path(root)
    if not root_path.is_dir():
        return records
    for jsonl_path in sorted(root_path.glob("*.jsonl")):
        split_name = jsonl_path.stem
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                payload.setdefault("packet_source_path", str(jsonl_path))
                payload.setdefault("packet_source_line", line_index)
                key = _packet_key(payload, fallback_split=split_name)
                if key is not None:
                    records.setdefault(key, payload)
    return records


def _matching_packet(
    records: dict[tuple[str, str, str], JsonDict],
    *,
    split_name: str,
    episode: HumanCentricEpisode,
    mission_id: str,
) -> JsonDict | None:
    scenario_id = str(episode.payload.get("scenario_id", episode.episode_id))
    candidates = (
        (split_name, scenario_id, mission_id),
        (split_name, episode.episode_id, mission_id),
        ("", scenario_id, mission_id),
        ("", episode.episode_id, mission_id),
    )
    for key in candidates:
        packet = records.get(key)
        if packet is not None:
            return packet
    return None


def _packet_key(payload: JsonDict, *, fallback_split: str) -> tuple[str, str, str] | None:
    split = str(payload.get("split") or fallback_split)
    scenario_id = str(payload.get("scenario_id") or payload.get("episode_id") or "")
    mission_id = str(payload.get("mission_id") or "")
    if not mission_id:
        mission = payload.get("mission")
        if isinstance(mission, dict):
            mission_id = str(mission.get("mission_id") or "")
    if not scenario_id or not mission_id:
        return None
    return (split, scenario_id, mission_id)


def _packet_id(
    payload: JsonDict | None,
    *,
    split_name: str,
    episode: HumanCentricEpisode,
    mission_id: str,
    suffix: str,
) -> str:
    if payload:
        for key in ("observation_packet_id", "hidden_gt_packet_id", "packet_id", "id"):
            value = payload.get(key)
            if value:
                return str(value)
    return f"{split_name}:{episode.episode_id}:{mission_id}:{suffix}"


def _action_mask(episode: HumanCentricEpisode, mission: JsonDict) -> JsonDict:
    mission_type = str(mission.get("mission_type", ""))
    valid_action_types = list(DEFAULT_ACTION_TYPES)
    if mission_type in {"deliver_to_human", "serve_queue", "mission_stream"}:
        preferred = "assign_mission"
    elif mission_type == "human_guided_uncertain_region":
        preferred = "interact"
    else:
        preferred = "set_subgoal"
    return {
        "valid_action_types": valid_action_types,
        "preferred_action_type": preferred,
        "available_robot_ids": [
            str(robot.get("robot_id", ""))
            for robot in _dict_list(episode.payload.get("robots"))
            if robot.get("robot_id")
        ],
        "released_mission_ids": [str(mission.get("mission_id", ""))],
    }


def _reward_terms(mission_result: JsonDict) -> JsonDict:
    return {
        "mission_success": _float_bool(mission_result.get("success")),
        "deadline_success": _float_bool(mission_result.get("deadline_success")),
        "collision_free": 1.0
        if int(_number(mission_result.get("collision_count"), 0.0)) == 0
        else 0.0,
        "weighted_mission_score": _optional_number(
            mission_result.get("weighted_mission_score")
        ),
    }


def _collision_flags(mission_result: JsonDict) -> JsonDict:
    return {
        "collision_count": int(_number(mission_result.get("collision_count"), 0.0)),
        "wrong_human_contact_count": int(
            _number(mission_result.get("wrong_human_contact_count"), 0.0)
        ),
        "robot_robot_collision_count": int(
            _number(mission_result.get("robot_robot_collision_count"), 0.0)
        ),
        "collision_free": int(_number(mission_result.get("collision_count"), 0.0)) == 0,
    }


def _social_law_flags(mission_result: JsonDict) -> JsonDict:
    return {
        "active_social_law_ids": _string_list(mission_result.get("active_social_law_ids")),
        "social_law_success": mission_result.get("social_law_success"),
        "personal_space_violation_count": int(
            _number(mission_result.get("personal_space_violation_count"), 0.0)
        ),
        "pedestrian_yield_violation_count": int(
            _number(mission_result.get("pedestrian_yield_violation_count"), 0.0)
        ),
        "group_region_violation_count": int(
            _number(mission_result.get("group_region_violation_count"), 0.0)
        ),
        "queue_order_violation_count": int(
            _number(mission_result.get("queue_order_violation_count"), 0.0)
        ),
    }


def _assignment_state(payload: JsonDict, mission: JsonDict) -> JsonDict:
    events = payload.get("event_log", {})
    if isinstance(events, dict):
        events = events.get("events", [])
    if not isinstance(events, list):
        events = []
    mission_id = str(mission.get("mission_id", ""))
    assignment_events = [
        event
        for event in events
        if isinstance(event, dict)
        and str(event.get("mission_id", "")) == mission_id
        and str(event.get("event_type", "")) in {"robot_assignment", "assign_mission"}
    ]
    return {
        "assigned_robot_id": str(mission.get("assigned_robot_id") or ""),
        "assignment_event_count": len(assignment_events),
        "release_time": _optional_number(mission.get("release_time")),
        "deadline": _optional_number(mission.get("deadline")),
        "priority": _optional_number(mission.get("priority")),
    }


def _public_instruction(mission: JsonDict) -> str:
    if isinstance(mission.get("instruction"), str):
        return str(mission["instruction"])
    metadata = _dict_value(mission.get("metadata"))
    for key in ("instruction", "coarse_instruction", "mission_description"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    robot_instructions = metadata.get("robot_instructions")
    if isinstance(robot_instructions, dict):
        for value in robot_instructions.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _public_scene_assets(scene_assets: JsonDict) -> JsonDict:
    allowed = {}
    for key in ("dataset", "scene_dir", "scene_graph", "semantic_map"):
        value = scene_assets.get(key)
        if value:
            allowed[key] = value
    return allowed


def _warning_record(
    failure_type: str,
    split_name: str,
    episode: HumanCentricEpisode,
    mission_id: str,
    message: str,
) -> JsonDict:
    return {
        "schema_version": DAGGER_ITERATION_SCHEMA_VERSION,
        "severity": "warning",
        "failure_type": failure_type,
        "split": split_name,
        "episode_id": episode.episode_id,
        "scenario_id": str(episode.payload.get("scenario_id", episode.episode_id)),
        "mission_id": mission_id,
        "message": message,
    }


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> Path:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _count_by_key(rows: Iterable[JsonDict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _dict_list(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _float_bool(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return None
