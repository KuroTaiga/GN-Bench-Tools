"""DAgger sample collector for human-centric mission episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..scenario_adapter import HumanCentricEpisode
from .mission_handlers import (
    MissionDaggerContext,
    MissionDaggerHandler,
    build_default_mission_registry,
)
from .model_adapters import ModelDaggerAdapter, build_default_model_adapter_registry
from .history import DaggerHistorySelector
from .schema import DaggerCollectionConfig, HumanDaggerSample, JsonDict


class HumanDaggerCollector:
    """Build model-neutral and model-specific human DAgger records."""

    def __init__(
        self,
        mission_handlers: dict[str, MissionDaggerHandler] | None = None,
        model_adapters: dict[str, ModelDaggerAdapter] | None = None,
        config: DaggerCollectionConfig | None = None,
    ) -> None:
        self.mission_handlers = mission_handlers or build_default_mission_registry()
        self.model_adapters = model_adapters or build_default_model_adapter_registry()
        self.config = config or DaggerCollectionConfig()
        self.history_selector = DaggerHistorySelector(self.config)

    def build_sample(
        self,
        episode: HumanCentricEpisode,
        observation: JsonDict,
        model_action: JsonDict,
        *,
        model_family: str,
        step_index: int = 0,
        time_s: float | None = None,
        metrics_snapshot: JsonDict | None = None,
        collection_context: JsonDict | None = None,
    ) -> HumanDaggerSample:
        mission = _select_mission(episode.payload, observation, model_action)
        mission_type = str(mission.get("mission_type", ""))
        handler = self.mission_handlers.get(mission_type, MissionDaggerHandler())
        context = MissionDaggerContext(
            episode_id=episode.episode_id,
            scenario_id=str(episode.payload.get("scenario_id", episode.episode_id)),
            payload=episode.payload,
            mission=mission,
            observation=observation,
            model_action=model_action,
            metrics_snapshot=metrics_snapshot or {},
        )
        validation = handler.validate(context)
        oracle = handler.oracle_correction(context, validation)
        return HumanDaggerSample(
            episode_id=context.episode_id,
            scenario_id=context.scenario_id,
            mission_type=context.mission_type,
            mission_id=context.mission_id,
            step_index=int(step_index),
            time_s=float(time_s if time_s is not None else observation.get("time", 0.0)),
            model_family=model_family,
            observation=observation,
            model_action=model_action,
            validation=validation,
            oracle=oracle,
            mission_context=handler.mission_context(context),
            metrics_snapshot=metrics_snapshot or {},
            collection_config=self.config.to_json_dict(),
            collection_context=collection_context or {},
        )

    def render_sample(self, sample: HumanDaggerSample, model_family: str | None = None) -> JsonDict:
        family = model_family or sample.model_family
        adapter = self.model_adapters[family]
        rendered = adapter.render(sample)
        rendered["source_sample"] = {
            "episode_id": sample.episode_id,
            "mission_id": sample.mission_id,
            "faults": [fault.fault_type for fault in sample.validation.faults],
            "collection_config": sample.collection_config,
            "collection_context": sample.collection_context,
            "schema_version": sample.schema_version,
        }
        return rendered

    def select_history_indices(self, history_length: int) -> list[int]:
        return self.history_selector.select_indices(history_length)

    def observe_step(
        self,
        episode: HumanCentricEpisode,
        observation: JsonDict,
        model_action: JsonDict,
        *,
        model_family: str,
        step_index: int | None = None,
        time_s: float | None = None,
        post_step_info: JsonDict | None = None,
        metrics_snapshot: JsonDict | None = None,
        history_frames: list[Any] | None = None,
        output_path: str | Path | None = None,
        append_only_trainable: bool = True,
    ) -> HumanDaggerSample:
        """Build and optionally persist a DAgger sample for one rollout step."""

        merged_metrics = _merge_metrics(metrics_snapshot, post_step_info)
        collection_context = _collection_context(
            post_step_info=post_step_info,
            history_frame_count=len(history_frames) if history_frames is not None else None,
            history_indices=self.select_history_indices(len(history_frames))
            if history_frames is not None
            else None,
        )
        sample = self.build_sample(
            episode,
            observation,
            model_action,
            model_family=model_family,
            step_index=int(
                step_index
                if step_index is not None
                else observation.get("step_index", model_action.get("step_index", 0))
            ),
            time_s=time_s,
            metrics_snapshot=merged_metrics,
            collection_context=collection_context,
        )
        if output_path is not None and (sample.trainable or not append_only_trainable):
            self.append_jsonl(output_path, sample, model_family=model_family)
        return sample

    def append_jsonl(
        self,
        path: str | Path,
        sample: HumanDaggerSample,
        *,
        model_family: str | None = None,
    ) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = self.render_sample(sample, model_family=model_family)
        with output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(rendered, sort_keys=True) + "\n")


def _select_mission(
    payload: JsonDict,
    observation: JsonDict,
    model_action: JsonDict,
) -> JsonDict:
    missions = _dict_list(payload.get("missions"))
    if not missions:
        return {}

    action_mission_id = str(_dict_value(model_action.get("payload")).get("mission_id", ""))
    if action_mission_id:
        mission = _find_mission(missions, action_mission_id)
        if mission is not None:
            return mission

    released = _dict_list(observation.get("released_missions"))
    for released_mission in released:
        mission_id = str(released_mission.get("mission_id", ""))
        mission = _find_mission(missions, mission_id)
        if mission is not None:
            return mission

    return sorted(
        missions,
        key=lambda mission: (
            _number(mission.get("release_time"), 0.0),
            str(mission.get("mission_id", "")),
        ),
    )[0]


def _find_mission(missions: list[JsonDict], mission_id: str) -> JsonDict | None:
    for mission in missions:
        if str(mission.get("mission_id", "")) == mission_id:
            return mission
    return None


def _dict_list(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_value(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _merge_metrics(
    metrics_snapshot: JsonDict | None,
    post_step_info: JsonDict | None,
) -> JsonDict:
    merged: JsonDict = {}
    if isinstance(metrics_snapshot, dict):
        merged.update(metrics_snapshot)
    if not isinstance(post_step_info, dict):
        return merged
    replay = _dict_value(post_step_info.get("replay"))
    replay_metrics = _dict_value(replay.get("metrics"))
    if replay_metrics:
        merged.update(replay_metrics)
    reward_components = _dict_value(post_step_info.get("reward_components"))
    if reward_components:
        merged["reward_components"] = reward_components
    post_metrics = _dict_value(post_step_info.get("metrics"))
    if post_metrics:
        merged.update(post_metrics)
    return merged


def _collection_context(
    *,
    post_step_info: JsonDict | None,
    history_frame_count: int | None,
    history_indices: list[int] | None,
) -> JsonDict:
    context: JsonDict = {}
    if history_frame_count is not None:
        context["history_frame_count"] = int(history_frame_count)
        context["history_indices"] = history_indices or []
    if isinstance(post_step_info, dict):
        accepted_action = _dict_value(post_step_info.get("accepted_action"))
        if accepted_action:
            context["accepted_action"] = accepted_action
        if "already_done" in post_step_info:
            context["already_done"] = bool(post_step_info.get("already_done"))
    return context


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default
