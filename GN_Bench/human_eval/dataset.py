"""GN-Bench dataset wrapper for NavDP human-centric scenario artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from GN_Bench.config import Config
from GN_Bench.core.dataset import ALL_SCENES_MASK, Dataset
from GN_Bench.core.registry import registry

from .scenario_adapter import HumanCentricEpisode, NavDPScenarioAdapter


@registry.register_dataset(name="HumanCentric-v0")
class HumanCentricDataset(Dataset[HumanCentricEpisode]):
    """Loads NavDP scenario JSON files as GN-Bench episodes."""

    episodes: list[HumanCentricEpisode]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self.config = config
        if config is None:
            return

        adapter = NavDPScenarioAdapter(navdp_root=_cfg_get(config, "NAVDP_ROOT"))
        for source_path in _source_paths_from_config(config):
            self.episodes.extend(adapter.load_path(source_path))

        self.episodes.sort(key=lambda episode: episode.episode_id)
        self._filter_content_scenes(config)

    @classmethod
    def check_config_paths_exist(cls, config: Config) -> bool:
        for source_path in _source_paths_from_config(config):
            if not Path(str(source_path)).expanduser().exists():
                navdp_root = _cfg_get(config, "NAVDP_ROOT")
                if not navdp_root:
                    return False
                if not (Path(str(navdp_root)).expanduser() / str(source_path)).exists():
                    return False
        return True

    def _filter_content_scenes(self, config: Config) -> None:
        content_scenes = _cfg_get(config, "CONTENT_SCENES", [ALL_SCENES_MASK])
        if isinstance(content_scenes, str):
            content_scenes = [content_scenes]
        content_scene_ids = {str(scene) for scene in content_scenes}
        if ALL_SCENES_MASK in content_scene_ids:
            return

        self.episodes = [
            episode
            for episode in self.episodes
            if episode.raw_scene_id in content_scene_ids
            or Path(episode.scene_id).name in content_scene_ids
            or episode.scene_id in content_scene_ids
        ]


def _source_paths_from_config(config: Config) -> list[str]:
    scenario_paths = _cfg_get(config, "SCENARIO_PATHS")
    if scenario_paths:
        if isinstance(scenario_paths, str):
            return [scenario_paths]
        return [str(path) for path in scenario_paths]

    for key in ("SPLIT_MANIFEST", "DATA_PATH", "SCENARIO_PATH"):
        value = _cfg_get(config, key)
        if value:
            return [str(value)]
    return []


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
