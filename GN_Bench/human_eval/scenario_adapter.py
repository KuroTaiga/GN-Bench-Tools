"""Adapter from NavDP datagen scenario JSON to GN-Bench episodes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import attr

from GN_Bench.core.dataset import Episode


JsonDict = dict[str, Any]


@attr.s(auto_attribs=True, kw_only=True)
class HumanCentricEpisode(Episode):
    """GN-Bench-side episode wrapper for one generated human scenario."""

    ref_json: str = attr.ib(default="")
    scenario_path: str = attr.ib(default="")
    navdp_root: str = attr.ib(default="")
    raw_scene_id: str = attr.ib(default="")
    dataset: str = attr.ib(default="")
    split: str = attr.ib(default="")
    schema_version: str = attr.ib(default="")
    mission_type: str = attr.ib(default="")
    scene_assets: JsonDict = attr.ib(factory=dict)
    payload: JsonDict = attr.ib(factory=dict)


class NavDPScenarioAdapter:
    """Loads NavDP-generated scenarios and exposes GN-Bench episode records."""

    def __init__(self, navdp_root: str | Path | None = None) -> None:
        self.navdp_root = Path(navdp_root).expanduser() if navdp_root else None

    def load_episode(
        self,
        scenario_path: str | Path,
        *,
        base_dir: str | Path | None = None,
        split: str | None = None,
    ) -> HumanCentricEpisode:
        """Load one NavDP scenario JSON as a GN-Bench evaluation episode."""

        path = self.resolve_path(scenario_path, base_dir=base_dir)
        payload = json.loads(path.read_text(encoding="utf-8"))
        root = self._resolve_root(path)
        scene_assets = payload.get("scene_assets", {})
        if not isinstance(scene_assets, dict):
            scene_assets = {}

        raw_scene_id = str(payload.get("scene_id", ""))
        scene_id = self._episode_scene_id(raw_scene_id, scene_assets, root, path.parent)
        start_position, start_rotation = self._episode_start_state(payload)
        dataset = str(scene_assets.get("dataset", ""))
        if not dataset:
            metadata = payload.get("metadata", {})
            if isinstance(metadata, dict):
                dataset = str(metadata.get("dataset", ""))

        return HumanCentricEpisode(
            episode_id=str(payload.get("scenario_id", path.stem)),
            scene_id=scene_id,
            start_position=start_position,
            start_rotation=start_rotation,
            info={
                "human_scenario": payload,
                "scenario_path": str(path),
                "raw_scene_id": raw_scene_id,
                "schema_version": str(payload.get("schema_version", "")),
            },
            ref_json=str(path),
            scenario_path=str(path),
            navdp_root=str(root) if root else "",
            raw_scene_id=raw_scene_id,
            dataset=dataset,
            split=str(split or payload.get("split") or ""),
            schema_version=str(payload.get("schema_version", "")),
            mission_type=self._mission_type(payload),
            scene_assets=scene_assets,
            payload=payload,
        )

    def load_split(self, split_manifest_path: str | Path) -> list[HumanCentricEpisode]:
        """Load a split manifest produced by NavDP datagen.

        Supported producer shapes include:
        - {"examples": [{"path": "..."}]}
        - {"scenarios": [{"path": "..."}]}
        - {"splits": {"val_seen": [{"path": "..."}], ...}}
        - [{"path": "..."}]
        """

        manifest_path = self.resolve_path(split_manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episodes: list[HumanCentricEpisode] = []
        for split_name, entries in self._scenario_entries_by_split(manifest).items():
            for entry in entries:
                scenario_path = self._entry_path(entry)
                if scenario_path is None:
                    continue
                episodes.append(
                    self.load_episode(
                        scenario_path,
                        base_dir=manifest_path.parent,
                        split=split_name,
                    )
                )
        return episodes

    def load_split_by_name(
        self,
        split_manifest_path: str | Path,
    ) -> dict[str, list[HumanCentricEpisode]]:
        """Load a split manifest while preserving split labels."""

        manifest_path = self.resolve_path(split_manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episodes_by_split: dict[str, list[HumanCentricEpisode]] = {}
        for split_name, entries in self._scenario_entries_by_split(manifest).items():
            episodes: list[HumanCentricEpisode] = []
            for entry in entries:
                scenario_path = self._entry_path(entry)
                if scenario_path is None:
                    continue
                episodes.append(
                    self.load_episode(
                        scenario_path,
                        base_dir=manifest_path.parent,
                        split=split_name,
                    )
                )
            episodes_by_split[split_name] = episodes
        return episodes_by_split

    def load_path(self, source_path: str | Path) -> list[HumanCentricEpisode]:
        """Load a scenario, manifest, or directory of scenario JSON files."""

        path = self.resolve_path(source_path)
        if path.is_dir():
            episodes: list[HumanCentricEpisode] = []
            for candidate in sorted(path.rglob("*.json")):
                if candidate.name.endswith("_cornercase_metadata.json"):
                    continue
                if self._looks_like_scenario(candidate):
                    episodes.append(self.load_episode(candidate))
            return episodes

        payload = json.loads(path.read_text(encoding="utf-8"))
        if self._is_scenario_payload(payload):
            return [self.load_episode(path)]
        return self.load_split(path)

    def resolve_path(
        self,
        raw_path: str | Path,
        *,
        base_dir: str | Path | None = None,
    ) -> Path:
        """Resolve NavDP paths that may be absolute or repo-root relative."""

        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path

        candidates: list[Path] = []
        if base_dir is not None:
            base = Path(base_dir).expanduser()
            candidates.append(base / path)
            candidates.extend(parent / path for parent in base.parents)
        if self.navdp_root is not None:
            candidates.append(self.navdp_root / path)
        candidates.append(Path.cwd() / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else path

    def _resolve_root(self, scenario_path: Path) -> Path | None:
        if self.navdp_root is not None:
            return self.navdp_root
        for parent in [scenario_path.parent, *scenario_path.parents]:
            if (parent / "README_data_structure.md").is_file():
                return parent
            if (parent / "configs" / "benchmark" / "human_centric_mission_stream").is_dir():
                return parent
        return None

    def _episode_scene_id(
        self,
        raw_scene_id: str,
        scene_assets: JsonDict,
        navdp_root: Path | None,
        scenario_dir: Path,
    ) -> str:
        scene_dir = scene_assets.get("scene_dir")
        if isinstance(scene_dir, str) and scene_dir.strip():
            base = navdp_root or scenario_dir
            resolved = self.resolve_path(scene_dir, base_dir=base)
            return str(resolved)
        return raw_scene_id

    @staticmethod
    def _episode_start_state(payload: JsonDict) -> tuple[list[float], list[float]]:
        robots = payload.get("robots", [])
        if not isinstance(robots, list):
            robots = []
        robot = next((item for item in robots if isinstance(item, dict)), {})
        pose = robot.get("start_map_pose", {}) if isinstance(robot, dict) else {}
        if not isinstance(pose, dict):
            pose = {}
        return (
            [float(_number(pose.get("x"), 0.0)), float(_number(pose.get("y"), 0.0))],
            [float(_number(pose.get("yaw"), 0.0))],
        )

    @staticmethod
    def _mission_type(payload: JsonDict) -> str:
        missions = payload.get("missions", [])
        if not isinstance(missions, list):
            return ""
        mission_types = sorted(
            {
                str(mission.get("mission_type", ""))
                for mission in missions
                if isinstance(mission, dict) and mission.get("mission_type")
            }
        )
        if len(mission_types) == 1:
            return mission_types[0]
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("mission_review_group"):
            return str(metadata["mission_review_group"])
        return ",".join(mission_types)

    @staticmethod
    def _scenario_entries(manifest: Any) -> list[Any]:
        entries: list[Any] = []
        for split_entries in NavDPScenarioAdapter._scenario_entries_by_split(manifest).values():
            entries.extend(split_entries)
        return entries

    @staticmethod
    def _scenario_entries_by_split(manifest: Any) -> dict[str, list[Any]]:
        if isinstance(manifest, list):
            return {"unsplit": manifest}
        if not isinstance(manifest, dict):
            return {}
        splits = manifest.get("splits")
        if isinstance(splits, dict):
            return {
                str(split_name): entries
                for split_name, split_payload in splits.items()
                if (entries := NavDPScenarioAdapter._entry_list(split_payload))
            }

        entries = NavDPScenarioAdapter._entry_list(manifest)
        grouped: defaultdict[str, list[Any]] = defaultdict(list)
        default_split = str(manifest.get("split") or manifest.get("name") or "unsplit")
        for entry in entries:
            split_name = default_split
            if isinstance(entry, dict) and entry.get("split"):
                split_name = str(entry["split"])
            grouped[split_name].append(entry)
        return dict(grouped)

    @staticmethod
    def _entry_list(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        for key in ("examples", "scenarios", "episodes", "paths"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []

    @staticmethod
    def _entry_path(entry: Any) -> str | Path | None:
        if isinstance(entry, (str, Path)):
            return entry
        if not isinstance(entry, dict):
            return None
        for key in ("path", "scenario_path", "json_path", "ref_json"):
            value = entry.get(key)
            if value:
                return str(value)
        return None

    @staticmethod
    def _is_scenario_payload(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and isinstance(payload.get("missions"), list)
            and bool(payload.get("scenario_id"))
        )

    def _looks_like_scenario(self, path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return self._is_scenario_payload(payload)


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default
