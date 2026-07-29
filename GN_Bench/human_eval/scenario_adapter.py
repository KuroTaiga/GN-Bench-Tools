"""Adapter from NavDP datagen scenario JSON to GN-Bench episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class HumanCentricEpisode:
    """GN-Bench-side episode wrapper for one generated human scenario."""

    episode_id: str
    scene_id: str
    scenario_path: Path
    payload: JsonDict = field(default_factory=dict)
    # TODO: Map payload fields into GN_Bench.core.dataset.Episode once the
    # scenario schema is stable.


class NavDPScenarioAdapter:
    """Loads NavDP-generated scenarios and exposes GN-Bench episode records."""

    def load_episode(self, scenario_path: str | Path) -> HumanCentricEpisode:
        """Load one NavDP scenario JSON as a GN-Bench evaluation episode."""

        path = Path(scenario_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return HumanCentricEpisode(
            episode_id=str(payload.get("scenario_id", path.stem)),
            scene_id=str(payload.get("scene_id", "")),
            scenario_path=path,
            payload=payload,
        )

    def load_split(self, split_manifest_path: str | Path) -> list[HumanCentricEpisode]:
        """Load a split manifest produced by NavDP datagen.

        TODO: Finalize manifest schema. Expected first version can be:
        {"scenarios": [{"path": "..."}]}.
        """

        manifest_path = Path(split_manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episodes: list[HumanCentricEpisode] = []
        for entry in manifest.get("scenarios", []):
            scenario_path = Path(entry["path"])
            if not scenario_path.is_absolute():
                scenario_path = manifest_path.parent / scenario_path
            episodes.append(self.load_episode(scenario_path))
        return episodes
