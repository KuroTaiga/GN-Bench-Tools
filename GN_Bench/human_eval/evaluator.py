"""Deterministic evaluator skeleton for human-centric GN-Bench episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .scenario_adapter import HumanCentricEpisode


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ReplayResult:
    """Output record for one deterministic sync replay."""

    episode_id: str
    metrics: JsonDict = field(default_factory=dict)
    events: list[JsonDict] = field(default_factory=list)
    success: bool | None = None
    # TODO: Add per-mission breakdown and violation traces for paper figures.


class HumanCentricEvaluator:
    """Evaluates NavDP-generated scenarios inside GN-Bench.

    Purpose: this is the authoritative evaluation path for benchmark reporting
    and RL reward computation. It should remain deterministic for sync-mode
    replay.
    """

    def replay(self, episode: HumanCentricEpisode) -> ReplayResult:
        """Run deterministic sync replay for one episode.

        TODO: Consume timestamped robot/human trajectories, compute mission
        metrics, navigation metrics, and social metrics.
        """

        return ReplayResult(
            episode_id=episode.episode_id,
            metrics={},
            events=[],
            success=None,
        )

    def evaluate_split(self, episodes: list[HumanCentricEpisode]) -> list[ReplayResult]:
        """Evaluate a list of episodes with deterministic ordering."""

        return [self.replay(episode) for episode in episodes]
