"""History-window selection for human-centric DAgger records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .schema import DaggerCollectionConfig


@dataclass(frozen=True)
class DaggerHistorySelection:
    """Selected past context for one DAgger query."""

    indices: list[int]
    frames: list[Any] = field(default_factory=list)
    positions: list[Any] = field(default_factory=list)
    rotations: list[Any] = field(default_factory=list)


class DaggerHistorySelector:
    """Select fixed-length past context using the DAgger collection config."""

    def __init__(self, config: DaggerCollectionConfig | None = None) -> None:
        self.config = config or DaggerCollectionConfig()

    def select_indices(self, history_length: int) -> list[int]:
        """Return source indices for the configured past-frame window."""

        if history_length <= 0:
            return []
        eligible_length = int(history_length)
        if self.config.history_end == "previous" and eligible_length > 1:
            eligible_length -= 1

        if self.config.history_sampling == "uniform":
            return _uniform_indices(eligible_length, self.config.past_frames)
        return _recent_indices(eligible_length, self.config.past_frames)

    def select_triplets(
        self,
        *,
        history_frames: Sequence[Any],
        history_positions: Sequence[Any] | None = None,
        history_rotations: Sequence[Any] | None = None,
    ) -> DaggerHistorySelection:
        """Select frames/poses/rotations using the same indices."""

        indices = self.select_indices(len(history_frames))
        return DaggerHistorySelection(
            indices=indices,
            frames=[history_frames[index] for index in indices],
            positions=_select_optional(history_positions, indices),
            rotations=_select_optional(history_rotations, indices),
        )


def _uniform_indices(history_length: int, past_frames: int) -> list[int]:
    if history_length <= 0:
        return []
    if history_length == 1:
        return [0] * past_frames
    if past_frames == 1:
        return [history_length - 1]
    return [
        int((step * (history_length - 1)) // (past_frames - 1))
        for step in range(past_frames)
    ]


def _recent_indices(history_length: int, past_frames: int) -> list[int]:
    if history_length <= 0:
        return []
    start = max(0, history_length - past_frames)
    indices = list(range(start, history_length))
    while len(indices) < past_frames:
        indices.insert(0, indices[0])
    return indices


def _select_optional(values: Sequence[Any] | None, indices: list[int]) -> list[Any]:
    if values is None:
        return []
    return [values[index] for index in indices]
