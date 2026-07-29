"""Social metric skeletons for human-centric evaluation."""

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
        # TODO: Implement min distance, near misses, collisions, and duration.
        return []


class FFormationMetrics:
    """Computes conversation-group social violations."""

    def compute(self, replay_state: JsonDict) -> list[MetricValue]:
        # TODO: Implement protected-center crossing and group splitting.
        return []


class QueueMetrics:
    """Computes queue order and segmentation violations."""

    def compute(self, replay_state: JsonDict) -> list[MetricValue]:
        # TODO: Implement queue cutting, segmentation, and service order.
        return []


class PedestrianFlowMetrics:
    """Computes flow-following and right-side-yield violations."""

    def compute(self, replay_state: JsonDict) -> list[MetricValue]:
        # TODO: Implement reverse-flow traversal and yield failures.
        return []


class VulnerableHumanMetrics:
    """Computes larger-buffer penalties for vulnerable humans."""

    def compute(self, replay_state: JsonDict) -> list[MetricValue]:
        # TODO: Implement larger buffer violation and weighted near-miss penalty.
        return []
