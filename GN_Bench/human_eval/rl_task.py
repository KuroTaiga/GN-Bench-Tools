"""RL task skeleton for human-centric GN-Bench evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evaluator import HumanCentricEvaluator, ReplayResult
from .scenario_adapter import HumanCentricEpisode


JsonDict = dict[str, Any]


@dataclass
class HumanCentricRLState:
    """Mutable RL episode state extracted from a human-centric scenario."""

    episode: HumanCentricEpisode
    step_index: int = 0
    last_observation: JsonDict = field(default_factory=dict)
    last_result: ReplayResult | None = None
    # TODO: Add robot assignment state, mission queue state, and human state.


class HumanCentricRLTask:
    """GN-Bench task wrapper for training/evaluating RL agents."""

    def __init__(self, evaluator: HumanCentricEvaluator | None = None) -> None:
        self.evaluator = evaluator or HumanCentricEvaluator()
        self.state: HumanCentricRLState | None = None

    def reset(self, episode: HumanCentricEpisode) -> JsonDict:
        """Initialize an RL episode from a NavDP-generated scenario."""

        self.state = HumanCentricRLState(episode=episode)
        # TODO: Convert scenario start state into GN-Bench simulator observation.
        return {}

    def step(self, action: JsonDict) -> tuple[JsonDict, float, bool, JsonDict]:
        """Advance one RL step.

        TODO: Integrate with GN_Bench.core.embodied_task once action and reward
        contracts are selected.
        """

        if self.state is None:
            raise RuntimeError("HumanCentricRLTask.reset must be called before step")
        self.state.step_index += 1
        reward = 0.0
        done = False
        info: JsonDict = {"todo": "implement RL transition and reward"}
        return {}, reward, done, info
