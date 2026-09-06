"""Human-centric evaluation and RL skeleton for GN-Bench.

This package is the GN0/GN-Bench-Tools implementation track. It should consume
scenario artifacts produced by NavDP datagen and expose deterministic evaluation
plus RL task integration.
"""

from .baseline_runner import DEFAULT_POLICY_NAMES, POLICY_CLASSES, run_policies_on_episode
from .dagger import DaggerCollectionConfig, HumanDaggerCollector
from .evaluator import HumanCentricEvaluator, ReplayResult
from .dataset import HumanCentricDataset
from .rl_task import HumanCentricRLTask
from .results import (
    CANONICAL_METRIC_KEYS,
    replay_result_row,
    summarize_replay_result_rows,
    write_replay_result_rows,
)
from .scenario_adapter import HumanCentricEpisode, NavDPScenarioAdapter

__all__ = [
    "HumanCentricDataset",
    "HumanCentricEpisode",
    "HumanCentricEvaluator",
    "DaggerCollectionConfig",
    "HumanDaggerCollector",
    "DEFAULT_POLICY_NAMES",
    "CANONICAL_METRIC_KEYS",
    "POLICY_CLASSES",
    "HumanCentricRLTask",
    "NavDPScenarioAdapter",
    "ReplayResult",
    "replay_result_row",
    "run_policies_on_episode",
    "summarize_replay_result_rows",
    "write_replay_result_rows",
]
