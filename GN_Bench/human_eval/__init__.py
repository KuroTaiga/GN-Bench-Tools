"""Human-centric evaluation and RL skeleton for GN-Bench.

This package is the GN0/GN-Bench-Tools implementation track. It should consume
scenario artifacts produced by NavDP datagen and expose deterministic evaluation
plus RL task integration.
"""

from .baseline_runner import POLICY_CLASSES, run_policies_on_episode
from .dagger import DaggerCollectionConfig, HumanDaggerCollector
from .evaluator import HumanCentricEvaluator, ReplayResult
from .dataset import HumanCentricDataset
from .rl_task import HumanCentricRLTask
from .scenario_adapter import HumanCentricEpisode, NavDPScenarioAdapter

__all__ = [
    "HumanCentricDataset",
    "HumanCentricEpisode",
    "HumanCentricEvaluator",
    "DaggerCollectionConfig",
    "HumanDaggerCollector",
    "POLICY_CLASSES",
    "HumanCentricRLTask",
    "NavDPScenarioAdapter",
    "ReplayResult",
    "run_policies_on_episode",
]
