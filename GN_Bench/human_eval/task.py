"""GN-Bench task registration for human-centric replay and future sim rollout."""

from __future__ import annotations

from typing import Any

from GN_Bench.core.embodied_task import EmbodiedTask
from GN_Bench.core.registry import registry

from .evaluator import HumanCentricEvaluator
from .rl_task import HumanCentricRLTask
from .scenario_adapter import HumanCentricEpisode


JsonDict = dict[str, Any]


@registry.register_task(name="HumanCentricTask-v0")
class HumanCentricTask(EmbodiedTask):
    """Human-centric task entry point with replay-backed reset/step helpers."""

    def __init__(
        self,
        config: Any | None = None,
        sim: Any | None = None,
        dataset: Any | None = None,
        evaluator: HumanCentricEvaluator | None = None,
    ) -> None:
        self.replay_task = HumanCentricRLTask(evaluator=evaluator)
        self._config = config
        self._sim = sim
        self._dataset = dataset
        self._is_episode_active = False
        if config is not None and sim is not None:
            super().__init__(config=config, sim=sim, dataset=dataset)

    def reset_replay(self, episode: HumanCentricEpisode) -> JsonDict:
        """Reset the deterministic replay-backed task surface."""

        self._is_episode_active = True
        return self.replay_task.reset(episode)

    def step_replay(self, action: JsonDict) -> tuple[JsonDict, float, bool, JsonDict]:
        """Apply a replay-backed JSON action without simulator rendering."""

        observation, reward, done, info = self.replay_task.step(action)
        self._is_episode_active = not done
        return observation, reward, done, info

    def overwrite_sim_config(self, sim_config: Any, episode: HumanCentricEpisode) -> Any:
        """Map NavDP episode references into the GN-Bench simulator config."""

        if hasattr(sim_config, "defrost"):
            sim_config.defrost()
        sim_config.SCENE = episode.scene_id
        sim_config.REF_JSON = episode.ref_json
        if hasattr(sim_config, "freeze"):
            sim_config.freeze()
        if episode.start_position is not None and episode.start_rotation is not None:
            try:
                agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
                agent_cfg = getattr(sim_config, agent_name)
            except (AttributeError, IndexError, KeyError, TypeError):
                return sim_config
            if hasattr(agent_cfg, "defrost"):
                agent_cfg.defrost()
            agent_cfg.START_POSITION = episode.start_position
            agent_cfg.START_ROTATION = episode.start_rotation
            agent_cfg.IS_SET_START_STATE = True
            if hasattr(agent_cfg, "freeze"):
                agent_cfg.freeze()
        return sim_config

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        state = self.replay_task.state
        if state is not None:
            return not state.done
        return self._is_episode_active
