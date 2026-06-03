from typing import Any, Dict, List, Optional

import attr
from gymnasium import spaces

from GN_Bench.core.registry import registry
from GN_Bench.core.simulator import Observations, Sensor
from GN_Bench.core.utils import not_none_validator
from GN_Bench.tasks.nav.nav import NavigationEpisode, NavigationTask


@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: Optional[List[str]] = None


@attr.s(auto_attribs=True, kw_only=True)
class VLNEpisode(NavigationEpisode):
    reference_path: List[List[float]] = attr.ib(
        default=None, validator=not_none_validator
    )
    instruction: InstructionData = attr.ib(default=None, validator=not_none_validator)
    trajectory_id: int = attr.ib(default=None, validator=not_none_validator)


@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        # Use Dict space for text instruction instead of Discrete(0)
        # Discrete(0) is invalid in gymnasium (requires n > 0)
        self.observation_space = spaces.Dict({"text": spaces.Text(max_length=1000)})

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self, observations: Dict[str, Observations], episode: VLNEpisode, **kwargs
    ):
        return {
            "text": episode.grounded_instruction,
        }

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_task(name="VLN-v0")
class VLNTask(NavigationTask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
