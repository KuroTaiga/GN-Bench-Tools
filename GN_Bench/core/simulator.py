import abc
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import attr
import numpy as np
import torch
from gymnasium import Space, spaces

from GN_Bench.config import Config
from GN_Bench.core.dataset import Episode

VisualObservation = Union[torch.Tensor, np.ndarray]


@attr.s(auto_attribs=True)
class ActionSpaceConfiguration(metaclass=abc.ABCMeta):
    config: Config

    @abc.abstractmethod
    def get(self) -> Any:
        raise NotImplementedError


class SensorTypes(Enum):
    NULL = 0
    COLOR = 1
    DEPTH = 2
    NORMAL = 3
    SEMANTIC = 4
    PATH = 5
    POSITION = 6
    FORCE = 7
    TENSOR = 8
    TEXT = 9
    MEASUREMENT = 10
    HEADING = 11
    TACTILE = 12
    TOKEN_IDS = 13


class Sensor(metaclass=abc.ABCMeta):
    uuid: str
    config: Config
    sensor_type: SensorTypes
    observation_space: Space

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = kwargs["config"] if "config" in kwargs else None
        if hasattr(self.config, "UUID"):
            # We allow any sensor config to override the UUID
            self.uuid = self.config.UUID
        else:
            self.uuid = self._get_uuid(*args, **kwargs)
        self.sensor_type = self._get_sensor_type(*args, **kwargs)
        self.observation_space = self._get_observation_space(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        raise NotImplementedError

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    @abc.abstractmethod
    def get_observation(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class Observations(Dict[str, Any]):
    def __init__(self, sensors: Dict[str, Sensor], *args: Any, **kwargs: Any) -> None:
        data = [
            (uuid, sensor.get_observation(*args, **kwargs))
            for uuid, sensor in sensors.items()
        ]
        super().__init__(data)


class RGBSensor(Sensor, metaclass=abc.ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> VisualObservation:
        raise NotImplementedError


class DepthSensor(Sensor, metaclass=abc.ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> VisualObservation:
        raise NotImplementedError


class SemanticSensor(Sensor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "semantic"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> VisualObservation:
        raise NotImplementedError


class BumpSensor(Sensor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "bump"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.FORCE

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class SensorSuite:
    sensors: Dict[str, Sensor]
    observation_spaces: spaces.Dict

    def __init__(self, sensors: Iterable[Sensor]) -> None:
        self.sensors = OrderedDict()
        ordered_spaces: OrderedDict[str, Space] = OrderedDict()
        for sensor in sensors:
            assert sensor.uuid not in self.sensors, (
                "'{}' is duplicated sensor uuid".format(sensor.uuid)
            )
            self.sensors[sensor.uuid] = sensor
            ordered_spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces = spaces.Dict(spaces=ordered_spaces)

    def get(self, uuid: str) -> Sensor:
        return self.sensors[uuid]

    def get_observations(self, *args: Any, **kwargs: Any) -> Observations:
        return Observations(self.sensors, *args, **kwargs)


@attr.s(auto_attribs=True)
class AgentState:
    position: Optional["np.ndarray"] = None
    rotation: Optional["np.ndarray"] = None


@attr.s(auto_attribs=True)
class ShortestPathPoint:
    position: List[Any]
    rotation: List[Any]
    action: Optional[int] = None


class Simulator:
    GN_Bench_config: Config

    def __init__(self, *args, **kwargs) -> None:
        pass

    @property
    def sensor_suite(self) -> SensorSuite:
        raise NotImplementedError

    @property
    def action_space(self) -> Space:
        raise NotImplementedError

    def reset(self) -> Observations:
        raise NotImplementedError

    def step(self, action, *args, **kwargs) -> Observations:
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def reconfigure(self, config: Config) -> None:
        raise NotImplementedError

    def geodesic_distance(
        self,
        position_a: Sequence[float],
        position_b: Union[Sequence[float], Sequence[Sequence[float]]],
        episode: Optional[Episode] = None,
    ) -> float:
        raise NotImplementedError

    def get_agent_state(self, agent_id: int = 0) -> AgentState:
        r"""..

        :param agent_id: id of agent.
        :return: state of agent corresponding to :p:`agent_id`.
        """
        raise NotImplementedError

    def get_observations_at(
        self,
        position: List[float],
        rotation: List[float],
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        raise NotImplementedError

    def sample_navigable_point(self) -> List[float]:
        r"""Samples a navigable point from the simulator. A point is defined as
        navigable if the agent can be initialized at that point.

        :return: navigable point.
        """
        raise NotImplementedError

    def is_navigable(self, point: List[float]) -> bool:
        raise NotImplementedError

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        raise NotImplementedError

    def get_straight_shortest_path_points(
        self, position_a: List[float], position_b: List[float]
    ) -> List[List[float]]:

        raise NotImplementedError

    @property
    def up_vector(self) -> "np.ndarray":
        raise NotImplementedError

    @property
    def forward_vector(self) -> "np.ndarray":
        raise NotImplementedError

    def render(self, mode: str = "rgb") -> Any:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def previous_step_collided(self) -> bool:
        raise NotImplementedError

    def __enter__(self) -> "Simulator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
