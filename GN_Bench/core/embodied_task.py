from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from GN_Bench.config import Config
from GN_Bench.core.dataset import Dataset, Episode
from GN_Bench.core.simulator import Observations, SensorSuite, Simulator
from GN_Bench.core.spaces import ActionSpace, EmptySpace, Space


class Action:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return

    def reset(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def step(self, *args: Any, **kwargs: Any) -> Observations:
        raise NotImplementedError

    @property
    def action_space(self) -> Space:
        raise NotImplementedError


class SimulatorTaskAction(Action):
    def __init__(
        self, *args: Any, config: Config, sim: Simulator, **kwargs: Any
    ) -> None:
        self._config = config
        self._sim = sim

    @property
    def action_space(self):
        return EmptySpace()

    def reset(self, *args: Any, **kwargs: Any) -> None:
        return None

    def step(self, *args: Any, **kwargs: Any) -> Observations:
        raise NotImplementedError


class Measure:
    _metric: Any
    uuid: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.uuid = self._get_uuid(*args, **kwargs)
        self._metric = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def get_metric(self):
        return self._metric


class Metrics(dict):
    def __init__(self, measures: Dict[str, Measure]) -> None:
        data = [(uuid, measure.get_metric()) for uuid, measure in measures.items()]
        super().__init__(data)


class Measurements:
    measures: Dict[str, Measure]

    def __init__(self, measures: Iterable[Measure]) -> None:
        self.measures = OrderedDict()
        for measure in measures:
            assert measure.uuid not in self.measures, (
                "'{}' is duplicated measure uuid".format(measure.uuid)
            )
            self.measures[measure.uuid] = measure

    def reset_measures(self, *args: Any, **kwargs: Any) -> None:
        for measure in self.measures.values():
            measure.reset_metric(*args, **kwargs)

    def update_measures(self, *args: Any, **kwargs: Any) -> None:
        for measure in self.measures.values():
            measure.update_metric(*args, **kwargs)

    def get_metrics(self) -> Metrics:
        return Metrics(self.measures)

    def _get_measure_index(self, measure_name):
        return list(self.measures.keys()).index(measure_name)

    def check_measure_dependencies(self, measure_name: str, dependencies: List[str]):
        measure_index = self._get_measure_index(measure_name)
        for dependency_measure in dependencies:
            assert (
                dependency_measure in self.measures
            ), f"""{measure_name} measure requires {dependency_measure}
                listed in tje measures list in the config."""

        for dependency_measure in dependencies:
            assert measure_index > self._get_measure_index(
                dependency_measure
            ), f"""{measure_name} measure requires be listed after {dependency_measure}
                in tje measures list in the config."""


class EmbodiedTask:
    _config: Any
    _sim: Optional[Simulator]
    _dataset: Optional[Dataset]
    _is_episode_active: bool
    measurements: Measurements
    sensor_suite: SensorSuite

    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        from GN_Bench.core.registry import registry

        self._config = config
        self._sim = sim
        self._dataset = dataset

        self.measurements = Measurements(
            self._init_entities(
                entity_names=config.MEASUREMENTS,
                register_func=registry.get_measure,
                entities_config=config,
            ).values()
        )

        self.sensor_suite = SensorSuite(
            self._init_entities(
                entity_names=config.SENSORS,
                register_func=registry.get_sensor,
                entities_config=config,
            ).values()
        )

        self.actions = self._init_entities(
            entity_names=config.POSSIBLE_ACTIONS,
            register_func=registry.get_task_action,
            entities_config=self._config.ACTIONS,
        )
        self._action_keys = list(self.actions.keys())

    def _init_entities(
        self, entity_names, register_func, entities_config=None
    ) -> OrderedDict:
        if entities_config is None:
            entities_config = self._config

        entities = OrderedDict()
        for entity_name in entity_names:
            entity_cfg = getattr(entities_config, entity_name)
            entity_type = register_func(entity_cfg.TYPE)
            assert entity_type is not None, (
                f"invalid {entity_name} type {entity_cfg.TYPE}"
            )
            entities[entity_name] = entity_type(
                sim=self._sim,
                config=entity_cfg,
                dataset=self._dataset,
                task=self,
            )
        return entities

    def reset(self, episode: Episode):
        observations = self._sim.reset()
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode, task=self
            )
        )

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)

        return observations

    def step(self, action: Dict[str, Any], episode: Episode):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]

        # If action is a dict (e.g. NavDP state), bypass task action lookup and pass directly to sim
        if isinstance(action_name, dict):
            observations = self._sim.step(action_name)
        else:
            if isinstance(action_name, (int, np.integer)):
                action_name = self.get_action_name(action_name)
            assert action_name in self.actions, (
                f"Can't find '{action_name}' action in {self.actions.keys()}."
            )

            task_action = self.actions[action_name]
            observations = task_action.step(**action["action_args"], task=self)

        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )

        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        return observations

    def get_action_name(self, action_index: int):
        if action_index >= len(self.actions):
            raise ValueError(f"Action index '{action_index}' is out of range.")
        return self._action_keys[action_index]

    @property
    def action_space(self) -> Space:
        return ActionSpace(
            {
                action_name: action_instance.action_space
                for action_name, action_instance in self.actions.items()
            }
        )

    def overwrite_sim_config(self, sim_config: Config, episode: Episode) -> Config:
        raise NotImplementedError

    def _check_episode_is_active(
        self,
        *args: Any,
        action: Union[int, Dict[str, Any]],
        episode: Episode,
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError

    @property
    def is_episode_active(self):
        return self._is_episode_active

    def seed(self, seed: int) -> None:
        return
