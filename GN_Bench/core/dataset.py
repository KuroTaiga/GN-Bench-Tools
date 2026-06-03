import copy
import json
import os
import random
from itertools import groupby
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import attr
import numpy as np
from numpy import ndarray

from GN_Bench.config import Config
from GN_Bench.core.utils import not_none_validator

ALL_SCENES_MASK = "*"


@attr.s(auto_attribs=True, kw_only=True)
class Episode:
    episode_id: str = attr.ib(default=None, validator=not_none_validator)
    scene_id: str = attr.ib(default=None, validator=not_none_validator)
    start_position: List[float] = attr.ib(default=None, validator=not_none_validator)
    start_rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    info: Optional[Dict[str, Any]] = None
    _shortest_path_cache: Any = attr.ib(init=False, default=None)

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items() if k not in {"_shortest_path_cache"}
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__["_shortest_path_cache"] = None


T = TypeVar("T", bound=Episode)


class Dataset(Generic[T]):
    episodes: List[T]

    @staticmethod
    def scene_from_scene_path(scene_path: str) -> str:
        return os.path.splitext(os.path.basename(scene_path))[0]

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        assert cls.check_config_paths_exist(config)  # type: ignore[attr-defined]
        dataset = cls(config)  # type: ignore[call-arg]
        return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    @classmethod
    def build_content_scenes_filter(cls, config) -> Callable[[T], bool]:
        scenes_to_load = set(config.CONTENT_SCENES)

        def _filter(ep: T) -> bool:
            return (
                ALL_SCENES_MASK in scenes_to_load
                or cls.scene_from_scene_path(ep.scene_id) in scenes_to_load
            )

        return _filter

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def scene_ids(self) -> List[str]:
        return sorted({episode.scene_id for episode in self.episodes})

    def get_scene_episodes(self, scene_id: str) -> List[T]:
        return list(filter(lambda x: x.scene_id == scene_id, iter(self.episodes)))

    def get_episodes(self, indexes: List[int]) -> List[T]:
        return [self.episodes[episode_id] for episode_id in indexes]

    def get_episode_iterator(self, *args: Any, **kwargs: Any) -> Iterator:
        return EpisodeIterator(self.episodes, *args, **kwargs)

    def to_json(self) -> str:
        class DatasetJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()

                return (
                    obj.__getstate__() if hasattr(obj, "__getstate__") else obj.__dict__
                )

        result = DatasetJSONEncoder().encode(self)
        return result

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        raise NotImplementedError

    def filter_episodes(self, filter_fn: Callable[[T], bool]) -> "Dataset":
        new_episodes = []
        for episode in self.episodes:
            if filter_fn(episode):
                new_episodes.append(episode)
        new_dataset = copy.copy(self)
        new_dataset.episodes = new_episodes
        return new_dataset

    def get_splits(
        self,
        num_splits: int,
        episodes_per_split: Optional[int] = None,
        remove_unused_episodes: bool = False,
        collate_scene_ids: bool = True,
        sort_by_episode_id: bool = False,
        allow_uneven_splits: bool = False,
    ) -> List["Dataset"]:
        if self.num_episodes < num_splits:
            raise ValueError("Not enough episodes to create those many splits.")

        if episodes_per_split is not None:
            if allow_uneven_splits:
                raise ValueError(
                    "You probably don't want to specify allow_uneven_splits"
                    " and episodes_per_split."
                )

            if num_splits * episodes_per_split > self.num_episodes:
                raise ValueError("Not enough episodes to create those many splits.")

        new_datasets = []

        if episodes_per_split is not None:
            stride = episodes_per_split
        else:
            stride = self.num_episodes // num_splits
        split_lengths = [stride] * num_splits

        if allow_uneven_splits:
            episodes_left = self.num_episodes - stride * num_splits
            split_lengths[:episodes_left] = [stride + 1] * episodes_left
            assert sum(split_lengths) == self.num_episodes

        num_episodes = sum(split_lengths)

        rand_items = np.random.choice(self.num_episodes, num_episodes, replace=False)
        if collate_scene_ids:
            scene_ids: Dict[str, List[int]] = {}
            for rand_ind in rand_items:
                scene = self.episodes[rand_ind].scene_id
                if scene not in scene_ids:
                    scene_ids[scene] = []
                scene_ids[scene].append(rand_ind)
            rand_items = []
            list(map(rand_items.extend, scene_ids.values()))
        ep_ind = 0
        new_episodes = []
        for nn in range(num_splits):
            new_dataset = copy.copy(self)  # Creates a shallow copy
            new_dataset.episodes = []
            new_datasets.append(new_dataset)
            for _ii in range(split_lengths[nn]):
                new_dataset.episodes.append(self.episodes[rand_items[ep_ind]])
                ep_ind += 1
            if sort_by_episode_id:
                new_dataset.episodes.sort(key=lambda ep: ep.episode_id)
            new_episodes.extend(new_dataset.episodes)
        if remove_unused_episodes:
            self.episodes = new_episodes
        return new_datasets


class EpisodeIterator(Iterator):
    def __init__(
        self,
        episodes: Sequence[T],
        cycle: bool = True,
        shuffle: bool = False,
        group_by_scene: bool = True,
        max_scene_repeat_episodes: int = -1,
        max_scene_repeat_steps: int = -1,
        num_episode_sample: int = -1,
        step_repetition_range: float = 0.2,
        seed: int = None,
    ) -> None:
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        # sample episodes
        if num_episode_sample >= 0:
            episodes = np.random.choice(episodes, num_episode_sample, replace=False)

        if not isinstance(episodes, list):
            episodes = list(episodes)

        self.episodes = episodes
        self.cycle = cycle
        self.group_by_scene = group_by_scene
        self.shuffle = shuffle

        if shuffle:
            random.shuffle(self.episodes)

        if group_by_scene:
            self.episodes = self._group_scenes(self.episodes)

        self.max_scene_repetition_episodes = max_scene_repeat_episodes
        self.max_scene_repetition_steps = max_scene_repeat_steps

        self._rep_count = -1  # 0 corresponds to first episode already returned
        self._step_count = 0
        self._prev_scene_id: Optional[str] = None

        self._iterator = iter(self.episodes)

        self.step_repetition_range = step_repetition_range
        self._set_shuffle_intervals()

    def __iter__(self) -> "EpisodeIterator":
        return self

    def __next__(self) -> Episode:
        self._forced_scene_switch_if()

        next_episode = next(self._iterator, None)
        if next_episode is None:
            if not self.cycle:
                raise StopIteration

            self._iterator = iter(self.episodes)

            if self.shuffle:
                self._shuffle()

            next_episode = next(self._iterator)

        if (
            self._prev_scene_id != next_episode.scene_id
            and self._prev_scene_id is not None
        ):
            self._rep_count = 0
            self._step_count = 0

        self._prev_scene_id = next_episode.scene_id
        return next_episode

    def _forced_scene_switch(self) -> None:
        grouped_episodes = [
            list(g) for k, g in groupby(self._iterator, key=lambda x: x.scene_id)
        ]

        if len(grouped_episodes) > 1:
            grouped_episodes = grouped_episodes[1:] + grouped_episodes[0:1]

        self._iterator = iter(sum(grouped_episodes, []))

    def _shuffle(self) -> None:
        assert self.shuffle
        episodes = list(self._iterator)

        random.shuffle(episodes)

        if self.group_by_scene:
            episodes = self._group_scenes(episodes)

        self._iterator = iter(episodes)

    def _group_scenes(
        self, episodes: Union[Sequence[Episode], List[Episode], ndarray]
    ) -> List[T]:
        assert self.group_by_scene

        scene_sort_keys: Dict[str, int] = {}
        for e in episodes:
            if e.scene_id not in scene_sort_keys:
                scene_sort_keys[e.scene_id] = len(scene_sort_keys)

        return sorted(episodes, key=lambda e: scene_sort_keys[e.scene_id])  # type: ignore[arg-type]

    def step_taken(self) -> None:
        self._step_count += 1

    @staticmethod
    def _randomize_value(value: int, value_range: float) -> int:
        return random.randint(
            int(value * (1 - value_range)), int(value * (1 + value_range))
        )

    def _set_shuffle_intervals(self) -> None:
        if self.max_scene_repetition_episodes > 0:
            self._max_rep_episode = self.max_scene_repetition_episodes
        else:
            self._max_rep_episode = None

        if self.max_scene_repetition_steps > 0:
            self._max_rep_step = self._randomize_value(
                self.max_scene_repetition_steps, self.step_repetition_range
            )
        else:
            self._max_rep_step = None

    def _forced_scene_switch_if(self) -> None:
        do_switch = False
        self._rep_count += 1

        if (
            self._max_rep_episode is not None
            and self._rep_count >= self._max_rep_episode
        ):
            do_switch = True

        if self._max_rep_step is not None and self._step_count >= self._max_rep_step:
            do_switch = True

        if do_switch:
            self._forced_scene_switch()
            self._set_shuffle_intervals()
