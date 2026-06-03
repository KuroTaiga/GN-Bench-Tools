#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from collections.abc import Collection
from typing import Dict, List, Union

import gymnasium
from gymnasium import Space


class EmptySpace(Space):
    def sample(self):
        return None

    def contains(self, x):
        if x is None:
            return True
        return False

    def __repr__(self):
        return "EmptySpace()"


class ActionSpace(gymnasium.spaces.Dict):
    def __init__(self, spaces: Union[List, Dict]):
        if isinstance(spaces, dict):
            self.spaces = OrderedDict(sorted(spaces.items()))
        if isinstance(spaces, list):
            self.spaces = OrderedDict(spaces)
        self.actions_select = gymnasium.spaces.Discrete(len(self.spaces))

    @property
    def n(self) -> int:
        return len(self.spaces)

    def sample(self):
        action_index = self.actions_select.sample()
        return {
            "action": list(self.spaces.keys())[action_index],
            "action_args": list(self.spaces.values())[action_index].sample(),
        }

    def contains(self, x):
        if not isinstance(x, dict) or "action" not in x:
            return False
        if x["action"] not in self.spaces:
            return False
        if not self.spaces[x["action"]].contains(x.get("action_args", None)):
            return False
        return True

    def __repr__(self):
        return (
            "ActionSpace("
            + ", ".join([k + ":" + str(s) for k, s in self.spaces.items()])
            + ")"
        )


class ListSpace(Space):
    def __init__(
        self,
        space: Space,
        min_seq_length: int = 0,
        max_seq_length: int = 1 << 15,
    ):
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.space = space
        self.length_select = gymnasium.spaces.Discrete(max_seq_length - min_seq_length)

    def sample(self):
        seq_length = self.length_select.sample() + self.min_seq_length
        return [self.space.sample() for _ in range(seq_length)]

    def contains(self, x):
        if not isinstance(x, Collection):
            return False

        if not (self.min_seq_length <= len(x) <= self.max_seq_length):
            return False

        return all(self.space.contains(el) for el in x)

    def __repr__(self):
        return (
            f"ListSpace({self.space}, min_seq_length="
            f"{self.min_seq_length}, max_seq_length={self.max_seq_length})"
        )
