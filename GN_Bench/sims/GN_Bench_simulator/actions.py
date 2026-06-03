from enum import Enum
from typing import Dict, Optional, Any
import attr


from GN_Bench.core.registry import registry
from GN_Bench.core.simulator import ActionSpaceConfiguration
from GN_Bench.core.utils import Singleton


@attr.s(auto_attribs=True)
class ActuationSpec:
    amount: float
    constraint: Optional[str] = None


@attr.s(auto_attribs=True)
class PyRobotNoisyActuationSpec(ActuationSpec):
    robot: str = "LoCoBot"
    controller: str = "ILQR"
    noise_multiplier: float = 1.0


@attr.s(auto_attribs=True)
class ActionSpec:
    name: str
    actuation: Optional[ActuationSpec] = None


class _DefaultGNBenchSimActions(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


@attr.s(auto_attribs=True, slots=True)
class GNBenchSimActionsSingleton(metaclass=Singleton):
    _known_actions: Dict[str, int] = attr.ib(init=False, factory=dict)

    def __attrs_post_init__(self):
        for action in _DefaultGNBenchSimActions:
            self._known_actions[action.name] = action.value

    def extend_action_space(self, name: str) -> int:
        assert name not in self._known_actions, "Cannot register an action name twice"
        self._known_actions[name] = len(self._known_actions)
        return self._known_actions[name]

    def has_action(self, name: str) -> bool:
        return name in self._known_actions

    def __getattr__(self, name):
        return self._known_actions[name]

    def __getitem__(self, name):
        return self._known_actions[name]

    def __len__(self):
        return len(self._known_actions)

    def __iter__(self):
        return iter(self._known_actions)


GNBenchSimActions: GNBenchSimActionsSingleton = GNBenchSimActionsSingleton()


@registry.register_action_space_configuration(name="v0")
class InteriorGSSimV0ActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            GNBenchSimActions.STOP: ActionSpec("stop"),
            GNBenchSimActions.MOVE_FORWARD: ActionSpec(
                "move_forward",
                ActuationSpec(amount=self.config.FORWARD_STEP_SIZE),
            ),
            GNBenchSimActions.TURN_LEFT: ActionSpec(
                "turn_left",
                ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            GNBenchSimActions.TURN_RIGHT: ActionSpec(
                "turn_right",
                ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }


@registry.register_action_space_configuration(name="v1")
class InteriorGSSimV1ActionSpaceConfiguration(InteriorGSSimV0ActionSpaceConfiguration):
    def get(self):
        config = super().get()
        new_config = {
            GNBenchSimActions.LOOK_UP: ActionSpec(
                "look_up",
                ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            GNBenchSimActions.LOOK_DOWN: ActionSpec(
                "look_down",
                ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
        }

        config.update(new_config)

        return config


@registry.register_action_space_configuration(name="pyrobotnoisy")
class GNBenchSimPyRobotActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            GNBenchSimActions.STOP: ActionSpec("stop"),
            GNBenchSimActions.MOVE_FORWARD: ActionSpec(
                "pyrobot_noisy_move_forward",
                PyRobotNoisyActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE,
                    robot=self.config.NOISE_MODEL.ROBOT,
                    controller=self.config.NOISE_MODEL.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MODEL.NOISE_MULTIPLIER,
                ),
            ),
            GNBenchSimActions.TURN_LEFT: ActionSpec(
                "pyrobot_noisy_turn_left",
                PyRobotNoisyActuationSpec(
                    amount=self.config.TURN_ANGLE,
                    robot=self.config.NOISE_MODEL.ROBOT,
                    controller=self.config.NOISE_MODEL.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MODEL.NOISE_MULTIPLIER,
                ),
            ),
            GNBenchSimActions.TURN_RIGHT: ActionSpec(
                "pyrobot_noisy_turn_right",
                PyRobotNoisyActuationSpec(
                    amount=self.config.TURN_ANGLE,
                    robot=self.config.NOISE_MODEL.ROBOT,
                    controller=self.config.NOISE_MODEL.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MODEL.NOISE_MULTIPLIER,
                ),
            ),
            GNBenchSimActions.LOOK_UP: ActionSpec(
                "look_up",
                ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            GNBenchSimActions.LOOK_DOWN: ActionSpec(
                "look_down",
                ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            # The perfect actions are needed for the oracle planner
            "_forward": ActionSpec(
                "move_forward",
                ActuationSpec(amount=self.config.FORWARD_STEP_SIZE),
            ),
            "_left": ActionSpec(
                "turn_left",
                ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            "_right": ActionSpec(
                "turn_right",
                ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }
