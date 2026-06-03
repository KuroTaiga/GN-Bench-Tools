import collections
from typing import Any, Callable, DefaultDict, Optional, Type

from GN_Bench.core.dataset import Dataset
from GN_Bench.core.embodied_task import Action, EmbodiedTask, Measure
from GN_Bench.core.simulator import ActionSpaceConfiguration, Sensor, Simulator
from GN_Bench.core.utils import Singleton


class Registry(metaclass=Singleton):
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(to_register, assert_type), (
                    "{} must be a subclass of {}".format(to_register, assert_type)
                )
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_task(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("task", to_register, name, assert_type=EmbodiedTask)

    @classmethod
    def register_simulator(
        cls, to_register: None = None, *, name: Optional[str] = None
    ):
        return cls._register_impl("sim", to_register, name, assert_type=Simulator)

    @classmethod
    def register_sensor(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("sensor", to_register, name, assert_type=Sensor)

    @classmethod
    def register_measure(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("measure", to_register, name, assert_type=Measure)

    @classmethod
    def register_task_action(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("task_action", to_register, name, assert_type=Action)

    @classmethod
    def register_dataset(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("dataset", to_register, name, assert_type=Dataset)

    @classmethod
    def register_action_space_configuration(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        return cls._register_impl(
            "action_space_config",
            to_register,
            name,
            assert_type=ActionSpaceConfiguration,
        )

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_task(cls, name: str) -> Type[EmbodiedTask]:
        return cls._get_impl("task", name)

    @classmethod
    def get_task_action(cls, name: str) -> Type[Action]:
        return cls._get_impl("task_action", name)

    @classmethod
    def get_simulator(cls, name: str) -> Type[Simulator]:
        return cls._get_impl("sim", name)

    @classmethod
    def get_sensor(cls, name: str) -> Type[Sensor]:
        return cls._get_impl("sensor", name)

    @classmethod
    def get_measure(cls, name: str) -> Type[Measure]:
        return cls._get_impl("measure", name)

    @classmethod
    def get_dataset(cls, name: str) -> Type[Dataset]:
        return cls._get_impl("dataset", name)

    @classmethod
    def get_action_space_configuration(
        cls, name: str
    ) -> Type[ActionSpaceConfiguration]:
        return cls._get_impl("action_space_config", name)


registry = Registry()
