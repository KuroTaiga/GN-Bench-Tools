from GN_Bench.config import Config, get_config
from GN_Bench.core.dataset import Dataset
from GN_Bench.core.logging import logger
from GN_Bench.core.registry import registry  # noqa : F401
from GN_Bench.datasets import make_dataset


_LAZY_EXPORTS = {
    "EmbodiedTask": ("GN_Bench.core.embodied_task", "EmbodiedTask"),
    "Env": ("GN_Bench.core.env", "Env"),
    "Measure": ("GN_Bench.core.embodied_task", "Measure"),
    "Measurements": ("GN_Bench.core.embodied_task", "Measurements"),
    "RLEnv": ("GN_Bench.core.env", "RLEnv"),
    "Sensor": ("GN_Bench.core.simulator", "Sensor"),
    "SensorSuite": ("GN_Bench.core.simulator", "SensorSuite"),
    "SensorTypes": ("GN_Bench.core.simulator", "SensorTypes"),
    "Simulator": ("GN_Bench.core.simulator", "Simulator"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = __import__(module_name, fromlist=[attribute_name])
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "Agent",
    "Benchmark",
    "Challenge",
    "Config",
    "Dataset",
    "EmbodiedTask",
    "Env",
    "get_config",
    "logger",
    "make_dataset",
    "Measure",
    "Measurements",
    "RLEnv",
    "Sensor",
    "SensorSuite",
    "SensorTypes",
    "Simulator",
]
