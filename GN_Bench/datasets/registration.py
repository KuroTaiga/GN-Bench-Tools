from GN_Bench.core.logging import logger
from GN_Bench.core.registry import registry


def make_dataset(id_dataset, **kwargs):
    logger.info("Initializing dataset {}".format(id_dataset))
    _dataset = registry.get_dataset(id_dataset)
    assert _dataset is not None, "Could not find dataset {}".format(id_dataset)

    return _dataset(**kwargs)  # type: ignore
