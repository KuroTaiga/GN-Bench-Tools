from GN_Bench.core.logging import logger
from GN_Bench.core.registry import registry
from GN_Bench.sims.GN_Bench_simulator import _try_register_GN_Bench_sim


def make_sim(id_sim, **kwargs):
    logger.info("initializing sim {}".format(id_sim))
    _sim = registry.get_simulator(id_sim)
    assert _sim is not None, "Could not find simulator with name {}".format(id_sim)
    return _sim(**kwargs)


_try_register_GN_Bench_sim()
