from GN_Bench.core.logging import logger
from GN_Bench.core.registry import registry
from GN_Bench.tasks.nav import _try_register_nav_task
from GN_Bench.tasks.vln import _try_register_vln_task


def _try_register_human_centric_task():
    try:
        import GN_Bench.human_eval.task  # noqa: F401
    except ImportError as import_error:
        from GN_Bench.core.embodied_task import EmbodiedTask

        @registry.register_task(name="HumanCentricTask-v0")
        class HumanCentricTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise import_error


def make_task(id_task, **kwargs):
    logger.info("Initializing task {}".format(id_task))
    _task = registry.get_task(id_task)
    assert _task is not None, "Could not find task with name {}".format(id_task)

    return _task(**kwargs)


_try_register_nav_task()
_try_register_vln_task()
_try_register_human_centric_task()
