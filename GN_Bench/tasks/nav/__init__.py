from GN_Bench.core.embodied_task import EmbodiedTask
from GN_Bench.core.registry import registry


def _try_register_nav_task():
    try:
        from GN_Bench.tasks.nav.nav import NavigationTask  # noqa
    except ImportError as e:
        navtask_import_error = e

        @registry.register_task(name="Nav-v0")
        class NavigationTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise navtask_import_error
