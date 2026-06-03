from GN_Bench.core.embodied_task import EmbodiedTask
from GN_Bench.core.registry import registry


def _try_register_vln_task():
    try:
        from GN_Bench.tasks.vln.vln import VLNTask  # noqa: F401
    except ImportError as e:
        vlntask_import_error = e

        @registry.register_task(name="VLN-v0")
        class VLNTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise vlntask_import_error
