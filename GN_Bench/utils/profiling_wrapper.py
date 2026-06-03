from contextlib import ContextDecorator

try:
    from GN_Bench_sim.utils import profiling_utils
except ImportError:
    profiling_utils = None


def configure(capture_start_step=-1, num_steps_to_capture=-1):
    r"""Wrapper for GN_Bench_sim profiling_utils.configure"""
    if profiling_utils:
        profiling_utils.configure(capture_start_step, num_steps_to_capture)


def on_start_step():
    r"""Wrapper for GN_Bench_sim profiling_utils.on_start_step"""
    if profiling_utils:
        profiling_utils.on_start_step()


def range_push(msg: str):
    r"""Wrapper for GN_Bench_sim profiling_utils.range_push"""
    if profiling_utils:
        profiling_utils.range_push(msg)


def range_pop():
    r"""Wrapper for GN_Bench_sim profiling_utils.range_pop"""
    if profiling_utils:
        profiling_utils.range_pop()


class RangeContext(ContextDecorator):
    r"""Annotate a range for profiling. Use as a function decorator or in a with
    statement. See GN_Bench_sim profiling_utils.
    """

    def __init__(self, msg: str):
        self._msg = msg

    def __enter__(self):
        range_push(self._msg)
        return self

    def __exit__(self, *exc):
        range_pop()
        return False
