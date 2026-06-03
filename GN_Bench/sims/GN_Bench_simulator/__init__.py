from GN_Bench.core.registry import registry


def _try_register_GN_Bench_sim():

    try:
        from GN_Bench.sims.GN_Bench_simulator import GN_Bench_simulator

        from GN_Bench.sims.GN_Bench_simulator import actions

    except ImportError as e:
        print(f"CRITICAL ERROR: Could not import GN_Bench_simulator python file: {e}")
        raise e
