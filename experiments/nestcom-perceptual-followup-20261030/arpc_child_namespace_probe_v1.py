"""Picklable import-only probe executed by the real Inductor subprocess pool."""
def probe():
    import os, statistics, sysconfig
    from pathlib import Path
    from torch._inductor.runtime.benchmarking import benchmarker
    p = Path(statistics.__file__).resolve()
    assert p == (Path(sysconfig.get_path('stdlib'))/'statistics.py').resolve()
    assert statistics.median([3, 1, 2]) == 2
    return dict(pid=os.getpid(), statistics=str(p), median=2, model_calls=0,
                PHY_calls=0, timing_calls=0, GPU_calls=0)
