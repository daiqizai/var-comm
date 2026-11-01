"""CPU qualification of the actual compile subprocess namespace, no model calls."""
from stdlib_statistics_v1 import bind_stdlib_statistics
standard_statistics = bind_stdlib_statistics()
from followup_common import *
import traceback

def main(request, expected):
    assert sha(request) == expected
    r=read(request); [checked(x) for x in r['code']]
    out=Path(r['output']); out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,set(r['cpu_affinity']))
    try:
        cfg=cr(r['config']); sys.path.insert(0,cfg['arpc_public']['environment'])
        sys.path.insert(0,str(R/'experiments/five-method-timing-20261020'))
        import native
        native.paths()
        # python_subprocess_env propagates sys.path into the child PYTHONPATH.
        sys.path.insert(0,str(E/'arpc_child_stdlib_v1'))
        import torch
        assert torch.__version__=='2.10.0+cu130'
        from torch._inductor.compile_worker.subproc_pool import SubprocPool
        from arpc_child_namespace_probe_v1 import probe
        pool=SubprocPool(1)
        try:
            result=pool.submit(probe).result(timeout=90)
            assert result['pid'] != os.getpid()
        finally:
            pool.shutdown()
        save(out/'completion.json',dict(status='COMPLETE',statistics_stdlib=standard_statistics,
             actual_compile_subprocess=result,torch=torch.__version__,model_calls=0,
             PHY_calls=0,timing_calls=0,GPU_calls=0,automatic_retry=False))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),model_calls=0,
             PHY_calls=0,timing_calls=0,GPU_calls=0,automatic_retry=False)); raise

if __name__=='__main__': main(*sys.argv[1:])
