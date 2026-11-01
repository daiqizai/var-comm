"""Namespace-only RX correction; immutable v3 TX timings are reused."""
from stdlib_statistics_v1 import bind_stdlib_statistics
statistics_identity=bind_stdlib_statistics()
from followup_common import *
import runpy

if __name__=='__main__':
    request,expected=sys.argv[1:]
    assert sha(request)==expected
    r=read(request); [checked(p) for p in r['code']]
    assert r['group']['method']=='ARPC' and r['direction']=='rx'
    assert statistics_identity==r['statistics_stdlib']
    q=cr(r['child_namespace_qualification'])
    assert q['status']=='COMPLETE' and q['model_calls']==q['PHY_calls']==q['GPU_calls']==0
    sys.path.insert(0,str(E/'arpc_child_stdlib_v1'))
    # This insertion is propagated by torch's python_subprocess_env. No compiler
    # options, numerical route, model, PHY, thread counts or timer code change.
    runpy.run_path(str(E/'timing_end_to_end_arpc_ipc_v2.py'),run_name='__main__')
