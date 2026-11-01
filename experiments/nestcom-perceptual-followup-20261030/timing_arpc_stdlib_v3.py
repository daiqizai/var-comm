"""Prepared, NOT RUN: preserve original ARPC worker, prevent stdlib shadowing."""
from stdlib_statistics_v1 import bind_stdlib_statistics
statistics_identity=bind_stdlib_statistics()

from followup_common import *
import runpy

if __name__=='__main__':
    request,expected=sys.argv[1:]
    assert sha(request)==expected
    r=read(request);[checked(p) for p in r['code']]
    assert r['group']['method']=='ARPC'
    assert statistics_identity==r['statistics_stdlib']
    # No changes to model, precision, compile mode, PHY, or timer boundaries.
    runpy.run_path(str(E/'timing_end_to_end_arpc_ipc_v2.py'),run_name='__main__')
