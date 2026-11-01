"""Dependency-only wrapper around immutable v2 timing mathematics and boundaries."""
from followup_common import *
from timing_dependency_v4 import activate
import runpy


if __name__ == '__main__':
    request, expected = sys.argv[1:]
    assert sha(request) == expected
    r = read(request); [checked(p) for p in r['code']]
    if r['group']['method'] == 'ARPC':
        sys.path.insert(0, cr(r['config'])['arpc_public']['environment'])
    activate(r['sionna_package'])
    runpy.run_path(str(E/'timing_end_to_end_namespace_v2.py'), run_name='__main__')
