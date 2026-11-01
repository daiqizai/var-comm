"""Resource-only BPG adapter revision, preserving original timed mathematics."""
from followup_common import *
from timing_dependency_v4 import activate

if __name__=='__main__':
    request,expected=sys.argv[1:]
    assert sha(request)==expected
    r=read(request);[checked(p) for p in r['code']]
    assert r['group']['method']=='BPG_LDPC' and len(r['cpu_affinity'])==2
    q=cr(r['BPG_two_core_qualification'])
    assert q['status']=='COMPLETE' and q['synthetic_only'] and q['real_source_calls']==0
    assert q['engineering_encode_calls']==q['engineering_decode_calls']==2
    shim=q['adapter'];checked(shim)
    original=(E/'timing_end_to_end_identity_v5.py').read_text()
    old="shim=B/'five_method_timing_20261020_v1/bpg_threads_qualification_v1/affinity_cpu_count.so'\n assert sha(shim)=='953375c8aa9c3ddd8f3313f99425ee2fe33a08b838492b0fee965dd04211494d'"
    new="shim=checked(BPG_TWO_CORE_SHIM)\n assert sha(shim)==BPG_TWO_CORE_SHIM['sha256']"
    assert original.count(old)==1
    revised=original.replace(old,new)
    assert ast.dump(ast.parse(revised.replace(new,old)))==ast.dump(ast.parse(original))
    activate(r['sionna_package'])
    namespace=dict(__name__='__main__',__file__=str(E/'timing_end_to_end_identity_v5.py'),BPG_TWO_CORE_SHIM=shim)
    exec(compile(revised,str(E/'timing_end_to_end_identity_v5.py')+'::BPG_TWO_CORE_V1','exec'),namespace)
