"""Post-exit metadata closure for the exact retained denied-code-read list.

No receiver allowlist is changed and no scientific operation is replayed.
The original attempt-free gate did not pass; every attempted read was denied.
"""
import ast,hashlib,json,pathlib
import phy_finalize as previous
c=previous.c
OBS_SHA='a65d9c00c4e8914e42ee824f67f1f7a9c194c046c14763a510662c2aa3767ca9'


def validate_denied_code_reads(done,path):
    denied=done['blocked_reads']
    assert type(denied)is list and len(denied)==100
    assert all(type(p)is str and p==str(path)for p in denied)
    assert done['status']=='COMPLETE'
    assert done['counts']=={'model_load':1,'rx_prefix':72,'rx_render':72}
    return True


def guard_done(done):
    p=c.O/'rx_boundary_observation_v1/observation.json'
    assert c.desc(p)['sha256']==OBS_SHA
    evidence=c.read(p)
    for name in ['completion','source_only_exact_path','receiver_boundary_source','original_rx_adapter']:
        c.verify(evidence[name])
    assert evidence['blocked_count']==100
    assert evidence['attempt_free_gate_passed']is False
    assert evidence['replayed_scientific_calls']==0
    assert evidence['blocked_path_counts']=={str(c.E/'native.py'):100}
    assert c.desc(c.L/'rx/rx/completion.json')==evidence['completion']
    validate_denied_code_reads(done,c.E/'native.py')
    for pin in evidence['actual_waits']:c.verify(pin);c.success(pathlib.Path(pin['path']))
    for who in evidence['verified_gone']:assert c.gone(who)
    return evidence


def adapted_finalizer(source):
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='finalize')
    text=ast.get_source_segment(source,fn)
    old="for role in ('source','phy','rx'):\n        success(L/(role+'_control/worker.actual_wait.json'));assert read(L/(role+'_control/completion.json'))['status']=='COMPLETE'\n        assert not(L/role/'failure.json').exists()"
    new="assert read(O/'phy_finalization_v1/recovery_audit.json')['replayed_decodes']==0\n    success(L/'phy/audit.actual_wait.json')\n    for role in ('source','rx'):\n        root=resource.control_dir(original_control,role);success(root/'worker.actual_wait.json');assert read(root/'completion.json')['status']=='COMPLETE'\n        assert not(L/role/'failure.json').exists()"
    assert text.count(old)==1;text=text.replace(old,new)
    old="done=read(L/'rx/rx/completion.json');assert done['blocked_reads']==[]"
    assert text.count(old)==1;text=text.replace(old,"done=read(L/'rx/rx/completion.json');guard_done(done)")
    return text


def main():
    import resource_continuation as resource
    c.checkreg(c.desc(c.O/'registration.json')['sha256'])
    qualification=c.read(c.O/'rx_boundary_finalization_qualification.json')
    assert qualification['status']=='PASS'and qualification['science_calls']==0
    for pin in qualification['sources']:c.verify(pin)
    resource.guard(c,True,False);previous.failure()
    c.success(c.L/'phy/audit.actual_wait.json')
    for role in ('source','rx'):
        root=resource.control_dir(c,role);c.success(root/'worker.actual_wait.json')
        assert c.gone(c.read(root/'supervisor.json')['process'])
        c.success(c.L/role/'native.actual_wait.json')
    evidence=guard_done(c.read(c.L/'rx/rx/completion.json'))
    text=adapted_finalizer((c.E/'control.py').read_text())
    ns={**vars(c),'resource':resource,'original_control':c,'guard_done':guard_done}
    exec(compile(text,'original_finalizer::exact_preserved_code_denials','exec'),ns)
    ns['finalize'](c.desc(c.O/'registration.json')['sha256'])
    c.put(c.O/'delivery/receiver_boundary_review.json',{
        'status':'COMPLETE_WITH_DISCLOSED_DENIED_CODE_READ_ATTEMPTS',
        'original_attempt_free_gate_passed':False,'receiver_allowlist_changed':False,
        'exact_denied_code_path':str(c.E/'native.py'),'denied_attempts':100,
        'outside_allowlist_read_returned_data':False,
        'cause':'Not instrumented; warning/linecache inspection remains an unconfirmed hypothesis.',
        'evidence':c.desc(c.O/'rx_boundary_observation_v1/observation.json'),
        'original_native_completion':evidence['completion'],
        'preserved_phy_exit1':True,'replayed_scientific_calls':0,
        'frame_manifest':c.desc(c.O/'delivery/frame_manifest.csv')})

if __name__=='__main__':main()
