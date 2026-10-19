"""Post-exit audit of fixed RX images; preserve original audit failure.

The persisted rejected opens name only the registered public RX wrapper source.
No file access is permitted retroactively and no model call is available here.
"""
import ast,collections,fcntl,hashlib,time
from common import *
import rx_audit,rx_control as c

EXPECTED='5590c2764da21856625214f6d58641aa041656521076ae906d2b39faa51faf95'
DONE_SHA='6e1e841fad457b62450d956ee667bb63cb2d2da37f1dbec3733ddc0027aa4d7b'
WRAPPER_SHA='a5a259bbc6e23b3199bb1da7f4d05fee7c4362b68af1de580d118ab8a47a4cc6'
RUN=L/'rx_v1';OUT=O/'rx_v1';REPORT=O/'rx_finalization_v1'

def validate_denied(paths,path,pin,expected_count=100):
    assert isinstance(paths,list) and len(paths)==expected_count and expected_count>0
    assert all(type(x)is str and x==str(path) for x in paths)
    assert pin['path']==str(path) and pin['sha256']==WRAPPER_SHA
    return dict(collections.Counter(paths))

def old_wait(path,rc):
    v=read(path);assert type(v['returncode'])is int and v['returncode']==rc
    assert not any(v.get(k)for k in ('timed_out','terminated','killed','metadata_error'))
    assert gone(v['process']);return desc(path)

def audit_source(raw):
    original=rx_audit.adapted(raw)
    old="done['status']=='COMPLETE' and not done['blocked_reads']"
    new="done['status']=='COMPLETE' and approved_public_source_denials(done['blocked_reads'])"
    assert original.count(old)==1
    changed=original.replace(old,new)
    assert changed.replace(new,old)==original
    return changed

def boundary_proof(raw):
    tree=ast.parse(raw)
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='install_rx_boundary')
    audit=next(n for n in f.body if isinstance(n,ast.FunctionDef)and n.name=='audit')
    branch=next(n for n in ast.walk(audit)if isinstance(n,ast.If)and ast.unparse(n.test).startswith('not any('))
    assert len(branch.body)==2 and ast.unparse(branch.body[0])=='blocked.append(str(p))'
    assert isinstance(branch.body[1],ast.Raise)and ast.unparse(branch.body[1].exc.func)=='PermissionError'
    assert 'sys.addaudithook(audit)'in ast.unparse(f)
    return {'hook_ast_sha256':hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest(),
            'out_of_allowlist_open_rejected_before_data':True,'boundary_changed':False,
            'attempt_free_gate_passed':False,'rejected_file_role':'already registered public RX wrapper source',
            'caller_stack_recorded':False,'caller_stack_cause':'unknown; no claim of a specific introspection caller'}

def main():
    assert desc(OUT/'registration.json')['sha256']==EXPECTED
    assert desc(RUN/'rx/completion.json')['sha256']==DONE_SHA
    assert not REPORT.exists() and not(OUT/'completion.json').exists()
    reg=read(OUT/'registration.json');assert reg['host']==os.uname().nodename
    for pin in reg['source_files']+reg['inputs']+reg['weights']:verify(pin)
    wrapper=next(p for p in reg['source_files']if p['path']==str(E/'rx_native.py'))
    done=read(RUN/'rx/completion.json');denied=validate_denied(done['blocked_reads'],E/'rx_native.py',wrapper)
    assert done['status']=='COMPLETE' and done['counts']=={'model_load':1,'rx_prefix':1184,'rx_render':1184}
    waits=[old_wait(RUN/'rx.actual_wait.json',0),old_wait(RUN/'audit.actual_wait.json',1),old_wait(L/'rx_dispatch_control_v1/child.actual_wait.json',1)]
    identities=[RUN/'owner.launch.json',RUN/'rx.launch.json',RUN/'audit.launch.json',L/'rx_dispatch_control_v1/supervisor.json']
    for p in identities:assert gone(read(p)['process'])
    failures=[RUN/'failure.json',L/'rx_dispatch_control_v1/failure.json']
    assert all(p.is_file()for p in failures)
    q=L/'rx_finalization_qualification_v1';assert read(q/'completion.json')['status']=='PASS'
    check_wait(q/'tests.actual_wait.json')
    for p in read(q/'completion.json')['sources']:verify(p)
    native=next(p for p in reg['source_files']if p['path']==str(PE/'a1_native.py'));verify(native)
    proof=boundary_proof((PE/'a1_native.py').read_text())
    from rx_native import adapt
    adapted_native=adapt((PE/'a1_native.py').read_bytes())
    assert sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='install_rx_boundary' for n in ast.walk(ast.parse(adapted_native)))==1
    code=audit_source((PE/'a3_rx_v1/audit.py').read_bytes())
    evidence=[desc(p)for p in identities+failures]+waits+[desc(RUN/'rx/completion.json'),desc(OUT/'registration.json')]
    lock=(PL/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    REPORT.mkdir()
    c.success=c.check_wait
    sys.argv=[str(Path(__file__)),'--expected-registration',EXPECTED]
    exec(compile(code,str(PE/'a3_rx_v1/audit.py')+'::public_wrapper_denial_lineage','exec'),
         {'__name__':'__main__','__file__':str(PE/'a3_rx_v1/audit.py'),
          'approved_public_source_denials':lambda paths:bool(validate_denied(paths,E/'rx_native.py',wrapper))})
    final=read(OUT/'completion.json');assert final['unresolved']==0 and final['logical_frames']==3600
    for pin in final['outputs']:verify(pin)
    for pin in evidence:verify(pin)
    report={'status':'PASS_WITH_PRESERVED_ORIGINAL_AUDIT_EXIT1','reviewer':'root','scientific_calls':0,
            'original_waits':waits,'original_evidence':evidence,'persisted_denied_open_count':100,
            'persisted_denied_paths':denied,'denied_public_source':wrapper,'boundary':proof,
            'qualification':desc(q/'completion.json'),'source_files':[desc(Path(__file__)),desc(E/'rx_audit.py'),native],
            'completion':desc(OUT/'completion.json'),'outputs':final['outputs'],'unresolved':0,
            'original_audit_returncode':1,'original_owner_returncode':1,'native_rx_returncode':0,
            'reason':'All native calls exited successfully. Original post-exit no-denied-opens audit rejected 100 persisted opens of the exact registered public wrapper source. Those opens were denied before data; no private path appears. Original failure remains. The remaining original pixel/status/count audit ran unchanged, with no regeneration or truth correction.'}
    put(REPORT/'recovery_audit.json',report)
    print(json.dumps({'recovery':desc(REPORT/'recovery_audit.json'),'completion':final}))

if __name__=='__main__':main()
