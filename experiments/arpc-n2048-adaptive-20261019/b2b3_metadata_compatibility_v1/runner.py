"""Explicit one-attempt metadata prepare, then a separate host closure and guard."""
from metadata import *
import argparse, importlib.util, traceback
import lineage as entropy_lineage

def directory_action(phase,root,out):
    assert phase in ('B2','B3') and root==AL/phase and out==AO/phase
    if phase=='B2':check_residue(root,out,True)
    else:
        assert not root.exists()and not out.exists();safe(root);safe(out)
        root.mkdir(parents=True);out.mkdir(parents=True)

def lineage_action(phase,root,value):
    assert root==AL/phase
    p=root/'native_identity_compatibility.json'
    if phase=='B2':
        verify(expected(p,*RESIDUE[p.name]));assert read(p)==value
    else:put(p,value)

def checked_registration(phase,expected_sha):
    d=pin(AO/phase/'registration.json');assert d['sha256']==expected_sha
    r=load(d);assert r['schema']=='N2048_B2B3_NATIVE_ANALYSIS_V1' and r['phase']==phase
    for p in original_pins():assert p in r['code'];verify(p)
    for p in r['code']:verify(p)
    verify(r['native_identity_compatibility']);verify(r['transport_registration']);verify(r['transport_completion'])
    return d,r

def install(native,report,report_descriptor):
    source=verify(expected(A/'native.py',*ORIGINAL['native.py'])).read_text()
    tree,proof=transformed(source)
    native.revision_directories=directory_action
    native.revision_lineage=lineage_action
    native.revision_assert_entropy=lambda c,routes:entropy_lineage.validate_candidate(c,routes,report,report_descriptor)
    exec(compile(tree,str(HERE/'runner.py')+'::verified_metadata_revision','exec'),native.__dict__)
    return proof

def scientific_import_guard(event,args):
    if event=='import'and args[0].split('.')[0]in {'torch','tensorflow','sionna','transformers','diffusers','lpips'}:
        raise AssertionError('No model/scientific import in metadata prepare: '+args[0])

def prepare(phase,transport_sha,auth,actual_control):
    authorization=validate_authorization(auth);assert phase in authorization['allowed_prepare_phases']
    require_prepare_host(load(authorization['diagnosis']),os.uname().nodename)
    actual_control=safe(actual_control)
    assert actual_control==L/'b2b3_analysis_root_v1'/(phase+'_prepare_metadata_compatibility_v1')
    assert not(actual_control/'actual_wait.json').exists()
    control=CONTROL/phase;assert not control.exists();safe(control)
    tp=O/'b2b3_revision_v1'/phase/'registration.json'
    assert pin(tp)['sha256']==transport_sha
    if phase=='B2':
        assert authorization['B2_transport_registration']==pin(tp)
        check_failure_audit();check_residue(AL/'B2',AO/'B2',True)
    else:
        assert not(AL/'B3').exists()and not(AO/'B3').exists()
        guard_closed(authorization['B2_closed_revision'],None)
    control.mkdir(parents=True)
    argv=[sys.executable,'-B',*sys.argv]
    # sys.argv already starts with runner.py; -B is retained explicitly because
    # Python removes interpreter switches from sys.argv.
    claim=dict(schema='B2B3_EXACT_METADATA_PREPARE_ATTEMPT_V1',phase=phase,
        process=identity(os.getpid()),parent_process=identity(os.getppid()),argv=argv,
        authorization=auth,original_code=authorization['original_code'],adapter_code=authorization['adapter_code'],
        preserved_failure_audit=FAILURE_AUDIT,actual_control=str(actual_control),
        transport_registration=pin(tp),started_unix=time.time(),automatic_retry=False)
    put(control/'claim.json',claim)
    try:
        sys.addaudithook(scientific_import_guard)
        report=entropy_lineage.build();put(control/'entropy_metadata_compatibility.json',report)
        rd=pin(control/'entropy_metadata_compatibility.json')
        sys.path.insert(0,str(A));spec=importlib.util.spec_from_file_location('native',A/'native.py')
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        proof=install(native,report,rd);put(control/'source_adaptation_proof.json',proof)
        native.prepare(phase,transport_sha)
        reg=pin(AO/phase/'registration.json');checked_registration(phase,reg['sha256'])
        check_residue(AL/'B2',AO/'B2',False)
        put(control/'completion.json',dict(status='METADATA_PREPARE_COMPLETE_WAIT_NOT_YET_CLOSED',phase=phase,
            process=claim['process'],parent_process=claim['parent_process'],argv=argv,
            authorization=auth,claim=pin(control/'claim.json'),original_code=authorization['original_code'],
            adapter_code=authorization['adapter_code'],registration=reg,transport_registration=pin(tp),
            entropy_metadata_compatibility=rd,source_adaptation_proof=pin(control/'source_adaptation_proof.json'),
            preserved_failure_audit=FAILURE_AUDIT,preserved_residue=[expected(AL/'B2'/n,*d)for n,d in RESIDUE.items()],
            original_registration_fields_unchanged=True,actual_control=str(actual_control),
            scientific_calls=0,finished_unix=time.time()))
        print(json.dumps(dict(registration=reg,completion=pin(control/'completion.json'),
            requires_actual_wait_and_host_closure=True)),flush=True)
    except BaseException:
        put(control/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False,
            original_failure_untouched=True,original_sources_untouched=True));raise

def close(phase,registration_sha,auth):
    authorization=validate_authorization(auth);control=CONTROL/phase
    public=PUBLIC/phase
    assert not(control/'failure.json').exists()and not public.exists()
    comp=read(control/'completion.json');assert comp['authorization']==auth and comp['phase']==phase
    reg,_=checked_registration(phase,registration_sha);assert comp['registration']==reg
    claim=load(comp['claim']);assert claim['authorization']==auth
    assert comp['original_code']==authorization['original_code']and comp['adapter_code']==authorization['adapter_code']
    ac=Path(comp['actual_control']);assert ac==L/'b2b3_analysis_root_v1'/(phase+'_prepare_metadata_compatibility_v1')
    w=read(ac/'actual_wait.json');launch=read(ac/'launch.json')
    assert launch['argv']==comp['argv']
    check_wait(w);assert same_process(launch['process'],w['process'])and gone(launch['parent_process'])
    v=dict(status='CLOSED_ON_PREPARE_HOST',host=os.uname().nodename,observer=identity(os.getpid()),
        child=launch['process'],parent=launch['parent_process'],child_gone=True,parent_gone=True,
        observed_unix=time.time(),actual_wait=pin(ac/'actual_wait.json'),launch=pin(ac/'launch.json'))
    assert_closed_observation(v,w,launch,comp,os.uname().nodename)
    check_residue(AL/'B2',AO/'B2',False);reject_unknown_failure(phase)
    safe(public);public.mkdir(parents=True)
    put(public/'closure_observation.json',v)
    receipt=dict(schema='B2B3_CLOSED_METADATA_REVISION_V1',status='CLOSED',phase=phase,authorization=auth,
        registration=reg,completion=pin(control/'completion.json'),claim=comp['claim'],
        closure_observation=pin(public/'closure_observation.json'),actual_wait=v['actual_wait'],launch=v['launch'],
        stdout=pin(ac/'stdout'),stderr=pin(ac/'stderr'),original_code=authorization['original_code'],
        adapter_code=authorization['adapter_code'],entropy_metadata_compatibility=comp['entropy_metadata_compatibility'],
        source_adaptation_proof=comp['source_adaptation_proof'],preserved_failure_audit=FAILURE_AUDIT,
        preserved_residue=comp['preserved_residue'],scientific_calls=0)
    put(public/'closed_revision.json',receipt)
    print(json.dumps(pin(public/'closed_revision.json')),flush=True)

def guard_closed(d,expected_registration):
    # d is the externally pinned descriptor, not an unbound path or payload.
    receipt=load(d);assert receipt['schema']=='B2B3_CLOSED_METADATA_REVISION_V1'and receipt['status']=='CLOSED'
    phase=receipt['phase'];assert phase in ('B2','B3')
    assert Path(d['path'])==PUBLIC/phase/'closed_revision.json'
    auth=validate_authorization(receipt['authorization'])
    assert receipt['original_code']==auth['original_code']and receipt['adapter_code']==auth['adapter_code']
    assert receipt['preserved_failure_audit']==FAILURE_AUDIT
    reg,r=checked_registration(phase,receipt['registration']['sha256']);assert reg==receipt['registration']
    if expected_registration is not None:assert reg['sha256']==expected_registration
    comp=load(receipt['completion']);claim=load(receipt['claim']);obs=load(receipt['closure_observation'])
    w=load(receipt['actual_wait']);launch=load(receipt['launch'])
    assert comp['registration']==reg and comp['authorization']==receipt['authorization']==claim['authorization']
    assert comp['claim']==receipt['claim']and comp['phase']==claim['phase']==phase
    assert comp['argv']==claim['argv']and same_process(comp['process'],claim['process'])
    assert same_process(comp['parent_process'],claim['parent_process'])
    assert comp['actual_control']==claim['actual_control']==str(L/'b2b3_analysis_root_v1'/(phase+'_prepare_metadata_compatibility_v1'))
    assert obs['actual_wait']==receipt['actual_wait']and obs['launch']==receipt['launch']
    assert_closed_observation(obs,w,launch,comp,os.uname().nodename)
    for item in receipt['preserved_residue']+[receipt['stdout'],receipt['stderr'],receipt['source_adaptation_proof']]:verify(item)
    assert receipt['preserved_residue']==[expected(AL/'B2'/n,*v)for n,v in RESIDUE.items()]
    proof=load(receipt['entropy_metadata_compatibility'])
    assert proof['status']=='THREE_KNOWN_ENTROPY_SCHEMAS_VERIFIED'
    assert proof['inline_rows']==6540 and proof['new_schema_counts']=={'T1B_NEW_RX':332,'T1C_NEW_RX':75}
    assert proof['model_routes']==r['model_routes']and proof['static']==r['static']
    for p in proof['evidence']:verify(p)
    check_residue(AL/'B2',AO/'B2',False);reject_unknown_failure(phase)
    return dict(status='EXACT_CLOSED_METADATA_REVISION_VALID',phase=phase,registration=reg,
        closed_revision=d,observer_host=os.uname().nodename,prepare_host=obs['host'],
        only_retired_failure=expected(AL/'B2/failure.json',*RESIDUE['failure.json']),
        remote_PID_absence_not_used=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','close','guard'])
    p.add_argument('--phase',choices=['B2','B3'],required=True);p.add_argument('--expected-registration',required=True)
    p.add_argument('--authorization');p.add_argument('--authorization-sha256');p.add_argument('--authorization-bytes',type=int)
    p.add_argument('--actual-control');p.add_argument('--expected-runner',required=True)
    p.add_argument('--closed-revision');p.add_argument('--closed-revision-sha256');p.add_argument('--closed-revision-bytes',type=int)
    a=p.parse_args();assert pin(Path(__file__).resolve())['sha256']==a.expected_runner
    if a.role=='guard':
        d=expected(a.closed_revision,a.closed_revision_bytes,a.closed_revision_sha256)
        assert Path(d['path'])==PUBLIC/a.phase/'closed_revision.json'
        print(json.dumps(guard_closed(d,a.expected_registration)));return
    auth=expected(a.authorization,a.authorization_bytes,a.authorization_sha256)
    if a.role=='prepare':prepare(a.phase,a.expected_registration,auth,a.actual_control)
    else:close(a.phase,a.expected_registration,auth)

if __name__=='__main__':main()
