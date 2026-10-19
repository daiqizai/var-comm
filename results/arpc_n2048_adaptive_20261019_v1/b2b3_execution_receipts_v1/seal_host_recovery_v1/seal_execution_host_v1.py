"""Root-only seal relocation. Preserves frozen science; no automatic retry.

Author preparation only until root independently reviews and deploys evidence.
The existing analysis_execute.metadata_revision_guard must acknowledge the
single exact failed B2 seal using failure_guard_code() before this is invoked.
"""
from pathlib import Path
import hashlib,json,sys

H=Path(__file__).resolve().parent
R='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM'
L='/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_n2048_adaptive_20261019_v1'
O=R+'/results/arpc_n2048_adaptive_20261019_v1'
P=O+'/b2b3_execution_receipts_v1/seal_host_recovery_v1'
MAIN_HOST='aibox-rfbd2ce574fd-5df9766c45-vz6d9'
LEO_HOST='aibox-rb80bf7d8e38-dc54b8fc7-zrgl7'
B2_REG=dict(path=O+'/b2b3_analysis_v1/B2/registration.json',bytes=80237,sha256='9039b73b40c19422dfc0104af062f086707bd1a14f10a47e4a4caaadd95db1e6')
NATIVE=dict(path=R+'/experiments/arpc-n2048-adaptive-20261019/b2b3_analysis_v1/native.py',bytes=34279,sha256='d02813fe2cdb3be74129c9a7cd7afd0474307278daa75f5aab644ad38088a889')
DIAG=dict(path=P+'/diagnosis.json',bytes=10698,sha256='0f9c7da36a128b34b58bb9e972113c47477162937261e4d329fa6037dd051599')
FAILED_OBS=dict(path=P+'/main_failure_observation.json',bytes=1564,sha256='b244cb8d17e77cad289ee473e6758320ecf417e9a7983679051a5580b9cdffeb')
FAILED_FILES={
 'actual_wait.json':(282,'49aa5ca1a53ff5275cb2a69abc8aad316e802210a43f9fda882a6c35e0c7eca5'),
 'launch.json':(755,'f563eb02f66e408045c56be03d815bd60dd7f8d61684ac0fda9e7a2987f102e1'),
 'supervisor.json':(311,'07d673ac4f0a7c609033cef54820e78b9d9a1fff077cb2481502c9ccc8132a4f'),
 'stdout':(0,'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
 'stderr':(1217,'64921564cf80436a43a54243240811898c3d7a7db3f173664d14476e7a5c31cf')}


def failure_guard_code():
    """Exact historical failure acknowledgement, never a replacement wait0.

    Root can call this inside only the B2/seal_none branch of its metadata
    guard and then continue that branch. All other failure rules remain intact.
    It deliberately does not require outputs to remain absent forever: the
    separate run preflight does that before the one relocated seal attempt.
    """
    return """seal_failure_diagnosis=%r;seal_failure_observation=%r
seal_native_source=%r;seal_failed_registration=%r;seal_failed_files=%r
for d in (seal_failure_diagnosis,seal_failure_observation,seal_native_source,seal_failed_registration):verify(d)
diag=read(seal_failure_diagnosis['path']);fo=read(seal_failure_observation['path'])
assert diag['schema']=='B2_SEAL_EXECUTION_HOST_DIAGNOSIS_V1' and diag['status']=='METADATA_HOST_MISMATCH_CONFIRMED_NO_PARTIAL_OUTPUTS'
assert diag['read_only'] is True and diag['failed_phase']=='B2'
assert diag['failed_host']==%r and diag['correct_metadata_host']==%r
assert diag['analysis_registration']==seal_failed_registration
assert diag['frozen_source_changes']==diag['new_model_calls']==diag['new_score_calls']==diag['new_channel_calls']==diag['new_seal_calls']==0
assert diag['leo_all_runtime_descriptors_match'] is True and diag['actual_reference_runtime_canonical_equal'] is True
assert len(diag['no_partial_outputs'])==5 and all(x['exists'] is False for x in diag['no_partial_outputs'])
for d in diag['frozen_sources']:verify(d)
failed_stage=ROOT/'B2_seal_none';old_descriptors={}
for name,(size,sha256)in seal_failed_files.items():
 d=dict(path=str(failed_stage/name),bytes=size,sha256=sha256);verify(d);old_descriptors[name]=d
assert not(failed_stage/'failure.json').exists()
fw=read(old_descriptors['actual_wait.json']['path']);fl=read(old_descriptors['launch.json']['path']);fs=read(old_descriptors['supervisor.json']['path'])
assert type(fw['returncode'])is int and fw['returncode']==1
assert fw['metadata_error']is None and all(fw[k]is False for k in ('timed_out','terminated','killed'))
assert fw['process']==fl['process'] and fl['parent_process']==fs['process']
assert fw['process']['pid']==3940558 and fw['process']['start_ticks']==209507359
assert fl['parent_process']['pid']==3940443 and fl['parent_process']['start_ticks']==209506951
assert fs['parent_process']['pid']==3940441 and fs['parent_process']['start_ticks']==209506941
assert fw['process']['host']==fl['parent_process']['host']==fs['parent_process']['host']==%r
assert fl['argv']==[CPY,'-B',seal_native_source['path'],'seal','--phase','B2','--expected-registration',seal_failed_registration['sha256']]
assert fl['timeout_seconds']==1800
assert (failed_stage/'stderr').read_text().endswith('AssertionError: /usr/bin/python3.12\\n')
assert fo['status']=='FAILED_SEAL_CLOSED_ON_EXECUTION_HOST' and fo['actual_returncode']==1 and fo['science_calls']==0
assert fo['host']==fw['process']['host'] and fo['child']==fw['process'] and fo['parent']==fl['parent_process'] and fo['outer_parent']==fs['parent_process']
assert fo['child_gone']is True and fo['parent_gone']is True and fo['outer_parent_gone']is True
assert fo['actual_wait']==old_descriptors['actual_wait.json'] and fo['launch']==old_descriptors['launch.json'] and fo['supervisor']==old_descriptors['supervisor.json']
assert fo['observed_unix']>=fo['not_before_unix']==fl['started_unix']+fw['seconds']
if fo['host']==os.uname().nodename:
 assert gone(fo['child'])and gone(fo['parent'])and gone(fo['outer_parent'])
retained_phase_failure=dict(path=str(AL/'B2/failure.json'),bytes=999,sha256='5bb9c1abfa1c1b656d916e4cca384af224c7b732547e8246e8bcced76efaa656');verify(retained_phase_failure)
seal_failure_ack=dict(schema='EXACT_B2_SEAL_HOST_FAILURE_ACK_V1',returncode=1,scientific_calls=0,
 diagnosis=seal_failure_diagnosis,closure_observation=seal_failure_observation,registration=seal_failed_registration,
 native_source=seal_native_source,original_receipts=old_descriptors,retained_phase_failure=retained_phase_failure,
 failure_retained_not_success=True)
"""%(DIAG,FAILED_OBS,NATIVE,B2_REG,FAILED_FILES,MAIN_HOST,LEO_HOST,MAIN_HOST)


def _helper_pin():
    p=Path(__file__);b=p.read_bytes()
    return dict(path=P+'/seal_execution_host_v1.py',bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def _closed_helpers():
    return """def strict_zero(w):
 assert type(w['returncode'])is int and w['returncode']==0
 assert w['metadata_error']is None and all(w[k]is False for k in ('timed_out','terminated','killed'))
def closed_pair(wp,lp,op):
 wd,ld,od=pin(wp),pin(lp),pin(op);w,l,ob=read(wp),read(lp),read(op)
 strict_zero(w);assert w['process']==l['process']
 assert ob['status']=='CLOSED_ON_EXECUTION_HOST' and ob['actual_wait']==wd and ob['launch']==ld
 assert ob['child']==w['process'] and ob['parent']==l['parent_process']
 assert ob['host']==ob['child']['host']==ob['parent']['host']
 assert ob['child_gone']is True and ob['parent_gone']is True
 assert ob['observed_unix']>=ob['not_before_unix'] and ob['not_before_unix']==l['started_unix']+w['seconds']
 if ob['host']==os.uname().nodename:assert gone(ob['child'])and gone(ob['parent'])
 assert not Path(wp).with_name('failure.json').exists()
 return dict(actual_wait=wd,launch=ld,host_observation=od)
"""


def run(host,phase,expected):
    """Single explicit root invocation; never dispatches models or next stage."""
    from analysis_execute import remote,metadata_revision_guard
    assert host=='leo-tactile' and phase in ('B2','B3') and len(expected)==64
    tag='analysis_'+phase+'_seal_execution_host_v1'
    assert all(not(H/(tag+'.'+suffix)).exists()for suffix in ('stdout','stderr'))
    # This existing root guard must first incorporate only failure_guard_code's
    # exact B2 historical exception. No guard text is weakened in this helper.
    guard=metadata_revision_guard(phase)
    code=guard+_closed_helpers()+"""phase=%r;expected=%r;helper=%r;native_source=%r;receipt_root=Path(%r)
verify(helper);verify(native_source)
registration=pin(AO/phase/'registration.json');assert registration['sha256']==expected
reg=read(registration['path']);assert reg['phase']==phase
for d in reg['code']:verify(d)
assert native_source in reg['code']
metadata_revision_runner.reject_unknown_failure(phase)
if phase=='B2':
 assert registration==seal_failed_registration
 predecessor=seal_failure_ack
else:
 assert not(ROOT/'B3_seal_none').exists()
 predecessor=None
out=ROOT/(phase+'_seal_execution_host_v1');assert not out.exists()
assert not(receipt_root/phase).exists()
output_paths=[AL/phase/n for n in ('post_exit_native_audit.json','actual_native_objects.json','actual_logical_pixels.json','native_manifest.json')]+[AO/phase/'native_completion.json']
assert all(not p.exists()for p in output_paths)
native_closures={}
for name in ('nest','arpc'):
 control=AL/'controls'/(phase+'_'+name);native_run=AL/phase/name
 assert not(native_run/'rx/failure.json').exists() and not(control/'failure.json').exists()
 if reg['new_native_calls'][name]==0:
  z=read(control/'zero_call_completion.json');assert z['new_model_calls']==0
  native_closures[name]=dict(zero_call=pin(control/'zero_call_completion.json'));continue
 inner=closed_pair(control/'actual_wait.json',control/'launch.json',control/'closure_observation.json')
 outer=ROOT/(phase+'_native_'+name)
 outer_closed=closed_pair(outer/'actual_wait.json',outer/'launch.json',ROOT/(phase+'_observe_'+name)/'outer_closure_observation.json')
 cp=native_run/'rx/completion.json';c=read(cp);assert c['status']=='COMPLETE'
 assert len(c['outputs'])==reg['new_native_calls'][name]
 if name=='nest':assert c['ledger']['unresolved']==0
 else:assert c['blocked_reads']==[]
 native_closures[name]=dict(inner=inner,outer=outer_closed,completion=pin(cp))
assert reg['new_native_calls']['arpc']>0,'A phase without new ARPC calls requires a separately explicit root route.'
cfgp=AL/phase/'arpc/configuration.json';cfg=read(cfgp)
assert cfg['phase']==phase and cfg['name']=='arpc' and cfg['host']==os.uname().nodename==%r
assert cfg['registration']==registration and cfg['public']['path']==str(AL/phase/'arpc/public.json')
verify(cfg['public']);public=read(cfg['public']['path']);assert public['run']==str(AL/phase/'arpc')
assert cfg['runtime_metadata']['path']==str(AL/phase/'arpc/runtime_metadata.json');verify(cfg['runtime_metadata']);meta=read(cfg['runtime_metadata']['path'])
verify(reg['native_identity_compatibility']);lineage=read(reg['native_identity_compatibility']['path'])
verify(lineage['environment_policy']);policy=read(lineage['environment_policy']['path'])
assert meta['environment_policy']==lineage['environment_policy'] and meta['reference_runtime']==lineage['arpc_reference_runtime']
for d in (meta['actual_wait'],meta['launch'],meta['stdout'],meta['stderr'],meta['reference_runtime']):verify(d)
mw,ml=read(meta['actual_wait']['path']),read(meta['launch']['path']);strict_zero(mw)
assert mw['process']==ml['process'] and mw['process']['host']==ml['parent_process']['host']==cfg['host']
assert gone(mw['process'])and gone(ml['parent_process'])
assert json.loads(Path(meta['stdout']['path']).read_text())==meta['metadata'] and Path(meta['stderr']['path']).read_bytes()==b''
v=meta['metadata'];actual=read(AL/phase/'arpc/rx/runtime.json');reference=read(meta['reference_runtime']['path'])
assert v['executable']==policy['base_python']=='/usr/bin/python3'
assert v['python']==actual['python'] and v['torch']==actual['torch']
assert v['torch_build']['__version__']==actual['torch'] and v['model_imports']==[]
assert v['overlay']==policy['target'] and policy['preserve_base_torch']is True
assert {x['name']:x['version']for x in v['packages']}==dict(x.split('==')for x in policy['packages']+['flash-attn==2.8.3.post2'])
runtime_descriptors=[v[k]for k in ('executable_binary','torch_version_file','torch_init')]
for package in v['packages']:
 for k in ('METADATA','RECORD'):
  assert Path(package[k]['path']).is_relative_to(Path(policy['target']));runtime_descriptors.append(package[k])
for d in runtime_descriptors:verify(d)
def canonical_runtime(x):
 x=json.loads(json.dumps(x))
 for k in ('pid','gpu_name','gpu_uuid','started_unix'):x.pop(k,None)
 x['native_arguments'].pop('cache_dir',None);return x
assert canonical_runtime(actual)==canonical_runtime(reference)
argv=[CPY,'-B',native_source['path'],'seal','--phase',phase,'--expected-registration',expected]
lock=(L/'adaptive_execution.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out.mkdir()
put(out/'supervisor.json',dict(process=identity(os.getpid()),parent_process=identity(os.getppid()),argv=sys.argv,source=helper,invocation='ROOT_HELPER_GENERATED_STDIN'))
put(out/'prepared.json',dict(schema='B2B3_SEAL_ON_CONFIGURATION_HOST_V1',phase=phase,registration=registration,
 native_source=native_source,helper=helper,actual_host=os.uname().nodename,argv=argv,timeout_seconds=1800,
 metadata_route=metadata_route,metadata_guard=metadata_guard,original_failure=predecessor,native_closures=native_closures,
 arpc_configuration=pin(cfgp),runtime_metadata=cfg['runtime_metadata'],runtime_descriptors=runtime_descriptors,
 output_paths_absent_before_launch=[str(p)for p in output_paths],new_model_calls=0,new_score_calls=0,new_channel_calls=0))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
try:
 wait_child(out,argv,env,1800)
 c=read(AO/phase/'native_completion.json');assert c['status']=='COMPLETE' and c['registration']==registration
 put(out/'completion.json',dict(status='WAIT_ZERO_HOST_OBSERVATION_PENDING',phase=phase,
  actual_wait=pin(out/'actual_wait.json'),native_completion=pin(AO/phase/'native_completion.json'),science_calls=0))
except BaseException:
 import traceback
 put(out/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False,new_model_calls=0));raise
finally:lock.close()
print(json.dumps(dict(phase=phase,actual_wait=pin(out/'actual_wait.json'),native_completion=pin(AO/phase/'native_completion.json'),closure_observation_pending=True)))
"""%(phase,expected,_helper_pin(),NATIVE,P,LEO_HOST)
    remote(host,code,tag,timeout=2100)


def observe(host,phase,expected):
    """Separate post-exit observation; only then publish exact success route."""
    from analysis_execute import remote,metadata_revision_guard
    assert host=='leo-tactile' and phase in ('B2','B3') and len(expected)==64
    tag='analysis_'+phase+'_seal_execution_host_observe_v1'
    assert all(not(H/(tag+'.'+suffix)).exists()for suffix in ('stdout','stderr'))
    target=H/('seal_route_'+phase+'.json');assert not target.exists()
    code=metadata_revision_guard(phase)+_closed_helpers()+"""phase=%r;expected=%r;helper=%r;receipt_root=Path(%r);native_source=%r
verify(helper);verify(native_source);rp=AO/phase/'registration.json';assert pin(rp)['sha256']==expected
out=ROOT/(phase+'_seal_execution_host_v1');prepared=read(out/'prepared.json');verify(prepared['helper'])
assert prepared['helper']==helper and prepared['registration']==pin(rp) and prepared['phase']==phase
assert prepared['native_source']==native_source and prepared['actual_host']==os.uname().nodename==%r
w,l,s=read(out/'actual_wait.json'),read(out/'launch.json'),read(out/'supervisor.json');strict_zero(w)
assert w['process']==l['process'] and l['parent_process']==s['process']
assert l['argv']==prepared['argv']==[CPY,'-B',native_source['path'],'seal','--phase',phase,'--expected-registration',expected]
assert l['timeout_seconds']==prepared['timeout_seconds']==1800
assert w['process']['host']==l['parent_process']['host']==os.uname().nodename
assert gone(w['process'])and gone(l['parent_process'])and gone(s['parent_process'])
assert not(out/'failure.json').exists()
completion=read(out/'completion.json');assert completion['status']=='WAIT_ZERO_HOST_OBSERVATION_PENDING' and completion['phase']==phase
assert completion['actual_wait']==pin(out/'actual_wait.json') and completion['science_calls']==0
verify(completion['native_completion']);c=read(completion['native_completion']['path'])
assert c['status']=='COMPLETE' and c['phase']==phase and c['registration']==pin(rp)
assert c['process_closure_status']=='ALL_NATIVE_PROCESSES_GONE'
verify(c['manifest']);m=read(c['manifest']['path']);assert m['inputs']==c['inputs'] and m['outputs']==c['outputs']
for d in m['inputs']+m['outputs']:verify(d)
assert completion['native_completion']['path']==str(AO/phase/'native_completion.json')
assert not(out/'closure_observation.json').exists() and not(receipt_root/phase).exists()
now=time.time();assert now>=l['started_unix']+w['seconds']
ob=dict(schema='FINAL_PUBLICATION_PROCESS_OBSERVATION_V1',status='CLOSED_ON_EXECUTION_HOST',phase=phase,
 host=os.uname().nodename,observer_host=os.uname().nodename,execution_host=os.uname().nodename,
 child=w['process'],parent=l['parent_process'],processes=[{k:i[k]for k in ('pid','start_ticks','host')}for i in (w['process'],l['parent_process'])],
 child_gone=True,parent_gone=True,all_process_generations_gone=True,actual_wait=pin(out/'actual_wait.json'),wait=pin(out/'actual_wait.json'),
 launch=pin(out/'launch.json'),argv=l['argv'],not_before_unix=l['started_unix']+w['seconds'],observed_unix=now,
 observer_source=helper,observer_source_kind='ROOT_HELPER_GENERATED_STDIN',registration=pin(rp),native_source=native_source,new_scientific_calls=0)
put(out/'closure_observation.json',ob)
route=dict(schema='B2B3_SEAL_EXECUTION_HOST_ROUTE_V1',status='CLOSED',phase=phase,execution_host=os.uname().nodename,
 registration=pin(rp),native_source=native_source,root_helper=helper,actual_wait=pin(out/'actual_wait.json'),launch=pin(out/'launch.json'),
 host_observation=pin(out/'closure_observation.json'),supervisor=pin(out/'supervisor.json'),prepared=pin(out/'prepared.json'),
 completion=pin(out/'completion.json'),native_completion=completion['native_completion'],manifest=c['manifest'],
 stdout=pin(out/'stdout'),stderr=pin(out/'stderr'),original_failure_audit=prepared['original_failure'],scientific_calls=0,
 previous_failure_is_not_relabelled_success=True,automatic_retry=False)
destination=receipt_root/phase;destination.mkdir();put(destination/'route.json',route)
print(json.dumps(dict(route=pin(destination/'route.json'),content=route)))
"""%(phase,expected,_helper_pin(),P,NATIVE,LEO_HOST)
    remote(host,code,tag,timeout=2100)
    payload=json.loads((H/(tag+'.stdout')).read_text())
    with target.open('x',encoding='utf8',newline='\n')as f:json.dump(payload,f,indent=2);f.write('\n')


if __name__=='__main__':
    role=sys.argv[1]
    if role=='run':run(*sys.argv[2:])
    elif role=='observe':observe(*sys.argv[2:])
    else:raise ValueError(role)
