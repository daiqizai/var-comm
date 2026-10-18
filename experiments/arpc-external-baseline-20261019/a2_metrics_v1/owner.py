"""Separate H800 metric registration/actual wait after clean RX is closed."""
import argparse,copy,ctypes,fcntl,json,os,pathlib,subprocess,sys,time,traceback
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');E=R/'experiments/arpc-external-baseline-20261019'
sys.path.insert(0,str(E));import a1_owner as base
from a1_owner import L,desc,verify,put,identity,gone,gpus
RT=R.parent.parent/'var_comm_runtime_20261010';PY=RT/'envs/frozen_visual_py310/bin/python'
OLD=R/'experiments/entropy-partial-mismatch-20261010/scripts';CODE=E/'a2_metrics_v1'
O=base.O/'a2_metrics_v1';RUN=L/'a2_metrics_v1';REG=O/'registration.json';CLEAN=base.O/'a2_clean_v1'
CONFIG=RT/'qualification/h800_ep_pilot_metrics_v1_attempt1/registered/request.json'
CONFIG_SHA='b01586e1fe39daad1b47a8d102949c6003136dc58036bab47bf6dab9c2cd0347'
CLEAN_REG='668f252cebbc8a283e7d990dbda3499ebc0cd1e8152e9d2d50374e11caedb555'
def closed_inputs():
    assert desc(CLEAN/'registration.json')['sha256']==CLEAN_REG
    review=json.loads((CLEAN/'root_closure_review.json').read_text());assert review['status']=='PASS' and review['all_processes_gone'] is True
    assert review['registration_sha256']==CLEAN_REG and review['new_failure_files']==[]
    for d in review['receipts']:verify(d)
    complete=json.loads((CLEAN/'completion.json').read_text());assert complete['total_clean_states']==700 and complete['unresolved']==0
    for d in complete['outputs']:verify(d)
    return [desc(CLEAN/n) for n in ['registration.json','root_closure_review.json','completion.json','clean_states.json','closed_files.json']]
def prepare():
    assert not RUN.exists() and not O.exists();inputs=closed_inputs()
    q=L/'a2_metrics_qualification_v1';wait=json.loads((q/'actual_wait.json').read_text())
    assert type(wait['returncode']) is int and wait['returncode']==0 and not wait['timed_out'] and gone(wait['process'])
    assert json.loads((q/'completion.json').read_text())['status']=='PASS'
    assert desc(CONFIG)['sha256']==CONFIG_SHA;config=copy.deepcopy(json.loads(CONFIG.read_text())['metric_config'])
    sys.path.insert(0,str(OLD));import h800_ep_pilot_metrics_v1 as old
    assert desc(OLD/'h800_ep_pilot_metrics_v1.py')['sha256']=='6bde19df620a7dbf391f5d406daa3b4c1de2a66d051c990c26ae9a2474c1ec8d'
    assets=old.bound_assets(config)
    gpu=next(x for x in gpus() if x['index']==0);assert gpu['free_mib']>=8192 and gpu['utilization']<=25
    assert gpu['uuid']=='GPU-c38d7822-f973-c389-37c9-9ffa97610ac8'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID')
    code="""import ctypes,json
lib=ctypes.CDLL('libcuda.so.1')
assert lib.cuInit(0)==0
d=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(d),0)==0
pci=ctypes.create_string_buffer(64);assert lib.cuDeviceGetPCIBusId(pci,64,d)==0
u=(ctypes.c_ubyte*16)();assert lib.cuDeviceGetUuid(ctypes.byref(u),d)==0
print(json.dumps({'visible_index':0,'pci_bus_id':pci.value.decode().lower(),'cuda_uuid_hex':bytes(u).hex()}))
"""
    result=subprocess.run([str(PY),'-B','-c',code],env=env,capture_output=True,timeout=30);assert result.returncode==0,result.stderr
    mapping=json.loads(result.stdout);assert mapping['pci_bus_id']=='0000:63:00.0'
    resource={'gpu':gpu,'cuda_admission':mapping,'start_free_gib':8,'running_free_gib':2,'allocator_gib':6,'threads':6,'interop_threads':2}
    config['runtime_identity']={'current_host':os.uname().nodename,'device':resource,'historical_config_source':desc(CONFIG),
      'numeric_flags':old.FLAGS,'expected_torch':'2.11.0+cu128','no_cross_hardware_timing_claim':True}
    source_files=[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()]
    source_files += [desc(E/'a1_owner.py'),desc(E/'cuda_recovery_v1/runtime.py')]
    source_files += [desc(p) for p in sorted({pathlib.Path(m.__file__) for m in sys.modules.values() if getattr(m,'__file__',None) and pathlib.Path(m.__file__).is_relative_to(OLD)})]
    source_files += [desc(d['path']) for d in config['sources'].values()]
    inputs += [desc(CONFIG),desc(q/'actual_wait.json'),desc(q/'completion.json')]
    RUN.mkdir();O.mkdir();put(RUN/'mapping_preparation.json',{'actual_returncode':result.returncode,'stdout':result.stdout.decode(),'stderr':result.stderr.decode(),'model_calls':0})
    reg={'schema':'ARPC_A2_CLEAN_METRICS_REGISTRATION_V1','created_unix':time.time(),'host':os.uname().nodename,
      'source_files':source_files,'inputs':inputs,'clean_states':desc(CLEAN/'clean_states.json'),'metric_config':config,'assets':assets,
      'resource':resource,'interpreter':desc(PY),'scope':'700 fixed cal100 clean pairs plus100 failure-baseline pairs; no test population',
      'limits':{'source_DINO':100,'logical_pairs':800,'model_loads':2,'LPIPS_pairs':800,'LPIPS_backbone_forwards':1600,'DINO_reconstruction':800,
        'channel':0,'VAR':0,'training':0,'bootstrap':0},'wall_seconds':3600,'successor':None,
      'failure_policy':'Unknown failure retained; no automatic retry or successful score replay'}
    put(REG,reg);print(json.dumps(desc(REG)))
def run(expected):
    assert desc(REG)['sha256']==expected;reg=json.loads(REG.read_text());assert reg['host']==os.uname().nodename
    closed_inputs()
    for d in reg['source_files']+reg['inputs']+[reg['interpreter']]:verify(d)
    assert not (RUN/'owner.launch.json').exists() and not (RUN/'failure.json').exists()
    gpu=reg['resource']['gpu'];now=next(x for x in gpus() if x['uuid']==gpu['uuid']);assert now['free_mib']>=8192 and now['utilization']<=25
    lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock=(L/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'argv':sys.argv,'registration_sha256':expected,'gpu':now,'started_unix':time.time()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
      OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env.pop('PYTHONPATH',None);base.RUN=RUN
    try:
        base.run_stage('score',[str(PY),'-B',str(CODE/'score.py'),'--registration',str(REG),'--sha256',expected],env,reg['wall_seconds'])
        complete=json.loads((O/'completion.json').read_text());assert complete['registration_sha256']==expected and complete['unresolved']==0
        for d in complete['outputs']:verify(d)
        put(RUN/'completion.json',{'status':'PASS_CAL100_METRICS','actual_wait':desc(RUN/'score.actual_wait.json'),'result':desc(O/'completion.json'),'ended_unix':time.time()})
    except BaseException:
        put(RUN/'failure.json',{'error':traceback.format_exc(),'ended_unix':time.time(),'automatic_retry':False});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='prepare':prepare()
    else:assert a.expected_registration;run(a.expected_registration)
if __name__=='__main__':main()
