"""400-source-only native execution; original A3/author math and public inputs."""
import argparse,fcntl,subprocess,time,traceback
from common import *
import a1_owner as parent
RUN=L/'source_v1';OUT=O/'source_v1';REG=OUT/'registration.json'
def prepare():
    assert not RUN.exists() and not OUT.exists()
    assert os.uname().nodename==read(PO/'a3_source_v1/registration.json')['host'],'Use the established native host/environment'
    qual=L/'qualification_v1';check_wait(qual/'tests.actual_wait.json',live=False)
    q=read(qual/'completion.json');assert q['status']=='PASS'
    for d in q['sources']:verify(d)
    base=read(O/'registration.json')
    for d in base['inputs']+[base['population'],base['scope']]:verify(d)
    old=read(PO/'a3_source_v1/registration.json')
    for d in old['source_files']+old['weights']:verify(d)
    remaining=read(base['population']['path'])['remaining400'];assert len(remaining)==400
    for s in remaining:verify(s['archive'])
    sources=[dict(s,rank=s['source_index'])for s in remaining]
    prior=read(PL/'a3_source_v1/public.json');cfg=dict(prior)
    candidates=[g for g in gpus()if g['free_mib']>=65536 and g['utilization']<=25]
    assert candidates,'No available GPU satisfying the unchanged native resource bounds'
    gpu=max(candidates,key=lambda g:(g['free_mib'],-g['index']))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']))
    probe="""import ctypes,json
lib=ctypes.CDLL('libcuda.so.1');assert lib.cuInit(0)==0
d=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(d),0)==0
pci=ctypes.create_string_buffer(64);assert lib.cuDeviceGetPCIBusId(pci,64,d)==0
u=(ctypes.c_ubyte*16)();assert lib.cuDeviceGetUuid(ctypes.byref(u),d)==0
print(json.dumps({'pci_bus_id':pci.value.decode().lower(),'cuda_uuid_hex':bytes(u).hex()}))
"""
    admission=json.loads(subprocess.check_output([str(CPY),'-B','-c',probe],env=env,text=True))
    bus=subprocess.check_output(['nvidia-smi','-i',str(gpu['index']),'--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip().lower()
    assert admission['pci_bus_id'].split(':')[-2:]==bus.split(':')[-2:]
    admission.update(visible_index=gpu['index'],nvml_uuid=gpu['uuid'])
    cfg.update(schema='ARPC_COMMON500_REMAINING400_SOURCE_V1',run=str(RUN),output=str(OUT),resource={**prior['resource'],'gpu':gpu},cuda_admission=admission)
    static=lambda x:{k:v for k,v in x.items()if k not in ('schema','run','output','resource','cuda_admission')}
    assert static(cfg)==static(prior)
    RUN.mkdir();OUT.mkdir();put(RUN/'public.json',cfg);put(OUT/'sources.json',{'new_sources':sources,'population':'common500_remaining400','original_indices_preserved':True})
    code=[desc(E/n)for n in ('common.py','source_control.py','source_native.py','execute.py')]
    reg={'schema':'ARPC_COMMON500_SOURCE400_EXECUTION_V1','host':os.uname().nodename,'created_unix':time.time(),'inputs':[desc(O/'registration.json'),base['population'],base['scope'],desc(PL/'a3_source_v1/public.json'),desc(PO/'a3_source_v1/registration.json'),desc(RUN/'public.json'),desc(OUT/'sources.json'),desc(qual/'completion.json')],'source_files':old['source_files']+code,'weights':old['weights'],'public':desc(RUN/'public.json'),'sources':desc(OUT/'sources.json'),'limits':{'caption':400,'tokenizer':400,'tx_probability':400,'reference_render':0,'endpoint_streams':2800,'wireless':0,'metrics':0},'stage_seconds':{'tx':7200,'audit':300},'phase_wall_seconds':7800,'native_python':sys.executable,'numeric_protocol_unchanged':True}
    put(REG,reg);print(json.dumps(desc(REG)))
def audit(expected):
    assert desc(REG)['sha256']==expected;check_wait(RUN/'tx.actual_wait.json')
    sources=read(OUT/'sources.json')['new_sources'];calls=RUN/'tx/calls';counts={}
    for p in calls.glob('*.reserved.json'):
        k=read(p)['kind'];counts[k]=counts.get(k,0)+1
        assert p.with_name(p.name.replace('.reserved.','.complete.')).is_file()
    assert not list(calls.glob('*.failed.json'))
    assert counts=={'model_load':2,'caption':400,'tokenizer':400,'tx_probability':400}
    done=read(RUN/'tx/completion.json');assert done['counts']==counts and done['blocked_reads']==[]and done['outputs']==[]
    rows=[];catalogue=[]
    for s in sources:
        i=s['source_index'];d=read(RUN/'tx'/f'{i:02d}_source.json')
        assert d['source_id']==s['source_id'] and d['source_index']==i and len(d['lengths'])==7
        streams=[]
        for e in range(1,8):
            pin=desc(RUN/'tx'/f'{i:02d}_e{e}.arc');item=d['lengths'][e-1]
            assert item['endpoint']==e and item['payload_sha256']==pin['sha256']and item['payload_bits']==8*pin['bytes']
            streams.append({'endpoint':e,'wire':pin,'payload_bits':8*pin['bytes'],'origin':'COMMON500_EXTENSION_SINGLE_TX_PASS'})
            rows.append({'source_id':s['source_id'],'source_index':i,'endpoint':e,'payload_bits':8*pin['bytes'],'wire_sha256':pin['sha256']})
        catalogue.append({'source_id':s['source_id'],'source_index':i,'source':s['archive'],'streams':streams})
    put(OUT/'stream_catalogue.json',{'schema':'ARPC_REMAINING400_STREAMS_V1','registration_sha256':expected,'sources':catalogue})
    write_csv(OUT/'source_lengths.csv',rows)
    put(OUT/'completion.json',{'status':'SOURCE400_CLOSED_NO_WIRELESS_YET','registration_sha256':expected,'sources':400,'streams':2800,'counts':counts,'unresolved':0,'new_images':0,'new_packets':0,'new_scores':0,'outputs':[desc(OUT/n)for n in ('stream_catalogue.json','source_lengths.csv')]})
def run(expected):
    assert desc(REG)['sha256']==expected;reg=read(REG);assert reg['host']==os.uname().nodename and sys.executable==reg['native_python']
    for d in reg['source_files']+reg['weights']+reg['inputs']:verify(d)
    assert not(RUN/'owner.launch.json').exists()
    cfg=read(RUN/'public.json');gpu=next(g for g in gpus()if g['uuid']==cfg['resource']['gpu']['uuid'])
    assert gpu['free_mib']>=65536 and gpu['utilization']<=25
    lock=(PL/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    glock=(PL/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'registration_sha256':expected,'started_unix':time.time(),'gpu':gpu})
    parent.RUN=RUN
    env=dict(os.environ,PYTHONPATH=str(PL/'environment_v1')+':'+str(PE),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(cfg['cuda_admission']['visible_index']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    try:
        parent.run_stage('tx',[sys.executable,'-B',str(E/'source_native.py'),'tx','--public',str(RUN/'public.json'),'--sources',reg['sources']['path']],env,reg['stage_seconds']['tx'])
        parent.run_stage('audit',[sys.executable,'-B',str(E/'source_control.py'),'audit','--expected-registration',expected],dict(env,CUDA_VISIBLE_DEVICES=''),reg['stage_seconds']['audit'])
        put(RUN/'completion.json',{'status':'SOURCE400_CLOSED','output':desc(OUT/'completion.json'),'actual_tx_wait':desc(RUN/'tx.actual_wait.json'),'actual_audit_wait':desc(RUN/'audit.actual_wait.json')})
    except BaseException:put(RUN/'failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','run','audit']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='prepare':prepare()
    elif a.role=='run':run(a.expected_registration)
    else:audit(a.expected_registration)
if __name__=='__main__':main()
