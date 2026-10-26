"""Bounded Kodak24 ARPC stages; no implicit successor or successful-call replay."""
import argparse,csv,fcntl,hashlib,json,os,pathlib,shutil,signal,subprocess,sys,time,traceback
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
BASE=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
E=R/'experiments/kodak-current-20261026/arpc_v1';O=R/'results/kodak_current_20261026_v1/arpc_v1';L=BASE/'kodak_current_20261026_v1/arpc_v1'
PE=R/'experiments/arpc-external-baseline-20261019';PO=R/'results/arpc_external_baseline_20261019_v1';PL=BASE/'arpc_external_20261019_v1'
EO=R/'results/arpc_common500_extension_20261019_v1';EL=BASE/'arpc_common500_extension_20261019_v1'
CPY=R.parent.parent/'var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
sys.path.insert(0,str(PE))
from a1_owner import desc,verify,put,identity,gone,gpus
POLICIES={'4':'e7_q2_r2-3','10':'e7_q4_r2-3','19':'e7_q6_r5-6'}
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
def csvout(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
def success(p):
    w=read(p);assert type(w['returncode'])is int and w['returncode']==0 and not any(w.get(k) for k in ('timed_out','terminated','killed','metadata_error')),p
    return w
def checkreg(expected):
    p=O/'registration.json';assert desc(p)['sha256']==expected;r=read(p)
    for d in r['source_files']+r['inputs']:verify(d)
    return r
def gpu_admission(index):
    gpu=next(g for g in gpus()if g['index']==index);assert gpu['free_mib']>=65536 and gpu['utilization']<=25,gpu
    probe="""import ctypes,json
lib=ctypes.CDLL('libcuda.so.1');assert lib.cuInit(0)==0
d=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(d),0)==0
pci=ctypes.create_string_buffer(64);assert lib.cuDeviceGetPCIBusId(pci,64,d)==0
u=(ctypes.c_ubyte*16)();assert lib.cuDeviceGetUuid(ctypes.byref(u),d)==0
print(json.dumps({'pci_bus_id':pci.value.decode().lower(),'cuda_uuid_hex':bytes(u).hex()}))
"""
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index))
    a=json.loads(subprocess.check_output([str(CPY),'-B','-c',probe],env=env,text=True))
    bus=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip().lower()
    assert a['pci_bus_id'].split(':')[-2:]==bus.split(':')[-2:]
    a.update(visible_index=index,nvml_uuid=gpu['uuid']);return gpu,a

def prepare(config_sha,gpu_index):
    assert not(O/'registration.json').exists() and not L.exists()
    root=O.parent/'scope_registration.json';assert desc(root)['sha256']==config_sha
    population=O.parent/'population.json';pop=read(population);sources=pop['sources'];assert len(sources)==24
    assert [(s['source_index'],s['source_id'])for s in sources]==[(i,f'kodak/kodim{i+1:02d}') for i in range(24)]
    old=read(EO/'source_v1/registration.json')
    oldlink=read(PO/'a3_link_v1/registration.json')
    source_files={d['path']:d for d in old['source_files']+oldlink['source_files']}
    # Include the exact previous mathematical adapters, not historical controls.
    for n in ('source_native.py','rx_native.py','plan.py'):
        p=R/'experiments/arpc-common500-extension-20261019'/n;source_files[str(p)]=desc(p)
    for p in sorted(E.glob('*')):
        if p.is_file():source_files[str(p)]=desc(p)
    for d in list(source_files.values())+old['weights']:verify(d)
    q=read(O/'qualification.json');assert q['status']=='PASS' and q['science_calls']==0
    for p in q['files']:verify(p)
    gpu,admission=gpu_admission(gpu_index)
    cfg=read(EL/'source_v1/public.json')
    cfg.update(schema='ARPC_KODAK24_SOURCE_V1',run=str(L/'source'),output=str(O/'source'),resource={**cfg['resource'],'gpu':gpu},cuda_admission=admission)
    L.mkdir(parents=True);(L/'source').mkdir();(L/'source_inputs').mkdir();(O/'source').mkdir()
    import numpy as np
    from PIL import Image
    src=[]
    for s in sources:
        p=pathlib.Path(s['png']['path']);assert desc(p)['sha256']==s['png']['sha256']
        pixels=np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8).transpose(2,0,1).copy()
        assert pixels.shape==(3,256,256) and sha(pixels.tobytes())==s['pixels_chw_uint8_sha256']
        target=L/'source_inputs'/f"{s['source_index']:02d}.npz"
        with target.open('xb') as f:np.savez(f,pixels=pixels)
        src.append({**s,'rank':s['source_index'],'archive':desc(target)})
    put(O/'source/sources.json',{'new_sources':src,'population':'Kodak24 fixed original center crops','no_resize':True})
    put(L/'source/public.json',cfg)
    reg={'schema':'ARPC_KODAK24_BOUNDED_V1','created_unix':time.time(),'native_host':os.uname().nodename,'native_python':'/usr/bin/python3','native_python_version':sys.version,'gpu_index':gpu_index,'source_files':list(source_files.values()),'weights':old['weights'],'inputs':[desc(root),desc(population),desc(O/'qualification.json'),desc(EO/'source_v1/registration.json'),desc(PO/'a3_link_v1/registration.json'),desc(EL/'source_v1/public.json'),desc(EO/'host3781_admission_v2/admission.json'),desc(PO/'a2b_metrics_v1/frozen_policies.json')],'public':desc(L/'source/public.json'),'sources':desc(O/'source/sources.json'),'backend_identity':oldlink['backend_identity'],'limits':{'caption':24,'tokenizer':24,'tx_probability':24,'endpoint_streams':168,'reference_render':0,'logical_frames':216,'header':216,'body':216,'rx_prefix':216,'rx_render':216,'metrics':0},'noise_seeds':[2001,2002,2003],'noise_rule':'original ep.standard_noise(source_id,noise_seed) * 10**(-snr_db/20)','counter_rule':'3000000 + 96*source_index + paid_profile_id','policies':POLICIES,'generation_seed':0,'tau':0.5,'N':1024,'header_symbols':68,'body_symbols':956,'Es':2,'calibration':False,'automatic_retry':False}
    put(O/'registration.json',reg);print(json.dumps(desc(O/'registration.json')))

def stage_wait(where,name,argv,env,seconds):
    start=time.time();timed=terminated=killed=False;error=None
    with (where/(name+'.stdout')).open('xb')as out,(where/(name+'.stderr')).open('xb')as err:
        p=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True)
        try:ident=identity(p.pid)
        except BaseException:
            error=traceback.format_exc();p.terminate();p.wait();raise
        put(where/(name+'.launch.json'),{'process':ident,'parent_process':identity(os.getpid()),'argv':argv,'started_unix':start,'timeout_seconds':seconds})
        try:rc=p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed=terminated=True;os.killpg(p.pid,signal.SIGTERM)
            try:rc=p.wait(timeout=30)
            except subprocess.TimeoutExpired:killed=True;os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
        put(where/(name+'.actual_wait.json'),{'process':ident,'returncode':rc,'seconds':time.time()-start,'timed_out':timed,'terminated':terminated,'killed':killed,'metadata_error':error})
        assert type(rc)is int and rc==0 and not timed and gone(ident),name

def audit_source(reg):
    source=read(reg['sources']['path'])['new_sources'];run=L/'source';success(run/'native.actual_wait.json')
    done=read(run/'tx/completion.json');assert done['counts']=={'model_load':2,'caption':24,'tokenizer':24,'tx_probability':24} and done['outputs']==[] and done['blocked_reads']==[]
    for p in (run/'tx/calls').glob('*.reserved.json'):assert p.with_name(p.name.replace('.reserved.','.complete.')).exists()
    assert not list((run/'tx/calls').glob('*.failed.json'))
    catalogue=[];rows=[]
    for s in source:
        i=s['source_index'];d=read(run/'tx'/f'{i:02d}_source.json');assert d['source_id']==s['source_id'] and d['source_index']==i and len(d['lengths'])==7
        assert d['pixels_sha256']==s['pixels_chw_uint8_sha256']
        streams=[]
        for e in range(1,8):
            pin=desc(run/'tx'/f'{i:02d}_e{e}.arc');item=d['lengths'][e-1];assert item['endpoint']==e and item['payload_sha256']==pin['sha256'] and item['payload_bits']==pin['bytes']*8
            streams.append({'endpoint':e,'wire':pin,'payload_bits':pin['bytes']*8,'origin':'KODAK24_SINGLE_TX_PASS'})
            rows.append({'source_id':s['source_id'],'source_index':i,'endpoint':e,'payload_bits':pin['bytes']*8,'payload_sha256':pin['sha256']})
        catalogue.append({'source_id':s['source_id'],'source_index':i,'source':s['archive'],'streams':streams})
    put(O/'source/stream_catalogue.json',{'sources':catalogue});csvout(O/'source/source_lengths.csv',rows)
    put(O/'source/completion.json',{'status':'COMPLETE','counts':done['counts'],'sources':24,'streams':168,'unresolved':0,'new_wireless_frames':0,'new_reference_renders':0,'outputs':[desc(O/'source/stream_catalogue.json'),desc(O/'source/source_lengths.csv')]})

def worker(role,expected):
    reg=checkreg(expected);run=L/role;run.mkdir(exist_ok=True)
    assert not(run/'worker.launch.json').exists()
    put(run/'worker.launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'registration_sha256':expected})
    native=role in ['source','rx']
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4'if native else'2',MKL_NUM_THREADS='4'if native else'2',OPENBLAS_NUM_THREADS='4'if native else'2',CUDA_VISIBLE_DEVICES='')
    lock=(PL/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if native:
        assert os.uname().nodename==reg['native_host'] and sys.executable==reg['native_python'] and sys.version==reg['native_python_version']
        for d in reg['weights']:verify(d)
        gpu=next(g for g in gpus()if g['index']==reg['gpu_index']);assert gpu['free_mib']>=65536 and gpu['utilization']<=25,gpu
        glock=(PL/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env.update(PYTHONPATH=str(PL/'environment_v1')+':'+str(PE),CUDA_VISIBLE_DEVICES=str(reg['gpu_index']))
    else:env.pop('PYTHONPATH',None)
    try:
        if role=='source':
            verify(reg['public']);verify(reg['sources'])
            stage_wait(run,'native',['/usr/bin/python3','-B',str(E/'native.py'),'tx','--public',reg['public']['path'],'--sources',reg['sources']['path']],env,3600)
            audit_source(reg)
        elif role=='phy':
            success(L/'source/native.actual_wait.json');assert read(O/'source/completion.json')['unresolved']==0
            out=O/'phy';out.mkdir()
            nomination={'decisions':{k:{'proposed_candidates':[v]}for k,v in POLICIES.items()}}
            put(out/'frozen_nomination.json',nomination)
            phyreg={'source_files':reg['source_files'],'streams':desc(O/'source/stream_catalogue.json'),'nomination':desc(out/'frozen_nomination.json'),'backend_identity':reg['backend_identity']}
            put(out/'registration.json',phyreg);physha=desc(out/'registration.json')['sha256']
            for s in ('tx','rx','audit'):
                argv=[str(CPY),'-B',str(E/'phy.py'),s]+([]if s=='rx'else['--expected-registration',physha])
                stage_wait(run,s,argv,env,1200)
        elif role=='rx':
            success(L/'phy/audit.actual_wait.json');assert read(O/'phy/completion.json')['unresolved']==0
            source_jobs=read(L/'phy/rx/actual_source_jobs.json');assert len(source_jobs)<=216
            wires=run/'wire_inputs';wires.mkdir();jobs=[]
            for j in source_jobs:
                verify(j['input']);p=wires/(j['key']+'.arc');p.write_bytes(pathlib.Path(j['path']).read_bytes())
                jobs.append({**j,'path':str(p),'input':desc(p)})
            put(wires/'jobs.json',jobs)
            cfg=read(reg['public']['path']);cfg.update(schema='ARPC_KODAK24_RX_V1',run=str(run),output=str(O/'rx'),new_job_count=len(jobs))
            put(run/'public.json',cfg)
            stage_wait(run,'native',['/usr/bin/python3','-B',str(E/'native.py'),'rx','--public',str(run/'public.json')],env,3600)
        else:raise ValueError(role)
        put(run/'completion.json',{'status':'COMPLETE','role':role,'registration_sha256':expected,'ended_unix':time.time()})
    except BaseException:put(run/'failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise

def launch(role,expected):
    checkreg(expected);control=L/(role+'_control');assert not control.exists();control.mkdir()
    put(control/'launcher.json',{'process':identity(os.getpid()),'argv':sys.argv,'registration_sha256':expected,'time':time.time()})
    with (control/'supervisor.stdout').open('xb')as out,(control/'supervisor.stderr').open('xb')as err:
        p=subprocess.Popen([sys.executable,'-B',str(E/'control.py'),'supervisor','--stage',role,'--expected-registration',expected],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    put(control/'dispatch.json',{'process':identity(p.pid),'parent_process':identity(os.getpid()),'argv':[sys.executable,'-B',str(E/'control.py'),'supervisor','--stage',role,'--expected-registration',expected]})
    print(json.dumps(desc(control/'dispatch.json')))
def supervisor(role,expected):
    control=L/(role+'_control');put(control/'supervisor.json',{'process':identity(os.getpid()),'argv':sys.argv})
    try:
        stage_wait(control,'worker',[sys.executable,'-B',str(E/'control.py'),'worker','--stage',role,'--expected-registration',expected],dict(os.environ),4200)
        put(control/'completion.json',{'status':'COMPLETE','stage':role,'actual_wait':desc(control/'worker.actual_wait.json')})
    except BaseException:put(control/'failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise

def finalize(expected):
    reg=checkreg(expected)
    for role in ('source','phy','rx'):
        success(L/(role+'_control/worker.actual_wait.json'));assert read(L/(role+'_control/completion.json'))['status']=='COMPLETE'
        assert not(L/role/'failure.json').exists()
    done=read(L/'rx/rx/completion.json');assert done['blocked_reads']==[]
    outputs={r['key']:r for r in done['outputs']}
    jobs=read(L/'rx/wire_inputs/jobs.json');assert set(outputs)=={j['key']for j in jobs}
    import numpy as np
    target=O/'delivery';target.mkdir();native=L/'delivery';native.mkdir()
    gray=native/'gray.npz'
    with gray.open('xb')as f:np.savez(f,float_output=np.full((3,256,256),.5,dtype=np.float32))
    rows=[]
    for r in read(O/'phy/logical_frames.json'):
        item=outputs.get(r['received_source_state']);status=r['actual_status']
        if item and item.get('status')!='SOURCE_PARSE_FAIL':p=pathlib.Path(item['path'])
        else:
            if item:status='SOURCE_PARSE_FAIL'
            p=gray
        with np.load(p,allow_pickle=False)as z:a=z['float_output'].copy()
        assert a.shape==(3,256,256) and a.dtype==np.float32 and np.isfinite(a).all()
        rawsha=sha(a.tobytes())
        if p!=gray:assert rawsha==item['float_sha256']
        rows.append({'method':'ARPC_ADAPTED','source_index':r['source_index'],'source_id':r['source_id'],'snr_db':r['snr_db'],'noise_seed':r['noise_seed'],'rx_status':status,'image_path':str(p),'image_key':'float_output','image_file_sha256':desc(p)['sha256'],'image_array_sha256':rawsha,'array_hash_convention':'sha256(array.tobytes(C)); float32 CHW [3,256,256]','existing_psnr_db':'','existing_lpips_alex':'','target_endpoint':r['target_endpoint'],'selected_endpoint':r['endpoint'],'actual_rx_endpoint':r['actual_rx_endpoint'],'source_state':r['received_source_state'],'accepted_wrong_information':r['accepted_wrong_information']})
    assert len(rows)==216 and len({(r['source_index'],r['snr_db'],r['noise_seed'])for r in rows})==216
    csvout(target/'frame_manifest.csv',rows)
    # Native prefix/model/render call reservations must all be closed.
    pending=[]
    for p in (L/'rx/rx/calls').glob('*.reserved.json'):
        if not p.with_name(p.name.replace('.reserved.','.complete.')).exists():pending.append(str(p))
    assert not pending and not list((L/'rx/rx/calls').glob('*.failed.json'))
    put(target/'completion.json',{'status':'ARPC_KODAK24_SOURCE_LINK_RX_COMPLETE','registration_sha256':expected,'logical_frames':216,'per_snr':{str(s):sum(r['snr_db']==s for r in rows)for s in [4,10,19]},'source_counts':read(O/'source/completion.json')['counts'],'link_counts':read(O/'phy/completion.json'),'rx_counts':done['counts'],'unique_rx_payloads':len(jobs),'unique_native_renders':done['counts'].get('rx_render',0),'gray_logical_frames':sum(r['image_path']==str(gray)for r in rows),'unresolved':0,'new_metrics':0,'outputs':[desc(target/'frame_manifest.csv')],'ended_unix':time.time()})
    print(json.dumps(desc(target/'completion.json')))
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','launch','supervisor','worker','finalize']);p.add_argument('--stage',choices=['source','phy','rx']);p.add_argument('--expected-registration');p.add_argument('--config-sha');p.add_argument('--gpu',type=int,default=1);a=p.parse_args()
    if a.role=='prepare':prepare(a.config_sha,a.gpu)
    elif a.role=='launch':launch(a.stage,a.expected_registration)
    elif a.role=='supervisor':supervisor(a.stage,a.expected_registration)
    elif a.role=='worker':worker(a.stage,a.expected_registration)
    else:finalize(a.expected_registration)
if __name__=='__main__':main()
