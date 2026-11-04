from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
from owned_process import run_role,proc_identity

L=Path(sys.argv[1]);S=Path(__file__).resolve().parent;U=L/'dependencies/RDVQ-a2744eedd803d9e20b55b4b0c935ec211a07d184'
O=L/'p0_chain_v2';O.mkdir(exist_ok=False)
lock=(L/'p0_gpu0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
identity=dict(**proc_identity(os.getpid()),started=time.time());(O/'owner.json').write_text(json.dumps(identity))
try:
    old=json.loads((L/'p0_tx_v1/actual_wait.json').read_text());assert old['returncode']==1 and old['child_gone']
    assert not (L/'p0_public_inputs').exists()
    bindings=json.loads((L/'p0_source_bindings_v3.json').read_text())
    for p,h in bindings.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    # Admission uses actual completed prerequisite receipts, never a stale PID alone.
    cpu=json.loads((L/'cpu_codec_headers_v2/actual_wait.json').read_text());assert cpu['returncode']==0 and cpu['child_gone']
    assets=json.loads((L/'asset_subset_v3/p0_ready.json').read_text());assert len(assets['images'])==8
    ckpt=json.loads((L/'checkpoint.json').read_text());assert ckpt['sha256']=='827aaab038626bb306fc30fac37634eece418dd02bd4617f88e65d8d19e92294'
    uuid='GPU-a9818dee-2885-eacd-d344-8b1f35b82798'
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpu0=next(x for x in gpu_rows.splitlines() if x.split(',')[0].strip()=='0').split(',')
    assert gpu0[1].strip()==uuid and int(gpu0[2])<100 and int(gpu0[3])==0
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);assert uuid not in apps
    protocol=dict(revision_reason='NVML UUID differs from CUDA runtime UUID; index 0 verified by PCI 61:00.0; old failure before model forward',stage='P0',scope='8 fixed development sources x all 28 native slice endpoints + NULL',max_gpu_device_hours=16,
        chain_gpu_wall_cap_hours=8,source_packet_version=1,source_packet_wrapper_bits=72,source_selection=assets,
        checkpoint=ckpt,upstream_commit='a2744eedd803d9e20b55b4b0c935ec211a07d184',
        numeric_route='FP32/TF32off/mathSDPA',suffix='deterministic argmax',batch_size=1,gpu_uuid=uuid,cpus=[0,1],
        no_new_phy_events=True,no_training=True,registered=time.time(),bindings=bindings)
    (O/'registration.json').write_text(json.dumps(protocol,indent=2))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',
        TORCH_EXTENSIONS_DIR=str(L/'extensions_headers_v2'),CPATH=str(L/'dependencies/python310_headers/Include')+':'+str(L/'dependencies/python310_headers'),
        PYTHONNOUSERSITE='1',PYTHONPATH=str(L/'environment/site')+':'+str(U)+':'+str(S),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    ledger=[]
    for mode in ['tx','rx','audit']:
        command=[sys.executable,'-u',str(S/'p0_worker_v3.py'),mode,'--root',str(L),'--checkpoint',ckpt['path']]
        stage_env=env.copy()
        if mode=='audit':stage_env['CUDA_VISIBLE_DEVICES']=''
        ledger.append(run_role(L,'p0_'+mode+'_v2',command,stage_env,[0,1],[] if mode=='audit' else [uuid],300 if mode=='audit' else 14400,bindings))
        (O/'resource_ledger.json').write_text(json.dumps(ledger,indent=2))
    (O/'complete.json').write_text(json.dumps(dict(passed=True,finished=time.time(),gpu_hours=sum(x['charged_gpu_device_hours'] for x in ledger),p0_throughput_pending=True),indent=2))
except BaseException:
    (O/'failure.txt').write_text(traceback.format_exc());raise
