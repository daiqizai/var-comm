"""Separate one-source m4 RX diagnostic; never repairs the failed v5 verdict.

prepare binds existing witnesses only. Explicit run owns one bounded GPU child.
The decoder receives bits and m=4; expected tokens are read only after it returns.
"""
from __future__ import annotations
import argparse
import ast
import csv
import io
import math
import shutil
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import leo_whole_gate_v1 as g

SCHEMA = 'LEO_SINGLE_RX_DIAGNOSTIC_V2'
PASS = 'PASS_SELECTED_NEW_HOST_GPU_SOURCE0_M4_RX_ONLY'
PARENT_SHA = '454c51046aba4f784549a7c8de1b993e6ff86db21538b9a189366ddb22e45150'
CAPS = dict(model_load=1, encoder=0, source_tx=0, source_rx=1,
            var_render=0, prior_scale=4, decoder_forward=0)
WITNESS_SHA = {
    'worker_failure.json':'1bcf9ef57e6ef616fe3477ef3a219d703e8fdde0402eb06c34bf2a6418b05cc7',
    'actual_child_wait.json':'0923eff6767843d1cc6bf76f9f3a0661baef44acea81e0f3b3ad5ab1a7c8696a',
    'cdf_trace_at_failure.json':'f36b4507451be9022fdf90899594820a9d5fa3996f6f3dab248ac4cfdfe5be6c',
    'checks/s0_encoder.json':'fbd09a62d199d43006be96969cd1d2ec52d1387b6cc3cbf8c2b79732cd301dac',
    'checks/s0_tx_m4.json':'2fe7102771ca58448522018e666e710f40504f7612eb25ef464792dbfba34a12',
    'checks/s0_tx_m4.npz':'ef72681edf1c82995d37118a3f57f9f7626eeec7563b60ad20e178c5348006b5',
}
TOOLS = ('leo_whole_gate_v1.py','leo_whole_math_v1.py','leo_shared_preflight.py','leo_runtime_verify_v1.py')
DEVICE_RECEIPT_SHA = '4d5f942482ba1560e5e6cecc1d43b6d30e031717f69cb4fc055189144dd547c0'
MATH_SHA = 'db4349f936b2cd0c267932cc286e2661e1f7d56dd10d1c4e4a5d7d90ec279252'
DEVICES = {
    6: ('GPU-686eb88b-dc4f-0a1d-6a5b-5e152f8783aa','GPU-d464ff16-56e0-39d3-a3f4-c9f8e2bca66a','0000:a3:00.0'),
    7: ('GPU-997386e0-610a-e6ec-db3f-9c22722121f1','GPU-15779e90-9005-00b7-e1ac-bd4df8948d46','0000:a4:00.0'),
}

class ResourceBusy(RuntimeError): pass

class DeviceAdapter:
    """Explicit new execution device; old gate module and parent stay unchanged."""
    def __init__(self,pin,index):
        g.require(type(index) is int and index in DEVICES,'Only explicitly registered GPU6 or GPU7')
        g.require(set(pin)=={'path','sha256'} and pin['sha256']==DEVICE_RECEIPT_SHA,'Exact zero-model identity receipt required')
        receipt=g.checked_json(g.inside(pin['path']),pin['sha256'])
        g.require(receipt['status']=='READ_ONLY_ENUMERATION_NO_MODEL' and receipt['model_calls']==receipt['explicit_tensor_calls']==0,'Identity observation scope changed')
        self.index=index;self.UUID,self.CUDA_UUID,self.PCI_BUS_ID=DEVICES[index]
        drivers=[r for r in receipt['CUDA_driver_devices'] if r['driver_ordinal_unmasked']==index]
        rows=[r for r in csv.reader(io.StringIO(receipt['NVML_csv'])) if int(r[0].strip())==index]
        g.require(len(drivers)==len(rows)==1,'Ambiguous selected device observation')
        g.require(g.normalize_gpu_uuid(drivers[0]['CUDA_UUID'])==self.CUDA_UUID and
            g.normalize_pci_bus_id(drivers[0]['PCI'])==self.PCI_BUS_ID and rows[0][1].strip()==self.UUID and
            g.normalize_pci_bus_id(rows[0][2].strip())==self.PCI_BUS_ID,'Observed dual device identities disagree')
    def __getattr__(self,name):return getattr(g,name)
    def cuda_environment_binding(self):return dict(CUDA_VISIBLE_DEVICES=str(self.index),CUDA_DEVICE_ORDER='PCI_BUS_ID')
    def device_identity_binding(self):
        return dict(nvml_gpu_index=self.index,nvml_uuid=self.UUID,cuda_uuid=self.CUDA_UUID,pci_bus_id=self.PCI_BUS_ID,
            NVML_and_CUDA_UUID_namespaces_differ=True,**self.cuda_environment_binding())
    def check_cuda_environment(self,environment=None):
        environment=os.environ if environment is None else environment
        g.require(all(environment.get(k)==v for k,v in self.cuda_environment_binding().items()),'Explicit selected ordinal/order required')
    def validate_cuda_device_identity(self,actual_uuid,actual_pci):
        g.require(g.normalize_gpu_uuid(actual_uuid)==self.CUDA_UUID and g.normalize_pci_bus_id(actual_pci)==self.PCI_BUS_ID,'Actual CUDA device does not match selected UUID/PCI')
        return dict(actual_cuda_uuid=self.CUDA_UUID,actual_pci_bus_id=self.PCI_BUS_ID,
            configured_nvml_index=self.index,configured_nvml_uuid=self.UUID,same_physical_PCI=True)
    def query_nvml_device_identity(self):
        argv=['/usr/bin/nvidia-smi','--id='+self.UUID,'--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits']
        value=subprocess.run(argv,check=True,capture_output=True,text=True,timeout=10)
        rows=list(csv.reader(io.StringIO(value.stdout.strip())))
        g.require(len(rows)==1 and len(rows[0])==3,'Exactly one selected NVML record required')
        index,uuid,pci=[x.strip() for x in rows[0]]
        g.require(int(index)==self.index and uuid==self.UUID and g.normalize_pci_bus_id(pci)==self.PCI_BUS_ID,'Fresh NVML device identity changed')
        return dict(argv=argv,actual_nvml_index=self.index,actual_nvml_uuid=uuid,actual_pci_bus_id=self.PCI_BUS_ID)
    def resource_snapshot(self,shared,prelaunch=False):
        started=time.time()
        def counters():
            v=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
            return sum(v),v[3],v[4]
        before=counters()
        argv=['/usr/bin/nvidia-smi','--id='+self.UUID,'--query-gpu=uuid,index,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits']
        result=subprocess.run(argv,check=True,capture_output=True,text=True,timeout=10)
        time.sleep(max(0,.25-(time.time()-started)));after=counters();ticks=after[0]-before[0]
        g.require(ticks>0,'No advancing CPU counters')
        cpu=shared.cpu_snapshot();cpu.update(host_busy_percent=100*(ticks-(after[1]-before[1])-(after[2]-before[2]))/ticks,
            host_iowait_percent=100*(after[2]-before[2])/ticks)
        gpu=shared.parse_gpu(result.stdout,self.UUID)
        g.require(gpu['index']==self.index,'Wrong selected GPU resource record')
        snapshot=dict(started_unix=started,completed_unix=time.time(),gpu=gpu,cpu=cpu,
            personal_filesystem_free_bytes=shutil.disk_usage(g.BASE).free,personal_quota_verified=False,gpu_query_argv=argv)
        g.require(0<=time.time()-started<=15,'Resource snapshot stale')
        checks=[(gpu['free_MiB']>=20*1024,'Less than20GiB GPU margin'),
            (not prelaunch or gpu['utilization_percent']<=50,'Selected GPU busy'),
            (cpu['effective_cpu_cores']>=2 and len(cpu['allowed_cpus'])>=2,'Insufficient CPU capacity'),
            (math.isfinite(cpu['host_busy_percent']) and 0<=cpu['host_busy_percent']<=95,'CPU pressure above95%'),
            (cpu['available_memory_bytes']>=2*(1<<30),'Less than2GiB host margin'),
            (snapshot['personal_filesystem_free_bytes']>=10*(1<<30),'Less than10GiB filesystem margin')]
        for ok,message in checks:
            if not ok:
                error=ResourceBusy(message);error.resource_snapshot=snapshot;raise error
        return snapshot,dict(cpu_affinity=cpu['allowed_cpus'][:2],threads=2,workers=1,shared_device=True)

def device_from_request(r): return DeviceAdapter(r['device_receipt'],r['target_index'])

def adapt_backend(module,device):
    """One import becomes explicit dependency injection; all mathematics identical."""
    path=Path(module.__file__);g.require(g.sha(path)==MATH_SHA,'Bound original mathematical implementation changed')
    tree=ast.parse(path.read_text());original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Backend')
    changed=copy.deepcopy(original);init=next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    matches=[(i,n) for i,n in enumerate(init.body) if isinstance(n,ast.Import) and
             len(n.names)==1 and n.names[0].name=='leo_whole_gate_v1' and n.names[0].asname=='g']
    g.require(len(matches)==1,'Exactly one original device-gate import required')
    index,node=matches[0];init.body[index]=ast.copy_location(ast.Assign(targets=[ast.Name(id='g',ctx=ast.Store())],value=ast.Name(id='_explicit_device_adapter',ctx=ast.Load())),node)
    restored=copy.deepcopy(changed)
    next(n for n in restored.body if isinstance(n,ast.FunctionDef) and n.name=='__init__').body[index]=copy.deepcopy(node)
    before=ast.dump(original,include_attributes=False);after=ast.dump(restored,include_attributes=False)
    g.require(before==after,'Device adaptation changed original mathematics')
    namespace=dict(vars(module));namespace['_explicit_device_adapter']=device
    exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['Backend'],dict(schema='EXPLICIT_DEVICE_DEPENDENCY_ADAPTER_V1',original_math_sha256=MATH_SHA,
        original_AST_sha256=g.hashlib.sha256(before.encode()).hexdigest(),inverse_restored_AST_sha256=g.hashlib.sha256(after.encode()).hexdigest(),
        only_change='Backend.__init__ import g replaced by explicitly registered device adapter',
        old_gate_globals_changed=False,parent_request_changed=False,device_binding=device.device_identity_binding())

def wait_prelaunch(device,shared,out,spec,owner_started,wait_seconds=120):
    g.require(wait_seconds==120,'Only fixed120s prelaunch observation window')
    end=min(time.monotonic()+wait_seconds,owner_started+spec['max_seconds'])
    while True:
        g.require(not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),'Stopped before child launch')
        g.require(time.time()<spec['deadline_unix'] and time.monotonic()<end,'Prelaunch resource window exhausted; no child started')
        try:
            result=device.resource_snapshot(shared,prelaunch=True)
            g.require(time.time()<spec['deadline_unix'] and time.monotonic()<end,'Prelaunch snapshot crossed resource deadline; no child started')
            return result
        except ResourceBusy as error:
            with (out/'prelaunch_resource_wait.jsonl').open('a') as f:f.write(json.dumps(dict(time_unix=time.time(),reason=str(error),snapshot=error.resource_snapshot,child_started=False))+'\n')
            remaining=min(end-time.monotonic(),spec['deadline_unix']-time.time())
            g.require(remaining>0,'Prelaunch resource window exhausted; no child started')
            time.sleep(min(5,remaining))

def descriptor(path): return dict(path=str(path),sha256=g.sha(path))

def inherited_spec(parent, deadline, seconds):
    g.require(type(deadline) in (int,float) and 0<seconds<=900 and time.time()<deadline,'Fresh at-most15-minute diagnostic deadline required')
    spec=copy.deepcopy(parent['spec'])
    # The original full-gate scope remains an input validation contract only.
    # Its counters are never used by this independent execution ledger.
    spec.update(deadline_unix=deadline,max_seconds=seconds)
    return spec

def cdf_witness(trace):
    rows=trace['new_host_TX_RX_traces']
    g.require(trace['source_index']==0 and trace['current_endpoint']==['TX',9] and len(rows)==9,'Wrong completed v5 TX trace')
    g.require({r['scale'] for r in rows}==set(range(9)),'Duplicate/missing TX scales')
    for row in rows:
        g.require(row['source']==0 and row['role']=='TX' and row['m']==9 and g.SHA_RE.fullmatch(row['cdf_sha256']), 'Invalid TX hash witness')
    return {r['scale']:r['cdf_sha256'] for r in rows if r['scale']<4}

def witness_metadata(parent_path):
    root=Path(parent_path).parent/'run';pins={};data={}
    for name,digest in WITNESS_SHA.items():
        path=g.inside(root/name)
        g.require(path.is_file() and g.sha(path)==digest,'v5 witness bytes changed: '+name)
        pins[name]=dict(path=str(path),sha256=digest)
        if name.endswith('.json'):data[name]=g.checked_json(path,digest)
    failed=data['worker_failure.json'];counts=failed['counts']
    expected=dict(model_load=1,encoder=1,source_tx=1,source_rx=0,var_render=0,prior_scale=9,decoder_forward=0)
    g.require(failed['request_sha256']==PARENT_SHA and failed['mismatch_layer']=='bitstream' and
              failed['status']=='STOPPED_NO_RETRY' and counts['completed']==counts['reserved']==expected and counts['unresolved']==0,'v5 failure scope changed')
    wait=data['actual_child_wait.json']
    g.require(wait['actual_child_waited'] and wait['child_exit_code']==1 and wait['automatic_retry'] is False,'v5 actual failed wait required')
    enc=data['checks/s0_encoder.json'];bits=data['checks/s0_tx_m4.json']
    g.require(enc['exact'] is True and enc['actual_shape']==enc['expected_shape']==[680] and enc['actual_sha256']==enc['expected_sha256'],'v5 encoder witness not exact')
    g.require(bits['name']=='s0_tx_m4' and bits['exact'] is False and bits['actual_dtype']=='uint8' and bits['actual_shape']==[311],'Wrong actual m4 stream witness')
    return dict(pins=pins,expected_new_TX_CDF_hashes=cdf_witness(data['cdf_trace_at_failure.json']),actual_bit_array_sha256=bits['actual_sha256'])

def check_tools(bindings):
    g.require(set(bindings)==set(TOOLS)|{Path(__file__).name},'Incomplete diagnostic implementation closure')
    for name,digest in bindings.items():g.require(g.sha(Path(__file__).with_name(name))==digest,'Diagnostic tool changed: '+name)

def registration(path,digest):
    r=g.checked_json(g.inside(path),digest)
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS,'Wrong single RX registration')
    g.require(r['prelaunch_wait_seconds']==120,'Prelaunch observation window changed')
    g.require(r['source_index']==0 and r['source_id']==g.IDS[0] and r['m']==4 and
              r['automatic_successor'] is False and r['production_admission'] is False and r['timing_measurement'] is False,
              'Diagnostic scope or continuation permission changed')
    g.require(r['parent_request']['sha256']==PARENT_SHA and r['device_identity_binding']==device_from_request(r).device_identity_binding(),'Wrong v5 parent or device binding')
    check_tools(r['tool_bindings'])
    parent=g.prepared(r['parent_request']['path'],PARENT_SHA)
    expected=inherited_spec(parent,r['spec']['deadline_unix'],r['spec']['max_seconds'])
    g.require(r['spec']==expected,'Only the independent deadline/window may differ from v5 input spec')
    observed=witness_metadata(r['parent_request']['path'])
    # JSON stringifies integer keys; comparison stays in that canonical domain.
    g.require(json.loads(json.dumps(observed,sort_keys=True))==r['witness'],'Saved v5 witness binding changed')
    return r

def prepare(a):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh diagnostic registration required')
    parent_path=g.inside(a.parent_request);g.require(a.parent_request_sha256==PARENT_SHA,'This diagnostic only admits the fixed v5 attempt')
    parent=g.prepared(parent_path,PARENT_SHA)
    device_pin=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256)
    device=DeviceAdapter(device_pin,a.target_index)
    spec=inherited_spec(parent,a.deadline_unix,a.max_seconds);witness=witness_metadata(parent_path)
    # Metadata/source-file validation only; no model, source image or token array load.
    _,_,_,_,extra=g.validate_spec(spec,verify_environment=False)
    out.mkdir(parents=True)
    request=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',parent_request=dict(path=str(parent_path),sha256=PARENT_SHA),
        spec=spec,caps=CAPS,witness=witness,source_index=0,source_id=g.IDS[0],m=4,
        device_identity_binding=device.device_identity_binding(),device_receipt=device_pin,target_index=a.target_index,prelaunch_wait_seconds=120,created_unix=time.time(),
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        purpose='Saved new-platform GPU5 TX decoded independently on explicitly selected GPU6 or GPU7; original v5 cross-host FAIL unchanged',
        budget_scope='New independent diagnostic ledger; no change to v5 or EP32/48 budgets',
        preparation_model_calls=0,preparation_source_arrays_read=0,automatic_successor=False,
        old_CDF='UNOBSERVABLE',production_admission=False,timing_measurement=False)
    g.write(out/'request.json',request);return descriptor(out/'request.json')

def actual_bits(witness):
    import numpy as np
    pin=witness['pins']['checks/s0_tx_m4.npz'];p=g.inside(pin['path'])
    g.require(g.sha(p)==pin['sha256'],'Actual v5 bit witness changed')
    with np.load(p,allow_pickle=False) as z: bits=z['actual'].copy() # Never load z['expected'].
    validate_bits(bits,witness['actual_bit_array_sha256']);return bits

def validate_bits(bits,digest):
    import numpy as np
    g.require(bits.dtype==np.uint8 and bits.shape==(311,) and np.isin(bits,(0,1)).all() and g.image_sha(bits)==digest,'Wrong saved actual311-bit vector')

def expected_tokens_after_decode(record,resolver):
    import numpy as np
    asset=record['asset']
    with np.load(resolver.path(asset['archive']),allow_pickle=False) as z:tokens=z['tokens'].copy()
    g.require(tokens.dtype==np.int64 and tokens.shape==(680,) and ((tokens>=0)&(tokens<4096)).all(),'Frozen source token array malformed')
    g.require(g.hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==asset['tokens_sha256'],'Frozen source token pin differs')
    return tokens[:g.OFFSETS[4]].copy()

def exercise(bits,tx_hashes,backend,ledger,expected_reader,out):
    """RX receives no expected token array or TX probability table."""
    g.require(set(tx_hashes)==set(range(4)) and all(g.SHA_RE.fullmatch(x) for x in tx_hashes.values()),'Four external CDF hash assertions required')
    backend.tx_cdfs=dict(tx_hashes) # Only digests; TracedProvider computes probabilities itself.
    rx=ledger.call('source_rx',lambda:backend.source_rx(bits.copy(),4),source=0,m=4)
    g.require(rx['canonical'] is True and rx['zero_extension_reads']==30,'Actual independent RX canonical parse failed')
    rows=backend.traces
    g.require(len(rows)==4 and [r['scale'] for r in rows]==list(range(4)) and
              all(r['source']==0 and r['role']=='RX' and r['m']==4 and r['cdf_sha256']==tx_hashes[r['scale']] for r in rows),'Independent RX CDF trace differs')
    # Expected tokens enter this function only after the actual decoder returned.
    out=Path(out);out.mkdir()
    check=g.exact(rx['received_tokens'],expected_reader(),'s0_rx_m4',out)
    g.write(out/'cdf_hash_assertions.json',dict(expected_new_TX_hashes=tx_hashes,actual_RX=rows,
        probabilities_shared_with_RX=False,comparison_tokens_supplied_to_RX=False,old_CDF='UNOBSERVABLE'))
    return check

def worker_identity(out,digest,owner_pid):
    g.require(os.getppid()==owner_pid and owner_pid>0,'No live admitted owner')
    intent=json.loads((out/'intent.json').read_text());g.require(intent['owner_pid']==owner_pid and intent['request_sha256']==digest,'Wrong owner intent')
    until=time.monotonic()+2
    while not (out/'child_started.json').exists() and time.monotonic()<until:time.sleep(.01)
    launch=json.loads((out/'child_started.json').read_text())
    expected=['-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(out.parent/'request.json'),'--request-sha256',digest,'--owner-pid',str(owner_pid)]
    g.require(launch['pid']==os.getpid() and launch['owner_pid']==owner_pid and launch['request_sha256']==digest and launch['argv'][1:]==expected,'Actual child identity differs')
    return launch

def run(a):
    g.require(sys.platform.startswith('linux'),'Actual owner requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256);spec=r['spec'];out=path.parent/'run';shared=g.helper();device=device_from_request(r);child=None;started=time.monotonic()
    with shared.owner_lock(g.inside(g.BASE/('controls/leo_whole_gate_v1/GPU'+str(device.index)+'.owner.lock'))):
        g.require(not out.exists(),'Diagnostic already claimed; no replay');out.mkdir()
        g.write(out/'intent.json',dict(owner_pid=os.getpid(),request_sha256=a.request_sha256,started_unix=time.time(),caps=CAPS))
        try:
            _,_,_,_,extra=g.validate_spec(spec)
            snapshot,ad=wait_prelaunch(device,shared,out,spec,started,r['prelaunch_wait_seconds']);g.write(out/'prelaunch_resources.json',snapshot)
            g.write(out/'prelaunch_device_identity.json',device.query_nvml_device_identity())
            _,controlled=shared.controlled_environment(g.BASE,out,6);env=dict(controlled);(out/'home').mkdir()
            libraries=extra['environment']['LD_LIBRARY_PATH']
            for d in libraries.split(':'):g.require(g.inside(d).is_dir(),'Unadmitted native library directory')
            env.update(PATH=str(Path(spec['python']).parent),LD_LIBRARY_PATH=libraries,HOME=str(out/'home'),LANG='C.UTF-8',LC_ALL='C.UTF-8',
                VIRTUAL_ENV=str(Path(spec['python']).parent.parent),CUBLAS_WORKSPACE_CONFIG=':4096:8',**device.cuda_environment_binding())
            g.write(out/'controlled_environment.json',env)
            argv=[spec['python'],'-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(path),'--request-sha256',a.request_sha256,'--owner-pid',str(os.getpid())]
            seconds=min(spec['max_seconds']-(time.monotonic()-started),spec['deadline_unix']-time.time());g.require(seconds>0,'Deadline expired before launch')
            with (out/'child.log').open('x') as log:
                child=subprocess.Popen(argv,env=env,cwd=spec['project_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    preexec_fn=shared.child_limits(ad['cpu_affinity'],seconds,2))
                g.write(out/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),request_sha256=a.request_sha256,argv=argv,cpu_affinity=ad['cpu_affinity'],started_unix=time.time()))
                waited=shared.wait_owned(child,seconds)
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'Diagnostic child failed; preserve ledger without retry')
            done=g.checked_json(out/'worker_completion.json',g.sha(out/'worker_completion.json'))
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and done['counts']['completed']==CAPS and done['counts']['unresolved']==0,'Diagnostic completion/counters differ')
            registration(path,a.request_sha256);g.validate_spec(spec)
            g.require(time.monotonic()-started<spec['max_seconds'] and time.time()<spec['deadline_unix'],'Owner final validation exceeded registered deadline')
            result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,worker_completion=descriptor(out/'worker_completion.json'),
                actual_children_waited=True,worker_exit_codes=[child.returncode],actual_wait=waited,caps=CAPS,
                v5_cross_host_verdict='FAIL_UNCHANGED',production_admission=False,timing_measurement=False,automatic_successor=False)
            g.write(out/'completion.json',result);return result
        except BaseException as e:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            g.write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),
                actual_child_waited=child is not None and child.poll() is not None,child_exit_code=None if child is None else child.returncode,
                v5_cross_host_verdict='FAIL_UNCHANGED'));raise

def worker(a):
    g.require(sys.platform.startswith('linux'),'Actual worker requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256);device=device_from_request(r);device.check_cuda_environment();spec=r['spec'];out=path.parent/'run'
    launch=worker_identity(out,a.request_sha256,a.owner_pid);g.require(launch['argv'][0]==spec['python'],'Wrong frozen interpreter')
    stopped=False;started=time.monotonic();ledger=None;backend=None;shared=g.helper();last=0.
    def stop(*args):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==a.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),'Diagnostic stop or parent loss')
        g.require(time.time()<spec['deadline_unix'] and time.monotonic()-started<spec['max_seconds'],'Diagnostic deadline exceeded')
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>=2:
            snapshot,_=device.resource_snapshot(shared);last=time.monotonic()
            with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256,started_unix=time.time()))
        resolver,environment,records,_,extra=g.validate_spec(spec);boundary()
        bits=actual_bits(r['witness']);ledger=g.Ledger(out/'calls',guard,CAPS)
        module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_leo_single_rx_original_math')
        Backend,ast_proof=adapt_backend(module,device)
        g.write(out/'device_adapter_ast_proof.json',ast_proof)
        backend=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
        hashes={int(k):v for k,v in r['witness']['expected_new_TX_CDF_hashes'].items()}
        check=exercise(bits,hashes,backend,ledger,lambda:expected_tokens_after_decode(records[0],resolver),out/'checks')
        backend.close();resolver.reverify();environment.reverify();registration(path,a.request_sha256);guard()
        counts=ledger.summary();g.require(counts['completed']==CAPS and counts['unresolved']==0,'Single RX actual budget not closed exactly')
        result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,counts=counts,source_index=0,m=4,check=check,
            old_CDF='UNOBSERVABLE',old_Direct='UNOBSERVABLE',v5_cross_host_verdict='FAIL_UNCHANGED',
            original32_and48_budgets_unchanged=True,comparison_tokens_supplied_to_RX=False,probability_tables_shared=False,
            production_admission=False,timing_measurement=False,automatic_successor=False)
        g.write(out/'worker_completion.json',result);return result
    except BaseException as e:
        if backend is not None:backend.failure_evidence()
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,error=repr(e),traceback=traceback.format_exc(),
            counts=None if ledger is None else ledger.summary(),v5_cross_host_verdict='FAIL_UNCHANGED'));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--parent-request',required=True);q.add_argument('--parent-request-sha256',required=True)
    q.add_argument('--device-receipt',required=True);q.add_argument('--device-receipt-sha256',required=True);q.add_argument('--target-index',required=True,type=int,choices=(6,7))
    q.add_argument('--out',required=True);q.add_argument('--deadline-unix',required=True,type=float);q.add_argument('--max-seconds',default=900,type=int)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',required=True,type=int)
    a=p.parse_args();result=prepare(a) if a.command=='prepare' else run(a) if a.command=='run' else worker(a)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
