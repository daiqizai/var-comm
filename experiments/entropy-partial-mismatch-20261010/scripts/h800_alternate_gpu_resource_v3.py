"""Alternate same-host H800 admission with original current-PID credit and math.
GPU3 may start below80% utilization under explicit user-authorized scheduling;
other cards retain50%. No scientific call budgets or numeric flags change.

No model, tensor, allocator-setting, numerical or source-population operation is
performed here. Original CPU/host/disk and memory checks are unchanged; only
the explicitly selected GPU3 prelaunch utilization threshold changes.
"""
from __future__ import annotations
import math
import csv
import io
import socket
import os
from pathlib import Path
import sys
import time
import h800_single_rx_v1 as host
import leo_whole_math_v1 as original_math

g=host.g
GiB=1<<30
POLICY_BASE=dict(schema='H800_ALTERNATE_SHARED_RESOURCE_POLICY_V3',
 prelaunch_free_bytes=20*GiB,prelaunch_utilization_max=50,runtime_free_bytes=4*GiB,
 runtime_free_plus_own_reserved_bytes=20*GiB,own_allocator_cap_bytes=16*GiB,
 own_credit='minimum_of_current_process_same_device_reserved_before_and_after_NVML_sample',
 uninitialized_or_absent_torch_credit_bytes=0,external_or_unknown_memory_credit_bytes=0,
 CPU_host_disk_gates_unchanged=True,scientific_calls=0)
DEVICE_SHA='9e158c4b1634d0ebb4e19fcbdf456e35c60afd41c78fdb0395637652192abc9a'
DEVICE_PATH=host.RT/'receipts/h800_all_devices_readonly_20261011_v1/completion.json'

def policy(index):
    g.require(type(index) is int and index in (0,1,2,3),'Observed same-host H800 required')
    return dict(POLICY_BASE,target_index=index,prelaunch_utilization_max=80 if index==3 else 50,
        prelaunch_utilization_comparison='strictly_below' if index==3 else 'at_most',
        scheduling_authorization='Explicit alternate shared GPU3 scheduling; original mathematical protocol unchanged')

MATH_SHA='db4349f936b2cd0c267932cc286e2661e1f7d56dd10d1c4e4a5d7d90ec279252'
ResourceBusy=host.original_rx.ResourceBusy


def validate_policy(request):
    g.require(request['target_index'] in (0,1,2,3) and request['resource_policy']==policy(request['target_index']) and
        request['resource_guard']==dict(path=str(Path(__file__)),sha256=g.sha(__file__)),
        'Frozen selected-H800 scheduling policy and implementation changed')


def own_allocator_observation(device,torch_module=None):
    """Observe the current process; never initialize CUDA or import torch."""
    result=dict(pid=os.getpid(),observed_unix=time.time(),target_index=device.index,
        initialized=False,reserved_bytes=0,allocated_bytes=0,peak_reserved_bytes=0,
        identity_verified=False,external_memory_counted=False)
    g.require(device.index in (0,1,2,3) and os.getpid()==device.creator_pid,'selected H800 same-process adapter required')
    torch_module=sys.modules.get('torch') if torch_module is None else torch_module
    if torch_module is None or not torch_module.cuda.is_initialized():return result
    device.check_cuda_environment()
    cuda=torch_module.cuda
    g.require(cuda.device_count()==1 and cuda.current_device()==0,'Exactly one selected visible H800 required')
    props=cuda.get_device_properties(0)
    g.require(g.sha(original_math.__file__)==MATH_SHA,'Original identity query changed')
    identity=device.validate_cuda_device_identity(str(getattr(props,'uuid','')),original_math.cuda_driver_pci_bus_id())
    values=dict(reserved_bytes=cuda.memory_reserved(0),allocated_bytes=cuda.memory_allocated(0),
        peak_reserved_bytes=cuda.max_memory_reserved(0))
    g.require(all(type(v) is int and v>=0 for v in values.values()),'Invalid current-process allocator observation')
    g.require(values['allocated_bytes']<=values['reserved_bytes']<=values['peak_reserved_bytes']<=16*GiB,
        'Current-process allocator exceeds frozen16GiB cap')
    result.update(values,initialized=True,identity_verified=True,actual_identity=identity,logical_device=0)
    return result


def assess_snapshot(snapshot,before,after,prelaunch):
    """Original memory/CPU gates plus explicitly authorized GPU3 utilization."""
    gpu=snapshot['gpu'];cpu=snapshot['cpu']
    g.require(gpu['index'] in (0,1,2,3),'Only selected H800 admitted by resource correction')
    own=0
    if not prelaunch and before['initialized'] and after['initialized']:
        g.require(before['pid']==after['pid']==os.getpid() and before['identity_verified'] and after['identity_verified'] and
            before['actual_identity']==after['actual_identity'],'Allocator identity changed during sample')
        own=min(before['reserved_bytes'],after['reserved_bytes'])
    g.require(type(own) is int and 0<=own<=16*GiB,'Invalid own allocator credit')
    free=gpu['free_MiB']*(1<<20)
    checks=[(free>=20*GiB if prelaunch else free>=4*GiB,'GPU absolute free memory margin'),
        (prelaunch or free+own>=20*GiB,'GPU free plus verified own reserved margin'),
        (not prelaunch or (gpu['utilization_percent']<80 if gpu['index']==3 else gpu['utilization_percent']<=50),'Selected GPU busy'),
        (cpu['effective_cpu_cores']>=2 and len(cpu['allowed_cpus'])>=2,'Insufficient CPU capacity'),
        (math.isfinite(cpu['host_busy_percent']) and 0<=cpu['host_busy_percent']<=95,'CPU pressure above95%'),
        (cpu['available_memory_bytes']>=2*GiB,'Less than2GiB host margin'),
        (snapshot['personal_filesystem_free_bytes']>=10*GiB,'Less than10GiB filesystem margin')]
    snapshot['resource_guard']=dict(policy=policy(gpu['index']),phase='prelaunch' if prelaunch else 'running',
        own_before=before,own_after=after,credited_own_reserved_bytes=own,
        effective_free_plus_own_reserved_bytes=free+own,checks=[dict(passed=bool(ok),name=name) for ok,name in checks])
    for ok,message in checks:
        if not ok:
            error=ResourceBusy(message);error.resource_snapshot=snapshot;raise error
    return snapshot,dict(cpu_affinity=cpu['allowed_cpus'][:2],threads=2,workers=1,shared_device=True)


class DeviceAdapter(host.DeviceAdapter):
    def __init__(self,pin,index):
        g.require(type(index) is int and index in (0,1,2,3),'Explicit observed H800 index required')
        g.require(pin==dict(path=str(DEVICE_PATH),sha256=DEVICE_SHA),'Exact all-H800 enumeration required')
        receipt=g.checked_json(g.inside(pin['path']),pin['sha256'])
        g.require(receipt['status']=='READ_ONLY_ENUMERATION_NO_MODEL' and receipt['model_calls']==receipt['explicit_tensor_calls']==0 and
            receipt['hostname']==socket.gethostname(),'Same host zero-model enumeration required')
        drivers=[r for r in receipt['CUDA_driver_devices'] if r['driver_ordinal_unmasked']==index]
        rows=[r for r in csv.reader(io.StringIO(receipt['NVML_csv'])) if int(r[0].strip())==index]
        g.require(len(drivers)==len(rows)==1 and 'H800' in rows[0][3],'Unique actual H800 device required')
        self.index=index;self.UUID=g.normalize_gpu_uuid(rows[0][1].strip())
        self.CUDA_UUID=g.normalize_gpu_uuid(drivers[0]['CUDA_UUID']);self.PCI_BUS_ID=g.normalize_pci_bus_id(drivers[0]['PCI'])
        g.require(g.normalize_pci_bus_id(rows[0][2].strip())==self.PCI_BUS_ID,'CUDA/NVML PCI mismatch')
        self.creator_pid=os.getpid();self.evidence_directory=None;self.request_sha=None

    def bind_resource_evidence(self,out,request_sha):
        path=g.inside(out);g.require(path.is_dir() and not path.is_symlink(),'Existing owner evidence directory required')
        g.require(g.SHA_RE.fullmatch(request_sha),'Resource evidence requires exact request SHA')
        self.evidence_directory=path;self.request_sha=request_sha

    def persist_failure(self,error,snapshot):
        g.require(self.evidence_directory is not None,'Resource failure must have owner evidence destination')
        directory=self.evidence_directory/'resource_guard_failures';directory.mkdir(exist_ok=True)
        value=dict(schema='H800_ALTERNATE_RESOURCE_GATE_FAILURE_V3',pid=os.getpid(),request_sha256=self.request_sha,
            error=repr(error),snapshot=snapshot,scientific_caps_changed=False)
        path=directory/(str(os.getpid())+'_'+str(time.time_ns())+'.json')
        with path.open('x') as handle:
            g.json.dump(value,handle,sort_keys=True,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
        error.resource_snapshot=snapshot;error.resource_failure_receipt=str(path)

    def resource_snapshot(self,shared,prelaunch=False):
        snapshot=dict(policy=policy(self.index),phase='prelaunch' if prelaunch else 'running',collector_complete=False)
        try:
            g.require(self.evidence_directory is not None,'Resource owner evidence binding is required before sampling')
            before=own_allocator_observation(self)
            try:
                snapshot,_=host.original_rx.DeviceAdapter.resource_snapshot(self,shared,prelaunch=prelaunch)
            except ResourceBusy as old_error:
                # Re-evaluate all original gates from the same complete snapshot.
                # The running self-memory credit and registered GPU3 launch
                # utilization threshold are the only changed decisions.
                snapshot=old_error.resource_snapshot
            after=own_allocator_observation(self)
            snapshot['collector_complete']=True
            g.require(0<=time.time()-snapshot['started_unix']<=15,'Resource snapshot stale')
            return assess_snapshot(snapshot,before,after,prelaunch)
        except BaseException as error:
            self.persist_failure(error,snapshot);raise
