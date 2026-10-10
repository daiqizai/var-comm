"""One saved leo TX decoded on an explicitly observed H800 host; no successor.

The frozen PFS runtime and mathematical code stay byte-identical. System native
libraries are separately bound to an actual CUDA-hidden import on this host.
Historical native receipts remain historical evidence, never current-host facts.
"""
from __future__ import annotations
import argparse
import copy
import csv
import io
from pathlib import Path
import types
import leo_whole_gate_v1 as original_gate
import leo_single_rx_diagnostic_v2 as original_rx

g = original_gate
SCHEMA = 'H800_SINGLE_RX_DIAGNOSTIC_V1'
PASS = 'PASS_H800_SOURCE0_M4_RX_ONLY'
PROJECTION_SHA = 'df9adc9cb2b13aa7f24137a7b1c95e1f548d0defe88ff9ef5deb4d51b79eede6'
DEVICE_SHA = '2b147901568b3cdbd9c48a9917fa962aa3c5520829cd0987ea84a3ca2b7f235f'
NATIVE_SHA = 'eb641b1f3139960b67b6342417eba09fb0aaed26784c93f99c1d85c9437a4868'
RT = g.BASE/'var_comm_runtime_20261010'
NATIVE_PATH = RT/'qualification/h800_native_import_v1/binding.json'
TOOLS = original_rx.TOOLS + ('leo_single_rx_diagnostic_v2.py',)
SOURCE_SHA = {
    'leo_whole_gate_v1.py':'06c8f1659eed5d67c9bcd368a91f6c35628d83136e4a9ff4e2fbe356cb65f81b',
    'leo_whole_math_v1.py':'db4349f936b2cd0c267932cc286e2661e1f7d56dd10d1c4e4a5d7d90ec279252',
    'leo_shared_preflight.py':'0f7473aa003991edca43a4587bbd03dfef0955e13f85cf23ffaff1dae1cf66ab',
    'leo_runtime_verify_v1.py':'77a059e9c8f4d4328b338274ef2dd8580c6634bd3e1d5ba0e6613b7237422981',
    'leo_single_rx_diagnostic_v2.py':'08835b22a2c4378cd462ae68690d2d00e2865971c130d4456dc21122d6df6de7',
}


def clone_function(function, namespace):
    """Reuse an unchanged code object with explicit dependencies, no globals patch."""
    copied=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    copied.__kwdefaults__=function.__kwdefaults__
    return copied


def historical_native_normalization(facts, by_path, projection_rows, base=g.BASE):
    """Reconstruct pinned historical metadata without assuming this host is leo."""
    g.require(len(facts)==len(projection_rows),'Historical native fact count differs')
    result=[]
    for old,row in zip(facts,projection_rows):
        expected=dict(row);lexical=expected.pop('observed_lexical_path')
        expected['path']=lexical
        g.require(old==expected,'Historical native fact differs from pinned projection')
        if Path(old['path']).is_relative_to(Path(base)):
            # Personal bytes still follow the original normalizer and exact map.
            actual=g.normalize_native_facts([old],by_path,base)
            g.require(actual==[row],'Personal native projection differs')
        result.append(copy.deepcopy(row))
    return result


def validate_native_document(document, projection_sha=PROJECTION_SHA, base=g.BASE):
    g.require(document.get('schema')=='H800_NEW_HOST_NATIVE_BINDING_V1' and
        document.get('status')=='CPU_IMPORT_OBSERVED_NOT_NUMERICAL','Actual H800 CPU import receipt required')
    for name in ('model_calls','tensor_calls','checkpoint_reads','source_image_reads'):
        g.require(type(document.get(name)) is int and document[name]==0,'Import scope changed: '+name)
    g.require(document.get('CUDA_VISIBLE_DEVICES')=='' and
        document.get('original_runtime_projection_sha256')==projection_sha,'Hidden import/projection binding differs')
    waited=document['actual_child_wait']
    g.require(waited.get('actual_wait') is True and waited.get('returncode')==0 and
        waited.get('interrupted_or_timeout') is False,'H800 import was not actually completed')
    rows=document['files'];g.require(isinstance(rows,list) and 1<=len(rows)<=128,'Finite actual native closure required')
    seen=set();answer=[]
    for row in rows:
        g.require(set(row)=={'path','sha256','bytes'} and g.SHA_RE.fullmatch(row['sha256']) and
            type(row['bytes']) is int and row['bytes']>=0,'Invalid actual native file pin')
        p=Path(row['path'])
        g.require(p.is_absolute() and '..' not in p.parts and not p.is_relative_to(Path(base)) and
            str(p.resolve(strict=True))==row['path'] and not p.is_symlink() and
            '.so' in p.name and row['path'] not in seen,'Canonical external native file required')
        before=p.stat();g.require(p.is_file() and before.st_size==row['bytes'] and g.sha(p)==row['sha256'],
            'Actual H800 native file bytes differ: '+str(p))
        after=p.stat();g.require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Native file changed during verification')
        seen.add(row['path']);answer.append(dict(row))
    return answer


class HostGate:
    """Original validation code; only historical/current host facts are separated."""
    def __getattr__(self,name):return getattr(g,name)
    def native_pin(self):return dict(path=str(NATIVE_PATH),sha256=NATIVE_SHA)
    def validate_spec(self,spec,base=g.BASE,verify_environment=True):
        g.require(spec['environment']['sha256']==PROJECTION_SHA,'Frozen PFS projection changed')
        env=g.checked_json(g.inside(spec['environment']['path'],base),PROJECTION_SHA)
        native=g.checked_json(g.inside(NATIVE_PATH,base),NATIVE_SHA)
        current=validate_native_document(native,PROJECTION_SHA,base)
        namespace=dict(vars(g))
        namespace['normalize_native_facts']=lambda facts,by_path,base=base:historical_native_normalization(
            facts,by_path,env['host_native_library_facts'],base)
        namespace['runtime_from_receipt']=clone_function(g.runtime_from_receipt,namespace)
        class ActualHostRuntime(g.RuntimeFiles):
            def __init__(self,original,base=base):
                super().__init__(original,base)
                # All personal original/generated checks remain in inherited reverify.
                self.historical_host_native=copy.deepcopy(self.host_native)
                self.host_native=copy.deepcopy(current)
        namespace['RuntimeFiles']=ActualHostRuntime
        def historical_addendum(runtime,pin,projection_sha256,base=base):
            g.require(projection_sha256==PROJECTION_SHA and
                pin['sha256']=='a0d1cc3dfb2e55cfd5dfcd74e027fb0d3d8747d484186d36d448be9e4ed9be90',
                'Historical leo native addendum differs')
            old=g.checked_json(g.inside(pin['path'],base),pin['sha256'])
            g.require(old['schema']=='LEO_NEW_HOST_NATIVE_ADDENDUM_V1' and
                old['original_runtime_projection_sha256']==PROJECTION_SHA and
                old['model_calls']==old['tensor_calls']==0,'Historical native addendum provenance differs')
            runtime.historical_host_native_addendum=old
        namespace['attach_host_native_addendum']=historical_addendum
        result=clone_function(g.validate_spec,namespace)(spec,base,verify_environment)
        result[-1]['actual_host_native_binding']=dict(receipt=self.native_pin(),files=current,
            historical_projection_unchanged=True,historical_native_facts_are_not_current_host_facts=True,
            PFS_original_bytes_exact=True,old_host_byte_identity_claimed=False)
        return result


HOST = HostGate()


class DeviceAdapter(original_rx.DeviceAdapter):
    def __init__(self,pin,index):
        g.require(type(index) is int and index in (0,2),'Only observed H800 GPU0 or GPU2')
        g.require(set(pin)=={'path','sha256'} and pin['sha256']==DEVICE_SHA,'Fixed H800 observation required')
        receipt=g.checked_json(g.inside(pin['path']),pin['sha256'])
        g.require(receipt['status']=='READ_ONLY_ENUMERATION_NO_MODEL' and
            receipt['model_calls']==receipt['explicit_tensor_calls']==0,'Device observation scope changed')
        devices=receipt['CUDA_driver_devices']
        drivers=[r for r in devices if r['driver_ordinal_unmasked']==index]
        rows=[r for r in csv.reader(io.StringIO(receipt['NVML_csv'])) if int(r[0].strip())==index]
        g.require(len(drivers)==len(rows)==1,'Ambiguous observed H800 device')
        self.index=index;self.UUID=g.normalize_gpu_uuid(rows[0][1].strip())
        self.CUDA_UUID=g.normalize_gpu_uuid(drivers[0]['CUDA_UUID']);self.PCI_BUS_ID=g.normalize_pci_bus_id(drivers[0]['PCI'])
        g.require(g.normalize_pci_bus_id(rows[0][2].strip())==self.PCI_BUS_ID,'CUDA/NVML PCI mismatch')
    def __getattr__(self,name):return getattr(HOST,name)
    def device_identity_binding(self):
        return dict(nvml_gpu_index=self.index,nvml_uuid=self.UUID,cuda_uuid=self.CUDA_UUID,pci_bus_id=self.PCI_BUS_ID,
            NVML_and_CUDA_UUID_namespaces_differ=self.UUID!=self.CUDA_UUID,**self.cuda_environment_binding())


def engine():
    for name,digest in SOURCE_SHA.items():
        g.require(g.sha(Path(__file__).with_name(name))==digest,'Original implementation changed: '+name)
    namespace=dict(vars(original_rx))
    namespace.update(g=HOST,__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,TOOLS=TOOLS,DeviceAdapter=DeviceAdapter)
    for name,value in vars(original_rx).items():
        if isinstance(value,types.FunctionType) and value.__module__==original_rx.__name__:
            namespace[name]=clone_function(value,namespace)
    old_registration=namespace['registration']
    def registration(path,digest):
        r=old_registration(path,digest)
        g.require(r.get('actual_host_native_binding')==HOST.native_pin(),'H800 native receipt binding changed')
        return r
    namespace['registration']=registration
    def prepare(args):
        out=g.inside(args.out);g.require(not out.exists(),'Fresh H800 diagnostic registration required')
        parent_path=g.inside(args.parent_request)
        g.require(args.parent_request_sha256==original_rx.PARENT_SHA,'Only fixed v5 witness parent allowed')
        parent=g.prepared(parent_path,original_rx.PARENT_SHA)
        pin=dict(path=str(g.inside(args.device_receipt)),sha256=args.device_receipt_sha256)
        device=DeviceAdapter(pin,args.target_index)
        spec=namespace['inherited_spec'](parent,args.deadline_unix,args.max_seconds)
        witness=namespace['witness_metadata'](parent_path)
        HOST.validate_spec(spec,verify_environment=False)
        out.mkdir(parents=True)
        request=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',
            parent_request=dict(path=str(parent_path),sha256=original_rx.PARENT_SHA),spec=spec,
            caps=dict(original_rx.CAPS),witness=witness,source_index=0,source_id=g.IDS[0],m=4,
            device_identity_binding=device.device_identity_binding(),device_receipt=pin,target_index=args.target_index,
            prelaunch_wait_seconds=120,created_unix=g.time.time(),
            tool_bindings={name:g.sha(Path(__file__).with_name(name)) for name in (*TOOLS,Path(__file__).name)},
            actual_host_native_binding=HOST.native_pin(),
            purpose='Saved leo GPU5 TX decoded independently on an observed H800 device; historical failures unchanged',
            budget_scope='Separate one-RX H800 diagnostic; no change to v5, leo v2 or EP32/48 budgets',
            preparation_model_calls=0,preparation_source_arrays_read=0,automatic_successor=False,
            old_CDF='UNOBSERVABLE',production_admission=False,timing_measurement=False)
        path=out/'request.json';g.write(path,request);return namespace['descriptor'](path)
    namespace['prepare']=prepare
    return types.SimpleNamespace(**namespace)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--parent-request',required=True);q.add_argument('--parent-request-sha256',required=True)
    q.add_argument('--device-receipt',required=True);q.add_argument('--device-receipt-sha256',required=True)
    q.add_argument('--target-index',required=True,type=int,choices=(0,2));q.add_argument('--out',required=True)
    q.add_argument('--deadline-unix',required=True,type=float);q.add_argument('--max-seconds',default=900,type=int)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',required=True,type=int)
    args=p.parse_args();run=engine();value=run.prepare(args) if args.command=='prepare' else run.run(args) if args.command=='run' else run.worker(args)
    print(g.json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
