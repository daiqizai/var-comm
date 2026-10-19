"""Select an available admitted device for RX; all native computation is unchanged."""
import ast,hashlib,subprocess
from common import *
import rx_prepare_v2 as schema
SCHEMA_SHA='78e05ff397ac94dfecfa0e82352c250f3587947a0a9181d88b46c0360831441c'

def choose_gpu(rows):
    candidates=[g for g in rows if g['free_mib']>=65536 and g['utilization']<=25]
    assert candidates,'No GPU satisfies unchanged startup64GiB/util25 bounds'
    return max(candidates,key=lambda g:(g['free_mib'],-g['index']))

def select_device(cfg):
    gpu=choose_gpu(gpus())
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
    cfg['resource']={**cfg['resource'],'gpu':gpu};cfg['cuda_admission']=admission
    return gpu

def adapt(raw):
    assert desc(E/'rx_prepare_v2.py')['sha256']==SCHEMA_SHA
    original=schema.adapt(raw)
    a="gpu=cfg['resource']['gpu'];now=next(g for g in gpus()if g['uuid']==gpu['uuid']);assert now['free_mib']>=65536 and now['utilization']<=25"
    b="gpu=select_device(cfg)"
    c="code+=[desc(E/n)for n in ('rx_prepare_v2.py','execute_rx_v2.py')]"
    d=c+"\n    code+=[desc(E/n)for n in ('rx_prepare_v3.py','execute_rx_v3.py','test_rx_device_v1.py')]\n    inputs+=[desc(L/'rx_device_qualification_v1/completion.json'),desc(L/'rx_prepare_control_v2/child.actual_wait.json'),desc(L/'rx_prepare_control_v2/failure.json')]"
    assert original.count(a)==original.count(c)==1
    text=original.replace(a,b).replace(c,d)
    assert text.replace(d,c).replace(b,a)==original
    fn=lambda s:next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef)and x.name=='run')
    assert ast.dump(fn(text))==ast.dump(fn(raw.decode()))
    return text

def main():
    assert sys.argv[1:]==['prepare']
    assert not(L/'rx_v1').exists()and not(O/'rx_v1').exists()
    for version in (1,2):
        old=L/f'rx_prepare_control_v{version}';w=read(old/'child.actual_wait.json')
        assert type(w['returncode'])is int and w['returncode']==1
        assert gone(w['process'])and gone(read(old/'supervisor.json')['process'])
    assert "KeyError: 'unresolved'"in(L/'rx_prepare_control_v1/child.stderr').read_text()
    err=(L/'rx_prepare_control_v2/child.stderr').read_text()
    assert 'rx_control.py::clean_closure_schema_v2' in err and 'line 33, in prepare' in err and err.rstrip().endswith('AssertionError')
    for name in ('rx_schema_qualification_v1','rx_device_qualification_v1'):
        q=read(L/name/'completion.json');assert q['status']=='PASS'and q['scientific_calls']==0
        for p in q['sources']:verify(p)
    ns={'__name__':'_rx_device_v3','__file__':str(E/'rx_control.py'),'normalized_clean_review':schema.normalized_clean_review,'select_device':select_device}
    exec(compile(adapt((E/'rx_control.py').read_bytes()),str(E/'rx_control.py')+'::schema_device_v3','exec'),ns)
    ns['prepare']()

if __name__=='__main__':main()
