"""PCI-joined NVML/CUDA identity; CUDA Driver queries only, no context or model."""
import ctypes,json,os,subprocess,sys,uuid
from pathlib import Path
def canonical_pci(s):
 domain,bus,tail=s.strip().lower().split(':');device,function=tail.split('.')
 return '%04x:%02x:%02x.%x'%(int(domain,16),int(bus,16),int(device,16),int(function,16))
def select_mapping(records,nvml):
 pci=canonical_pci(nvml['pci']);matches=[r for r in records if canonical_pci(r['pci'])==pci]
 assert len(matches)==1,('CUDA_PCI_MAPPING_NOT_UNIQUE',pci)
 r=matches[0];assert type(r['ordinal'])is int and r['ordinal']>=0
 assert uuid.UUID(r['uuid'].removeprefix('GPU-'))
 # NVML UUID is intentionally not equated with the CUDA Driver UUID.
 return dict(nvml_index=int(nvml['index']),nvml_uuid=nvml['uuid'],pci=pci,cuda_ordinal=r['ordinal'],cuda_uuid='GPU-'+str(uuid.UUID(r['uuid'].removeprefix('GPU-'))),cuda_visible_devices=str(r['ordinal']),cuda_device_order='PCI_BUS_ID',source='NVML and CUDA Driver joined by exact PCI identity',model_imports=0,context_creation_calls=0)
def query_driver():
 assert os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID' and 'CUDA_VISIBLE_DEVICES' not in os.environ
 lib=ctypes.CDLL('libcuda.so.1');ci=ctypes.c_int
 def bind(name,args):
  f=getattr(lib,name);f.argtypes=args;f.restype=ci;return f
 init=bind('cuInit',[ctypes.c_uint]);count=bind('cuDeviceGetCount',[ctypes.POINTER(ci)]);device=bind('cuDeviceGet',[ctypes.POINTER(ci),ci]);pci=bind('cuDeviceGetPCIBusId',[ctypes.POINTER(ctypes.c_char),ci,ci]);getuuid=bind('cuDeviceGetUuid',[ctypes.c_void_p,ci])
 assert init(0)==0;n=ci();assert count(ctypes.byref(n))==0;rows=[]
 for ordinal in range(n.value):
  dev=ci();assert device(ctypes.byref(dev),ordinal)==0
  p=ctypes.create_string_buffer(64);assert pci(p,64,dev)==0
  u=(ctypes.c_ubyte*16)();assert getuuid(ctypes.byref(u),dev)==0
  rows.append(dict(ordinal=ordinal,pci=p.value.decode(),uuid=str(uuid.UUID(bytes=bytes(u)))))
 return rows
def query_clean():
 env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1');env.pop('CUDA_VISIBLE_DEVICES',None)
 p=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'driver'],env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=60)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
if __name__=='__main__':
 assert sys.argv[1:] == ['driver'];print(json.dumps(query_driver(),sort_keys=True))
