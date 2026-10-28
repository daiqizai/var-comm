"""Pinned variable-length LDPC and paid packet syntax for the fixed CAB scope."""
from block_common import *
from protocol import packet_module
NAMESPACE='NESTCOM_CAB_20261028_V1'

def rng(*parts):
 s=json.dumps([NAMESPACE,*parts],sort_keys=True,separators=(',',':'),allow_nan=False)
 return np.random.default_rng(int.from_bytes(hashlib.sha256(s.encode()).digest()[:16],'big'))

def counter(*parts):
 s=json.dumps([NAMESPACE,*parts],sort_keys=True,separators=(',',':'),allow_nan=False)
 return int.from_bytes(hashlib.sha256(s.encode()).digest()[:16],'big')

def runtime():
 sys.path.insert(0,str(R/'experiments/prior-aware-grouped-mcs-v1'))
 import ldpc_backend,uep_phy
 b=ldpc_backend.SionnaBackend('cpu',R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json');assert b.qualified
 t=b.torch;t.set_num_threads(2);t.set_num_interop_threads(1);t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False;t.backends.cudnn.benchmark=False;t.use_deterministic_algorithms(True);assert not t.cuda.is_initialized()
 return b,uep_phy,uep_phy.Header(R),packet_module()

def save_npz(path,**arrays):
 with Path(path).open('xb')as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())

def receive(b,phy,packet,y,g,snr,counters,group,empty,session):
 """RX sees only observations and public framing/session information."""
 t=b.torch;llr=phy.demap_torch(t.as_tensor(y,dtype=t.float32),snr,2)
 mask=t.as_tensor(phy.mask_batch(g['n'],counters,group,session),dtype=t.float32)
 hard=b.decode(llr*(1-2*mask),g['k'],g['n'],2).cpu().numpy().astype(np.uint8)
 decisions=[packet.parse_group(a,g['k'],explicit_empty=bool(e))for a,e in zip(hard,empty)]
 return hard,decisions
