"""Independent arbitrary-N L13+CRC16 adapter, original math unchanged."""
from common import *
import importlib.util,sys,ast,types,math,numpy as np
def original():
 sys.path.insert(0,str(R/'experiments/prior-aware-grouped-mcs-v1'));import ldpc_backend,uep_phy,uep_common
 backend=ldpc_backend.SionnaBackend('cpu',R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json');assert backend.qualified
 torch=backend.torch;torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);assert not torch.cuda.is_initialized()
 # Select unchanged original pure modulation definitions, not its raw12 API.
 p=types.SimpleNamespace(require=lambda v,m:None if v else (_ for _ in ()).throw(AssertionError(m)))
 path=R/'experiments/main-raw64-20261007/main_raw64_packet_adapter.py';nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('modulate','demap')];assert len(nodes)==2
 ns=dict(math=math,p=p);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 spec=importlib.util.spec_from_file_location('_LE2_original_paid_group',E1/'packet_two_group.py');packet=importlib.util.module_from_spec(spec);spec.loader.exec_module(packet)
 return backend,uep_phy,uep_common,uep_phy.Header(R),ns,packet
def rng(*parts):return np.random.default_rng(int.from_bytes(hashlib.sha256(canonical(['NESTCOM_LAYERED_EC_LE2_V1',*parts]).encode()).digest()[:16],'big'))
def receive(backend,legacy,mapping,packet,received,layout,snr,public_masks,explicit_empty):
 # Signature contains received symbols and public code/profile/session only.
 logits=mapping['demap'](received,snr,layout['q'],backend.torch,legacy)
 logits=logits*(1-2*public_masks.to(logits.dtype))
 hard=backend.decode(logits,layout['k'],layout['n'],layout['q']).cpu().numpy().astype(np.uint8)
 decisions=[packet.parse_group(h,layout['k'],explicit_empty=bool(empty)) for h,empty in zip(hard,explicit_empty)]
 return hard,decisions
def body_batch(runtime,layout,start,count,snr,lengths):
 backend,legacy,_,_,mapping,packet=runtime;t=backend.torch;gid=layout['geometry_id'];k,n,q=layout['k'],layout['n'],layout['q'];N=layout['N']
 ls=[int(lengths[(start+i)%len(lengths)]) for i in range(count)];assert all(0<=L<=min(8191,k-29) and L!=1 for L in ls)
 info=np.stack([packet.pack_group(rng('payload',gid,start+i).integers(0,2,L,dtype=np.uint8),k,explicit_empty=(L==0)) for i,L in enumerate(ls)])
 masks=np.stack([rng('public_scrambler',gid,start+i).integers(0,2,n,dtype=np.uint8) for i in range(count)]);mask=t.as_tensor(masks,device='cpu')
 code=backend.encode(info,n,q);wave=mapping['modulate']((code+mask)%2,q,t,legacy);assert tuple(wave.shape)==(count,N,2)
 noise=np.stack([rng('standard_AWGN',gid,start+i).standard_normal((N,2)).astype(np.float32) for i in range(count)]);received=wave+t.as_tensor(noise)*(10**(-float(snr)/20))
 hard,decisions=receive(backend,legacy,mapping,packet,received,layout,snr,mask,[L==0 for L in ls])
 # Caller persists actual RX evidence FIRST, then uses this returned TX array
 # in a separate audit. No oracle bit is sent back to receive/parse.
 return dict(hard=hard,accepted=np.array([x['accepted'] for x in decisions]),statuses=[x['status'] for x in decisions],declared_lengths=np.array([x.get('declared_length',-1) for x in decisions]),actual_energy=wave.square().sum((1,2)).cpu().numpy(),transmitted_lengths=np.array(ls)),info
def header_batch(runtime,start,count,snr):
 header=runtime[3];rows=[];pids=[]
 for i in range(start,start+count):
  pid=int(rng('paid_header_field',i).integers(0,4096));wave=np.asarray(header.transmit(pid),np.float32);assert wave.shape==(68,2)
  noisy=wave+rng('standard_header_AWGN',i).standard_normal((68,2)).astype(np.float32)*(10**(-float(snr)/20));r=header.receive(noisy,snr,codebook=None)
  rows.append(r);pids.append(pid)
 return rows,pids
