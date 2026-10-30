import os,sys,io
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from c_train_pair_v2 import make_schedule
import torch
ix,K=make_schedule(10000,2026103001);ix2,K2=make_schedule(10000,2026103001)
assert np.array_equal(ix,ix2)and np.array_equal(K,K2)
assert ix.shape==K.shape==(3000,4,8)and ix.min()==0 and ix.max()==9999
assert ((K[:,[0,2]]>0)&(K[:,[0,2]]<100)).all()and ((K[:,[1,3]]>0)&(K[:,[1,3]]<169)).all()
ns=np.random.get_state();state=dict(torch_rng=torch.get_rng_state(),numpy_state=dict(generator=ns[0],keys=ns[1].tolist(),position=ns[2],has_gauss=ns[3],cached_gaussian=ns[4]),numerics=dict(torch=str(torch.__version__)))
b=io.BytesIO();torch.save(state,b);b.seek(0);restored=torch.load(b,weights_only=True);assert torch.equal(restored['torch_rng'],state['torch_rng'])and restored['numpy_state']==state['numpy_state']
save(D/'c_train_preflight_v2.json',dict(status='PASS',new_v2_changes=['explicit admitted PCI/UUID proof','weights_only safe RNG checkpoint serialization'],training_algorithm_changed=False,matched_schedule_exact=True,schedule_sha256=asha(ix),masks_sha256=asha(K),checkpoint_safe_load=True,GPU_initialized=torch.cuda.is_initialized(),code=pin(E/'c_train_pair_v2.py')))
print(read(D/'c_train_preflight_v2.json'))
