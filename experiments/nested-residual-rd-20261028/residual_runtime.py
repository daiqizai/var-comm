"""Frozen native modules; direct cumulative quantized latent, no missing-token zero IDs."""
import sys,pathlib,ast,types,copy,hashlib,os,time,platform
import numpy as np
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/shared-codeword-fit-20261028'))
from fit_common import checked,read,sha,save,pin,asha,Ledger,ident,gone
from fit_codec import FitCodec,OFFSETS
SIZES=(1,2,3,4,5,6,8,10,13,16)
E=R/'experiments/nested-residual-rd-20261028'
O=R/'results/nested_residual_rd_20261028_v1'
L=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/nested_residual_rd_20261028_v1')
def definitions(path,names,ns):
    tree=ast.parse(pathlib.Path(path).read_text(encoding='utf-8-sig'));nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))and x.name in names]
    assert {x.name for x in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
def load_frozen(reg,out):
    import torch
    torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.cuda.set_device(0)
    assert torch.cuda.device_count()==1
    actual=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=6,interop_threads=2)
    routes=read(checked(reg['model_routes']));assert actual==routes['native_qualification']['numeric_settings']['runtime']
    pci=definitions(R/'experiments/entropy-partial-mismatch-20261010/scripts/leo_whole_math_v1.py',{'cuda_driver_pci_bus_id'},dict(ctypes=__import__('ctypes')))['cuda_driver_pci_bus_id']()
    assert pci.lower().lstrip('0')==reg['pci'].lower().lstrip('0')
    torch.cuda.set_per_process_memory_fraction((48<<30)/torch.cuda.get_device_properties(0).total_memory)
    paths=dict(var_source=routes['author_root'])
    for d in routes['author_files']+[routes['dist']]:checked(d)
    for d in routes['files']:
        p=checked(d);paths['vae_checkpoint'if p.name=='vae_ch160v4096z32.pth'else'var_checkpoint'if p.name=='var_d16.pth'else'decoder_checkpoint']=str(p)
    ns=definitions(R/'src/var_comm/next_scale_prior.py',{'load_models','state_sha256'},dict(torch=torch,Path=pathlib.Path,sys=sys,np=np,hashlib=hashlib,PATCH_NUMS=SIZES,VOCAB_SIZE=4096))
    vae,var=ns['load_models'](paths,torch.device('cuda:0'))
    cls=definitions(R/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py',{'ContinuousDecoder'},dict(torch=torch,nn=torch.nn,copy=copy))['ContinuousDecoder']
    dc=cls(vae).cuda();cp=torch.load(paths['decoder_checkpoint'],map_location='cpu',weights_only=True);dc.load_state_dict(cp['decoder'],strict=True);del cp
    dc.eval().requires_grad_(False);states={k:ns['state_sha256'](v)for k,v in [('vae',vae),('var',var),('decoder',dc)]};assert states==routes['models']
    prior=definitions(R/'experiments/scale-causal-partial-residual-20261002/partial_receiver.py',{'_Prior'},dict(torch=torch,np=np,SIZES=SIZES))['_Prior']
    native=types.SimpleNamespace(torch=torch,receiver=types.SimpleNamespace(_Prior=prior),loaded=dict(vae=vae,var=var,decoder=dc,device=torch.device('cuda:0'),identity=dict(models=states)),flags=dict(deterministic=True,matmul_tf32=False,cudnn_tf32=False,precision='highest'))
    save(out/'native_identity.json',dict(states=states,runtime=actual,pci=pci,GPU_name=torch.cuda.get_device_name(0),GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),TF32=False,precision='FP32'))
    return native,dc,ns['state_sha256']
def direct_latents(native,tokens):
    t=native.torch;q=native.loaded['vae'].quantize;tokens=t.as_tensor(tokens,dtype=t.long,device='cuda')
    if tokens.ndim==1:tokens=tokens[None]
    f=t.zeros(len(tokens),32,16,16,device='cuda');coarse=None
    with t.no_grad():
        for s in range(10):
            v=tokens[:,OFFSETS[s]:OFFSETS[s+1]];emb=q.embedding(v).transpose(1,2).reshape(-1,32,SIZES[s],SIZES[s])
            f,_=q.get_next_autoregressive_input(s,10,f,emb)
            if s==3:coarse=f.clone()
    return f,coarse
def coarse_latent(native,received):
    t=native.torch;q=native.loaded['vae'].quantize;v=t.as_tensor(received,dtype=t.long,device='cuda')[None];assert v.shape==(1,OFFSETS[4])
    f=t.zeros(1,32,16,16,device='cuda')
    with t.no_grad():
        for s in range(4):
            emb=q.embedding(v[:,OFFSETS[s]:OFFSETS[s+1]]).transpose(1,2).reshape(1,32,SIZES[s],SIZES[s]);f,_=q.get_next_autoregressive_input(s,10,f,emb)
    return f
def load_lpips(reg):
    import torch,lpips
    m=lpips.LPIPS(net='alex',pnet_rand=True,model_path=str(checked(reg['lpips_linear'])),verbose=False)
    w=torch.load(checked(reg['alexnet']),map_location='cpu',weights_only=True);m.net.load_state_dict({n:w['features.'+n.split('.',1)[1]]for n in m.net.state_dict()},strict=True)
    return m.cuda().eval().requires_grad_(False)
