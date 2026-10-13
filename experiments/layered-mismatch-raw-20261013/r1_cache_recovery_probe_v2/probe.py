"""Once-only RX cache/import/flags qualification; no tensor, model or channel work."""
from __future__ import annotations
import argparse, ast, collections, copy, hashlib, importlib.abc, json, os, platform, sys, tempfile, traceback, types
from pathlib import Path

sys.dont_write_bytecode=True
BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam')
ROOT=BASE/'code/liulu/code/VAR_COMM'
RT=BASE/'code/liulu/var_comm_runtime_20261010'
TEMP_ROOT=BASE/'liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013'
OUT=TEMP_ROOT/'r1_cache_recovery_probe_v2'
SOURCE=ROOT/'experiments/layered-mismatch-raw-20261013/r1_execution_resource_v2/receiver_visual.py'
SOURCE_SHA='55cd58863018e30ce92e501148c225fd4b412ac1c99c274d3ea0b63f964e2a9f'
BUNDLE=ROOT/'results/layered_mismatch_raw_20261013/r1_execution_v1/registered/model_bundle.json'
BUNDLE_SHA='c1d25aaf76fdb36098e225c8dba1157b0d4ea92c9afccb39f348e2121c87a0a0'
FLAGS=dict(matmul_tf32=False,cudnn_tf32=False,precision='highest',cudnn_benchmark=False,
           cudnn_deterministic=True,deterministic=True,threads=6,interop_threads=2)
SCHEMA='R1_RX_CACHE_ZERO_MODEL_QUALIFICATION_V2'
FACTORIES=frozenset('tensor as_tensor from_numpy frombuffer empty empty_like empty_strided zeros zeros_like ones ones_like full full_like rand rand_like randn randn_like randint randint_like randperm arange range linspace logspace eye scalar_tensor sparse_coo_tensor sparse_csr_tensor sparse_csc_tensor sparse_bsr_tensor sparse_bsc_tensor _empty_affine_quantized _empty_per_channel_affine_quantized'.split())

def require(value,message):
    if not value:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def pin(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)

def checked(value):
    path=Path(value['path']);require(path.is_absolute() and path.is_file(),'Missing bound software metadata')
    require(sha(path)==value['sha256'],'Bound file changed: '+str(path))
    if 'bytes' in value:require(type(value['bytes']) is int and value['bytes']==path.stat().st_size,'Bound size changed')
    return path

def save(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def absent(path):
    try:Path(path).lstat()
    except FileNotFoundError:return
    raise FileExistsError('Existing probe is preserved; no implicit retry: '+str(path))

def extracted_definitions(text):
    """Exact original firewall, numeric reader and eight pre-CUDA flag statements."""
    tree=ast.parse(text)
    firewall=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SourceReadFirewall')
    numeric=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='numeric_flags')
    model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IsolatedModels')
    init=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    calls={'torch.set_num_threads','torch.set_num_interop_threads','torch.use_deterministic_algorithms','torch.set_float32_matmul_precision'}
    targets={'torch.backends.cuda.matmul.allow_tf32','torch.backends.cudnn.allow_tf32','torch.backends.cudnn.benchmark','torch.backends.cudnn.deterministic'}
    nodes=[]
    for node in init.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in calls:nodes.append(node)
        if isinstance(node,ast.Assign) and len(node.targets)==1 and ast.unparse(node.targets[0]) in targets:nodes.append(node)
    require(len(nodes)==8,'Original numeric flag prefix changed')
    expected=ast.parse('torch.set_num_threads(6)\ntorch.set_num_interop_threads(2)\ntorch.backends.cuda.matmul.allow_tf32=False\ntorch.backends.cudnn.allow_tf32=False\ntorch.backends.cudnn.benchmark=False\ntorch.backends.cudnn.deterministic=True\ntorch.use_deterministic_algorithms(True)\ntorch.set_float32_matmul_precision("highest")').body
    require([ast.dump(n,include_attributes=False) for n in nodes]==[ast.dump(n,include_attributes=False) for n in expected], 'Numeric statements differ from frozen prefix')
    function=ast.parse('def apply_numeric_flags(torch):\n    pass').body[0];function.body=nodes
    module=ast.fix_missing_locations(ast.Module(body=[firewall,numeric,function],type_ignores=[]))
    proof={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in module.body}
    return module,proof

def make_definitions(text,source,base=BASE,runtime=RT):
    module,proof=extracted_definitions(text)
    ns=dict(Path=Path,sys=sys,os=os,collections=collections,s=types.SimpleNamespace(BASE=base,RT=runtime))
    exec(compile(module,str(source),'exec'),ns)
    return ns,proof

def private_cache(output):
    """Only fresh child directories of the already reserved private output."""
    require(output.is_absolute() and output.is_dir() and not output.is_symlink(),'Private output is invalid')
    cache=output/'runtime_cache';absent(cache);cache.mkdir()
    dirs={name:cache/name for name in ('tmp','inductor','triton')}
    for path in dirs.values():path.mkdir()
    bindings={'TMPDIR':str(dirs['tmp']),'TEMP':str(dirs['tmp']),'TMP':str(dirs['tmp']),
              'TORCHINDUCTOR_CACHE_DIR':str(dirs['inductor']),'TRITON_CACHE_DIR':str(dirs['triton'])}
    os.environ.update(bindings);tempfile.tempdir=bindings['TMPDIR']
    return bindings

class PrivateWrites:
    """Additional probe-only restriction; original RX firewall is unchanged."""
    def __init__(self,output):self.output=output.resolve();self.writes=collections.Counter();self.blocked=[]
    def hook(self,event,args):
        paths=[]
        if event=='open' and args and not isinstance(args[0],int):
            mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
            write=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool((flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if write:paths=[args[0]]
        elif event in {'os.mkdir','os.remove','os.rmdir'}:paths=[args[0]]
        elif event in {'os.rename','os.link','os.symlink'}:paths=list(args[:2])
        elif event.startswith(('subprocess.','socket.')):
            self.blocked.append(event);raise RuntimeError('No process/network request in cache probe: '+event)
        for value in paths:
            if not isinstance(value,(str,bytes,os.PathLike)):continue
            path=Path(os.fsdecode(value)).resolve()
            if path!=self.output and self.output not in path.parents:
                self.blocked.append(str(path));raise RuntimeError('Probe write outside private output: '+str(path))
            if event=='os.symlink':
                self.blocked.append(str(path));raise RuntimeError('No new PFS symlinks in cache probe')
            self.writes[str(path)]+=1

class RejectExecution(importlib.abc.Loader):
    """Preserve real spec discovery but refuse executing a scientific package."""
    def __init__(self,guard,fullname,original_loader):
        self.guard=guard;self.fullname=fullname;self.original_loader=original_loader
    def create_module(self,spec):self.guard.reject('channel_import_requests',self.fullname+':create_module')
    def exec_module(self,module):self.guard.reject('channel_import_requests',self.fullname+':exec_module')

class NoScience(importlib.abc.MetaPathFinder):
    """Reject observed tensor/model/weight/CUDA/channel requests, including imports."""
    def __init__(self):
        self.counts={k:0 for k in ('tensor_requests','model_constructor_requests','weight_load_requests','cuda_init_requests','channel_import_requests')}
        self.rejected=[];self.metadata_lookups=[]
    def reject(self,kind,detail):
        self.counts[kind]+=1;self.rejected.append(dict(kind=kind,detail=detail))
        raise RuntimeError('Zero-science probe rejected '+kind+': '+detail)
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] not in {'sionna','tensorflow','lpips','var_comm','models'}:return None
        # Torch's trace-rule discovery legitimately asks find_spec('tensorflow').
        # Delegate to every real later finder without executing any loader and
        # preserve real presence/absence, origin and package search locations.
        finders=list(sys.meta_path);require(self in finders,'Guard must be installed before metadata discovery')
        for finder in finders[finders.index(self)+1:]:
            find=getattr(finder,'find_spec',None)
            require(find is not None,'Unsupported legacy finder in qualified interpreter')
            spec=find(fullname,path,target)
            if spec is not None:
                self.metadata_lookups.append(dict(fullname=fullname,found=True,origin=spec.origin,
                    loader_type=None if spec.loader is None else type(spec.loader).__module__+'.'+type(spec.loader).__qualname__))
                protected=copy.copy(spec);protected.loader=RejectExecution(self,fullname,spec.loader)
                return protected
        self.metadata_lookups.append(dict(fullname=fullname,found=False,origin=None,loader_type=None))
        return None
    def inspect_call(self,module,name,filename=''):
        if module.startswith('torch'):
            if name in FACTORIES:self.reject('tensor_requests',module+'.'+name)
            if module.startswith('torch.serialization') and name=='load':self.reject('weight_load_requests',module+'.'+name)
            if module.startswith('torch.cuda') and name in {'_lazy_init','init','set_device'}:self.reject('cuda_init_requests',module+'.'+name)
            if module=='torch.nn.modules.module' and name=='__init__':self.reject('model_constructor_requests',filename+':'+name)
    def profile(self,frame,event,arg):
        if event=='call':self.inspect_call(str(frame.f_globals.get('__name__','')),frame.f_code.co_name,frame.f_code.co_filename)
        elif event=='c_call':self.inspect_call(str(getattr(arg,'__module__','')),str(getattr(arg,'__name__','')))

def metadata_bundle(bundle):
    require(bundle['schema']=='R1_MODEL_ONLY_BUNDLE_V1' and bundle['no_source_or_TX_paths'] is True,'Not the source-free model bundle')
    require(bundle['numeric_flags']==FLAGS and bundle['expected_runtime']['python']=='3.10.12'
            and bundle['expected_runtime']['torch']=='2.11.0+cu128' and bundle['expected_runtime']['numpy']=='2.2.6','Original runtime identity differs')
    pins=bundle['runtime_files']
    require(pins and all(set(p)=={'path','sha256','bytes'} and type(p['bytes']) is int for p in pins),'Unbound runtime metadata')
    return pins

def run(bundle_sha):
    require(sys.platform.startswith('linux') and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Original Linux interpreter with CUDA hidden required')
    require(platform.python_version()=='3.10.12' and not any(k in sys.modules for k in ('torch','numpy','sionna','tensorflow','lpips','models','var_comm')), 'Fresh original visual interpreter required')
    require(sha(SOURCE)==SOURCE_SHA,'Original RX source changed')
    require(bundle_sha==BUNDLE_SHA,'Explicit original source-free bundle SHA required')
    checked(dict(path=str(BUNDLE),sha256=bundle_sha));bundle=json.loads(BUNDLE.read_text());software=metadata_bundle(bundle)
    ns,proof=make_definitions(SOURCE.read_text(encoding='utf8'),SOURCE)
    absent(OUT);OUT.mkdir();bindings=private_cache(OUT)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='6'
    own=Path(__file__).resolve();firewall=ns['SourceReadFirewall']([p['path'] for p in software],OUT,[SOURCE,BUNDLE,own],{str(SOURCE):SOURCE_SHA,str(own):sha(own)})
    requests=NoScience();storage=PrivateWrites(OUT);sys.meta_path.insert(0,requests);sys.addaudithook(firewall.hook);sys.addaudithook(storage.hook)
    result=dict(schema=SCHEMA,status='STARTED_ENGINEERING_ONLY',source=pin(SOURCE),bundle=pin(BUNDLE),probe=pin(own),
        exact_extracted_AST=proof,cache_environment=bindings,tempfile_tempdir=tempfile.tempdir,
        model_or_channel_calls=0,qualification_frames=0,source_truth_reads=0,model_weight_bytes_read=0,
        actual_scientific_registration_created=False,CUDA_VISIBLE_DEVICES='',limits='Python request/open audit, not OS sandbox or numerical model qualification')
    try:
        require(sys.getprofile() is None,'Existing profiler would change this qualification')
        sys.setprofile(requests.profile)
        import numpy as np
        import torch
        require(not torch.cuda.is_initialized(),'CUDA initialized on import')
        result['cuda_initialized_before_flags']=torch.cuda.is_initialized()
        # A dispatcher mode denies all ATen operations during the exact original
        # settings. The pre-import profile also denies public tensor factories.
        from torch.utils._python_dispatch import TorchDispatchMode
        class DenyTensorOps(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                requests.reject('tensor_requests',str(func))
        with DenyTensorOps():ns['apply_numeric_flags'](torch)
        observed=ns['numeric_flags'](torch);require(observed==FLAGS,'Original flags do not match after setting')
        require(not torch.cuda.is_initialized(),'CUDA initialized by original flags')
        result['cuda_initialized_after_flags']=torch.cuda.is_initialized()
        require(torch.__version__=='2.11.0+cu128' and np.__version__=='2.2.6','Imported runtime version differs')
        original={str(Path(p['path']).resolve()):p for p in software};actual=[]
        for path in (sys.executable,np.__file__,torch.__file__,torch._C.__file__):
            resolved=str(Path(path).resolve());require(resolved in original,'Imported runtime not in pinned projection')
            value=dict(original[resolved],path=resolved);checked(value);actual.append(value)
        sys.setprofile(None)
        require(not any(requests.counts.values()) and not firewall.blocked and not storage.blocked,'Swallowed scientific or denied file request')
        require(tempfile.gettempdir()==bindings['TMPDIR'],'Temporary directory drifted')
        cache_paths=[str(p.relative_to(OUT)) for p in sorted((OUT/'runtime_cache').rglob('*'))]
        require(all(not p.is_symlink() for p in (OUT/'runtime_cache').rglob('*')),'Cache produced unexpected symbolic link')
        result.update(status='PASS_ZERO_MODEL_CACHE_IMPORT_FLAGS',numeric_flags=observed,actual_runtime_files=actual,
            blocked=firewall.blocked,opened_paths=dict(firewall.reads),request_counts=requests.counts,
            rejected_requests=requests.rejected,metadata_lookups=requests.metadata_lookups,cache_members=cache_paths,private_writes=dict(storage.writes),storage_blocked=storage.blocked,scientific_calls=0)
        save(OUT/'completion.json',result)
    except BaseException as error:
        sys.setprofile(None)
        result.update(status='FAILED_ENGINEERING_PROBE_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
            blocked=firewall.blocked,opened_paths=dict(firewall.reads),request_counts=requests.counts,rejected_requests=requests.rejected,metadata_lookups=requests.metadata_lookups,
            private_writes=dict(storage.writes),storage_blocked=storage.blocked)
        save(OUT/'failure.json',result);raise
    print(json.dumps(dict(status=result['status'],completion=pin(OUT/'completion.json'),scientific_calls=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model-bundle-sha256',required=True)
    args=parser.parse_args();run(args.model_bundle_sha256)
