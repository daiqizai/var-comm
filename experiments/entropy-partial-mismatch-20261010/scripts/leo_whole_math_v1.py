"""New assembly of SHA-bound original mathematical definitions, batch one.

Imported only by the explicitly owned GPU worker. No original loader/guard is
patched. AST-selected definitions preserve their original bodies verbatim.
"""
from __future__ import annotations
import ast
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace


def definitions(path, names, namespace):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if len(nodes)!=len(names) or {n.name for n in nodes}!=set(names):raise RuntimeError('Original definition missing')
    # No original module-level configuration, path assignment or model load runs.
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def native_linkage(environment,gate,maps_path=Path('/proc/self/maps'),previous=None):
    """Two read-only snapshots, no model calls or old-host identity claim."""
    paths=set()
    for line in Path(maps_path).read_text(encoding='utf-8').splitlines():
        parts=line.split(None,5)
        if len(parts)==6 and parts[5].startswith('/') and '.so' in Path(parts[5]).name:
            paths.add(parts[5])
    gate.require(paths,'Actual loaded native libraries were not observable')
    old_host={str(Path(r['path']).resolve()):r for r in environment.host_native}
    rows=[]
    for name in sorted(paths):
        p=Path(name);gate.require(p.is_file(),'Loaded native library unavailable: '+name)
        value=dict(path=name,sha256=gate.sha(p),bytes=p.stat().st_size)
        if name in environment.rows:
            expected=environment.rows[name];origin='original_runtime_file_projected'
        elif str(p.resolve()) in old_host:
            expected=old_host[str(p.resolve())];origin='already_observed_new_host_library'
        else:
            # CUDA-hidden import checks cannot observe every lazily loaded driver.
            # This allowlist admits observation only, never a claim of original bytes.
            gate.require(p.name.startswith(('libcuda.so','libnvidia-')),'Unregistered non-driver native library: '+name)
            expected=value;origin='additional_new_host_driver_observation'
        gate.require((value['sha256'],value['bytes'])==(expected['sha256'],expected['bytes']),'Actual loaded native library pin differs: '+name)
        rows.append(value|dict(origin=origin,old_host_byte_identity_claimed=False))
    if previous is not None:
        # Also reverify files which were dlclosed after the first observation.
        for old in previous['loaded_libraries']:
            p=Path(old['path']);gate.require(p.is_file() and p.stat().st_size==old['bytes'] and gate.sha(p)==old['sha256'],
                'Observed native library changed during this gate: '+str(p))
    return dict(schema='LEO_GATE_NATIVE_LINKAGE_OBSERVATION_V1',loaded_libraries=rows,
        previous_observed_files_reverified=previous is not None,old_host_closure_exact=False,
        new_driver_libraries_are_new_host_facts=True,model_calls=0)

def cuda_driver_pci_bus_id(driver=None):
    """Read logical device0 identity after torch initialization; no tensor/model calls."""
    driver=ctypes.CDLL('libcuda.so.1') if driver is None else driver
    get_device=driver.cuDeviceGet;get_device.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int];get_device.restype=ctypes.c_int
    get_pci=driver.cuDeviceGetPCIBusId;get_pci.argtypes=[ctypes.POINTER(ctypes.c_char),ctypes.c_int,ctypes.c_int];get_pci.restype=ctypes.c_int
    device=ctypes.c_int();code=get_device(ctypes.byref(device),0)
    if code!=0:raise RuntimeError('cuDeviceGet(0) failed: '+str(code))
    bus=ctypes.create_string_buffer(64);code=get_pci(bus,len(bus),device.value)
    if code!=0:raise RuntimeError('cuDeviceGetPCIBusId failed: '+str(code))
    return bus.value.decode('ascii')


class Backend:
    def __init__(self,resolver,environment,spec,extra,ledger,boundary,out):
        import numpy as np
        import torch
        import leo_whole_gate_v1 as g
        self.g,self.np,self.t=g,np,torch;self.ledger=ledger;self.boundary=boundary;self.out=Path(out)
        self.source_index=0;self.tx_cdfs={};self.received={};self.traces=[];self.current=None
        g.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS workspace environment changed')
        g.check_cuda_environment()
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            g.require(os.environ.get(key)=='6','Original six-thread environment required: '+key)
        torch.set_num_threads(6);torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        g.require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one CUDA device required')
        torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
        actual_uuid=str(getattr(props,'uuid',''))
        identity=g.validate_cuda_device_identity(actual_uuid,cuda_driver_pci_bus_id())
        g.write(self.out/'cuda_device_identity.json',dict(**identity,
            configured_device_binding=g.device_identity_binding(),CUDA_DEVICE_ORDER=os.environ['CUDA_DEVICE_ORDER'],
            CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],identity_queries_only=True))
        g.require(len(os.sched_getaffinity(0))==2,'Two-CPU scheduling mask required')
        cap=16*(1<<30);g.require(props.total_memory>cap,'Unexpected device memory')
        torch.cuda.set_per_process_memory_fraction(cap/props.total_memory,0)
        torch.cuda.reset_peak_memory_stats(0)
        observed=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
            cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads())
        g.require(observed==g.EXPECTED_RUNTIME,'Frozen scientific runtime differs: '+repr(observed))
        imported=[Path(sys.executable),Path(torch.__file__),Path(torch._C.__file__),Path(np.__file__)]
        for p in imported:environment.path_for_new(p)
        for p in sys.path:
            if p and Path(p).exists():g.inside(Path(p).absolute())
        g.write(self.out/'numerical_runtime.json',dict(**observed,actual_GPU_UUID=actual_uuid,device_identity=identity,device=props.name,
            device_total_bytes=props.total_memory,affinity=sorted(os.sched_getaffinity(0)),
            CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],FP32=True,TF32=False,
            deterministic=True,allocator_cap_bytes=cap,allocator_cap_scope='Only this process PyTorch allocator; not total device/library allocation isolation',
            shared_GPU=True,timing_measurement=False,old_runtime_exact=False,
            preserved_math_threads=True,scheduling_amendment='Original6/2 retained; CPU affinity restricted to2'))
        self.boundary()
        ns=dict(torch=torch,Path=Path,sys=sys,np=np,hashlib=hashlib,PATCH_NUMS=g.SIZES,VOCAB_SIZE=4096)
        definitions(resolver.path(g.OLD+'src/var_comm/next_scale_prior.py',g.CODE['src/var_comm/next_scale_prior.py']),
            {'load_models','state_sha256'},ns)
        self.state_sha=ns['state_sha256']
        paths=dict(var_source=extra['author_root'],vae_checkpoint=str(resolver.path(g.WEIGHTS['vae'],g.WEIGHT_SHA['vae'])),
                   var_checkpoint=str(resolver.path(g.WEIGHTS['var'],g.WEIGHT_SHA['var'])))
        # Explicitly import the reviewed new candidate from its registered path.
        # It is not represented as a recovered old-runtime file.
        dependency=extra['native_dependency_candidate']
        g.require('dist' not in sys.modules,'Unregistered preimported dist module')
        g.require(g.sha(dependency['dist'])==g.UPSTREAM_DIST_SHA,'Candidate dist changed before import')
        dist=g.import_file(dependency['dist'],'dist')
        g.require(Path(dist.__file__)==Path(dependency['dist']) and dist.get_device()=='cuda' and
                  dist.initialized() is False and not torch.distributed.is_initialized(),'Candidate distributed/device state differs')
        g.write(self.out/'native_dependency_admission.json',dict(**dependency,
            actual_module=str(dist.__file__),initialized=dist.initialized(),get_device=dist.get_device(),
            init_process_group_called=False,old_dependency_byte_identity_proven=False))
        device=torch.device('cuda:0');vae,var=ns['load_models'](paths,device)
        # Ensure every actually imported author model module came from the mapped original seven files.
        author_dir=Path(extra['author_root'])/'models'
        for name,mod in list(sys.modules.items()):
            if name=='models' or name.startswith('models.'):
                p=Path(mod.__file__);g.require(p.parent==author_dir and p.suffix=='.py','Unexpected author module origin')
                resolver.path(g.AUTHOR+p.name)
        dn=dict(torch=torch,nn=torch.nn,copy=copy)
        definitions(resolver.path(g.OLD+'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py'),{'ContinuousDecoder'},dn)
        checkpoint=torch.load(resolver.path(extra['decoder']),map_location='cpu',weights_only=True)
        decoder=dn['ContinuousDecoder'](vae).to(device);decoder.load_state_dict(checkpoint['decoder'],strict=True)
        del checkpoint
        self.models=dict(vae=vae,var=var,decoder=decoder)
        for model in self.models.values():model.eval().requires_grad_(False)
        self.verify_models()
        prior_ns=dict(torch=torch,np=np,SIZES=g.SIZES)
        definitions(resolver.path(g.OLD+'experiments/scale-causal-partial-residual-20261002/partial_receiver.py'),{'_Prior'},prior_ns)
        OriginalPrior=prior_ns['_Prior'];owner=self
        class CountedPrior(OriginalPrior):
            def logits(inner,k):
                return ledger.call('prior_scale',lambda:super(CountedPrior,inner).logits(k),source=owner.source_index,scale=k)
        class CountedDecoder:
            def __call__(inner,latent):
                return ledger.call('decoder_forward',lambda:decoder(latent),source=owner.source_index)
        flags=dict(deterministic=torch.are_deterministic_algorithms_enabled(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,precision=torch.get_float32_matmul_precision())
        self.native=SimpleNamespace(torch=torch,receiver=SimpleNamespace(_Prior=CountedPrior),
            loaded=dict(vae=vae,var=var,decoder=CountedDecoder(),device=device,identity=dict(models=dict(g.MODELS))),flags=flags)
        pn=dict(np=np,time=time,SIZES=g.SIZES,require=g.require,token_count=lambda m,K=0:g.OFFSETS[m]+K)
        definitions(resolver.path(g.OLD+'experiments/content-real-64qam-20261006/h_source_driver.py'),{'IndependentProvider','render_received'},pn)
        self.render_function=pn['render_received'];OriginalProvider=pn['IndependentProvider']
        core_path=resolver.path(g.OLD+'experiments/wcl-evidence-closure-20261009/scripts/t1_entropy_core.py')
        core=g.import_file(core_path,'t1_entropy_core')
        codec_module=g.import_file(resolver.path(g.OLD+'experiments/wcl-evidence-closure-20261009/scripts/t1_codec_runtime.py'),'_leo_old_codec')
        self.codec=codec_module.SourceCodec(spec['project_root'])
        codec_module.check_native(self.native)
        self.codec.native=self.native
        class TracedProvider(OriginalProvider):
            def cdf(inner):
                result=super().cdf();digest=g.image_sha(result)
                role,m=owner.current;entry=dict(source=owner.source_index,role=role,m=m,scale=inner.scale,cdf_sha256=digest)
                owner.traces.append(entry)
                if role=='TX':owner.tx_cdfs[inner.scale]=digest
                else:
                    # Equality is an external engineering assertion; no TX table supplies RX probabilities.
                    if owner.tx_cdfs.get(inner.scale)!=digest:
                        g.write(owner.out/'first_CDF_mismatch.json',entry|dict(expected_TX_CDF_sha256=owner.tx_cdfs.get(inner.scale),old_CDF_unobservable=True))
                        raise g.GateMismatch('new_host_TX_RX_CDF','New-host independent RX CDF differs from new TX; old CDF unavailable')
                return result
        self.codec.provider_factory=lambda:TracedProvider(self.native,SimpleNamespace(cdf_function=self.codec.entropy.probability_cdf))
        self.verify_models();self.boundary()
        self.environment=environment
        self.native_start=native_linkage(environment,g)
        g.write(self.out/'native_linkage_loaded.json',self.native_start)

    def verify_models(self):
        for name,model in self.models.items():
            self.g.require(self.state_sha(model)==self.g.MODELS[name],'Frozen model state changed: '+name)
            self.g.require(not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters()),'Model acquired training state')

    def encode_pixels(self,pixels):
        self.boundary();t=self.t;vae=self.models['vae']
        # Exact original encode_flat expression, cached uint8 CHW input, batch1.
        with t.no_grad():
            image=t.as_tensor(pixels[None],dtype=t.float32,device='cuda:0')/127.5-1
            f=vae.quant_conv(vae.encoder(image))
            tokens=t.cat(vae.quantize.f_to_idxBl_or_fhat(f,to_fhat=False),1)[0].cpu().numpy()
        return self.np.ascontiguousarray(tokens,dtype=self.np.int64)

    def source_tx(self,tokens):
        self.boundary();self.current=('TX',9);self.tx_cdfs={};self.received={}
        result=self.codec.encode('EC_VAR_WHOLE',tokens,modes=(4,5,6,7,8,9))
        self.g.require(set(self.tx_cdfs)==set(range(9)),'TX CDF trace incomplete');return result

    def source_rx(self,bits,m):
        self.boundary();self.current=('RX',m)
        result=self.codec.decode('EC_VAR_WHOLE',bits,m)
        self.received[m]=result['received_tokens'].copy();return result

    def render(self,m):
        self.boundary();self.g.require(m in self.received,'Render requires actual independent RX state')
        with self.t.no_grad():return self.render_function(self.native,self.received[m].copy(),m,0)

    def end_source(self):
        self.g.require(set(self.received)==set(range(4,10)),'Incomplete source RX endpoints')
        self.g.write(self.out/f'cdf_trace_source{self.source_index}.json',dict(new_host_TX_RX_traces=self.traces,old_CDF='UNOBSERVABLE_IN_SELECTED_SEED'))
        self.traces=[];self.tx_cdfs={};self.received={};self.source_index+=1;self.boundary()

    def close(self):
        self.t.cuda.synchronize();self.verify_models();self.boundary()
        self.g.write(self.out/'native_linkage_end.json',native_linkage(self.environment,self.g,previous=self.native_start))

    def failure_evidence(self):
        # No further model/GPU calls; retain the trace already observed before stopping.
        self.g.write(self.out/'cdf_trace_at_failure.json',dict(source_index=self.source_index,
            current_endpoint=self.current,new_host_TX_RX_traces=self.traces,
            old_CDF='UNOBSERVABLE_IN_SELECTED_SEED',additional_model_calls=0))
