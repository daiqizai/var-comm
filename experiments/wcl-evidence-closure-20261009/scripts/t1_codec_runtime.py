"""Actual entropy-only TX/RX interface for T1; never accepts RX truth.

The caller owns GPU admission, packet framing and all failure-output policy.
This module constructs no model unless build_frozen_native is explicitly called.
"""
from __future__ import annotations
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
from t1_entropy_core import OFFSETS,SIZES,load_integer_codec,read,require,sha,source_tokens,verify

MODEL_STATES={'vae':'6830fc533a34d52c8f765a7b212782a5b5ef6138992099e8a817ce920fe6fe2b',
              'var':'b82cb0855b24e1d0d8d0aacebbb2d2d6728c2dcc6509b667dfa697c9330347d5',
              'decoder':'bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1'}

class InvalidSourceStream(ValueError):
    """Actual received source syntax fails; caller may apply frozen fallback."""

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value; spec.loader.exec_module(value); return value

def check_native(native):
    for key,expected in MODEL_STATES.items():
        require(native.loaded['identity']['models'][key]==expected,'Frozen visual model state changed: '+key)
    for key,expected in {'deterministic':True,'matmul_tf32':False,'cudnn_tf32':False,'precision':'highest'}.items():
        require(native.flags[key]==expected,'Frozen visual numerical flag changed: '+key)

def build_frozen_native(root,h_completion,signal_handler):
    """EXPLICIT GPU construction; caller must have already admitted an owned job.

    Source implementations are compared with the original H completion. Frozen
    native construction checks original model identities and numerical flags.
    """
    root=Path(root); done=read(h_completion)
    require(done['status']=='H_FULL1000_SOURCE_CODEC_COMPLETE','Missing original H source identity')
    native_dir=root/'experiments/m1-n2048-full-grid-20261004'
    uep=root/'experiments/prior-aware-grouped-mcs-v1'
    needed=list(native_dir.glob('m1_*.py'))+[uep/'quality_driver.py',uep/'source_quality.py',
        root/'experiments/content-real-64qam-20261006/h_source_driver.py',
        root/'experiments/scale-causal-partial-residual-20261002/partial_receiver.py']
    bindings={**done['source_bindings'],**done['input_bindings']}
    for path in needed:
        require(str(path) in bindings,'Unbound original visual implementation: '+str(path));verify(path,bindings[str(path)])
    sys.path.insert(0,str(uep)); sys.path.insert(0,str(root/'src'))
    quality=module(uep/'quality_driver.py','_wcl_t1_original_quality_driver')
    native=quality.build_native(root,str(native_dir),signal_handler)
    check_native(native); return native

class SourceCodec:
    def __init__(self,root,static_completion=None,native=None):
        self.root=Path(root);self.entropy,self.Encoder,self.Decoder=load_integer_codec(root)
        self.native=None;self.static_cdf=None;self.static_identity=None
        if static_completion is not None:
            done=read(static_completion)
            require(done['status']=='T1_STATIC_TRAIN20K_CDF_COMPLETE' and done['source_count']==20000,
                    'Frozen original train20k CDF required')
            require(done['source_role']=='original_train20k' and done['calibration_used_for_fitting'] is False,
                    'Static CDF fitting population differs')
            require(len(done['outputs'])==1,'Static output binding ambiguous')
            p,s=next(iter(done['outputs'].items()));verify(p,s)
            with np.load(p,allow_pickle=False) as z:
                self.static_cdf=self.entropy.validate_cdf(z['cdf']).copy()
                require(self.static_cdf.shape==(10,4097),'Static table scale count differs')
                require(np.array_equal(self.static_cdf,self.entropy.probability_cdf(np.log(z['counts'].astype(np.float64)+.5))),
                        'Static CDF differs from registered counts and smoothing')
            self.static_identity=sha(static_completion)
        if native is not None:self.bind_native(native)

    def bind_native(self,native):
        check_native(native); self.native=native
        path=self.root/'experiments/content-real-64qam-20261006/h_source_driver.py'
        verify(path,'b38f8451ca2d94a8ceb4381df3647e327dba38e3d4cf969aa15597bc47c342ce')
        sys.path.insert(0,str(path.parent))
        original=module(path,'_wcl_t1_bound_h_source_driver')
        self.provider_factory=lambda:original.IndependentProvider(native,types.SimpleNamespace(cdf_function=self.entropy.probability_cdf))

    def _context(self,family):
        if family=='EC_STATIC_WHOLE':
            require(self.static_cdf is not None,'Static source model not loaded')
            return nullcontext(None)
        require(family=='EC_VAR_WHOLE' and self.native is not None,'VAR source model not loaded')
        return self.provider_factory()

    def _inference(self,family):
        return nullcontext() if family=='EC_STATIC_WHOLE' else self.native.torch.no_grad()

    def _cdf(self,family,provider,scale):
        cdf=np.broadcast_to(self.static_cdf[scale],(SIZES[scale]**2,4097)) if family=='EC_STATIC_WHOLE' else provider.cdf()
        return self.entropy.validate_cdf(cdf,rows=SIZES[scale]**2)

    def decode(self,family,payload_bits,m):
        """Return only recovered tokens and parse evidence; no source ID or truth.

        Caller has already recovered paid profile/length and checked packet CRC.
        Catch only InvalidSourceStream as source parsing failure. Model/software
        exceptions remain errors, not gray channel events.
        """
        require(type(m) is int and 4<=m<=9,'T1 registered complete prefixes m4..m9 only')
        a=np.asarray(payload_bits)
        if a.ndim!=1 or len(a)<2 or not np.isin(a,(0,1)).all():
            raise InvalidSourceStream('Invalid arithmetic bit vector')
        a=a.astype(np.uint8,copy=True)
        class Bounded(self.Decoder):
            def read_bit(inner):
                if inner.position>=len(inner.bits)+30:
                    raise InvalidSourceStream('Arithmetic terminal lookahead exceeded')
                return super().read_bit()
        decoder=Bounded(a); canonical=self.Encoder(); recovered=[]
        with self._inference(family),self._context(family) as provider:
            for scale in range(m):
                cdf=self._cdf(family,provider,scale)
                try: values=decoder.decode(cdf)
                except InvalidSourceStream: raise
                except ValueError as error: raise InvalidSourceStream(str(error)) from error
                canonical.encode(values,cdf); recovered.append(values.copy())
                if provider is not None:provider.advance(values.copy())
        if not np.array_equal(canonical.finish(),a):
            raise InvalidSourceStream('Noncanonical arithmetic termination or length')
        if decoder.position-len(a)!=30:raise InvalidSourceStream('Unexpected terminal lookahead')
        return dict(received_tokens=np.concatenate(recovered).astype(np.int64,copy=False),
            source_status='ARITHMETIC_SOURCE_DECODED',canonical=True,zero_extension_reads=30,
            transmitted_source_bits=len(a),source_family=family,m=m,K=0,
            receiver_truth_used=False,transmitter_probability_table_used=False)

    def encode(self,family,tokens,modes=(4,5)):
        """Fresh TX probability work. Includes prefix flush; no raw substitution."""
        a=source_tokens(tokens); modes=tuple(modes)
        require(modes and tuple(sorted(set(modes)))==modes and all(4<=m<=9 for m in modes),
                'Ordered registered complete prefix modes required')
        encoder=self.Encoder();result={}
        with self._inference(family),self._context(family) as provider:
            for scale in range(max(modes)):
                values=a[OFFSETS[scale]:OFFSETS[scale+1]];cdf=self._cdf(family,provider,scale)
                encoder.encode(values,cdf)
                if provider is not None:provider.advance(values.copy())
                if scale+1 in modes:
                    bits=np.asarray(encoder.finish(),dtype=np.uint8)
                    result[scale+1]=dict(bits=bits,raw_bits=int(12*OFFSETS[scale+1]),
                        arithmetic_bits=len(bits),flush_bits=len(bits)-len(encoder.bits))
        return result
