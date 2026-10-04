"""Pinned Sionna 5G NR implementation; no replacement LDPC construction/decoder."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import inspect
import os
from pathlib import Path
import numpy as np
from uep_common import identity,require,read,sha

DECODER = dict(cn_update='boxplus-phi',vn_update='sum',cn_schedule='flooding',
               num_iter=20,llr_max=20.0,hard_out=True,return_infobits=True,
               prune_pcm=True,harq_mode=False,precision='single',early_stop=False)
BIT_MAPPING='component_Gray_00_-3_01_-1_10_3_11_1_over_sqrt5;QPSK_0_plus1_1_minus1'
SCRAMBLING='public_PCG64_SHA256_protocol_session_frame_group_counter_after_rate_matching_v1'
POWER='fixed_constellation_average_Es2'

class SionnaBackend:
    def __init__(self,device='cpu',qualification=None):
        import sionna,torch
        from sionna.phy.fec.ldpc import LDPC5GEncoder,LDPC5GDecoder
        self.torch=torch;self.Encoder=LDPC5GEncoder;self.Decoder=LDPC5GDecoder;self.device=device
        require(sionna.__version__=='2.2.0','Pinned Sionna 2.2.0 required')
        self.identity=dict(implementation='Sionna_5G_LDPC',version=sionna.__version__,
            encoder_source_sha256=sha(inspect.getfile(LDPC5GEncoder)),decoder_source_sha256=sha(inspect.getfile(LDPC5GDecoder)),
            decoder=DECODER,bit_mapping=BIT_MAPPING,scrambling=SCRAMBLING,power_protocol=POWER,
            precision='float32',rate_matching='Sionna RV0; filler removal, first2Z and tail puncturing; supported n only',
            cuda=torch.version.cuda,torch=torch.__version__)
        self.qualified=False
        if qualification and Path(qualification).exists():
            q=read(qualification);require(q['backend_identity']==self.identity and q['status']=='PASS','LDPC qualification identity differs')
            self.qualified=True

    @lru_cache(maxsize=20000)
    def plan(self,k,n,num_bits_per_symbol):
        from profiles import UnsupportedConfiguration
        k,n,q=int(k),int(n),int(num_bits_per_symbol)
        if q not in (2,4) or n%q or k<12 or n<k:
            raise UnsupportedConfiguration('Single-codeblock k/n or modulation alignment invalid')
        try:e=self.Encoder(k,n,num_bits_per_symbol=q,device='cpu',precision='single')
        except (ValueError,AssertionError) as error:
            # This is a resource admission request. A thrown constructor is retained verbatim.
            raise UnsupportedConfiguration(type(error).__name__+': '+str(error)) from error
        require(e.k==k and e.n==n and e.k_ldpc-e.k_filler==k,'Encoder information dimensions differ')
        if n>e.n_cb_comp:
            raise UnsupportedConfiguration('Repetition outside this pinned encoder RV0 output is unsupported')
        layout=dict(k=k,n=n,k_ldpc=int(e.k_ldpc),k_filler=int(e.k_filler),n_cb=int(e.n_cb),
            n_cb_comp=int(e.n_cb_comp),bg=str(e._bg),z=int(e.z),code_blocks=1,
            mother_bits=int(e.n_ldpc),puncturing_bits=int(e.n_ldpc-e.k_filler-n),
            puncturing_first_2Z=int(2*e.z),puncturing_remaining=int(e.n_cb_comp-n),
            shortening_bits=int(e.k_filler),repetition_bits=0,modulation_padding_bits=0,
            interleaver_sha256=hashlib.sha256(e.out_int.cpu().numpy().tobytes()).hexdigest(),
            decoder_config=DECODER,bit_mapping=BIT_MAPPING,scrambling=SCRAMBLING,power_protocol=POWER,
            actual_effective_rate=k/n,implementation=self.identity)
        layout['layout_id']=identity(layout);return layout

    @lru_cache(maxsize=8)
    def codecs(self,k,n,q):
        e=self.Encoder(int(k),int(n),num_bits_per_symbol=int(q),precision='single',device=self.device)
        d=self.Decoder(e,**{k:v for k,v in DECODER.items() if k!='early_stop'},device=self.device)
        e.eval();d.eval();return e,d

    def encode(self,bits,n,q):
        t=self.torch;bits=t.as_tensor(bits,dtype=t.float32,device=self.device)
        with t.inference_mode():return self.codecs(bits.shape[-1],n,q)[0](bits)

    def decode(self,logits,k,n,q):
        t=self.torch;logits=t.as_tensor(logits,dtype=t.float32,device=self.device)
        with t.inference_mode():return self.codecs(k,n,q)[1](logits)

def make_backend():
    root=Path(os.environ.get('VAR_COMM_ROOT','/home/liulu/projects/VAR_COMM'))
    q=root/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json'
    return SionnaBackend('cpu',q)
