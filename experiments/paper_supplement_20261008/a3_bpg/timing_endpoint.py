"""Fresh native256 BPG endpoints for the independent A3 fixed16 measurement.

Construction and finish run outside timing. All source coding, QP search, real
FEC and complete-frame modulation are inside encode. Only actual received IQ
and public context enter receive. No codec or reconstruction disk cache is read.
"""
import base64
from pathlib import Path
import sys
import numpy as np


class BPGEndpoint:
    def __init__(self,r,calls,helper):
        import torch
        self.r,self.calls,self.helper,self.t=r,calls,helper,torch
        helper.configure(torch)
        b=r['bpg'];require,sha=helper.require,helper.sha
        cfg=helper.descriptor(b['original_request'])
        freeze=helper.descriptor(b['freeze'])
        parity=helper.descriptor(b['parity_completion'])
        build=helper.descriptor(b['build_receipt'])
        require(parity['status']=='COMPLETE' and parity['source_count']==20 and parity['case_count']==40
            and parity['complete_byte_identity_all'] and parity['exact_uint8_pixel_identity_all'], 'Actual20cal memory parity required')
        require(build['library']==parity['library'] and build['library_sha256']==parity['library_sha256']==sha(build['library']), 'Qualified memory library changed')
        require(freeze['status']=='BPG_POLICIES_FROZEN_ON_CALIBRATION_ONLY_V1'
            and freeze['holdout_used_for_selection'] is False
            and freeze['request_sha256']==b['original_request']['sha256'],'Original BPG policy/request changed')
        require(cfg['SNRs']==[1,4,7,10,13,19] and cfg['noise_seeds']==[2001,2002,2003], 'Frozen BPG working points changed')
        self.policies=freeze['policies'];self.cfg=cfg
        # Reuse the exact already-admitted pure Sionna package in the shared UM
        # runtime. This imports no RAW model and changes no scientific library.
        private=helper.read(r['private_sionna_manifest'])
        require(private['status']=='ACTUAL_ORIGINAL_SIONNA_PURE_PACKAGE_PRIVATE_COPY_VERIFIED_V1'
            and private['source']==r['sionna_source'] and private['private_parent']==r['private_sionna_parent'], 'Qualified private Sionna manifest required')
        for relative,row in private['entries'].items():
            require(sha(row['source'])==row['sha256']==sha(row['copy']), 'Private PHY package changed')
        require('sionna' not in sys.modules,'One private Sionna namespace per A3 method process')
        sys.path.insert(0,r['private_sionna_parent'])
        codec_path=next(p for p in cfg['codec_source_bindings'] if p.endswith('/bpg_codec.py'))
        helper.load(codec_path,'bpg_codec',{codec_path:cfg['codec_source_bindings'][codec_path]})
        phy=b['phy_module'];require(sha(phy['path'])==phy['sha256'],'Original BPG PHY wrapper changed')
        module=helper.load(phy['path'],'_a3_original_bpg_phy',{phy['path']:phy['sha256']})
        self.link=module.Link(cfg)
        mem=b['memory_module'];memory=helper.load(mem['path'],'_a3_bpg_memory_codec',{mem['path']:mem['sha256']})
        self.codec=memory.MemoryBPG(build['library'])
        class Ledger:
            def decode(self,phase,event_id,kind,request,callback):
                return calls.invoke('BPG_'+kind,callback)
        self.ledger=Ledger()
        self.library_path,self.library_sha=build['library'],build['library_sha256']
        self.gray=False;self.source_unfit=False;self.last_fit=None;self.last_outcome=None
        self.storage=dict(unique_loaded_parameters=0,unique_parameter_bytes=0,unique_buffer_bytes=0,
            weight_files=[],total_checkpoint_file_bytes=0,classical_codec=True,
            codec_library=str(build['library']),codec_library_bytes=Path(build['library']).stat().st_size,
            codec_library_sha256=build['library_sha256'],minimal_deployment_size_claimed=False,
            offloaded_unused_modules=[],full_vae_loaded=False,
            parameter_count_note='Classical BPG codec has no learned weights; actual linked codec library size is separate from model weights.',
            baseline_allocated_bytes=torch.cuda.memory_allocated(),baseline_reserved_bytes=torch.cuda.memory_reserved())

    def encode(self,item,meter):
        profile=self.policies[str(self.snr)]
        self.counter=(self.cfg['SNRs'].index(self.snr)*100+int(item['record']['source_index']))*3
        rgb=np.ascontiguousarray(item['pixels'].transpose(1,2,0))
        payload,self.last_fit=meter.call('TX_BPG_PNG_and_fresh_QP_search',lambda:self.codec.fit(rgb,profile['capacity_bytes']))
        self.last_source_bytes=None if payload is None else len(payload)
        self.source_unfit=payload is None;self.gray=self.source_unfit
        self.last_outcome=None
        if payload is None:
            return None
        return meter.call('TX_FEC_header_modulation',lambda:self.link.transmit(payload,profile,self.counter))

    def receive(self,observed,snr,meter):
        # The receiver does not accept the source image, QP, TX stream or TX MCS.
        outcome=meter.call('RX_BPG_PHY',lambda:self.link.receive(observed,float(snr),self.counter,
            self.ledger,'A3_TIMING',self.calls.case))
        self.last_outcome=outcome
        image=np.full((3,256,256),.5,dtype=np.float32);self.gray=True
        if outcome['body'] is not None and outcome['body']['parser_accepted']:
            received=base64.b64decode(outcome['body']['payload_b64'])
            decoded=meter.call('RX_BPG_memory_decode',lambda:self.codec.decode(received))
            if decoded is not None:
                image=np.ascontiguousarray(decoded.transpose(2,0,1),dtype=np.float32)/np.float32(255)
                self.gray=False
        return image

    def noise(self,wave,item,snr):
        return self.link.noisy(wave,'A3_DEVELOPMENT_TIMING',snr,int(item['record']['source_index']),2001)

    def finish(self):
        self.helper.require(self.helper.sha(self.library_path)==self.library_sha,'Memory codec changed during benchmark')
