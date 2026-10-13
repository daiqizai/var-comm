"""One frozen entropy probability pass per common500 source, all real endpoints."""
import argparse
import copy
import time
import traceback
import numpy as np
from common import *

def run(path,parent):
    r=registration(path);out=ART/'source';out.mkdir()
    guard=guard_for(r,out,parent);device,g=resource_device(r,out);last=0.;ledger=None;b=None
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>5:
            snapshot,_=device.resource_snapshot(g.helper());last=time.monotonic();status(out/'resource.json',snapshot)
    caps=dict(model_load=1,encoder=0,source_tx=500,source_rx=0,var_render=0,prior_scale=5000,decoder_forward=0)
    try:
        ledger=Ledger(out/'calls.sqlite',caps,guard);write(out/'launch.json',dict(pid=os.getpid(),registration=descriptor(path)))
        b,module,resolver,environment,_,g=backend(r,out,ledger,boundary)
        import ep_source_codec as partial
        codec=partial.PartialSourceCodec(ROOT);codec.native=b.native;codec.provider_factory=b.codec.provider_factory
        results=[];started=time.monotonic()
        for record in r['sources']:
            boundary();i=record['source_index'];p=checked(record['archive'])
            with np.load(p,allow_pickle=False) as z:tokens=z['tokens'].copy()
            require(tokens.dtype==np.int64 and tokens.shape==(680,) and hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==record['tokens_sha256'],'Frozen source tokens differ')
            b.source_index=i;b.current=('TX',10);b.traces=[];b.tx_cdfs={};b.received={}
            encoded=ledger.call('source_tx',lambda:codec.encode_endpoints(tokens,partial.all_endpoints()),source=i)
            require(set(encoded)==set(partial.all_endpoints()) and len(b.traces)==10,'Exact24 terminated streams / ten-scale TX required')
            dest=out/'sources'/f'{i:04d}';dest.mkdir(parents=True);arrays={};endpoints=[]
            for (m,K),item in encoded.items():
                bits=item['bits'];require(bits.dtype==np.uint8 and bits.ndim==1 and len(bits)==item['arithmetic_bits'] and np.isin(bits,(0,1)).all(),'Invalid terminated arithmetic stream')
                arrays[f'm{m}_K{K}']=bits;endpoints.append(dict(m=m,K=K,payload_bits=len(bits),bits_sha256=array_sha(bits)))
            ap=dest/'actual_streams.npz'
            with ap.open('xb') as f:np.savez(f,**arrays)
            meta=dict(source_index=i,source_id=record['source_id'],tokens_sha256=record['tokens_sha256'],archive=descriptor(ap),
                      endpoints=endpoints,CDF_trace=copy.deepcopy(b.traces),old_host_bits_used=False,old_host_CDF_used=False)
            write(dest/'actual_TX.json',meta);results.append(dict(source_index=i,source_id=record['source_id'],archive=meta['archive'],metadata=descriptor(dest/'actual_TX.json')))
            status(OUT/'source_status.json',dict(completed_sources=i+1,total_sources=500,elapsed_seconds=time.monotonic()-started,counts=ledger.summary()))
        b.close();resolver.reverify();environment.reverify();counts=ledger.summary();require(counts['reserved']==counts['completed']==caps,'Exact source budget closure')
        write(out/'streams.json',results);write(out/'completion.json',dict(status='COMPLETE_SOURCE500',source_count=500,streams=descriptor(out/'streams.json'),counts=counts,
            encoder_calls=0,source_tokens_reused=True,actual_new_source_TX=500,old_host_bitwise_claimed=False));ledger.close()
    except BaseException as e:
        if b is not None:b.failure_evidence()
        write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--parent',type=int,required=True);a=p.parse_args();run(a.registration,a.parent)
