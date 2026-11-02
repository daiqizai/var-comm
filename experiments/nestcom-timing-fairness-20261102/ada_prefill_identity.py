"""Two-source prefill identity diagnostic, not an optimized timing benchmark.

The original scientific receiver, timing workers, and caches are immutable.
Only four registered probability evaluations are made. Negative identity results
are valid diagnostic outcomes and never trigger a retry or parameter search.
"""
from pathlib import Path
import sys,os,json,time,traceback

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
from common import read,save,sha,checked,asha,Ledger
import numpy as np

def main(path,expected):
    assert sha(path)==expected
    r=read(path);[checked(p)for p in r['code']]
    out=Path(r['output']);out.mkdir(exist_ok=False)
    ledger=Ledger(out/'calls.sqlite',dict(model_load=1,prefill_identity=4))
    os.sched_setaffinity(0,{38,40});begin=time.time()
    try:
        from c_predclass_native import load_native
        from c_arithmetic import load_codec
        n=load_native(r['native'],out,ledger);t=n.torch;t.set_num_threads(2)
        assert str(t.cuda.get_device_properties(0).uuid)==r['CUDA_UUID']
        assert t.get_num_threads()==t.get_num_interop_threads()==2
        Encoder,_,cdf,proof=load_codec();save(out/'arithmetic_identity.json',proof)
        rows=[]
        with t.inference_mode():
            for s in r['sources']:
                tokens=np.load(checked(s['tokens']),allow_pickle=False)
                assert tokens.shape==(256,)and tokens.dtype==np.int64 and asha(tokens)==s['token_sha256']
                ref=read(checked(s['cdf_hashes']));assert len(ref)==256
                with np.load(checked(s['streams']),allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files}
                for branches in [1,2]:
                    assert time.time()<r['deadline_unix']and time.time()-begin<r['seconds']
                    key=s['source_id']+'/branches'+str(branches)
                    def call():
                        # Shifted known TX tokens: the class predicts q_0, q_i
                        # predicts q_(i+1). A strict causal mask prevents leakage.
                        with t.device('cuda:0'):
                            n.prior.setup_caches(max_batch_size=branches,max_seq_length=257,dtype=t.float32)
                        idx=t.tensor(np.repeat(tokens[None,:255],branches,axis=0),dtype=t.int64,device='cuda:0')
                        cond=t.tensor([s['predicted_class']]+([1000]if branches==2 else[]),dtype=t.int64,device='cuda:0')
                        pos=t.arange(256,dtype=t.int64,device='cuda:0')
                        mask=n.prior.causal_mask[:branches,None,pos]
                        assert mask.shape[-2:]==(256,264)
                        assert not bool(mask[...,t.triu(t.ones((256,264),device='cuda:0',dtype=t.bool),diagonal=1)].any())
                        with t.nn.attention.sdpa_kernel(t.nn.attention.SDPBackend.MATH):
                            logits,_=n.prior(idx,cond,input_pos=pos,mask=mask)
                        t.cuda.synchronize()
                        assert logits.shape==(branches,256,16384)and logits.dtype==t.float32 and bool(t.isfinite(logits).all())
                        # Use exactly the existing conditional CDF routine.
                        enc=Encoder();cdf_matches=[];stream_matches=[];length_deltas=[]
                        for i,v in enumerate(tokens):
                            table=n.cdf(logits[:,i:i+1,:],cdf)
                            cdf_matches.append(asha(table)==ref[i])
                            enc.encode(np.array([v],np.int64),table);bits=enc.finish()
                            original=streams['p%03d'%(i+1)]
                            stream_matches.append(np.array_equal(bits,original));length_deltas.append(len(bits)-len(original))
                        return dict(source_id=s['source_id'],source_index=s['source_index'],branches=branches,
                            conditional_CDF_matches=sum(cdf_matches),conditional_CDF_total=256,
                            independently_terminated_stream_matches=sum(stream_matches),stream_total=256,
                            first_CDF_mismatch=next((i for i,v in enumerate(cdf_matches)if not v),None),
                            first_stream_mismatch=next((i+1 for i,v in enumerate(stream_matches)if not v),None),
                            maximum_absolute_length_delta_bits=max(map(abs,length_deltas)),
                            all_CDFs_identical=all(cdf_matches),all_streams_identical=all(stream_matches))
                    v=ledger.call('prefill_identity',key,call);rows.append(v);save(out/('source_%04d_branches%d.json'%(s['source_index'],branches)),v)
        passed=all(r['all_CDFs_identical']and r['all_streams_identical']for r in rows)
        save(out/'completion.json',dict(status='COMPLETE',identity_qualification='PASS'if passed else'NON_EQUIVALENT_TO_FROZEN_INCREMENTAL_ROUTE',
             rows=rows,ledger=ledger.summary(),new_model_prefills=4,logical_probability_positions=1024,
             new_tokenizer=0,new_classifier=0,new_render=0,new_PHY=0,new_quality=0,
             torch_compile='NOT_RUN: first establish numerical/CDF compatibility',
             scope='Not timing or author-code reproduction. Tests one full-prefix call under frozen FP32/math route against preserved incremental CDFs/streams.',
             elapsed_seconds=time.time()-begin))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
