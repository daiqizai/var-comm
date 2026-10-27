"""Finite production RX shard. Registered received-only inputs and shared weights."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from c_arithmetic import load_codec
from c_predclass_native import load_native
from c_predclass_protocol import received_class,generation_seed
from protocol_rx import realize
import argparse,traceback,numpy as np


def main(role,path,expected):
    assert sha(path)==expected;r=read(path)
    for d in r['code']:checked(d)
    assert r['scope']['sha256']=='b41ed4361aab6606ce6263071865cf466bc5e09cfeb7b34efabd3e56da031a68'
    assert r['revision']['sha256']=='fb80b863dad31fa37cac9cc2e8d8ee4a7d5248071b3a73fdcc74c704517fc254'
    checked(r['scope']);checked(r['revision'])
    assert type(r['max_seconds'])is int and 0<r['max_seconds']<=43200
    inputs=cr(r['inputs']);assert 0<len(inputs)<=18000
    assert len({v['key']for v in inputs})==len(inputs)
    fields={'key','prefix','bits','payload_sha256','received_class_bits','container_sha256'}
    for v in inputs:
        assert set(v)==fields, 'Only received payload and paid prefix/class metadata permitted'
        bits=np.array(v['bits']);assert bits.ndim==1 and 2<=len(bits)<=8181 and np.isin(bits,(0,1)).all()
        cb=np.array(v['received_class_bits']);assert cb.shape==(10,)and np.isin(cb,(0,1)).all()
        received_class(cb.astype(np.uint8))
        assert type(v['prefix'])is int and 1<=v['prefix']<=256
        assert asha(bits.astype(np.uint8))==v['payload_sha256']
        assert asha(np.concatenate([cb.astype(np.uint8),bits.astype(np.uint8)]))==v['container_sha256']
        assert v['key']==digest(dict(prefix=v['prefix'],container_sha256=v['container_sha256']))
    assert r['caps']==dict(model_load=1,receive_and_render=len(inputs))
    out=L/role;out.mkdir(parents=True,exist_ok=False);runtime_cache(out)
    os.sched_setaffinity(0,set(range(6)));beg=time.monotonic()
    ledger=Ledger(out/'calls.sqlite',r['caps'])
    save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
    try:
        sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'))
        from runtime_boundary import firewall
        firewall(out,[d['path']for d in r['weights'].values()],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
        Encoder,Decoder,cdf,proof=load_codec();save(out/'arithmetic_proof.json',proof)
        n=load_native(r,out,ledger);rows=[];(out/'images').mkdir()
        with n.torch.inference_mode():
            for job in inputs:
                assert time.monotonic()-beg<r['max_seconds']and n.torch.cuda.mem_get_info()[0]>=2<<30
                bits=np.array(job['bits'],np.uint8);prefix=job['prefix'];key=job['key']
                predicted_class=received_class(np.array(job['received_class_bits'],np.uint8))
                seed=generation_seed(bits,prefix,predicted_class)
                def receive():
                    result=realize(n,Encoder,Decoder,cdf,bits,prefix,predicted_class,seed)
                    image=result.pop('image');received=result.pop('received_tokens');tokens=result.pop('completed_tokens')
                    assert image.dtype==np.float32 and image.shape==(3,256,256)and np.isfinite(image).all()
                    p=out/'images'/(key+'.npz')
                    with p.open('xb')as f:
                        np.savez(f,image=image,received_tokens=received,completed_tokens=tokens);f.flush();os.fsync(f.fileno())
                    record=dict(key=key,prefix=prefix,received_class=predicted_class,
                        payload_sha256=asha(bits),container_sha256=job['container_sha256'],generation_seed=seed,
                        image=pin(p),pixel_sha256=asha(image),received_tokens_sha256=asha(received),
                        source_truth_used=False,TX_CDF_used=False,**result)
                    save(p.with_suffix('.json'),record);return record
                rows.append(ledger.call('receive_and_render',key,receive))
                print('LLAMAGEN_PRODUCTION_RX_COMMITTED',len(rows),len(inputs),rows[-1]['status'],flush=True)
        assert ledger.summary()['unresolved']==0
        save(out/'outputs.json',rows)
        save(out/'completion.json',dict(status='COMPLETE',registration_sha256=expected,outputs=pin(out/'outputs.json'),
            received_outcomes=len(rows),accepted_source=sum(v['status']=='ACCEPTED_SOURCE'for v in rows),
            source_rejections=sum(v['status']=='FIXED_GRAY_SOURCE_REJECT'for v in rows),
            prior_forwards=sum(v['prior_forwards']for v in rows),
            prior_branch_forwards=sum(v['prior_branch_forwards']for v in rows),
            decoder_forwards=sum(v['decoder_forwards']for v in rows),
            ledger=ledger.summary(),source_reads=0,elapsed_seconds=time.monotonic()-beg))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True)
    p.add_argument('--expected-registration',required=True);a=p.parse_args()
    main(a.role,a.registration,a.expected_registration)
