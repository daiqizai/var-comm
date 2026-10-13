"""Only4500 new actual EC partial frames, original T1 common500 noise/counters."""
import argparse
import csv
import time
import traceback
import numpy as np
from common import *

def run(path,parent):
    r=registration(path);out=ART/'phy';out.mkdir();guard=guard_for(r,out,parent);ledger=None
    try:
        require(read(ART/'source/actual_wait.json')['returncode']==0,'Source actual exit0 required')
        done=read(ART/'source/completion.json');sources=read(checked(done['streams']));require(len(sources)==500,'Complete source500')
        import ep_phy,ep_plan,ep_source_codec,t1_phy
        from t2_ledger import Ledger
        import h800_ep48_link_core_v1 as link
        preq=read(checked(r['phy_request']));provider=preq['providers']['partial']
        runtime=ep_phy.create_runtime(ROOT,provider['qualification']['path'])
        require(len(runtime.profiles)==360 and runtime.catalogue['catalogue_sha256']==provider['catalogue_sha256'],'Complete public360 domain')
        require(not runtime.torch.cuda.is_initialized(),'CPU PHY only')
        ledger=Ledger(out/'packets.sqlite',sha(path),9000);frames=[];lengths=[];started=time.monotonic()
        lookup={(p['family'],p['m'],p['K'],p['q'],p['nominal_rate']):p for p in runtime.profiles.values()}
        write(out/'launch.json',dict(pid=os.getpid(),registration=descriptor(path),provider=provider))
        for item,record in zip(sources,r['sources']):
            i=record['source_index'];require(item['source_id']==record['source_id'],'Source order differs')
            streams=link.load_streams(item['archive'],read(checked(item['metadata'])),ep_source_codec)
            for s in SNRS:
                choice=r['policies'][str(s)];selected=ep_plan.select_stream({k:len(v) for k,v in streams.items()},choice)
                m,K=selected['m'],selected['K'];T=ep_plan.token_count(m,K);bits=streams[m,K]
                profile=lookup['EC_VAR_PARTIAL' if K else 'EC_VAR_WHOLE',m,K,choice['q'],choice['nominal_rate']]
                lengths.append(dict(source_index=i,source_id=record['source_id'],snr_db=s,target_m=choice['target_m'],target_K=choice['target_K'],
                    actual_m=m,actual_K=K,actual_T=T,arithmetic_bits=len(bits),capacity_bits=choice['source_capacity_bits'],raw_bits=12*T,
                    information_margin=choice['source_capacity_bits']-len(bits),capacity_utilization=len(bits)/choice['source_capacity_bits'],
                    source_length_saving=1-len(bits)/(12*T),fallback_count=len(selected['attempts'])-1,fallback_attempts=canonical(selected['attempts']),
                    stream_archive=item['archive']['path'],stream_sha256=array_sha(bits)))
                for n in SEEDS:
                    guard();idx=len(frames);fc=counter(i,s,n);wave,tx=runtime.transmit(profile['profile_id'],bits,fc)
                    standard=t1_phy.standard_noise(record['source_id'],n);noise=standard*10**(-s/20);observed=wave+noise
                    event=f'EMAIN_COMMON500_V1/source{i}/snr{s}/noise{n}'
                    actual=runtime.receive(observed,s,fc,runtime.profiles,ledger,event,phase='emain_common500_partial360')
                    frame=dict(frame_index=idx,method='EC_VAR_PARTIAL',source_index=i,source_id=record['source_id'],snr_db=s,noise_seed=n,N=1024,
                        public_frame_counter=fc,policy=choice,TX_profile=profile,actual_m=m,actual_K=K,actual_T=T,arithmetic_bits=len(bits),
                        capacity_bits=choice['source_capacity_bits'],transmission=tx,fallback=selected,actual_RX=actual,
                        payload_sha256=array_sha(bits),waveform_sha256=array_sha(wave),standard_noise_sha256=array_sha(standard),
                        observation_sha256=array_sha(observed),noise_sha256=array_sha(noise),source_truth_supplied_to_RX=False)
                    pp=out/'frames'/f'{idx:04d}.json';write(pp,frame);frames.append(descriptor(pp))
            status(OUT/'phy_status.json',dict(completed_frames=len(frames),total_frames=4500,elapsed_seconds=time.monotonic()-started,packet_ledger=ledger.snapshot()))
        require(len(frames)==4500 and ledger.snapshot()['unresolved']==0 and ledger.snapshot()['total']<=9000,'Full bounded PHY grid')
        write(out/'frames.json',frames)
        cp=OUT/'source_streams.csv'
        with cp.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(lengths[0]));w.writeheader();w.writerows(lengths)
        write(out/'completion.json',dict(status='COMPLETE_PHY4500',logical_frames=4500,frames=descriptor(out/'frames.json'),
            source_streams=descriptor(cp),packet_ledger=ledger.snapshot(),source_RX_calls=0,model_calls=0,CUDA_initialized=False));ledger.close()
    except BaseException as e:
        write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),packet_ledger=None if ledger is None else ledger.snapshot()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--parent',type=int,required=True);a=p.parse_args();run(a.registration,a.parent)
