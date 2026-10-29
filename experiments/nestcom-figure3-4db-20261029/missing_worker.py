"""Four new bounded visualization frames; frozen source and received-only RX."""
from pathlib import Path
import os,sys,traceback,hashlib
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
from common import *
L=B/'nestcom_fig3_4db_20261029_v1'
sys.path.insert(0,str(R/'experiments/five-method-timing-20261020'))
import native
import numpy as np

def whole_source(r,out):
    cfg=cr(r['timing_config']);ctx=native.nest_codec(cfg,out);n,codec,render,calls=ctx
    led=Ledger(out/'calls.sqlite',dict(source_encode=2));rows=[]
    for s in cr(r['sources']):
        with np.load(checked(s['archive']),allow_pickle=False)as z:tokens=z['tokens'].copy()
        assert hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==s['tokens_sha256']
        encoded=led.call('source_encode',s['source_id'],lambda:codec.encode_endpoints(tokens,[(7,0),(6,0),(5,0),(4,0)]))
        attempts=[dict(m=m,K=0,bits=len(encoded[m,0]['bits']),fits=len(encoded[m,0]['bits'])<=1245)for m in range(7,3,-1)]
        m=next(a['m']for a in attempts if a['fits']);bits=encoded[m,0]['bits']
        rows.append(dict(source_id=s['source_id'],source_index=s['source_index'],m=m,K=0,q=2,nominal_rate='2/3',payload=bits.tolist(),payload_sha256=asha(bits),length_checks=attempts,tokens_sha256=s['tokens_sha256']))
    save(out/'sources.json',rows)
    return dict(status='COMPLETE',sources=pin(out/'sources.json'),new_tokenizer=0,new_source_passes=2,model_calls=calls.counts,ledger=led.summary())

def link(r,out,method):
    native.paths()
    sys.path.insert(0,str(R/'experiments/wcl-adatoken-20261027'))
    from public_counter_rx import receive
    from t2_ledger import Ledger as PacketLedger
    if method=='WHOLE':
        import worker as old
        cfg=cr(r['timing_config']);rt=old.phy_runtime(cfg,'NEST',out)
        import ep_phy as phy
        sources=cr(r['whole_sources'])
    else:
        from c_predclass_frame import runtime,choose,source_projection
        phy,rt=runtime(r);sources=cr(r['ada_sources'])
    ledger=PacketLedger(out/'packets.sqlite',r['scope_sha256'],4)
    frames=[];inputs=[]
    try:
        for s in sources:
            if method=='WHOLE':
                p=next(p for p in rt.profiles.values()if (p['family'],p['m'],p['K'],p['q'],p['nominal_rate'])==('EC_VAR_WHOLE',s['m'],0,2,'2/3'))
                bits=np.array(s['payload'],np.uint8);attempts=s['length_checks'];seed=6201;base=14000000+4096*s['source_index']
            else:
                with np.load(checked(s['streams']),allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files}
                p,bits,attempts=choose(streams,list(rt.profiles.values()),2,'2/3',s['predicted_class']);seed=2001;base=15000000+4096*s['source_index']
            event=digest([r['noise_namespace'],method,s['source_id'],4,seed]);counter=base+p['profile_id']
            save(out/(event+'_reserved.json'),dict(event=event,source_id=s['source_id'],source_index=s['source_index'],counter=counter,seed=seed))
            wave,tx=rt.transmit(p['profile_id'],bits,counter);noise=phy.standard_noise(s['source_id'],seed);observed=wave+noise*10**(-4/20)
            assert wave.shape==observed.shape==(1024,2)
            op=out/(event+'.npz')
            with op.open('xb')as f:np.savez(f,observed=observed);f.flush();os.fsync(f.fileno())
            rx=receive(phy,rt,observed,4,base,ledger,event,phase='figure3_4db')
            if method=='WHOLE':
                body=rx['body'];accepted=bool(rx['header']['header_ok']and body and body['crc_accepted']and body['parser_accepted'])
                inp=dict(accepted=accepted,status=rx['status'],profile=rx['rx_profile'],payload=body['payload']if accepted else None)
                ip=out/(event+'_received.json');save(ip,inp);rxkey=event
                inputs.append(dict(key=event,source_index=s['source_index'],actual_input=pin(ip)))
            else:
                projection=source_projection(rx)
                if projection['status']=='SOURCE_INPUT':
                    payload=np.array(rx['body']['payload'],np.uint8);container_sha=asha(payload)
                    inp=dict(key=digest(dict(prefix=projection['prefix'],container_sha256=container_sha)),prefix=projection['prefix'],bits=projection['bits'],payload_sha256=projection['payload_sha256'],received_class_bits=payload[:10].tolist(),container_sha256=container_sha)
                    inputs.append(inp);rxkey=inp['key']
                else:rxkey=None
            frames.append(dict(method=method,source_id=s['source_id'],source_index=s['source_index'],snr_db=4,N=1024,noise_seed=seed,noise_namespace=r['noise_namespace'],standard_noise_sha256=asha(noise),event_id=event,public_base=base,transmission=tx,actual_RX=rx,rx_key=rxkey,observation=pin(op),waveform_sha256=asha(wave),total_energy=float(np.square(wave.astype(np.float64)).sum()),length_checks=attempts,source_truth_supplied_to_RX=False))
            save(out/(event+'.json'),frames[-1])
        assert len(frames)==2 and ledger.snapshot()['unresolved']==0
        save(out/'frames.json',frames);save(out/'received_inputs.json',inputs)
        return dict(status='COMPLETE',physical_events=2,logical_events=2,frames=pin(out/'frames.json'),received_inputs=pin(out/'received_inputs.json'),packet_ledger=ledger.snapshot())
    finally:ledger.close()

def whole_rx(r,out):
    cfg=cr(r['timing_config']);jobs=cr(r['inputs'])
    actuals=[(j,cr(j['actual_input']))for j in jobs]
    sys.path.insert(0,str(R/'experiments/five-method-timing-20261020/library_recovery_v1'))
    from adapter import adapted_worker
    w=adapted_worker();blocked=w.firewall(cfg,'NEST',out)
    ctx=native.nest_codec(cfg,out);rows=[];(out/'images').mkdir()
    for j,actual in actuals:
        save(out/(j['key']+'_reservation.json'),dict(key=j['key'],actual_input=j['actual_input']))
        image,status=w.native_rx(actual,cfg,'NEST',ctx,w.Clock(ctx[0].torch))
        p=out/'images'/(j['key']+'.npz')
        with p.open('xb')as f:np.savez(f,image=image);f.flush();os.fsync(f.fileno())
        rows.append(dict(key=j['key'],source_index=j['source_index'],status=status,image=pin(p),image_sha256=asha(image),source_truth_used=False))
        save(p.with_suffix('.json'),rows[-1])
    assert len(rows)==2 and not blocked;save(out/'outputs.json',rows)
    return dict(status='COMPLETE',outputs=pin(out/'outputs.json'),model_calls=ctx[3].counts,new_reconstructions=2,source_truth_used=False)

def main(role,path,expected):
    assert sha(path)==expected;r=read(path)
    for d in r['code']:checked(d)
    assert r['scope_sha256']==sha(r['scope_path'])
    assert role in ('whole_source','whole_phy','whole_rx','ada_phy')
    out=L/role;out.mkdir();runtime_cache(out);os.sched_setaffinity(0,set(range(6)))
    save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),registration_sha256=expected))
    try:
        if role=='whole_source':v=whole_source(r,out)
        elif role=='whole_rx':v=whole_rx(r,out)
        else:v=link(r,out,'WHOLE'if role=='whole_phy'else'ADA')
        save(out/'completion.json',v)
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
