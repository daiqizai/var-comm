"""Bounded real CPU PHY checks; no source models or metrics are loaded."""
from pathlib import Path
import json, sys, hashlib, time, traceback
import numpy as np
from n1024_null_v2 import *

def main(root,out,inputs):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    registration=dict(protocol=PROTOCOL,frame_cap=18,packet_cap=36,max_seconds=600,
        checks='12 paid-NULL full frames plus 3 existing payloads through old/new catalogues',
        noise='no added noise; declared SNR 40 dB for finite decoder LLR',
        scope='targeted CPU engineering checks, not a formal quality or reliability grid',
        source_model_calls=0,metric_calls=0,training_calls=0,
        files={str(Path(__file__).name):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'n1024_null_v2.py':hashlib.sha256(Path(__file__).with_name('n1024_null_v2.py').read_bytes()).hexdigest()},
        historical_source_pins=PINS,inputs=inputs)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    class Ledger:
        def __init__(self):self.calls=0
        def call(self,event,request,fn):
            assert self.calls<36 and time.monotonic()-started<600
            self.calls+=1;p=out/f'packet_{self.calls:02d}.json'
            p.write_text(json.dumps(dict(event=event,request=request,status='STARTED')))
            result=fn();p.write_text(json.dumps(dict(event=event,request=request,status='DONE',result=result)))
            return result
    ledger=Ledger(); records=[]
    try:
        phy,tb=load(root);rt=phy.Runtime(root)
        old=tb.extend_catalogue(rt.build_catalogue())
        # Match the exact catalogue used by the current N1024 m10/main lineage.
        published=json.loads((root/'results/t1b_m10_calibration_20261016_v1/public_catalogue.json').read_text())
        assert old['profiles']==published['profiles']
        new=extend_catalogue(phy,published)
        (out/'catalogue.json').write_text(json.dumps(new,indent=2)+'\n')
        # Preserve the original old-family functions in a separate namespace.
        src=root/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py'
        oldns=dict(__file__=str(src),__name__='_null_test_unchanged_original')
        exec(compile(src.read_bytes(),str(src),'exec'),oldns)
        for i,item in enumerate(inputs):
            fp=Path(item['frame']['path']);assert hashlib.sha256(fp.read_bytes()).hexdigest()==item['frame']['sha256']
            saved=json.loads(fp.read_text());rx=saved['actual_RX'];meta=saved['transmission']
            assert rx['body']['status']=='PAYLOAD_PARSED'
            bits=np.array(rx['body']['payload'],np.uint8)
            assert len(bits)==meta['arithmetic_bits']
            profile=published['profiles'][meta['profile_id']]
            target_m,target_K=profile['m'],profile['K']
            candidate=dict(target_m=target_m,target_K=target_K,q=profile['q'],nominal_rate=profile['nominal_rate'],source_capacity_bits=profile['k']-29)
            selected=choose({f'm{target_m}_K{target_K}':bits},candidate,new,'legacy')
            assert not selected['tx_no_fit'] and selected['profile']==profile and np.array_equal(selected['payload'],bits)
            outputs=[]
            for mode,cat,ns in [('old',published,oldns),('new',new,phy.__dict__)]:
                rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']}
                wave,tx=ns['transmit_frame'](rt,bits,profile['profile_id'],meta['public_frame_counter'],meta['session'])
                assert tx==meta, 'Historical full TX metadata differs'
                result=ns['receive_frame'](rt,wave,40,meta['public_frame_counter'],ledger,f'normal_{i}_{mode}',session=meta['session'])
                assert result['body']['status']=='PAYLOAD_PARSED'
                outputs.append((wave,result))
            assert np.array_equal(outputs[0][0],outputs[1][0]) and outputs[0][1]==outputs[1][1]
            assert received_projection(outputs[1][1])==received_projection(rx)
            # No neural replay: exact old source input permits existing image reuse.
            img=item['reconstruction_identity']['image_archive']
            assert hashlib.sha256(Path(img['path']).read_bytes()).hexdigest()==img['sha256']
            records.append(dict(check='normal_old_new_exact',source_input=received_projection(rx),profile_id=profile['profile_id'],historical_frame=item['frame'],image_reuse=img,neural_calls=0,waveform_sha256=meta['transmitted_frame_sha256']))
        rt.catalogue=new;rt.profiles={str(p['profile_id']):p for p in new['profiles']}
        nulls=[p for p in new['profiles'] if p['family']==NULL]
        assert len(nulls)==12
        for i,p in enumerate(nulls):
            cap=p['k']-29
            c=dict(target_m=4,target_K=0,q=p['q'],nominal_rate=p['nominal_rate'],source_capacity_bits=cap)
            selected=choose({'m4_K0':np.zeros(cap+1,np.uint8)},c,new,'legacy')
            assert selected['tx_no_fit'] and selected['profile']==p
            wave,tx=phy.transmit_frame(rt,selected['payload'],p['profile_id'],90000+i)
            assert wave.shape==(1024,2) and tx['header_symbols']==68 and tx['body_symbols']==956 and tx['total_symbols']==1024
            rx=phy.receive_frame(rt,wave,40,90000+i,ledger,f'null_{i}')
            projection=received_projection(rx);assert projection==dict(status='FIXED_GRAY',reason='EXPLICIT_NULL_PROFILE')
            image=fixed_gray(projection);assert image.shape==(3,256,256) and image.dtype==np.float32 and np.all(image==0.5)
            records.append(dict(check='no_fit_paid_full_frame',profile_id=p['profile_id'],q=p['q'],rate=p['nominal_rate'],transmission=tx,rx_projection=projection,gray_array_sha256=phy.array_sha(image)))
        # m10 preserves each family-specific chain; absent evidence still fails.
        for family in ('WHOLE','PARTIAL'):
            chain=tb.fallback(family);p=nulls[0];cap=p['k']-29
            streams={f'm{m}_K{k}':np.zeros(cap+1,np.uint8) for m,k in chain}
            c=dict(target_m=10,target_K=0,strategy_family=family,q=p['q'],nominal_rate=p['nominal_rate'],source_capacity_bits=cap)
            s=choose(streams,c,new,'m10');assert s['tx_no_fit']
            streams['m10_K0']=np.zeros(cap,np.uint8);s=choose(streams,c,new,'m10');assert not s['tx_no_fit'] and s['profile']['m']==10
            assert tb.choose({e:len(streams[f'm{e[0]}_K{e[1]}']) for e in chain},family,cap)==(10,0)
        try:choose({},c,new,'m10')
        except KeyError:pass
        else:raise AssertionError('Missing evidence must not be hidden as NULL')
        # Invalid/failed NULLs cannot bypass body acceptance; these are pure checks.
        valid=dict(header=dict(header_ok=True),rx_profile=dict(family=NULL),body=dict(crc_accepted=True,parser_accepted=True,payload=[1,0]))
        assert received_projection(valid)['reason']=='INVALID_NULL_PAYLOAD'
        valid['body'].update(crc_accepted=False,status='CRC_REJECT');assert received_projection(valid)['reason']=='CRC_REJECT'
        valid['header']['header_ok']=False;assert received_projection(valid)['reason']=='HEADER_REJECT'
        assert ledger.calls==36 and not rt.torch.cuda.is_initialized()
        summary=dict(status='PASS_TARGETED_PROTOCOL_CHECKS',protocol=PROTOCOL,actual_full_frame_encodes=18,actual_full_frame_receives=18,actual_packet_decodes=ledger.calls,normal_existing_payloads=3,normal_old_new_waveform_and_rx_identical=True,null_mcs_full_frames=12,source_model_calls=0,metric_calls=0,neural_replays=0,old_image_reuse=3,historical_protocol_files_unchanged=True,formal_quality_claim=False,elapsed_seconds=time.monotonic()-started,records=records)
        for rel,pin in PINS.items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==pin
        (out/'completion.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
    except BaseException:
        (out/'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),actual_packet_calls=ledger.calls,automatic_retry=False),indent=2)+'\n');raise

if __name__=='__main__':main(sys.argv[1],sys.argv[2],json.loads(Path(sys.argv[3]).read_text()))
