"""Independent metered N2048 raw qualification; never launches a successor."""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
import t2_pilot as h
import t6_raw_phy as phy
from t2_ledger import Ledger
from t6_plan import write_csv

SEED=20261009117


def prepare(a):
    done=h.read(a.metadata_completion);r=h.checked(h.desc(a.metadata_request));cat=h.read(a.candidate_catalogue)
    h.require(done['status']=='T6_RESOURCE_METADATA_COMPLETE_NOT_PHY_QUALIFIED'and done['request_sha256']==h.sha(a.metadata_request)
        and done['encoder_forward_calls']==done['packet_decodes']==0,'Actual constructor-only N2048 metadata required')
    h.require(h.sha(a.candidate_catalogue)==done['outputs'][str(Path(a.candidate_catalogue).resolve())],'Candidate catalogue not sealed')
    env=h.checked(r['environment_request']);bindings=dict(env['source_bindings']);bindings.update(r['source_bindings'])
    for name in('t6_raw_phy.py','t6_qualify_raw_phy.py','t6_plan.py','t2_pilot.py','t2_ledger.py'):
        p=str(Path(__file__).with_name(name).resolve());bindings[p]=h.sha(p)
    rt=phy.Runtime(r,cat,bindings);bykey={p['groups'][0]['phy_key']:p for p in cat['profiles']}
    cases=[dict(phy_key=k,profile_id=p['profile_id'],layout=p['groups'][0]['layout'])for k,p in sorted(bykey.items())]
    headers=list(range(len(cat['profiles'])))+[4095];cap=len(cases)*8+len(headers)
    out=Path(a.out).resolve();h.require(out.is_relative_to(Path(r['root'])/'outputs/WCL-EVIDENCE-CLOSURE-20261009'),'Independent WCL qualification output')
    request=dict(schema=phy.QUAL_SCHEMA,root=r['root'],out=str(out),metadata_request=h.desc(a.metadata_request),
        metadata_completion=h.desc(a.metadata_completion),candidate_catalogue=h.desc(a.candidate_catalogue),
        source_bindings=bindings,body_cases=cases,header_ids=headers,packet_cap=cap,body_decoder_calls=len(cases)*8,
        header_decoder_calls=len(headers),payload_seed=SEED,original_backend_identity=rt.backend.identity,
        raw64_backend_identity=rt.backend.identity64,catalogue_sha256=rt.catalogue.digest,
        qualification_rule='All unique real bodies:4noiseless+4AWGN60; every paid header plus1unknownID',
        neural_model_calls=0,source_images_read=0,deadline_unix=a.deadline_unix,max_seconds=21600,
        stop_files=[str(Path(r['root'])/'STOP'),str(out/'STOP')],automatic_successor=False)
    h.save(out/'request.json',request)
    return dict(status='T6_RAW_QUALIFICATION_REGISTERED_NOT_RUN',request=h.desc(out/'request.json'),packet_cap=cap,
        unique_body_layouts=len(cases),paid_header_cases=len(headers),new_packet_decodes=0)


def run(path):
    r=h.read(path);rh=h.sha(path);out=Path(r['out']);h.require(r['schema']==phy.QUAL_SCHEMA and r['payload_seed']==SEED,'Frozen new qualification required')
    h.require(r['packet_cap']==r['body_decoder_calls']+r['header_decoder_calls']==len(r['body_cases'])*8+len(r['header_ids']), 'Finite qualification grid differs')
    with h.lock(out/'qualification.lock'):
        cp=out/'completion.json'
        if cp.exists():
            done=h.read(cp);h.require(done['request']==h.desc(path)and done['status']=='PASS','Completed qualification differs')
            for p,v in done['outputs'].items():h.require(h.sha(p)==v,'Completed qualification output changed')
            return dict(status='REUSED_COMPLETE_QUALIFICATION',new_packet_decodes=0)
        meta=h.checked(r['metadata_request']);cat=h.checked(r['candidate_catalogue']);rt=phy.Runtime(meta,cat,r['source_bindings'])
        h.require(rt.catalogue.digest==r['catalogue_sha256']and rt.backend.identity==r['original_backend_identity']
            and rt.backend.identity64==r['raw64_backend_identity'],'Current actual backend differs')
        meter=Ledger(out/'packet_ledger.sqlite',rh,r['packet_cap']);started=time.monotonic();bodyrows=[];headerrows=[];outputs={}
        for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,h.stop)
        try:
            for j,case in enumerate(r['body_cases']):
                p=rt.catalogue.entry(case['profile_id']);g=p['groups'][0]
                h.require(g['phy_key']==case['phy_key']and g['layout']==case['layout'],'Registered actual body differs')
                for index in range(8):
                    h.guard(r,started);rng=np.random.default_rng(np.random.SeedSequence([SEED,j,index]));length=g['source_bits']
                    payload=np.zeros(length,np.uint8)if index%4==0 else np.ones(length,np.uint8)if index%4==1 else rng.integers(0,2,length,np.uint8)
                    ctr=j*8+index;wave,tx=phy.transmit_body(rt,p,payload,ctr,'WCL_T6_RAW_QUALIFICATION')
                    noise=np.zeros_like(wave)if index<4 else rng.standard_normal(wave.shape).astype(np.float32)*np.float32(.001)
                    observed=(wave+noise).astype(np.float32);event=f'{phy.QUAL_SCHEMA}/body{j:04d}/{index}'
                    rx=phy.receive_body(rt,observed,p,60.,ctr,meter,event,'qualification','WCL_T6_RAW_QUALIFICATION')
                    correct=rx['crc_accept']and rx['hard_payload']==payload.tolist()
                    row=dict(phy_key=g['phy_key'],layout_id=g['layout']['layout_id'],profile_id=p['profile_id'],q=phy.MODULATIONS[g['modulation']],
                        k=g['information_bits'],n=g['transmitted_bits'],index=index,noiseless=index<4,snr_db=60,
                        source_payload_sha256=phy.array_sha(payload),received_sha256=phy.array_sha(observed),noise_sha256=phy.array_sha(noise),
                        actual_received_payload_equal=correct,CRC_accept=rx['crc_accept'],body_energy=tx['body_energy'],event_id=event)
                    op=out/'body_cases'/f'{j:04d}_{index}.json';h.save(op,row);outputs[str(op)]=h.sha(op);bodyrows.append(row)
                    h.require(correct,'Actual N2048 raw packet roundtrip failed: '+event)
                print(h.canonical(dict(stage='T6_RAW_QUALIFICATION',completed_body_layouts=j+1,total=len(r['body_cases']))),flush=True)
            for pid in r['header_ids']:
                h.guard(r,started);wave=np.asarray(rt.header.transmit(pid),np.float64);event=f'{phy.QUAL_SCHEMA}/header{pid:04d}'
                rx=phy.receive_header(rt,wave,60.,meter,event,'qualification');known=str(pid)in rt.profiles
                correct=rx['header_crc_ok']is True and rx['header_fields_legal']is known and rx['header_ok']is known and rx['profile_id']==(pid if known else None)
                row=dict(test_profile_id=pid,known=known,correct=correct,received_sha256=phy.array_sha(wave),event_id=event,**rx)
                op=out/'header_cases'/f'{pid:04d}.json';h.save(op,row);outputs[str(op)]=h.sha(op);headerrows.append(row)
                h.require(correct,'Actual paid N2048 header roundtrip failed')
            snap=meter.snapshot();h.require(snap==dict(total=r['packet_cap'],unresolved=0,cap=r['packet_cap']),'Qualification ledger incomplete')
            for n,values in [('body_qualification.csv',bodyrows),('header_qualification.csv',headerrows)]:
                write_csv(out/n,values);outputs[str(out/n)]=h.sha(out/n)
            done=dict(status='PASS',schema=phy.QUAL_SCHEMA,request=h.desc(path),catalogue_sha256=rt.catalogue.digest,
                original_backend_identity=rt.backend.identity,raw64_backend_identity=rt.backend.identity64,
                actual_body_decodes=len(bodyrows),actual_header_decodes=len(headerrows),packet_decode_count=snap['total'],ledger=snap,
                N=2048,source_images_read=0,neural_model_calls=0,outputs=outputs,automatic_successor=False)
            h.save(cp,done);return {k:v for k,v in done.items()if k not in('outputs','original_backend_identity','raw64_backend_identity')}
        finally:meter.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);q=s.add_parser('prepare')
    for n in('metadata-request','metadata-completion','candidate-catalogue','out'):q.add_argument('--'+n,required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q=s.add_parser('run');q.add_argument('--request',required=True)
    a=p.parse_args();print(json.dumps(prepare(a)if a.command=='prepare'else run(a.request)))
