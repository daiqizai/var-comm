"""Actual constructor admission and metered finite N2048 entropy qualification."""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
import t2_pilot as h
import t6_entropy_phy as phy
from t2_ledger import Ledger
from t6_plan import write_csv
from t6_calibration_plan import entropy_candidates

SEED=20261009129


def prepare(a):
    family=h.read(a.entropy_family_selection)
    h.require(family['status']=='T1_SINGLE_ENTROPY_FAMILY_FROZEN_CALIBRATION_ONLY'
        and not family['holdout_used_for_selection']and not family['holdout_files_read'],'Common family frozen on N1024 calibration first')
    h.checked(family['original_policy_freeze']);root=Path(a.root).resolve();out=Path(a.out).resolve()
    h.require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009'),'New independent WCL namespace')
    bindings=phy.source_bindings(root);rt=phy.Runtime(root,family['family']);catalogue=rt.build_catalogue()
    h.require(len(catalogue['profiles'])==7*len(catalogue['layouts'])and len(catalogue['layouts'])>0,'Every legal MCS retains all seven prefix lengths')
    legal={(p['q'],p['nominal_rate'])for p in catalogue['profiles']};queries=entropy_candidates()
    admitted=[dict(c,actual_ldpc_layout_status='ACTUAL_CONSTRUCTOR_ADMITTED')for c in queries if(c['q'],c['nominal_rate'])in legal]
    rejected=[dict(c,actual_ldpc_layout_status='ACTUAL_CONSTRUCTOR_UNSUPPORTED')for c in queries if(c['q'],c['nominal_rate'])not in legal]
    h.require(len(admitted)+len(rejected)==48,'All original48 resource queries accounted for before quality')
    headers=[p['profile_id']for p in catalogue['profiles']]+[4095];cap=len(catalogue['layouts'])*8+len(headers)
    r=dict(schema=phy.QUAL_SCHEMA,root=str(root),out=str(out),family=family['family'],entropy_family_selection=h.desc(a.entropy_family_selection),
        catalogue=catalogue,source_bindings=bindings,input_bindings={rt.reference_qualification:h.sha(rt.reference_qualification)},
        backend_identity=rt.backend.identity,payload_seed=SEED,header_ids=headers,body_decoder_calls=len(catalogue['layouts'])*8,
        header_decoder_calls=len(headers),packet_cap=cap,admitted_candidates=admitted,unsupported_candidate_queries=rejected,
        deadline_unix=a.deadline_unix,max_seconds=7200,stop_files=[str(root/'STOP'),str(out/'STOP')],source_images_read=0,neural_model_calls=0,
        admission_rule='Actual encoder-constructor legality only, before image quality. Keep exact k and the uint13 cap; never approximate an unsupported layout.',automatic_successor=False)
    h.save(out/'constructor_catalogue.json',catalogue);h.save(out/'request.json',r)
    write_csv(out/'candidate_admission.csv',admitted+rejected)
    return dict(status='T6_ENTROPY_CONSTRUCTORS_ADMITTED_NOT_PACKET_QUALIFIED',request=h.desc(out/'request.json'),
        legal_layouts=len(catalogue['layouts']),legal_profiles=len(catalogue['profiles']),legal_candidates=len(admitted),
        unsupported_queries=len(rejected),packet_cap=cap,new_packet_decodes=0,new_model_calls=0)


def run(path):
    r=h.read(path);rh=h.sha(path);out=Path(r['out']);h.require(r['schema']==phy.QUAL_SCHEMA and r['payload_seed']==SEED,'Frozen entropy qualification')
    h.require(r['packet_cap']==8*len(r['catalogue']['layouts'])+len(r['header_ids']),'Exact finite qualification cap')
    phy.old.verify_bindings(r['source_bindings']);phy.old.verify_bindings(r['input_bindings']);h.checked(r['entropy_family_selection'])
    with h.lock(out/'qualification.lock'):
        cp=out/'completion.json'
        if cp.exists():
            d=h.read(cp);h.require(d['request']==h.desc(path)and d['status']=='PASS','Existing qualification differs');phy.old.verify_bindings(d['outputs'])
            return dict(status='REUSE_COMPLETED_N2048_ENTROPY_QUALIFICATION',new_packet_decodes=0)
        rt=phy.Runtime(r['root'],r['family']);rt.use_catalogue(r['catalogue']);h.require(rt.backend.identity==r['backend_identity'],'Actual backend differs')
        meter=Ledger(out/'packet_ledger.sqlite',rh,r['packet_cap']);began=time.monotonic();bodyrows=[];headerrows=[];outputs={}
        for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,h.stop)
        try:
            for j,layout in enumerate(r['catalogue']['layouts']):
                p=next(p for p in rt.catalogue['profiles']if p['layout_id']==layout['layout_id']);cap=p['source_capacity_bits']
                for index in range(8):
                    h.guard(r,began);rng=np.random.default_rng(np.random.SeedSequence([SEED,j,index]));mode=index%4
                    length=[2,cap,max(2,cap//2),cap-1][mode]
                    bits=np.zeros(length,np.uint8)if mode==0 else np.ones(length,np.uint8)if mode==1 else rng.integers(0,2,length,np.uint8)
                    ctr=j*8+index;wave,tx=phy.transmit_body(rt,bits,p,ctr,'WCL_T6_EC_QUALIFICATION')
                    noise=np.zeros_like(wave)if index<4 else rng.standard_normal(wave.shape).astype(np.float32)*np.float32(.001)
                    y=(wave+noise).astype(np.float32);event=f'{phy.QUAL_SCHEMA}/body{j:02d}/{index}'
                    rx=phy.receive_body(rt,y,p,60.,ctr,meter,event,'qualification','WCL_T6_EC_QUALIFICATION')
                    correct=rx['crc_accepted']and rx['parser_accepted']and rx['payload']==bits.tolist()and rx['decoded_bits']==phy.pack_body(bits,p).tolist()
                    row=dict(event_id=event,layout_id=layout['layout_id'],profile_id=p['profile_id'],q=p['q'],k=p['k'],n=p['n'],index=index,
                        noiseless=index<4,source_bits=length,correct=correct,CRC_accept=rx['crc_accepted'],body_energy=tx['body_energy'],received_sha256=phy.array_sha(y))
                    pp=out/'body_cases'/f'{j:02d}_{index}.json';h.save(pp,dict(row=row,tx=tx,rx=rx));outputs[str(pp)]=h.sha(pp);bodyrows.append(row)
                    h.require(correct,'Actual N2048 entropy body roundtrip failed')
                print(h.canonical(dict(stage='T6_EC_PHY_QUALIFICATION',completed_layouts=j+1,total=len(r['catalogue']['layouts']))),flush=True)
            for pid in r['header_ids']:
                h.guard(r,began);wave=np.asarray(rt.header.transmit(pid),np.float64);event=f'{phy.QUAL_SCHEMA}/header{pid:04d}'
                rx=phy.receive_header(rt,wave,60.,meter,event,'qualification');known=str(pid)in rt.profiles
                correct=rx['header_crc_ok']is True and rx['header_fields_legal']is known and rx['header_ok']is known and rx['profile_id']==(pid if known else None)
                row=dict(rx,test_profile_id=pid,known=known,correct=correct,event_id=event);pp=out/'header_cases'/f'{pid:04d}.json'
                h.save(pp,row);outputs[str(pp)]=h.sha(pp);headerrows.append(row);h.require(correct,'Actual N2048 entropy paid header roundtrip failed')
            snap=meter.snapshot();h.require(snap==dict(total=r['packet_cap'],unresolved=0,cap=r['packet_cap']),'Finite qualification not complete')
            for name,rows in [('body_qualification.csv',bodyrows),('header_qualification.csv',headerrows)]:
                write_csv(out/name,rows);outputs[str(out/name)]=h.sha(out/name)
            for name in('constructor_catalogue.json','candidate_admission.csv'):outputs[str(out/name)]=h.sha(out/name)
            done=dict(schema=phy.QUAL_SCHEMA,status='PASS',request=h.desc(path),family=r['family'],entropy_family_selection=r['entropy_family_selection'],
                backend_identity=rt.backend.identity,catalogue_sha256=rt.catalogue['catalogue_sha256'],profile_count=len(rt.profiles),
                actual_layout_count=len(rt.catalogue['layouts']),actual_body_decodes=len(bodyrows),actual_header_decodes=len(headerrows),
                packet_decode_count=snap['total'],ledger=snap,outputs=outputs,source_images_read=0,neural_model_calls=0,automatic_successor=False)
            h.save(cp,done);return {k:v for k,v in done.items()if k not in('outputs','backend_identity')}
        finally:meter.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);q=s.add_parser('prepare')
    for x in('root','entropy-family-selection','out'):q.add_argument('--'+x,required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q=s.add_parser('run');q.add_argument('--request',required=True)
    a=p.parse_args();print(h.canonical(prepare(a)if a.command=='prepare'else run(a.request)))
