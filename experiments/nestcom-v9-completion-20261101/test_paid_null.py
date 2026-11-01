"""Bounded current-directory regression; real CPU LDPC, no image inference."""
from paid_null import *
from opt_common import read,save,pin,sha,checked,Ledger,asha
from frame_worker import Packets
import os,time,traceback

def main(request,expected):
    assert sha(request)==expected;r=read(request);[checked(v) for v in r['code']]
    out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,{44,45})
    ledger=Ledger(out/'calls.sqlite',dict(header=21,body=20));start=time.monotonic()
    try:
        old=read(checked(r['catalogue']));cat=extend_catalogue(old);save(out/'catalogue.json',cat)
        phy,rt=runtime(checked(r['qualification']),cat);packets=Packets(ledger);checks=[];frames=[]
        def check(name,condition):
            assert condition,name
            checks.append(name)
        def event(name,p,b,info=None,header=None):
            counter=len(frames)+900000;wave,meta=rt.transmit(p['profile_id'],b,counter)
            if info is not None:
                code=rt.backend.encode(info[None],p['n'],p['q']);t=rt.torch
                mask=t.as_tensor(phy.scramble_mask(p['n'],counter,phy.WIRE_PROTOCOL)[None])
                body=phy.as_numpy(rt.mapping.modulate((code+mask)%2,p['q'],t,rt.legacy))[0]
                wave[68:]=body
            if header is not None:wave[:68]=header
            rx=rt.receive(wave,60,counter,rt.profiles,packets,name,phase='V9_PAID_NULL_REGRESSION')
            projection=received_projection(rx)
            save(out/(name+'.json'),dict(RX=rx,projection=projection,wave_sha256=asha(wave),transmission=meta))
            frames.append(name);return rx,projection
        check('old_records_exact',cat['profiles'][:3378]==old['profiles'])
        check('unique_ids_reserved_BPG',len({p['profile_id'] for p in cat['profiles']})==3385 and not {p['profile_id'] for p in cat['profiles']}&set(range(3000,3012)))
        for q in (2,4,6):
            for rate in ('1/2','2/3','3/4','5/6'):
                eligible=[p for p in cat['profiles']if(p['q'],p['nominal_rate'])==(q,rate)]
                streams={'m4_K0':np.zeros(eligible[0]['k']-28,np.uint8)}
                p,b,no_fit,_=choose(streams,cat,q=q,rate=rate,target=(4,0))
                rx,pr=event('null_q%d_r%s'%(q,rate.replace('/','-')),p,b)
                check('nofit_paid_null_q%d_%s'%(q,rate),no_fit and p['family']==NULL and rx['header']['header_ok']and rx['body']['parser_accepted']and pr==dict(status='FIXED_GRAY',reason='EXPLICIT_NULL_PROFILE'))
        q,rate=4,'2/3';cap=next(p['k']-29 for p in cat['profiles']if(p['q'],p['nominal_rate'])==(q,rate))
        streams={f'm{m}_K{k}':np.zeros(cap+1,np.uint8)for m,k in endpoints('token')}
        for chosen,target,label in [((4,2),(4,2),'exact_capacity'),((4,1),(4,2),'same_scale_fallback'),((4,24),(5,0),'cross_scale_fallback')]:
            a=dict(streams);a['m%d_K%d'%chosen]=np.zeros(cap,np.uint8)
            p,b,nf,attempts=choose(a,cat,q=q,rate=rate,target=target)
            check(label,(p['m'],p['K'])==chosen and len(b)==cap and not nf)
        try:choose({},cat,q=q,rate=rate,target=(4,0))
        except KeyError:check('missing_source_is_error',True)
        else:raise AssertionError('Missing evidence became NULL')
        try:choose({'m4_K0':np.array([0,2],np.uint8)},cat,q=q,rate=rate,target=(4,0))
        except AssertionError:check('invalid_source_is_error',True)
        else:raise AssertionError('Invalid evidence became NULL')
        reps=[next(p for p in old['profiles']if p['family']==f and (p['K']>0 if kind=='partial'else True) and (p['K']==1 if kind=='token'else True))for f,kind in [('EC_VAR_WHOLE','whole'),('EC_VAR_PARTIAL','partial'),('EC_VAR_PARTIAL','token'),('ADA_PREDICTED_CLASS_PREFIX','ada')]]
        for name,p in zip(('whole','quarter','token','ada'),reps):
            b=np.array([0,1]*20,np.uint8);rt.catalogue=old;rt.profiles={str(p['profile_id']):p for p in old['profiles']}
            w0,m0=rt.transmit(p['profile_id'],b,8000)
            rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']}
            w1,m1=rt.transmit(p['profile_id'],b,8000)
            check('normal_%s_byte_wave_exact'%name,np.array_equal(w0,w1)and m0==m1 and np.array_equal(phy.pack_body(b,p),phy.pack_body(b,rt.profiles[str(p['profile_id'])])))
            rx,pr=event('normal_'+name,p,b);check('normal_'+name+'_actual_decode',rx['body']['payload']==b.tolist())
        p=next(p for p in cat['profiles']if p['family']==NULL);b=np.zeros(2,np.uint8)
        rx,pr=event('invalid_null_payload',p,np.array([0,1],np.uint8));check('invalid_NULL_reject',pr['reason']=='INVALID_NULL_PAYLOAD')
        clean=phy.pack_body(b,p)
        bad=clean.copy();bad[-1]^=1
        rx,pr=event('bad_body_crc',p,b,info=bad);check('body_crc_reject',rx['body']['status']=='CRC_REJECT'and pr['status']=='FIXED_GRAY')
        bad=clean[:-16].copy();bad[:13]=0
        rx,pr=event('bad_length',p,b,info=phy.append_crc(bad));check('length_reject',rx['body']['invalid_reason']=='L_OUTSIDE_PROFILE_CAPACITY')
        bad=clean[:-16].copy();bad[15]=1
        rx,pr=event('bad_padding',p,b,info=phy.append_crc(bad));check('padding_reject',rx['body']['invalid_reason']=='NONZERO_KNOWN_PADDING')
        s=rt.header.s;hb=s.append_crc(s.indices_to_bits([p['profile_id']],12));hb[-1]^=1
        header=s.encode_packet(hb,68,with_crc=False)['symbols']
        rx,pr=event('bad_header_crc',p,b,header=header);check('header_crc_reject',not rx['header']['header_crc_ok']and rx['body']is None and pr['status']=='FIXED_GRAY')
        check('all_counts',len(frames)==21 and ledger.summary()['unresolved']==0)
        save(out/'completion.json',dict(status='PASS',checks=checks,frames=len(frames),ledger=ledger.summary(),seconds=time.monotonic()-start,catalogue=pin(out/'catalogue.json'),CUDA_initialized=rt.torch.cuda.is_initialized(),new_model_calls=0,old_18_frames_not_relabelled=True))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
