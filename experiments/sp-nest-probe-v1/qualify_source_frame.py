"""Two additional end-to-end engineering fixtures; no image/model quality claims."""
import argparse,hashlib,json,time,sys
from pathlib import Path
import numpy as np
from profiles import OFF,count
from qualify_split import Fixture,TOKEN_SHA,require
from split_prefixes import SplitPrefixes,select_profile
from source_frame import emit
from radio import Radio

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','tokens','out'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.out.exists(),'choose a fresh output file'); t=time.perf_counter();c=time.process_time(); r=Radio(a.repo)
    try:
        from t1_entropy_core import source_tokens
        with np.load(a.tokens,allow_pickle=False) as z: q=source_tokens(z['tokens']).copy()
        require(hashlib.sha256(b'int64:680\0'+q.astype('<i8').tobytes()).hexdigest()==TOKEN_SHA,'source mismatch')
        tx=SplitPrefixes(Fixture(r.Encoder,r.Decoder));_,bases,extras=tx.encode(q)
        rx=SplitPrefixes(Fixture(r.Encoder,r.Decoder)); inverse={v:k for k,v in r.directory.items() if k}
        choice=select_profile(bases,extras,6,(10,0),'2/3',4,'3/4',80,'QUARTER',inverse)
        require(choice is not None,'expected payload');pid,bb,eb=choice;j,m,k,*_=r.directory[pid]
        require(len(eb)>0,'expected a nonempty enhancement');wave,energy=emit(r,choice,10003);rr=r.receive(wave.copy(),60.,10003,rx)
        require(rr['status']=='BOTH_ACCEPTED' and rr['pid']==pid and np.array_equal(rr['B_bits'],bb) and np.array_equal(rr['E_bits'],eb) and np.array_equal(np.r_[rr['B'],rr['E']],q[:count((m,k))]),'paid packet reconstruction mismatch')
        w0,e0=emit(r,None,10004);zero=r.receive(w0.copy(),60.,10004,rx)
        require(zero['status']=='NULL_GRAY' and zero['pid']==0 and not zero['sic'] and zero['B'] is None,'NULL semantics')
        # The CRC/header fallback fixture is forced for framing coverage, not a real source NULL statistic.
        v=dict(status='PASS_ENGINEERING_ONLY',conditional_CDF='controlled numerical fixture, not VAR',noise_added=False,demapper_reference_snr_db=60,source_token_sha256=TOKEN_SHA,nonempty=dict(j=j,m=m,K=k,profile_id=pid,native_source_tokens_recovered=count((m,k)),B_bits=len(bb),E_bits=len(eb),symbols=len(wave),energy=energy,status=rr['status']),forced_paid_NULL=dict(symbols=len(w0),energy=e0,status=zero['status']),additional_engineering_waveforms=2,scientific_frames=0,neural_calls=0,gpu_hours=0,parent_cpu_seconds=time.process_time()-c,wall_seconds=time.perf_counter()-t)
        with a.out.open('x',encoding='utf8') as f: json.dump(v,f,indent=2);f.write('\n')
        print(json.dumps(v))
    finally:r.close()
if __name__=='__main__':main()
