"""SP0 deterministic 24-frame qualification. Never supplies source truth to receive()."""
import argparse,csv,hashlib,json,platform,time
from pathlib import Path
import numpy as np
from profiles import ALPHA,BR,EPHY,OFF,count,capacity,reachable_superset
from radio import Radio,llr,PAM

def digest(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
def write_csv(path,rows):
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class FixtureDecoder:
    """Public, full-support 4096-symbol integer CDF. Synthetic; not neural image input."""
    def __init__(self,enc,dec): self.enc,self.dec=enc,dec
    def table(self,n,offset):
        f=np.ones(4096,dtype=np.int64);f[(np.arange(256)+offset)%4096]+=(2**24-4096)//256
        return np.broadcast_to(np.r_[0,f.cumsum()],(n,4097))
    def encode(self,tokens,offset):
        en=self.enc();en.encode(tokens,self.table(len(tokens),offset));return en.finish()
    def decode(self,n,offset,bits):
        de=self.dec(bits); values=de.decode(self.table(n,offset))
        if de.position-len(bits)!=30 or not np.array_equal(self.encode(values,offset),bits): raise ValueError('noncanonical termination')
        return values
    @staticmethod
    def condition(b):return int(np.dot(np.arange(1,len(b)+1,dtype=np.int64),b)%4096)
    def decode_base(self,j,bits):return self.decode(OFF[j],0,bits)
    def decode_enhancement(self,j,end,b,bits):return self.decode(count(end)-OFF[j],self.condition(b),bits)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    if (args.out/'sp0_radio.json').exists() or (args.out/'sp0_frames.csv').exists():raise FileExistsError('protect completed outputs; choose a fresh destination')
    t0=time.time();c0=time.process_time();r=Radio(args.repo);rng=np.random.default_rng(20261119);src=FixtureDecoder(r.Encoder,r.Decoder)
    try:
        _,raw,witness=reachable_superset()
        catalog=[{'profile_id':0,'j':'','m':'','K':'','q_B':'','rate_B':'','q_E':'','rate_E':'','alpha_percent':'','grids':'NULL'}]
        for pid,pr in list(r.directory.items())[1:]:
            j,m,k,br,qe,er,ap=pr;catalog.append(dict(zip(catalog[0],[pid,j,m,k,2,br,qe,er,ap,'|'.join(sorted(witness[pr]))])))
        write_csv(args.out/'public_directory.csv',catalog)
        q=torch=r.be.torch
        x=rng.uniform(-2,2,(150,2)); errors=[]
        for snr in (4,13,19):
            ref=r.legacy.demap_torch(torch.as_tensor(x[None],dtype=torch.float64),snr,2)[0].numpy()
            for qe in (2,4,6):errors.append(float(np.max(np.abs(llr(x,2/10**(snr/10),2,qe,1.-1e-14)-ref))))
        assert max(errors)<1e-6
        layouts={f'{q}:{s}':r.be.plan(capacity(q,s)+29,956*q,q) for q,s in [(2,k) for k in BR]+list(EPHY)}
        frames=[]; last=None
        for i in range(24):
            a,br,(qe,er)=ALPHA[i%4],BR[i%3],EPHY[i%5]; opts=[]
            for pid,pr in list(r.directory.items())[1:]:
                j,m,k,b,v,e,al=pr;delta=count((m,k))-OFF[j]
                if (b,v,e,al)==(br,qe,er,a) and delta>0 and 8.01*OFF[j]+5<=capacity(2,br) and 8.01*delta+5<=capacity(qe,er):opts.append((pid,pr))
            pid,pr=opts[int(rng.integers(len(opts)))];j,m,k,_,_,_,_=pr
            bt=rng.integers(0,256,OFF[j],dtype=np.int64);et=(rng.integers(0,256,count((m,k))-OFF[j],dtype=np.int64)+src.condition(bt))%4096
            bb=src.encode(bt,0);eb=src.encode(et,src.condition(bt));ib=r.packet.pack_group(bb,capacity(2,br)+29);ie=r.packet.pack_group(eb,capacity(qe,er)+29)
            wave,energy=r.transmit(pid,ib,ie,i); rr=r.receive(wave.copy(),60.,i,FixtureDecoder(r.Encoder,r.Decoder));assert rr['pid']==pid and rr['status']=='BOTH_ACCEPTED' and rr['sic']
            for da,db in ((bb,rr['B_bits']),(eb,rr['E_bits']),(bt,rr['B']),(et,rr['E'])): assert np.array_equal(da,db)
            frames.append(dict(frame=i,profile_id=pid,j=j,m=m,K=k,alpha=a/100,rate_B=br,q_E=qe,rate_E=er,B_bits=len(bb),E_bits=len(eb),B_bits_sha256=digest(bb),E_bits_sha256=digest(eb),wave_sha256=digest(wave),**{'energy_'+x:v for x,v in energy.items()},rx=rr['status'],B_exact=True,E_exact=True,B_tokens_exact=True,E_tokens_exact=True));last=(pid,ib,ie,i,eb)
            print('qualified',i+1,'/ 24',flush=True)
        pid,ib,ie,c,eb=last;checks={};
        def check(name,w,expected):
            rr=r.receive(w,60.,c,FixtureDecoder(r.Encoder,r.Decoder));assert rr['status']==expected,(name,rr['status']);checks[name]=expected;return rr
        i1=ib.copy();i1[-1]^=1;w,_=r.transmit(pid,i1,ie,c);assert check('bad_base_crc',w,'BASE_CRC_REJECT_GRAY')['sic'] is False
        e1=ie.copy();e1[-1]^=1;w,_=r.transmit(pid,ib,e1,c);assert check('bad_enhancement_crc',w,'BASE_ONLY_CRC_REJECT')['B'] is not None
        e2=r.packet.pack_group(np.r_[eb,np.uint8(0)],len(ie));w,_=r.transmit(pid,ib,e2,c);check('noncanonical_enhancement_with_valid_crc',w,'BASE_ONLY_ARITHMETIC_REJECT')
        e3=ie.copy();e3[13+len(eb)]=1;e3[-16:]=r.packet.intbits(r.packet.crc16(e3[:-16]),16);w,_=r.transmit(pid,ib,e3,c);check('nonzero_padding_with_valid_crc',w,'BASE_ONLY_KNOWN_PADDING_REJECT')
        w[:68]=r.header.transmit(4095);check('unknown_public_id',w,'HEADER_REJECT_GRAY'); w[:68]=r.header.transmit(0);check('null_header',w,'NULL_GRAY')
        for p0,row in list(r.directory.items())[1:]:
            j,m,k,br,qe,er,ap=row
            if (m,k)==(j,0) and 8.01*OFF[j]+5<=capacity(2,br):break
        else:raise AssertionError('no empty-enhancement fixture')
        b=src.encode(rng.integers(0,256,OFF[j],dtype=np.int64),0);w,_=r.transmit(p0,r.packet.pack_group(b,capacity(2,br)+29),r.packet.pack_group(np.array([],dtype=np.uint8),capacity(qe,er)+29,True),c);assert not check('empty_enhancement',w,'BASE_PAID_EMPTY')['sic']
        write_csv(args.out/'sp0_frames.csv',frames)
        dump(args.out/'sp0_radio.json',dict(status='PASS_RADIO_FIXTURES_ONLY',source='synthetic full-support integer CDF; enhancement conditional on decoded base tokens; no image quality measured',awgn_added=False,demapper_reference_snr_db=60,random_seed=20261119,actual_noiseless_random_frames=len(frames),extra_failure_semantics_checks=checks,limit_max_abs_llr=max(errors),limit_threshold=1e-6,limit_test='alpha=1-1e-14, (4,13,19) dB, q_E=(2,4,6), original QPSK float64 reference',directory=len(r.directory),unpruned_including_null=raw,layouts=layouts,backend_identity=r.be.identity,wall_seconds=time.time()-t0,parent_cpu_seconds=time.process_time()-c0,python=platform.python_version(),gpu_hours=0,training_steps=0))
    finally:r.close()

if __name__=='__main__':main()
