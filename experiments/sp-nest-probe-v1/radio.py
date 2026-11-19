"""Reference physical channel. Only received symbols and public context cross the RX boundary."""
import hashlib, json, os, sys, subprocess
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from profiles import capacity, NBODY, OFF, count, public_directory

PAM={2:np.array([1.,-1.]),4:np.array([-3.,-1.,3.,1.])/np.sqrt(5),6:np.array([-7.,-5.,-1.,-3.,7.,5.,1.,3.])/np.sqrt(21)}

def mask(n,counter,layer):
    if layer not in ('B','E') or type(counter) is not int or counter<0: raise ValueError('invalid public scrambling context')
    key=('SP_NEST_V1:%s:%d'%(layer,counter)).encode('ascii')
    return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest()[:16],'big'))).integers(0,2,n,dtype=np.uint8)

def mod(bits,q):
    x=np.asarray(bits).reshape(-1,q//2);idx=x @ (1<<np.arange(q//2-1,-1,-1)); return PAM[q][idx].reshape(-1,2)

def llr(y,n0,q,qe=None,alpha=1.):
    """Positive log(P1/P0) for Sionna. n0 is total complex AWGN variance."""
    if not n0>0 or not 0<alpha<=1: raise ValueError('variance or power')
    p=PAM[q]; t=np.zeros(1) if qe is None else PAM[qe]
    levels=np.sqrt(alpha)*p[:,None]+np.sqrt(1-alpha)*t[None,:]
    sq=-np.square(np.asarray(y,dtype=np.float64).reshape(-1,1,1)-levels)/n0
    v=logsumexp(sq,axis=-1); labels=np.arange(len(p)); vals=[]
    for b in range(q//2-1,-1,-1):
        one=((labels>>b)&1).astype(bool)
        vals.append(logsumexp(v[:,one],axis=-1)-logsumexp(v[:,~one],axis=-1))
    return np.stack(vals,axis=-1).ravel()

def wslpath(p):
    s=Path(p).resolve().as_posix(); return '/mnt/'+s[0].lower()+s[2:]

class Radio:
    def __init__(self,root):
        root=Path(root).resolve(); exp=root/'experiments'
        for folder in ('prior-aware-grouped-mcs-v1','entropy-partial-mismatch-20261010/scripts','layered-entropy-feasibility-20261014','wcl-evidence-closure-20261009/scripts'):
            sys.path.insert(0,str(exp/folder))
        import ep_phy,ldpc_backend,uep_phy,packet_two_group
        from t1_entropy_core import load_integer_codec
        self.legacy=uep_phy; self.be=ep_phy.Backend(ldpc_backend.SionnaBackend('cpu')); self.be.torch.set_num_threads(1)
        self.packet=packet_two_group; self.header=uep_phy.Header(root); self.directory=public_directory()
        self.entropy,self.Encoder,self.Decoder=load_integer_codec(root); self.worker=None
        if os.name=='nt':
            self.worker=subprocess.Popen(['wsl','-e','/usr/bin/python3','-u',wslpath(Path(__file__).with_name('header_worker.py')),wslpath(root)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf8')

    def close(self):
        if self.worker is not None:
            self.worker.stdin.close(); rc=self.worker.wait(timeout=25)
            if rc: raise RuntimeError('Original header subprocess exit %s'%rc)

    def header_rx(self,y,snr):
        if self.worker is None: return self.header.receive(y,snr)
        self.worker.stdin.write(json.dumps({'y':y.tolist(),'snr':snr})+'\n'); self.worker.stdin.flush()
        line=self.worker.stdout.readline()
        if not line: raise RuntimeError('Original header worker stopped before header decision')
        return json.loads(line)

    def layer_tx(self,info,q,c,layer):
        n=NBODY*q; code=self.be.encode(np.array(info,copy=True)[None,:],n,q)[0].detach().cpu().numpy().astype(np.uint8)
        return mod(code^mask(n,c,layer),q)

    def hard(self,logits,k,q,c,layer):
        n=NBODY*q; signed=logits*(1.-2.*mask(n,c,layer)); raw=self.be.decode(signed[None,:],k,n,q)[0].detach().cpu().numpy()
        if raw.shape!=(k,) or not np.isin(raw,(0,1)).all(): raise RuntimeError('invalid decoder output')
        return raw.astype(np.uint8)

    def transmit(self,pid,ib,ie,counter):
        j,m,k,br,qe,er,ap=self.directory[pid]; a=ap/100
        if len(ib)!=capacity(2,br)+29 or len(ie)!=capacity(qe,er)+29: raise ValueError('wrong TX paid layer size')
        b=self.layer_tx(ib,2,counter,'B'); e=self.layer_tx(ie,qe,counter,'E'); body=np.sqrt(a)*b+np.sqrt(1-a)*e
        wave=np.vstack([self.header.transmit(int(pid)),body]); assert wave.shape==(1024,2)
        return wave,dict(B=float((b*b).sum(1).mean()),E=float((e*e).sum(1).mean()),body=float((body*body).sum(1).mean()),frame=float((wave*wave).sum(1).mean()),cross=float((b*e).sum(1).mean()))

    def receive(self,observed,snr_db,counter,source):
        y=np.asarray(observed); assert y.shape==(1024,2) and np.isfinite(y).all()
        h=self.header_rx(y[:68],snr_db); pid=h['profile_id']; out={'pid':pid,'B':None,'E':None,'B_bits':None,'E_bits':None,'sic':False}
        if not h['header_ok'] or pid not in self.directory: return dict(out,status='HEADER_REJECT_GRAY')
        if pid==0: return dict(out,status='NULL_GRAY')
        j,m,k,br,qe,er,ap=self.directory[pid]; a=ap/100; n0=2./10**(snr_db/10.); body=y[68:]; kb=capacity(2,br)+29; ke=capacity(qe,er)+29
        ib=self.hard(llr(body,n0,2,qe,a),kb,2,counter,'B'); b=self.packet.parse_group(ib,kb)
        if not b['accepted']: return dict(out,status='BASE_'+b['status']+'_GRAY')
        try: bt=source.decode_base(j,b['bits'])
        except ValueError: return dict(out,status='BASE_ARITHMETIC_REJECT_GRAY')
        out.update(B=bt,B_bits=b['bits']);
        if (j,0)==(m,k): return dict(out,status='BASE_PAID_EMPTY')
        rem=(body-np.sqrt(a)*self.layer_tx(ib,2,counter,'B'))/np.sqrt(1-a); out['sic']=True
        ie=self.hard(llr(rem,n0/(1-a),qe),ke,qe,counter,'E'); e=self.packet.parse_group(ie,ke)
        if not e['accepted']: return dict(out,status='BASE_ONLY_'+e['status'])
        try: et=source.decode_enhancement(j,(m,k),bt,e['bits'])
        except ValueError: return dict(out,status='BASE_ONLY_ARITHMETIC_REJECT')
        return dict(out,E=et,E_bits=e['bits'],status='BOTH_ACCEPTED')
