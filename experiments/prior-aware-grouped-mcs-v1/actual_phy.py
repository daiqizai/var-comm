"""Full paid waveform, actual header format discovery and continuous accepted prefix."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from uep_common import SIZES,require,rng,array_sha,identity
from uep_phy import Header,transmit_packet,receive_packet,continuous_prefix

def indices_to_bits(values):
    a=np.asarray(values,dtype=np.int64)
    require(a.ndim==1 and ((a>=0)&(a<4096)).all(),'12-bit VQ values required')
    return ((a[:,None]>>np.arange(11,-1,-1))&1).astype(np.uint8).reshape(-1)

def bits_to_indices(bits):
    b=np.asarray(bits,dtype=np.uint8)
    require(b.ndim==1 and len(b)%12==0 and np.isin(b,(0,1)).all(),'Aligned VQ bit vector')
    return (b.reshape(-1,12).astype(np.int64)*(1<<np.arange(11,-1,-1))).sum(1)

def serialize_groups(scales,profile,positions):
    """TX-only source-dependent operation. Position computation is paid separately."""
    m,K=int(profile['m']),int(profile['K']);j=profile['j']
    prefix=np.concatenate([np.asarray(s,dtype=np.int64) for s in scales[:m]])
    require(len(prefix)==sum(p*p for p in SIZES[:m]),'Invalid source scale shapes')
    if K:
        pos=np.asarray(positions,dtype=np.int64)
        require(pos.shape==(K,) and len(set(pos.tolist()))==K and pos.min()>=0 and pos.max()<SIZES[m]**2,'Actual entropy order positions required')
        prefix=np.concatenate((prefix,np.asarray(scales[m])[pos]))
    else:require(positions is None or len(positions)==0,'K0 has no partial positions')
    groups=[prefix] if profile['G']==1 else np.split(prefix,[sum(p*p for p in SIZES[:int(j)])])
    result=[indices_to_bits(v) for v in groups]
    require([len(v) for v in result]==[g['source_bits'] for g in profile['groups']],'Raw group accounting differs')
    return result

def accepted_state(profile,payloads,accepted):
    kept=continuous_prefix(payloads,accepted)
    if not kept:return dict(kind='gray',prefix=[],partial_values=[],m=0,K=0,accepted_groups=0)
    if profile['G']==2 and len(kept)==1:m,K=int(profile['j']),0
    else:m,K=int(profile['m']),int(profile['K'])
    values=bits_to_indices(np.concatenate(kept));count=sum(p*p for p in SIZES[:m])
    require(len(values)==count+K,'Received payload does not match actual decoded public format')
    prefix=[x.tolist() for x in np.split(values[:count],np.cumsum([p*p for p in SIZES[:m]])[:-1])]
    return dict(kind='tokens',prefix=prefix,partial_values=values[count:].tolist(),m=m,K=K,accepted_groups=len(kept))

class FrameLink:
    def __init__(self,root,backend,codebook):
        entries=codebook['entries'];self.codebook={str(int(p['profile_id'])):p for p in entries}
        require(len(self.codebook)==len(entries)<=4096,'Invalid public profile codebook')
        self.header=Header(root);self.backend=backend
        self.codebook_sha256=identity(entries)

    def transmit(self,profile_id,payloads,frame_counter):
        p=self.codebook[str(profile_id)];require(len(payloads)==p['G'],'Wrong group count')
        waves=[self.header.transmit(int(profile_id))];ledgers=[]
        for index,(payload,g) in enumerate(zip(payloads,p['groups'])):
            require(len(payload)==g['source_bits'],'TX source payload length differs')
            q=2 if g['modulation']=='QPSK' else 4
            w,ledger=transmit_packet(self.backend,np.asarray(payload)[None],g['transmitted_bits'],q,[frame_counter],index,'actual-body')
            waves.append(w[0].cpu().numpy().astype(np.float64));ledgers.append(ledger)
        if p['idle_symbols']:waves.append(np.ones((p['idle_symbols'],2),dtype=np.float64))
        wave=np.concatenate(waves);require(wave.shape==(p['N'],2),'Full paid N differs')
        return wave,dict(N=p['N'],E=float(np.square(wave).sum()),header_uses=68,
            body_uses=sum(g['symbols'] for g in p['groups']),idle_uses=p['idle_symbols'],groups=ledgers,
            waveform_sha256=array_sha(wave),profile_id=profile_id,frame_counter=frame_counter)

    def receive(self,received,N,snr,frame_counter):
        """No TX profile, source tokens, label, clean waveform or noise input exists."""
        y=np.asarray(received,dtype=np.float64);require(y.shape==(N,2),'Complete noisy waveform required')
        header=self.header.receive(y[:68],snr,self.codebook)
        event=dict(header,groups=[],N=N,received_sha256=array_sha(y),frame_counter=frame_counter,
                   codebook_sha256=self.codebook_sha256,state=dict(kind='gray',prefix=[],partial_values=[],m=0,K=0,accepted_groups=0))
        if not header['header_ok']:return event
        p=self.codebook[str(header['profile_id'])]
        if p['N']!=N:
            event.update(header_ok=False,header_fields_legal=False,reason='decoded_profile_budget_mismatch');return event
        cursor=68;payloads=[];accepts=[]
        for index,g in enumerate(p['groups']):
            q=2 if g['modulation']=='QPSK' else 4
            symbols=self.backend.torch.as_tensor(y[None,cursor:cursor+g['symbols']],dtype=self.backend.torch.float32,device=self.backend.device)
            payload,ok,_=receive_packet(self.backend,symbols,g['information_bits'],g['transmitted_bits'],q,snr,[frame_counter],index,'actual-body')
            cursor+=g['symbols'];payloads.append(payload[0]);accepts.append(bool(ok[0]))
            event['groups'].append(dict(index=index,crc_accept=bool(ok[0]),accepted_payload=payload[0].tolist() if ok[0] else None,
                decoded_payload_sha256=array_sha(payload[0]),source_bits=g['source_bits'],phy_key=g['phy_key']))
        event['state']=accepted_state(p,payloads,accepts)
        # Rejected hard bits have never been exposed to renderer or source-state cache.
        event['state_sha256']=identity(event['state']);return event

    def execute(self,profile_id,payloads,N,snr,frame_counter,noise_seed,source_id):
        """Simulation shell alone sees truth; it observes errors after RX completes."""
        wave,ledger=self.transmit(profile_id,payloads,frame_counter)
        noise=rng('real-link',N,int(noise_seed),str(source_id)).standard_normal(wave.shape)
        received=wave+noise*10**(-float(snr)/20)
        event=self.receive(received,N,snr,frame_counter)
        header_correct=event['header_ok'] and event['profile_id']==profile_id
        truth=dict(header_correct=bool(header_correct),header_false_accept=bool(event['header_ok'] and not header_correct),
                   noise_sha256=array_sha(noise),undetected_body_errors=0)
        if header_correct:
            for index,(g,payload) in enumerate(zip(event['groups'],payloads)):
                correct=g['crc_accept'] and np.array_equal(g['accepted_payload'],payload)
                g['truth_correct_after_receiver']=bool(correct)
                g['false_accept_after_receiver']=bool(g['crc_accept'] and not correct)
                truth['undetected_body_errors']+=int(g['false_accept_after_receiver'])
        return event,ledger,truth
