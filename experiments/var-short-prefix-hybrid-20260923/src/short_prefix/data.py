"""A hash-checked population adapter for the shared paired training engine."""
import numpy as np,torch
from var_comm.prefix_training_data import read_image_population
from var_comm.study import seeded_noise
from latent_enhancement.runtime import digest,verify_snapshot
from latent_enhancement_b.common import CACHE,sample_key_digest
from .common import OUT,read
from .protocol import config,allocation,standard_noise

class Population:
    def __init__(self,role,m=None,N=4084,nd=None,preflight=False):
        self.images,self.labels,self.ids,self.image_bindings=read_image_population(role)
        cfg=config();self.role=role;self.m=m;self.N=N;self.snrs=cfg['snrs_db']
        if preflight:
            if role!='calibration' or not m:raise ValueError('preflight scope')
            self.images=self.images[:100];self.labels=self.labels[:100];self.ids=self.ids[:100]
        self.seeds=cfg['training_base_seeds'] if role=='train' else cfg['calibration_seeds']
        self.ledger=allocation(m,N,nd) if m else None
        self.folder=OUT/('preflight' if preflight else 'cache')/f"N{N}_m{m}_ND{self.ledger['ND']}"/role if m else CACHE/role
        self.bindings={};values={};ids=[]
        if m:
            done=read(self.folder/'completion.json');reg=read(self.folder/'registration.json')
            verify_snapshot(reg['bindings'])
            if done['registration_sha256']!=digest(self.folder/'registration.json'):raise RuntimeError('cache registration hash')
            if done['sources']!=len(self.ids) or done['scope']!=reg['scope']:raise RuntimeError('cache completion scope')
            scope=reg['scope']
            if scope['ledger']!=self.ledger or scope['role']!=role or scope['snrs_db']!=self.snrs or scope['seeds']!=self.seeds or scope['precision']!=cfg['precision']:raise RuntimeError('cache execution scope')
            names=('F','B_TX','B_RX','G_RX','status','label');self.cache_identity=done
            self.bindings[str(self.folder/'completion.json')]=digest(self.folder/'completion.json')
        else:names=('F',);self.cache_identity={'source':'original F, checked shard hashes'}
        for p in sorted(self.folder.glob('shard_*.pt')):
            meta=read(p.with_suffix('.json'))
            if digest(p)!=meta['sha256']:raise RuntimeError('population file hash')
            self.bindings[str(p)]=meta['sha256'];v=torch.load(p,map_location='cpu',weights_only=True)
            current_ids=v['image_ids'];ids.extend(current_ids)
            if m:
                if done['hashes'].get(p.name)!=meta['sha256'] or v['identity']!=meta['identity'] or v['identity']['scope']!=scope:raise RuntimeError('cache shard identity')
                if v['identity']['sample_key_sha256']!=sample_key_digest(current_ids,self.snrs,self.seeds):raise RuntimeError('cache full sample keys')
                for k in ('B_RX','G_RX','status','label'):
                    if v[k].shape[:3]!=(len(current_ids),len(self.snrs),len(self.seeds)):raise RuntimeError('cache grid shape')
            for k in names:values.setdefault(k,[]).append(v[k])
        if ids!=self.ids or len(ids)!=(100 if preflight else cfg['data'][role]) or len(set(ids))!=len(ids):raise RuntimeError('source identity/count/order')
        self.values={k:torch.cat(v) for k,v in values.items()}
    def __len__(self):return len(self.ids)
    def batch(self,ids,si,ni,noise_seeds,device):
        b={'F':self.values['F'][ids].to(device),'target':self.images[ids].to(device).float()/255,'snr':torch.tensor(self.snrs)[si].to(device)}
        if self.m:
            b['B_TX']=self.values['B_TX'][ids].to(device)
            for k in ('B_RX','G_RX','status','label'):b[k]=self.values[k][ids,si,ni].to(device)
            noise=np.stack([standard_noise(self.ids[int(i)],s,self.ledger,'continuous') for i,s in zip(ids,noise_seeds)])
        else:noise=np.stack([seeded_noise(f'VAR-CONTINUOUS-{self.N}|'+self.ids[int(i)],int(s),(self.N,2)) for i,s in zip(ids,noise_seeds)])
        b['noise']=torch.as_tensor(noise,device=device,dtype=torch.float32)
        return b
