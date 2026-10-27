"""One-time metadata/noise preparation. No model, encoding, or packet decoding."""
from p1_common import *
import subprocess
def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='1f2a86a41b65fe90c0bf245076212b0a27852deb'
 assert not (O/'p1_registration.json').exists() and not (L/'link').exists()
 P=L/'preparation';P.mkdir(parents=True,exist_ok=False);O.mkdir(parents=True,exist_ok=True)
 regpath=R/'results/n1024_consistency_20261019_v1/repair_v1/registration.json';reg=read(regpath)
 poppath=R/'results/arpc_external_baseline_20261019_v1/sources.json';pop=read(poppath)['external_common100_v1'];ids={s['source_id']for s in pop};assert len(pop)==len(ids)==100
 allsources=cr(reg['sources']);sources=[s for s in allsources if s['source_id']in ids];assert len(sources)==100
 bankpath=R/'results/arpc_external_baseline_20261019_v1/a3_metrics_v1/bootstrap_identity.json';bank=read(bankpath);ix=np.load(checked(bank['indices']),allow_pickle=False);assert ix.shape==(10000,100) and set(bank['source_ids'])==ids
 lookup={s['source_id']:s for s in sources};sources=[lookup[sid]for sid in bank['source_ids']]
 allrows=cr(read(R/'results/n1024_consistency_20261019_v1/repair_v1/delivery/closure_audit.json')['metric_rows'])
 old={(v['source_id'],v['strategy_family'],v['noise_seed']):v for v in allrows if v['source_id']in ids and v['snr_db']==10};assert len(old)==600
 cat=cr(reg['catalogue']);pol=cr(reg['frozen_policies']);assert pol['policies']['10']['WHOLE']['winner']=='m9_q4_r2-3' and pol['policies']['10']['PARTIAL']['winner']=='m8_K126_q4_r2-3'
 imports();import ep_phy
 noise=[];jobs=[];images={};short=[]
 for rank,s in enumerate(sources):
  with np.load(checked(s['current_streams']),allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files}
  with np.load(checked(s['archive']),allow_pickle=False)as z:x=z['pixels'].copy()
  assert x.dtype==np.uint8 and hashlib.sha256(x.tobytes()).hexdigest()==s['preprocessing_id']
  refhash=ash(x.astype(np.float32)/np.float32(255));short.append({k:s[k]for k in ('source_index','source_id','archive','preprocessing_id','current_streams')}|dict(rank=rank,reference_sha256=refhash))
  for seed in (6201,6202,6203):noise.append(ep_phy.standard_noise(s['source_id'],seed))
  for fam in ('WHOLE','PARTIAL'):
   m,k,bits=choose(streams,fam);wirefam='EC_VAR_PARTIAL'if k else'EC_VAR_WHOLE'
   p=next(p for p in cat['profiles']if p['family']==wirefam and p['m']==m and p['K']==k and p['q']==4 and p['nominal_rate']=='2/3');assert p['k']==2549
   for seed in (6201,6202,6203):
    v=old[s['source_id'],fam,seed];assert (v['actual_m'],v['actual_K'],v['arithmetic_bits'])==(m,k,len(bits))
    f=cr(v['physical_frame']);rx=f['actual_RX'];tx=f['transmission'];assert tx['profile_id']==p['profile_id'] and tx['arithmetic_bits']==len(bits)
    a=v['reconstruction'];ev=a['evidence'];ev=cr(ev)if'path'in ev else ev;ev=ev.get('evidence',ev);assert ev['source_status']=='ARITHMETIC_SOURCE_DECODED'
    if a['image_archive']['path']not in images:
     with np.load(checked(a['image_archive']),allow_pickle=False)as z:y=z[a['image_key']].copy()
     if a.get('image_slot')is not None:y=y[a['image_slot']]
     assert ash(y)==a['image_sha256'];images[a['image_archive']['path']]=True
    for snr in (8,10,12):
     jobs.append(dict(source_id=s['source_id'],source_index=s['source_index'],rank=rank,family=fam,TX_policy_SNR=10,snr_db=snr,noise_seed=seed,noise_index=rank*3+seed-6201,public_frame_counter=tx['public_frame_counter'],session=tx['session'],profile_id=p['profile_id'],m=m,K=k,arithmetic_bits=len(bits),payload_sha256=ash(bits),old10=v))
 with (P/'standard_noise.npy').open('xb')as f:np.save(f,np.stack(noise),allow_pickle=False)
 save(P/'sources.json',short);save(P/'jobs.json',jobs);assert len(jobs)==1800
 r=dict(schema='WCL_TX_POLICY_SNR_MISMATCH_V1',basis_HEAD='1f2a86a41b65fe90c0bf245076212b0a27852deb',logical=1800,source_count=100,TX_policy_SNR=10,true_SNRs=[8,10,12],noise_seeds=[6201,6202,6203],RX_LLR='true noise variance',N=1024,Es=2,header_symbols=68,body_symbols=956,models_changed=False,policy_changed=False,training=False,source_probability_calls=0,tokenizer_calls=0,targets=TARGETS,fallback={f:chain(f)for f in TARGETS},source_list=pin(poppath),sources=pin(P/'sources.json'),jobs=pin(P/'jobs.json'),noise=pin(P/'standard_noise.npy'),noise_array_sha256=ash(np.stack(noise)),noise_shape=[300,1024,2],noise_semantics='one actual float64 standard (real,imag) vector per source/repeat, shared across families and scaled by 10**(-true_snr/20); independent SHA256 channel namespace',original_registration=pin(regpath),frozen_policy=reg['frozen_policies'],catalogue=reg['catalogue'],phy_qualification=reg['phy_qualification'],backend_identity=reg['backend_identity'],bootstrap=bank,bootstrap_identity=pin(bankpath),caps=dict(logical_receptions=1800,new_physical_max=1800,packet_decodes_max=3600,source_rx_max=1800,render_max=1800,unique_metric_pairs_per_metric_max=1800),max_seconds=10800,automatic_retry=False,reuse='old10 only with exact waveform/observation/noise/backend proof; deterministic RX by actual input key; metrics by source/pixel identity',code=[pin(p)for p in sorted(E.glob('*.py'))])
 save(O/'p1_registration.json',r);print(json.dumps(dict(registration=pin(O/'p1_registration.json'),sources=100,logical=1800,noise=300,source_calls=0)))
if __name__=='__main__':main()
