"""Bind original repeat-zero frames and seeds before threshold observations exist."""
from followup_common import *
def main():
 out=L/'E6_plan';out.mkdir(exist_ok=False);sources=read(L/'inventory_core/sources.json')[:100];sm={s['source_id']:s for s in sources};ids=list(sm)
 assert ids==read(D/'source_sets.json')['control100'];spec=read(D/'protocol.json')['E6_threshold']
 assert spec['noise_repeats_per_source']==20 and spec['total_real_frames']==30000
 # These fixed new draws change AWGN only, never public frame counters.
 seeds=list(range(73000000,73000020));save(out/'noise_registration.json',dict(seeds=seeds,namespace='E6_FIXED_POLICY_NOISE_V1',rule='same registered standard AWGN draws across SNR and methods; full source ID remains in original noise generator',source_order=ids,no_seed_selection=True))
 core={(v['method'],v['source_id']):v for v in read(L/'inventory_core/logical.json')if v['source_id']in sm and int(v['N'])==1024 and int(v['snr_db'])==10 and int(v['channel_repeat'])==0}
 baseline=read(OLD/'final_phy_1024_registration.json');dframe={v['physical_key']:v['frame']for v in read(OL/'final_phy_1024/frames.json')}
 whole={v['source_id']:v['physical_frame']for v in read(B/'n1024_consistency_20261019_v1/repair_v1/delivery/metric_rows_9000.json')if v['strategy_family']=='WHOLE'and v['snr_db']==10 and v['noise_seed']==6201 and v['source_id']in sm}
 ar=B/'adatoken_response_20261026_v1';areg=read(R/'results/adatoken_response_20261026_v1/c_predclass_common500_link_registration.json')
 af={v['source_id']:v['frame']for v in read(ar/'c_predclass_common500_rx_preparation/logical_frames.json')if v['snr_db']==10 and v['noise_seed']==6201 and v['source_id']in sm}
 arx={}
 for i in range(2):
  for v in read(ar/f'c_predclass_common500_rx_s{i:02d}/outputs.json'):arx[v['image']['path']]=v
 catalogues=dict(WHOLE=baseline['original_catalogue'],Token_NeST=baseline['catalogue'],Ada=areg['catalogue']);cats={m:cr(p)for m,p in catalogues.items()};donors=[]
 sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'));from n512_phy import received_projection
 for s in sources:
  for method,cm in [('WHOLE','WHOLE_final_receiver'),('Token_NeST','final_NeST'),('Ada','Ada_frozen')]:
   c=core[cm,s['source_id']];fp=whole[s['source_id']]if method=='WHOLE'else af[s['source_id']]if method=='Ada'else dframe[c['physical_key']]
   f=cr(fp);tx=f['transmission'];rx=f['actual_RX'];p=received_projection(rx);assert p['status']=='SOURCE_INPUT'
   b=np.array(rx['body']['payload'],np.uint8);h=f.get('TX_payload_sha256',f.get('payload_sha256',asha(b)));assert h==asha(b)
   profile=next(v for v in cats[method]['profiles']if v['profile_id']==tx['profile_id']);assert all(profile[k]==v for k,v in rx['rx_profile'].items())
   seed=None;image=image_spec(c['image'])
   if method=='Ada':
    rec=arx[image['path']];assert rec['pixel_sha256']==c['pixel_sha256']and rec['prefix']==p['m']and rec['received_class']==p['received_class']and rec['payload_sha256']==asha(np.array(p['bits'],np.uint8));seed=rec['generation_seed']
   key=digest(dict(method=method,source_id=s['source_id'],frame=fp))
   donors.append(dict(key=key,method=method,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],frame=fp,profile=profile,payload=b.tolist(),payload_sha256=h,transmission=tx,counter=tx['public_frame_counter'],session=tx['session'],projection=p,generation_seed=seed,image=image,pixel_sha256=c['pixel_sha256'],original_repeat=0,original_noise_seed=int(c['noise_seed'])))
 assert len(donors)==300
 save(out/'donors.json',donors);save(out/'completion.json',dict(status='PREPARED',donors=300,real_frames=30000,source_order=ids,catalogues=catalogues,qualification=baseline['qualification'],noise_registration=pin(out/'noise_registration.json'),actual_snr_db=spec['actual_snr_db'],new_PHY=0,new_models=0,transmitter_policy_unchanged=True,public_counter_session_preserved=True))
 print('E6_ORIGINAL_REPEAT0_BOUND',len(donors))
if __name__=='__main__':main()
