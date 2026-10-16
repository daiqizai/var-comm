"""Read closed original-calibration evidence; no Torch, model, PHY or metrics calls."""
from tb_common import *
import tb_plan as plan
import collections,subprocess,traceback

def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==HEAD
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=R,text=True).strip()
 T.mkdir(exist_ok=False);O.mkdir(exist_ok=False);out=T/'preparation';out.mkdir()
 save(out/'launch.json',dict(process=pid_identity(os.getpid()),parent_process=pid_identity(os.getppid()),argv=sys.argv,basis_HEAD=HEAD,scientific_calls=0))
 try:
  sr=read(FIRST_O/'source_registration.json');assert sha(FIRST_O/'source_registration.json')=='13c9057b42cdc9d36a8934b2479e8a9385073d5205402852a355e32803c60845'
  assert sha(FIRST_O/'m10/closure_audit.json')=='4138a73fa86f2546de7db6f18bbadd29a8c54673f6de86dc150cf943baaac449'
  sources=[];lengths=[];shas=[];proof=[]
  for i,s in enumerate(sr['sources']):
   assert i==s['source_index'];d=read(FIRST_T/'m10_source/streams'/f'{i:04d}_fixed_output.json');ld=read(FIRST_T/'m10_source/streams'/f'{i:04d}_length.json');rg=read(FIRST_T/'m10_source/streams'/f'{i:04d}_regression.json')
   assert ld['archive_sha256']==d['archive']['sha256'] and all(rg['old24_exact'].values()) and len(rg['old24_exact'])==24
   md=checkread(s['old_metadata']);ls={(v['m'],v['K']):v['payload_bits']for v in md['endpoints']};hs={(v['m'],v['K']):v['bits_sha256']for v in md['endpoints']}
   assert set(ls)==set(plan.ENDS);ls[10,0]=ld['arithmetic_bits'];hs[10,0]=ld['bits_sha256'];lengths.append(ls);shas.append(hs)
   sources.append(dict(s,current_streams=d['archive'],length_evidence=pin(FIRST_T/'m10_source/streams'/f'{i:04d}_length.json'),stream_endpoints=[dict(m=m,K=k,bits=ls[m,k],payload_sha256=hs[m,k])for m,k in ls]))
  save(out/'sources.json',sources);progress(out,dict(phase='sources',count=1000))
  # Actual completed owner chains, not request status labels.
  reqs={};workers={};controls=[]
  names=dict(pilot_phy='h800_ep_pilot_phy_v1_attempt1',full_phy='h800_ep_full_phy_v1_attempt1',pilot_visual='h800_ep_pilot_visual_v1_attempt1',full_visual='h800_ep_full_visual_v2_attempt1',pilot_metrics='h800_ep_pilot_metrics_v1_attempt1',full_metrics='h800_ep_full_metrics_v2_attempt1')
  for k,n in names.items():reqs[k],workers[k],cs=old_closed(n);controls+=cs
  assert reqs['pilot_phy']['phy_identity']==reqs['full_phy']['phy_identity']
  qual=checkread(reqs['full_phy']['phy_identity']['qualification']);old=read(qual['request_path'])['catalogue'];assert old['catalogue_sha256']=='687cc3b3760a46acca57afaa36b6b2f304c8a8df13d657bcd1f8c98d704e4a3e'
  catalogue=plan.extend_catalogue(old);save(O/'public_catalogue.json',catalogue)
  lookup={(p['m'],p['K'],p['q'],p['nominal_rate']):p for p in catalogue['profiles']if p['family']!='EC_STATIC_WHOLE'}
  oldpackets={};packetdata={};header_counts=collections.Counter();events={}
  allpins=[]
  for k in ('pilot_phy','full_phy'):allpins+=workers[k]['results']['actual_new_physical']
  allpins+=reqs['pilot_phy']['reuse48']['physical_packets']
  for ix,d in enumerate(allpins):
   if d['path']in packetdata:continue
   p=checkread(d);h=p['actual_RX']['header'];assert plan.directory_unchanged_on_old_event(h),('BLOCKED_OLD_ILLEGAL_ID_REQUIRES_EXPLICIT_REPLAY',d)
   assert p['full_public_receive_catalogue'] and not p['source_truth_supplied_to_RX']
   row=p['logical_event'];tx=p['transmission'];assert tx['public_frame_counter']==counter(row['source_index'],row['snr_db'],row['noise_seed'])
   key=event_lookup(row['source_index'],row['snr_db'],row['noise_seed'],tx['profile_id'],p['payload_sha256'])
   if key in oldpackets:assert packetdata[oldpackets[key]['path']]['actual_RX']==p['actual_RX']
   oldpackets[key]=d;packetdata[d['path']]=p;header_counts[canonical([h['header_crc_ok'],h['header_fields_legal']])]+=1
   if (ix+1)%2000==0:progress(out,dict(phase='old_packet_metadata',count=ix+1))
  save(out/'old_packet_index.json',oldpackets)
  header_proof=dict(old_profiles_byte_equal=True,old_profile_count=360,new_profile_count=372,old_unique_packets=len(packetdata),header_counts=dict(header_counts),old_crc_accepted_illegal_ids=0,affected_old_outputs=0,proof='Unchanged Header.receive decodes before catalogue membership; every old CRC-passing decoded ID is retained byte-exact.',header_source=pin(R/'experiments/prior-aware-grouped-mcs-v1/uep_phy.py'),old_controls=controls,zero_scientific_calls=True)
  save(O/'old_receive_domain_audit.json',header_proof)
  # Four changed source strategies get complete calibration at all three SNRs.
  # Ten MCS per family have zero changed prefixes on cal1000 and alias legacy maxima.
  cs=plan.candidates();changes=[];changed=[]
  for c in cs:
   ids=[i for i in range(1000)if plan.choose(lengths[i],c['strategy_family'],c['source_capacity_bits'])==(10,0)]
   if ids:changed.append(c)
   changes.append(dict(c,changed_source_indices=ids,changed_source_count=len(ids),treatment='FULL1000_THREE_NOISE'if ids else'EXACT_LEGACY_CALIBRATION_ALIAS'))
  assert len(changed)==4 and sum(x['changed_source_count']for x in changes)==44
  rows=[];newkeys=set();oldkeys=set();counts=collections.Counter()
  for i,s in enumerate(sources):
   for snr in SNRS:
    for seed in SEEDS:
     for c in changed:
      m,K=plan.choose(lengths[i],c['strategy_family'],c['source_capacity_bits']);p=lookup[m,K,c['q'],c['nominal_rate']];key=event_lookup(i,snr,seed,p['profile_id'],shas[i][m,K])
      prior=oldpackets.get(key)
      if prior:oldkeys.add(key)
      else:newkeys.add(key)
      status='VERIFIABLE_OLD_EVENT'if prior else'SAME_NEW_EVENT'if key in events else'NEW_EVENT'
      counts[status]+=1;events.setdefault(key,len(rows))
      rows.append(dict(frame_index=len(rows),source_index=i,source_id=s['source_id'],snr_db=snr,noise_seed=seed,public_frame_counter=counter(i,snr,seed),candidate_id=c['candidate_id'],strategy_family=c['strategy_family'],m=m,K=K,profile_id=p['profile_id'],profile_key=p['profile_key'],payload_sha256=shas[i][m,K],arithmetic_bits=lengths[i][m,K],event_lookup=key,old_physical=prior,first_logical_event=events[key]))
  assert len(rows)==36000 and not newkeys.intersection(oldkeys)
  save(out/'frames.json',rows)
  # Index closed image/metric results by actual RX input and both pixel identities.
  images={};pairs={};pairdata={};vis_ids={};metric_ids={};old_rows={}
  for label in ('pilot','full'):
   rq=reqs[label+'_visual'];vis_ids[label]=rq['visual_identity']
   md=checkread(workers[label+'_metrics']['metric_identity']);metric_ids[label]=md
   mr=checkread(workers[label+'_metrics']['results']['metric_rows']);old_rows[label]=workers[label+'_metrics']['results']['metric_rows']
   # One representative per physical packet is enough to map every actual source input.
   seenphys=set()
   for ix,row in enumerate(mr):
    pp=row['metric_pair_result'];pairkey=canonical(pp)
    if pairkey not in pairdata:
     pair=checkread(pp);assert pair['metric_identity']==md['identity'];assert pair['metrics']=={k:row[k]for k in METRICS};pairdata[pairkey]=pair
    pair=pairdata[pairkey]
    vr=checkread(row['reconstruction_result']);pr=vr['physical_frame'];pd=packetdata.get(pr['path'])
    if pd is None:pd=checkread(pr);packetdata[pr['path']]=pd
    rec=vr['reconstruction'];key=source_key(pd['actual_RX'])
    assert rec['actual_source_input_key']==digest(dict(visual_identity=rq['visual_identity'],kind='FIXED_GRAY')if key==digest({'kind':'FIXED_GRAY'})else dict(visual_identity=rq['visual_identity'],kind='ACTUAL_SOURCE_INPUT',family=pd['actual_RX']['rx_profile']['family'],m=pd['actual_RX']['rx_profile']['m'],K=pd['actual_RX']['rx_profile']['K'],payload=pd['actual_RX']['body']['payload']))
    if key in images:assert images[key]['image_sha256']==rec['image_sha256']
    else:images[key]=dict(image_archive=rec['image_archive'],image_sha256=rec['image_sha256'],reconstruction_result=row['reconstruction_result'],physical_frame=pr,original_visual_identity=rq['visual_identity'])
    assert pair['reference_archive']==sources[row['source_index']]['archive'] and pair['reconstruction_archive']==rec['image_archive']
    pk=canonical([row['source_index'],rec['image_sha256']]);item=dict(pair=pp,metric_pair_key=row['metric_pair_key'],metrics=pair['metrics'],image_sha256=rec['image_sha256'],source_id=row['source_id'],source_index=row['source_index'],reference_archive=pair['reference_archive'],reconstruction_archive=pair['reconstruction_archive'],metric_identity=pair['metric_identity'])
    if pk in pairs:assert pairs[pk]['metrics']==item['metrics']
    else:pairs[pk]=item
    if(ix+1)%5000==0:progress(out,dict(phase=label+'_image_metric_metadata',count=ix+1))
  assert vis_ids['pilot']==vis_ids['full'] and metric_ids['pilot']['identity']==metric_ids['full']['identity']
  save(out/'old_image_index.json',images);save(out/'old_pair_index.json',pairs)
  save(out/'inheritance.json',dict(requests={k:pin(Q/n/'registered/request.json')for k,n in names.items()},controls=controls,old_metric_rows=old_rows,old_visual_identity=vis_ids['full'],old_metric_identity=metric_ids['full'],model_routes=sr['model_routes'],metric_config=reqs['full_metrics']['metric_config'],old_full_winners=workers['full_metrics']['results']['full_winners'],old_policy=reqs['full_phy']['policy']))
  manifest=dict(schema='NESTCOM_T1B_CANDIDATE_DELTA_V1',scope='T1-B_ONLY_STOP_AFTER_DELIVERY',basis_HEAD=HEAD,old_candidates_retained=144,new_strategy_candidates=24,candidate_changes=changes,public_profiles_added=list(range(360,372)),shared_actual_m10_profile_across_families=True,selection_population='original_calibration1000',snrs=list(SNRS),noise_seeds=list(SEEDS),source_count=1000,logical_frames=36000,unique_new_physical=len(newkeys),new_header_packet_cap=len(newkeys),new_body_packet_cap=len(newkeys),new_packet_total_cap=2*len(newkeys),reuse_counts=dict(counts),unique_old_events=len(oldkeys),source_tx=0,tokenizer=0,standalone_qualification_images=0,source_rx_and_render_cap=len(newkeys),new_metric_pair_cap=36000,selection_rule='Retain all144 old pilot candidates and all12 old full finalists. Collapse20 exact cal1000 aliases. Fully evaluate all4 distinct m10 strategies at4/10/19 independent of pilot rank. Per-source three-noise mean then1000-source DINO mean; exact ties favor legacy then lexical IDs.',source_based_prescreen_only=True,old_receive_domain_audit=pin(O/'old_receive_domain_audit.json'),prior_clean_images_not_assumed_wireless_success=True,frames=pin(out/'frames.json'),sources=pin(out/'sources.json'),T1C=False,LE3=False,training_updates=0,automatic_followup=False)
  save(O/'candidate_delta_manifest.json',manifest)
  save(out/'completion.json',dict(status='METADATA_PREPARATION_COMPLETE_NOT_SCIENCE',manifest=pin(O/'candidate_delta_manifest.json'),old_images=len(images),old_pairs=len(pairs),new_scientific_calls=0))
  print(json.dumps({k:v for k,v in manifest.items()if k not in('candidate_changes','selection_rule')},indent=2),flush=True)
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),new_scientific_calls=0));raise
if __name__=='__main__':main()
