"""Test-only source/event delta and exact legacy evidence inventory. No science calls."""
from tc_common import *
import csv,numpy as np,collections,traceback
import tb_plan as plan
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')
def csvread(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def values(row):
 return {k:(float(row[k]) if k!=METRICS[-1] or str(row[k]) not in ('True','False') else float(str(row[k])=='True')) for k in METRICS}
def closed(stage):
 w=read(T/(stage+'_control/actual_wait.json'));assert w['actual_wait'] and type(w['returncode'])is int and w['returncode']==0 and not any(w[k] for k in ('timed_out','terminated','killed')) and w['metadata_error']is None
 assert gone(w['process']) and gone(w['parent_process']);return pin(T/(stage+'_control/actual_wait.json'))
def main():
 out=T/'link_preparation_v2';out.mkdir();save(out/'launch.json',dict(process=pid_identity(os.getpid()),source_actual_wait=closed('source'),new_scientific_calls=0))
 try:
  sd=read(T/'source/completion.json');sources=checkread(sd['sources']);assert sd['ledger']['unresolved']==0
  em=B/'entropy_main_common500_20261013_v1';eo=R/'results/entropy_main_common500_20261013';wr=T/'whole_evidence_restored'
  restored=read(wr/'restoration_manifest.json');restoremap={v['relative']:v for v in restored['files']}
  for v in restoremap.values():checked(v)
  cat=read(R/'results/t1b_m10_calibration_20261016_v1/public_catalogue.json');assert len(cat['profiles'])==372
  lookup={(p['m'],p['K'],p['q'],p['nominal_rate']):p for p in cat['profiles'] if p['family']!='EC_STATIC_WHOLE'}
  oldwhole=list(v for v in csvread(R/'results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/per_frame.csv') if v['family']=='EC_VAR_WHOLE' and int(v['snr_db'])==19);assert len(oldwhole)==1500
  oldpartial=list(v for v in csvread(eo/'frame_rows.csv') if int(v['snr_db'])==19);assert len(oldpartial)==1500
  partialframes=read(em/'visual/frames.json');pmap={}
  for d in partialframes:
   v=checkread(d)
   if v['snr_db']==19:pmap[v['source_index'],v['noise_seed']]=(d,v)
  bridge=B/'quality_metrics_supplement_20261014_v1/digital_input_bridge_v2';bm=read(bridge/'manifest.json');bf=checkread(bm['frame_index'])
  bmap={(v['source_index'],v['noise_seed']):v for v in bf if v['method']=='EC_VAR_WHOLE' and int(v['snr_db'])==19}
  sm=B/'quality_metrics_supplement_20261014_v1/common500_metrics_v1';suppdone=read(sm/'completion.json');supp=csvread(checked(suppdone['scores']));suppmap={(v['method'],int(v['source_index']),int(v['noise_seed'])):v for v in supp if int(v['snr_db'])==19 and v['method'] in ('EC_VAR_WHOLE','EC_VAR_PARTIAL')}
  old={};classes={};regs=read(eo/'registration_v4.json')
  for s in regs['sources']:classes['common500',s['source_index']]=s['evaluation_class_index']
  for family,rr in [('WHOLE',oldwhole),('PARTIAL',oldpartial)]:
   for row in rr:
    i=int(row['source_index']);seed=int(row['noise_seed'])
    if family=='WHOLE':
     rel=f'posthoc_render500_v1/physical_frames/{i:04d}/19_{seed}_{int(row["received_profile_id"]):03d}.json';pd=restoremap[rel];b=bmap[i,seed];rec=dict(image_archive=b['archive'],image_key=b['image_key'],image_slot=b['image_slot'],image_sha256=b['reconstruction_sha256'],old_row=pin(R/'results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/per_frame.csv'))
    else:
     vd,v=pmap[i,seed];pd=v['physical_frame'];rec=dict(v['reconstruction'],image_key='image',image_slot=None,old_row=vd)
    assert rec['image_sha256']==row['image_sha256'];sp=suppmap['EC_VAR_'+family,i,seed];assert sp['reconstruction_rgb_sha256']==rec['image_sha256']
    old['common500',i,seed,family]=dict(physical_frame=pd,image=rec,metrics=dict(values(row),**{k:float(sp[k]) for k in ('dists','dreamsim','ms_ssim')}),original_row=row,supplemental_row=sp)
  nd=read(Q/'h800_ep_new100_metrics_v3_attempt1/registered/run/worker_completion.json');nr=read(Q/'h800_ep_new100_metrics_v3_attempt1/registered/request.json');nrows=checkread(nd['results']['metric_rows'])
  for s in nr['records']:classes['new100',s['source_index']]=s['evaluation_class_index']
  for row in nrows:
   if row['snr_db']!=19 or row['method'] not in ('EC_VAR_WHOLE','EC_VAR_PARTIAL'):continue
   vr=checkread(row['reconstruction_result']);rec=dict(vr['reconstruction'],image_key='image',image_slot=None,old_row=row['reconstruction_result']);pair=checkread(row['metric_pair_result']);assert pair['metrics']=={k:row[k] for k in METRICS} and pair['reconstruction_archive']==rec['image_archive']
   old['new100',row['source_index'],row['noise_seed'],row['method'].removeprefix('EC_VAR_')]=dict(physical_frame=row['physical_frame'],image=rec,metrics=values(row),original_row=row)
  assert len(old)==3600
  rows=[];counts=collections.Counter();newkeys=set();changed_sources=set();oldbytes={};imagechecked=set();headers=collections.Counter()
  for s in sources:
   pop=s['population'];i=s['source_index'];j=s['source_global_index'];s['evaluation_class_index']=classes[pop,i]
   ls={(v['m'],v['K']):v['bits'] for v in s['old_endpoints']};hs={(v['m'],v['K']):v['payload_sha256'] for v in s['old_endpoints']};ls[10,0]=s['m10']['arithmetic_bits'];hs[10,0]=s['m10']['payload_sha256']
   for family in ('WHOLE','PARTIAL'):
    m,K=plan.choose(ls,family,4751);p=lookup[m,K,6,'5/6']
    for seed in ((6201,6202,6203) if pop=='common500' else (9301,9302,9303)):
     legacy=old[pop,i,seed,family];pkt=checkread(legacy['physical_frame']);ident=pkt.get('observation_identity',pkt);tx=pkt['transmission'];h=pkt['actual_RX']['header'];headers[canonical([h['header_crc_ok'],h['header_fields_legal']])]+=1
     assert ident['source_id']==s['source_id'] and ident['source_index']==i and ident['noise_seed']==seed and ident['snr_db']==19 and ident['public_frame_counter']==counter(pop,i,seed)
     same=tx['profile_id']==p['profile_id'] and ident['payload_sha256']==hs[m,K] and plan.directory_unchanged_on_old_event(h)
     # Reuse requires the old paid profile definition itself, not only its numeric ID.
     if same:assert tx['profile_key']==p['profile_key'] and tx['n']==5736 and tx['k']==4780 and tx['length_field_bits']==13 and tx['crc_bits']==16
     why='EXACT_SOURCE_AND_PAID_PROFILE_LEGAL_RX' if same else 'M10' if m==10 else 'LEGACY_BYTES_OR_RECEIVE_DOMAIN_CHANGED'
     eventkey=canonical([pop,i,seed,counter(pop,i,seed),p['profile_id'],hs[m,K]])
     if not same:newkeys.add(eventkey);changed_sources.add(j)
     counts[pop+'_'+family+'_'+why]+=1
     d=legacy['image']['image_archive']
     if d['path'] not in imagechecked:
      checked(d);imagechecked.add(d['path'])
     with np.load(d['path'],allow_pickle=False) as z:
      a=z[legacy['image']['image_key']];slot=legacy['image']['image_slot'];a=a if slot is None else a[int(slot)];assert a.dtype==np.float32 and a.shape==(3,256,256) and asha(a)==legacy['image']['image_sha256']
     rows.append(dict(frame_index=len(rows),population=pop,source_global_index=j,source_index=i,source_id=s['source_id'],snr_db=19,noise_seed=seed,strategy_family=family,candidate_id=f'm10_{family.lower()}_q6_r5-6',m=m,K=K,arithmetic_bits=ls[m,K],payload_sha256=hs[m,K],profile_id=p['profile_id'],profile_key=p['profile_key'],public_frame_counter=counter(pop,i,seed),event_key=eventkey,reuse=bool(same),reuse_reason=why,legacy=legacy))
   if (j+1)%50==0:print('INVENTORIED',j+1,flush=True)
  save(out/'sources.json',sources);save(out/'frames.json',rows)
  summary=dict(status='T1C_EXACT_DELTA_PREPARED_NO_NEW_PHY',logical_frames=len(rows),source_TX_completed=600,m10_fit=dict(common500=6,new100=1),logical_reused=sum(v['reuse'] for v in rows),logical_changed=sum(not v['reuse'] for v in rows),unique_new_physical=len(newkeys),packet_cap=2*len(newkeys),changed_source_indices=sorted(changed_sources),counts=dict(counts),old_header_counts=dict(headers),source_inputs=sd['sources'],restored_whole_metadata=pin(wr/'restoration_manifest.json'),sources=pin(out/'sources.json'),frames=pin(out/'frames.json'),old_new100_metric_rows=nd['results']['metric_rows'],common_supplemental_scores=suppdone['scores'],new_scientific_calls=0,policy_selection=False)
  save(out/'completion.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('sources','frames')},indent=2))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),new_scientific_calls=0));raise
if __name__=='__main__':main()
