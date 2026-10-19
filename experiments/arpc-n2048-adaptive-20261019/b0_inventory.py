"""Read completed source/cache metadata and verify actual source bytes. No models."""
from pathlib import Path
import json,hashlib,time,csv,os,sys,collections,traceback,subprocess
import numpy as np
from adaptive_plan import *
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
B=BASE/'baseline_strength_20261011';AO=R/'results/arpc_external_baseline_20261019_v1'
L=BASE/'arpc_n2048_adaptive_20261019_v1';O=R/'results/arpc_n2048_adaptive_20261019_v1';E=R/'experiments/arpc-n2048-adaptive-20261019'
pins={};verified={}
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canon(v).encode()).hexdigest()
def pin(path):
 path=Path(path);a=path.stat();raw=path.read_bytes();b=path.stat();assert(a.st_ino,a.st_mtime_ns,a.st_size)==(b.st_ino,b.st_mtime_ns,b.st_size)
 d=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());pins[str(path)]=d;return d
def read(path):
 path=Path(path);pin(path);return json.loads(path.read_text())
def verify(d):
 path=Path(d['path']);assert not path.is_symlink()
 if str(path) not in verified:verified[str(path)]=pin(path)
 got=verified[str(path)];assert got['sha256']==d['sha256'] and ('bytes' not in d or got['bytes']==d['bytes']),str(path)
 return got
def put(path,d):
 with path.open('x',encoding='utf8',newline='\n') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def table(path,rows):
 rows=iter(rows);first=next(rows)
 with path.open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(first));w.writeheader();w.writerow(first);w.writerows(rows)
def progress(phase,**v):print(json.dumps(dict(phase=phase,**v)),flush=True)
def main():
 assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()=='980d74c4fb08f279c31e7a559afb0cf15cc92057'
 assert not L.exists() and not O.exists()
 L.mkdir();O.mkdir();(L/'b0').mkdir();out=L/'b0'
 sources=read(AO/'sources.json');calids={x['source_id'] for x in sources['cal100']};testids={x['source_id'] for x in sources['external_common100_v1']}
 ns_cal=read(B/'t1b_m10_calibration_20261016_v1/preparation/sources.json')
 ns_test=[s for s in read(B/'t1c_m10_test_20261016_v1/source/sources.json') if s['population']=='common500' and s['source_id'] in testids]
 assert len(ns_cal)==1000 and len(ns_test)==100 and not({s['source_id'] for s in ns_cal}&testids)
 oldimages=read(B/'t1b_m10_calibration_20261016_v1/preparation/old_image_index.json')
 for stage,receiver in [('t1b_m10_calibration_20261016_v1','receiver_recovery_v1'),('t1c_m10_test_20261016_v1','receiver')]:
  more=read(B/stage/receiver/'images.json')
  for k,v in more.items():
   if k in oldimages:assert oldimages[k]['image_sha256']==v['image_sha256']
   else:oldimages[k]=v
 # Index full-object N2048 outputs as additional native-cache evidence.
 fullreg=read(R/'results/arpc_n2048_validation_20261019_v1/registration.json')
 nc=read(BASE/'arpc_n2048_validation_20261019_v1/formal/nest/rx/completion.json')
 native_full={v['source_input_key']:v for v in nc['outputs']}
 # Original T1-B four-metric cache, kept with exact metric/source/image identity.
 nestpairs=read(B/'t1b_m10_calibration_20261016_v1/preparation/old_pair_index.json')
 more=read(B/'t1b_m10_calibration_20261016_v1/metrics_recovery_v2/metric_rows.json')
 for x in more:
  k=canon([x['source_index'],x['image_sha256']])
  if k not in nestpairs:nestpairs[k]=dict(source_id=x['source_id'],source_index=x['source_index'],image_sha256=x['image_sha256'],pair=x['metric_pair_result'],metrics={m:x[m] for m in ('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')})
 streams=[];cache=[];costs=[];source_rows=[]
 for pop,ns in [('cal1000',ns_cal),('external_common100',ns_test)]:
  for j,s in enumerate(ns):
   source=verify(s['archive']);wire=verify(s['current_streams']);sid=s['source_id'];i=s['source_index']
   source_rows.append(dict(population=pop,source_id=sid,source_index=i,source=source,cal100=sid in calids))
   with np.load(wire['path'],allow_pickle=False)as z:
    assert set(z.files)=={f'm{m}_K{k}' for m,k in NEST_ENDPOINTS}
    for end,(m,k) in enumerate(NEST_ENDPOINTS):
     bits=z[f'm{m}_K{k}'];assert bits.dtype==np.uint8 and bits.ndim==1 and np.isin(bits,(0,1)).all()
     h=hashlib.sha256(bits.tobytes()).hexdigest();data=np.packbits(bits,bitorder='big').tobytes();length=len(bits);family=nest_family(end)
     # Identical complete received source input, not target-MCS/quality matching.
     key=digest(dict(kind='ACTUAL_SOURCE_INPUT',family=family,m=m,K=k,payload=bits.tolist()))
     im=oldimages.get(key);origin='ORIGINAL_ENTROPY_RX' if im else None
     if im is None and end==24:
      import struct
      fk=hashlib.sha256(struct.pack('>BH',0,length)+data).hexdigest();fr=native_full.get(fk)
      if fr:im=dict(image_archive=fr['image'],image_sha256=fr['float_sha256'],image_key='image',evidence=fr);origin='N2048_FULL_RX'
     row=dict(population=pop,source_id=sid,source_index=i,codec=0,endpoint=end,m=m,K=k,bit_length=length,object_bytes=len(data),object_sha256=hashlib.sha256(data).hexdigest(),bits_sha256=h,wire=wire,array_key=f'm{m}_K{k}',source=source,cal100=sid in calids)
     streams.append(row)
     pair=nestpairs.get(canon([i,im['image_sha256']])) if im and pop=='cal1000' else None
     cache.append(dict(population=pop,source_id=sid,codec=0,endpoint=end,image=im,score=pair,origin=origin,cache_identity='EXACT_ARITHMETIC_INPUT_AND_FROZEN_NATIVE' if im else 'MISSING'))
     costs.append(dict(population=pop,source_id=sid,codec=0,endpoint=end,object_bits=length,object_storage_bits=8*len(data),object_alignment_bits=(-length)%8,caption_bits=0,source_fields_and_native_finish_included=True))
   if(j+1)%100==0:progress('nest_sources',population=pop,completed=j+1)
 arc_images={}
 for row in read(AO/'a2_clean_v1/clean_states.json')['states']:arc_images[(row['wire']['sha256'],row['endpoint'])]=dict(image_archive=row['image'],image_key='float_output',evidence=row,origin='ARPC_CLEAN_INDEPENDENT_RX')
 for stage in ('a2b_rx_v1','a3_rx_v1'):
  for row in read(AO/stage/'actual_states.json'):
   if row['status']=='IMAGE_READY':arc_images[(row['input']['sha256'],row['endpoint'])]=dict(image_archive=row['image'],image_key='float_output',image_sha256=row['float_sha256'],evidence=row,origin=stage)
 # This full ARPC cache was independently decoded after completed pair PHY.
 ac=read(BASE/'arpc_n2048_validation_20261019_v1/formal/arpc_resource_v1/rx/completion.json')
 afull={v['key']:v for v in ac['outputs']}
 for j in fullreg['formal_jobs']:
  if j['codec']!=1:continue
  w=j['asset']['wire'];raw=Path(w['path']).read_bytes();import struct
  key=hashlib.sha256(struct.pack('>BH',1,8*len(raw))+raw).hexdigest();v=afull.get(key)
  if v and v.get('status')!='SOURCE_PARSE_FAIL':arc_images[(w['sha256'],7)]=dict(image_archive=pin(v['path']),image_key='float_output',image_sha256=v['float_sha256'],evidence=v,origin='N2048_FULL_RX')
 arc_quality={}
 for name in ('a2_metrics_v1/clean_quality.csv','a2b_metrics_v1/actual_pair_scores.csv','a3_metrics_v1/actual_pair_scores.csv'):
  p=AO/name;pin(p)
  with p.open()as f:rr=list(csv.DictReader(f))
  for x in rr:
   if 'image_float_sha256' in x or 'state' in x:arc_quality[(x['source_id'],x.get('image_float_sha256',x.get('state')))]=dict(values=x,evidence=pins[str(p)])
   if 'endpoint' in x and name.startswith('a2_metrics'):arc_quality[(x['source_id'],int(x['endpoint']))]=dict(values=x,evidence=pins[str(p)])
 for pop,stage in [('cal1000','a2b_source_v1'),('external_common100','a3_source_v1')]:
  cat=read(AO/stage/'stream_catalogue.json')['sources']
  for s in cat:
   sid=s['source_id'];source=verify(s['source']);expected=next(x for x in source_rows if x['population']==pop and x['source_id']==sid)
   assert source['sha256']==expected['source']['sha256']
   for st in s['streams']:
    w=verify(st['wire']);e=st['endpoint'];assert st['payload_bits']==8*w['bytes'];im=arc_images.get((w['sha256'],e))
    score=arc_quality.get((sid,im.get('image_sha256'))) if im else None
    if not score and sid in calids:score=arc_quality.get((sid,e))
    streams.append(dict(population=pop,source_id=sid,source_index=s['source_index'],codec=1,endpoint=e,bit_length=st['payload_bits'],object_bytes=w['bytes'],object_sha256=w['sha256'],wire=w,array_key=None,source=source,cal100=sid in calids))
    cache.append(dict(population=pop,source_id=sid,codec=1,endpoint=e,image=im,score=score,origin=im['origin'] if im else None,cache_identity='EXACT_SELF_DESCRIBING_OBJECT_AND_FROZEN_NATIVE' if im else 'MISSING'))
    costs.append(dict(population=pop,source_id=sid,codec=1,endpoint=e,object_bits=st['payload_bits'],object_storage_bits=st['payload_bits'],object_alignment_bits='included_in_native_object',caption_bits='included_in_native_object',source_fields_and_native_finish_included=True))
  progress('arpc_sources',population=pop,completed=len(cat))
 assert len(streams)==len(cache)==35200
 put(out/'source_catalogue.json',streams);put(out/'cache_catalogue.json',cache);put(out/'sources.json',source_rows)
 table(out/'source_cost_breakdown.csv',costs)
 coverage=[dict(population=s['population'],source_id=s['source_id'],codec=s['codec'],endpoint=s['endpoint'],cal100=s['source_id'] in calids,source_bytes_verified=True,clean_image_metadata_present=s['image'] is not None,clean_score_metadata_present=s['score'] is not None,actual_image_SHA_must_be_verified_before_reuse=True,origin=s['origin'] or '') for s in cache]
 table(out/'cache_coverage.csv',coverage);table(O/'layouts.csv',layouts())
 lookup=collections.defaultdict(dict)
 for s in streams:lookup[s['population'],s['source_id'],s['codec']][s['endpoint']]=s
 states=[];needed=collections.defaultdict(set);aliases=collections.defaultdict(list);gmap={g['layout_id']:g for g in layouts()}
 for codec in (0,1):
  for pol in policies(codec):
   g=gmap[pol['layout_id']];vector=[]
   for pop,ss in [('cal1000',ns_cal),('external_common100',ns_test)]:
    for s in ss:
     sid=s['source_id'];st=lookup[pop,sid,codec];end=choose(st,codec,pol['target'],g);v=st[end] if end is not None else None
     states.append(dict(population=pop,source_id=sid,codec=codec,policy_id=pol['policy_id'],target=pol['target'],actual_endpoint=end,layout_id=g['layout_id'],object_bits=v['bit_length'] if v else 0,object_bytes=v['object_bytes'] if v else 0,fragment0_bytes=(v['object_bytes']+1)//2 if v else 0,fragment1_bytes=v['object_bytes']//2 if v else 0,capacity_bytes=2*g['fragment_bytes'],source_no_fit=end is None,object_sha256=v['object_sha256'] if v else 'NULL'))
     if sid in calids:
      vector.append((sid,end,v['object_sha256'] if v else None));needed[codec].add((sid,end))
   aliases[codec,g['layout_id'],digest(vector)].append(pol['policy_id'])
 table(out/'candidate_source_states.csv',states)
 counts=[]
 for pop in ('cal100','cal1000','external_common100'):
  for codec in (0,1):
   cs=[s for s in coverage if s['codec']==codec and (s['cal100'] if pop=='cal100' else s['population']==pop)]
   counts.append(dict(population=pop,codec=codec,streams=len(cs),source_missing=0,images_metadata_present=sum(s['clean_image_metadata_present'] for s in cs),image_gaps=sum(not s['clean_image_metadata_present'] for s in cs),scores_metadata_present=sum(s['clean_score_metadata_present'] for s in cs),score_gaps=sum(not s['clean_score_metadata_present'] for s in cs)))
 table(O/'cache_summary.csv',counts)
 put(O/'scope_proposal.json',dict(schema='N2048_ADAPTIVE_B0_NET_SCOPE_V1',status='SOURCE_AUDIT_COMPLETE_IMAGE_REUSE_REQUIRES_FINAL_PIN_CHECK',basis_HEAD='980d74c4fb08f279c31e7a559afb0cf15cc92057',user_authorization='按照这个推进到底; staged B0 through gated B3 and delivery',source_streams=len(streams),source_passes_missing=0,source_probability_calls=0,tokenizer_calls=0,training=0,cache_summary=counts,cal100_policy_alias_groups=len(aliases),cal100_policies=384,qualification_caps=dict(objects=24,subframes=48,decoder_attempts=96),B2_caps=dict(logical_objects=24000,subframes=48000,decoder_attempts=96000),B3_caps=dict(logical_objects=1200,subframes=2400,decoder_attempts=4800),cal100_render_cap=sum(c['image_gaps'] for c in counts if c['population']=='cal100'),cal100_DINO_cap=sum(c['score_gaps'] for c in counts if c['population']=='cal100'),candidate_uncertainty_rule='At most two per method/SNR; do not silently exclude another plausible policy. Stop and report additional exact cost if unresolved.',fallback_note='Original m4..m9 K0 whole fallback; original partial targets all-lower fallback; m10 uses existing T1-B PARTIAL chain for main NeST interface. m10-WHOLE comparison is not a third method.',old19='Separate fixed-full historical reference; no new calibration or PHY',no_new_scientific_calls=True))
 put(O/'input_manifest.json',dict(files=list(pins.values()),source_arrays_and_objects_SHA_verified=True,images_not_rehashed_yet=True))
 files=[pin(p) for p in sorted(out.iterdir()) if p.is_file()]
 put(O/'b0_completion.json',dict(status='B0_METADATA_AND_ACTUAL_SOURCE_BYTES_COMPLETE',observed_unix=time.time(),runtime_files=files,scientific_calls=0,cache_summary=counts))
 print(json.dumps(dict(status='B0_COMPLETE',cache_summary=counts,aliases=len(aliases))),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  if L.exists() and not(L/'b0_failure.json').exists():put(L/'b0_failure.json',dict(error=traceback.format_exc(),science=0,automatic_retry=False))
  raise
