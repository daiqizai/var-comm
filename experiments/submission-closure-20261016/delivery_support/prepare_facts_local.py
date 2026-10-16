from pathlib import Path
import json,csv,hashlib,re,itertools
H=Path(__file__).resolve().parent;W=H.parents[1];S=H/'snapshot';O=S/'results/submission_closure_20261016_v1';F=H/'deliverable';F.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_bytes())
def pin(p):p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def csvsave(p,rs):
 with p.open('w',newline='',encoding='utf8')as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
invpath=W/'.research/wcl_evidence_closure_20261009/t6_actual/old_content_reference_inventory_v1.json';inv=read(invpath)
mr=read(O/'metric_registration.json');sr=read(O/'source_registration.json');new=read(O/'facts/new100_source_request_metadata.json')['data']['records'];dev=read(O/'facts/development100_manifest.json')['records'];com=read(S/'results/entropy_main_common500_20261013/e0_reuse/source_records.json')['records'];kp=W/'.research/generalization_kodak_20261009/data/dataset_manifest.json';kodak=read(kp)['records']
print('DEVROW',str(dev[0])[:1100]);print('INVROLE',inv['role_counts'])
sets={'train20k':[r['source_id']for r in inv['records']if 'train20k'in r['roles']], 'calibration1000':[r['source_id']for r in sr['sources']], 'development100':[r['source_id']for r in dev], 'common500':[r['source_id']for r in com], 'new100':[r['source_id']for r in new], 'Kodak24':[r['source_id']for r in kodak], 'HiFi_fixed16':[r['source_id']for r in mr['pairs']if r['method']=='RAW_PARTIAL']}
assert [len(x)for x in sets.values()]==[20000,1000,100,500,100,24,16]
def canon(s):
 m=re.search(r'ILSVRC2012_val_(\d+)',s);return 'imagenet-val:'+m.group(1)if m else s
hashes={r['source_id']:set(r['preprocessing_sha256'])for r in inv['records']}
for r in sr['sources']+dev+com:hashes.setdefault(r['source_id'],set()).add(r.get('preprocessing_id',r.get('pixel_sha256','')))
for r in new:hashes.setdefault(r['source_id'],set()).update([r['canonical_pixel_sha256'],r['horizontal_flip_pixel_sha256']])
for r in kodak:hashes[r['source_id']]={r['pixels_chw_uint8_sha256']}
for h in hashes.values():h.discard('')
rows=[]
for role,ids in sets.items():
 for i,s in enumerate(ids):rows.append(dict(role=role,role_index=i,source_id=s,canonical_source_id=canon(s),official_split='ImageNet training parquet'if role=='train20k'else'Kodak (no train/val designation)'if role=='Kodak24'else'ImageNet validation',canonical256_hashes=';'.join(sorted(hashes.get(s,[])))))
csvsave(F/'cohort_source_ids.csv',rows)
inter=[]
for a,b in itertools.combinations(sets,2):
 A={canon(s)for s in sets[a]};B={canon(s)for s in sets[b]};ha=set().union(*(hashes.get(s,set())for s in sets[a]));hb=set().union(*(hashes.get(s,set())for s in sets[b]));inter.append(dict(role_a=a,role_b=b,source_id_intersection=len(A&B),matching_available_canonical_hashes=len(ha&hb),a_sources_with_hash=sum(bool(hashes.get(s))for s in sets[a]),b_sources_with_hash=sum(bool(hashes.get(s))for s in sets[b]),shared_ids=';'.join(sorted(A&B)),scope='IDs and available recorded canonical/flip hashes; not universal original-content dedup'))
csvsave(F/'cohort_intersections.csv',inter);print('INTERSECTIONS',[(r['role_a'],r['role_b'],r['source_id_intersection'],r['matching_available_canonical_hashes'])for r in inter])
roles=[dict(role=k,count=len(v),previous_use={'train20k':'communication/Dc training; frozen after training','calibration1000':'Dc checkpoint and communication policy selection; not evaluation-only','development100':'historical development and diagnosis; not blind','common500':'historically evaluated frozen-policy benchmark; not a first blind test','new100':'content-excluded independent confirmation at original registration; now already evaluated','Kodak24':'all24 canonical center crops; raw transport generalization only; pretrained overlap not excluded','HiFi_fixed16':'predeclared development subset; descriptive fixed16, not common500'}[k])for k,v in sets.items()];csvsave(F/'cohort_roles.csv',roles)
save(F/'cohort_evidence.json',dict(status='METADATA_SOURCE_ID_AND_AVAILABLE_PIXEL_INTERSECTION_AUDIT',bindings=[pin(invpath),pin(kp),pin(O/'source_registration.json'),pin(O/'metric_registration.json'),pin(O/'facts/development100_manifest.json'),pin(O/'facts/new100_source_request_metadata.json'),pin(S/'results/entropy_main_common500_20261013/e0_reuse/source_records.json')],source_id_lists=pin(F/'cohort_source_ids.csv'),intersections=pin(F/'cohort_intersections.csv'),new_pixels_decoded=0,new_models=0,universal_or_pretrained_training_overlap_exclusion_claimed=False))
cat=read(W/'.research/main_raw64_20261007/assets/catalogue.json');audit=read(S/'experiments/entropy-partial-mismatch-20261010/reports/source_population_audit_v1.json');raw=[]
print('RAWPROFILE',str(cat['profiles'][0])[:1800])
for r in audit['frozen_raw_rows']:
 if r['snr_db']not in [4,10,19]:continue
 p=next(p for p in cat['profiles']if p['profile_id']==r['profile_id']);raw.append(dict(policy=r,profile=p))
save(F/'raw_frozen_profile_evidence.json',dict(catalogue=pin(W/'.research/main_raw64_20261007/assets/catalogue.json'),rows=raw))
csvsave(F/'fixed16_raw_native_assets.csv',[dict(method=r['method'],source_index=r['source_index'],source_id=r['source_id'],archive=r['archive']['path'],archive_sha256=r['archive']['sha256'],pixel_sha256=r['reconstruction_sha256'],precision='native_float32_RGB_256',N=1024,snr_db=13,channel_seed=r['noise_seed'],posterior_seed=23 if r['method']=='HiFiSwin200000'else'',shared_actual_RX='Swin200000 and HiFiSwin200000 only')for r in mr['pairs']])
fr=list(csv.DictReader((O/'fixed16/frame_scores.csv').open()));paired=[]
for i in mr['source_indices']:
 a=next(x for x in fr if x['source_index']==str(i)and x['method']=='RAW_PARTIAL')
 for m in ['P100000','Swin200000','HiFiSwin200000']:
  b=next(x for x in fr if x['source_index']==str(i)and x['method']==m)
  for metric in ['psnr_db','dists','dreamsim','ms_ssim']:paired.append(dict(source_index=i,source_id=a['source_id'],reference_method=m,metric=metric,raw_score=a[metric],reference_score=b[metric],raw_minus_reference=float(a[metric])-float(b[metric]),same_noise_or_y=False,new_metric_calls=0))
csvsave(F/'fixed16_source_paired_differences.csv',paired)
