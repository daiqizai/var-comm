"""Display-only extraction from hash-pinned, previously generated contact sheets."""
from pathlib import Path
import json,csv,hashlib,zipfile,os,time
from PIL import Image
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
D=R/'results/nestcom_perceptual_followup_20261030_v2/f1'
L=B/'fig3_candidates_20261102';L.mkdir(exist_ok=False);out=L/'tiles';out.mkdir()
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=list(csv.DictReader((D/'candidate_index.csv').open()))
pages=json.loads((D/'contact_sheet_index.json').read_text());pinmap={(v['snr_db'],v['rank_first']):v for v in pages}
eligible=[v for v in rows if v['both_metrics_both_comparators']=='True']
assert len(eligible)==166
cache={};records=[];reused_pages=[]
for v in eligible:
 snr,rank,idx=map(int,[v['snr_db'],v['rank'],v['source_index']]);first=(rank-1)//8*8+1;key=(snr,first)
 if key not in cache:
  pin=pinmap[key]['image'];p=Path(pin['path']);assert p.stat().st_size==pin['bytes']and sha(p)==pin['sha256'];cache[key]=Image.open(p).convert('RGB');reused_pages.append(pin)
 sheet=cache[key];y=40+((rank-1)%8)*290;assert y+256<=sheet.height
 for label,col in [('Source',0),('WHOLE',1),('Ada',4),('NeST',3)]:
  image=sheet.crop((col*256,y,(col+1)*256,y+256));assert image.size==(256,256)
  name=f's{snr:02d}_i{idx:03d}_{label}.png';p=out/name;image.save(p)
  records.append(dict(snr_db=snr,source_index=idx,source_id=v['source_id'],original_rank=rank,method=label,filename=name,sha256=sha(p),display_RGB_sha256=hashlib.sha256(image.tobytes()).hexdigest(),contact=pinmap[key]['image'],box=[col*256,y,(col+1)*256,y+256]))
# Validate extraction against the two existing full-resolution Fig.3 exports.
selected=B/'nestcom_perceptual_followup_20261030_v2/F1_selected'
for row,idx in [(1,84),(2,143)]:
 for label,old in [('Source','source'),('WHOLE','WHOLE_final_receiver'),('Ada','Ada_frozen'),('NeST','final_NeST')]:
  assert Image.open(out/f's07_i{idx:03d}_{label}.png').convert('RGB').tobytes()==Image.open(selected/f'row{row}_{old}.png').convert('RGB').tobytes()
receipt=dict(status='EXPORTED',rows=len(eligible),images=len(records),records=records,pages=reused_pages,first_repeat_only=True,all_field_of_view=True,pixel_changes='None: exact 256x256 RGB blocks extracted from the previously hash-pinned PNG contact sheets',new_model_calls=0,new_PHY=0,new_metrics=0,new_statistics=0,extraction_identity_checked_against_existing_Fig3_rows=[84,143])
(L/'extraction_identity.json').write_text(json.dumps(receipt,indent=2)+'\n')
z=L/'display_tiles.zip'
with zipfile.ZipFile(z,'x',zipfile.ZIP_DEFLATED,compresslevel=3)as pack:
 for p in sorted(out.glob('*.png')):pack.write(p,'tiles/'+p.name)
 pack.write(L/'extraction_identity.json','extraction_identity.json')
print(json.dumps(dict(path=str(z),sha256=sha(z),bytes=z.stat().st_size,rows=len(eligible),images=len(records),contact_pages_reused=len(cache))))
