"""Verify closed images, scalar-score provenance and detailed source costs."""
from shared import *
import numpy as np,csv
def table(path,rows):
 with path.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 out=L/'cal100_verified_cache_v1';out.mkdir();cs=read(L/'b0/cache_catalogue.json');ss=read(L/'b0/source_catalogue.json');src={(x['source_id'],x['codec'],x['endpoint']):x for x in ss if x['cal100']}
 verified={};results=[];missing=[];metadata={};cost=[]
 def once(d):
  if d['path'] not in verified:verified[d['path']]=pin(verify(d))
  assert verified[d['path']]['sha256']==d['sha256'];return Path(d['path'])
 for c in cs:
  key=(c['source_id'],c['codec'],c['endpoint'])
  if key not in src:continue
  s=src[key];im=c['image']
  if im is None:missing.append(s);continue
  archive=once(im['image_archive']);arraykey=im.get('image_key','image')
  with np.load(archive,allow_pickle=False)as z:a=z[arraykey].copy()
  assert a.dtype==np.float32 and a.shape==(3,256,256) and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
  raw=hashlib.sha256(a.tobytes()).hexdigest();structured=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
  if im.get('image_sha256'):assert im['image_sha256']==(structured if c['codec']==0 and c['origin']=='ORIGINAL_ENTROPY_RX' else raw)
  score=c['score'];assert score is not None
  if c['codec']==0:
   pair=read(once(score['pair']));assert score['image_sha256']==structured
   assert score['source_id']==s['source_id'] and score['source_index']==s['source_index']
   assert pair['metrics']==score['metrics']
   assert pair['reference_archive']['sha256']==s['source']['sha256'] and pair['reconstruction_archive']['sha256']==im['image_archive']['sha256']
   dino=float(score['metrics']['dinov2_vitl14_cosine']);proof=verified[score['pair']['path']]
  else:
   ev=im['evidence'];wire=ev.get('input',ev.get('wire'));assert once(wire).read_bytes()==once(s['wire']).read_bytes()
   once(score['evidence']);dino=float(score['values']['dinov2_vitl14_cosine']);proof=verified[score['evidence']['path']]
  assert np.isfinite(dino)
  results.append(dict(source_id=s['source_id'],source_index=s['source_index'],codec=c['codec'],endpoint=c['endpoint'],source=s['source'],source_object=s['wire'],object_sha256=s['object_sha256'],bit_length=s['bit_length'],image=verified[im['image_archive']['path']],array_key=arraykey,image_float_sha256=raw,dino=dino,score_evidence=proof,cache_provenance=c))
 # Reconstruct ARPC overhead only from exact old source metadata, no source encoder.
 for s in ss:
  if s['codec']!=1:continue
  w=Path(s['wire']['path']);mp=w.parent/(w.stem.rsplit('_e',1)[0]+'_source.json')
  if str(mp) not in metadata:metadata[str(mp)]=read(mp);verified[str(mp)]=pin(mp)
  m=metadata[str(mp)];assert m['source_id']==s['source_id'];v=m['lengths'][s['endpoint']-1]
  assert v['endpoint']==s['endpoint'] and v['payload_sha256']==s['object_sha256'] and v['payload_bits']==s['bit_length']
  fields=('caption_bits','caption_mode_bits','caption_length_bits','metadata_mode_bit','mode_and_boundary_bits','visual_bits_including_native_finishes','alignment_bits')
  assert sum(v[k] for k in fields)==s['bit_length']
  cost.append(dict(population=s['population'],source_id=s['source_id'],endpoint=s['endpoint'],payload_bits=s['bit_length'],**{k:v[k]for k in fields},tag_stream_length_bits=v['tag_stream_length_bits'],tag_arithmetic_bits_including_finish=v['tag_arithmetic_bits_including_finish'],mode_boundary_includes_tag_length_and_finish=True,metadata_path=str(mp),metadata_sha256=verified[str(mp)]['sha256']))
 put(out/'cache.json',results);put(out/'missing.json',missing);table(O/'arpc_source_cost_breakdown_detailed.csv',cost)
 put(out/'completion.json',dict(status='EXACT_CLOSED_CAL100_CACHE_VERIFIED',cached=len(results),missing=len(missing),cache=pin(out/'cache.json'),missing_plan=pin(out/'missing.json'),input_files=list(verified.values()),new_scientific_calls=0,source_cost=pin(O/'arpc_source_cost_breakdown_detailed.csv')))
 print(json.dumps(dict(cached=len(results),missing=len(missing),new_scientific_calls=0)))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=L/'cal100_verified_cache_v1'
  if p.exists():put(p/'failure.json',dict(error=traceback.format_exc(),new_scientific_calls=0))
  raise
