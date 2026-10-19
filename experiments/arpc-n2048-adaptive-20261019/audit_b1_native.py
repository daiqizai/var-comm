"""Post-exit source audit and lossless image contact sheet, no model calls."""
from shared import *
from adaptive_plan import NEST_ENDPOINTS,SIZES
import numpy as np
from PIL import Image,ImageDraw
def main():
 w=read(L/'controls/qualification_native/actual_wait.json');launch=read(L/'controls/qualification_native/launch.json');assert w['returncode']==0 and gone(w['process']) and gone(launch['parent_process'])
 nc=read(L/'qualification/native/rx/completion.json');assert nc['ledger']['unresolved']==0
 reuse=read(L/'qualification/native/reuse.json');outs={x['source_input_key']:dict(image=x['image'],array_key='image',float_sha256=x['float_sha256'],origin='NEW_ACTUAL_RX',evidence=x)for x in nc['outputs']}
 outs.update({x['key']:dict(image=x['image'],array_key=x['array_key'],float_sha256=x['float_sha256'],origin='CLOSED_EXACT_RX_REUSE',evidence=x)for x in reuse})
 frames=[x for x in read(O/'qualification/logical_frames.json')if x['case']=='clean'];src={(x['source_id'],x['codec'],x['endpoint']):x for x in read(L/'b0/source_catalogue.json')}
 rows=[];pictures={};original={}
 for f in frames:
  a=src[f['source_id'],f['codec'],f['endpoint']];x=outs[f['source_input_key']]
  with np.load(verify(x['image']),allow_pickle=False)as z:im=z[x['array_key']].copy()
  assert hashlib.sha256(im.tobytes()).hexdigest()==x['float_sha256']
  with np.load(verify(a['source']),allow_pickle=False)as z:tokens=z['tokens'].copy();pixels=z['pixels'].copy()
  if x['origin']=='NEW_ACTUAL_RX':
   m,K=NEST_ENDPOINTS[f['endpoint']];e=x['evidence'];assert e['status']=='SOURCE_DECODED' and np.array_equal(tokens[:sum(s*s for s in SIZES[:m])+K],e['received_tokens'])
  pictures[f['source_id'],f['codec']]=np.rint(im.transpose(1,2,0)*255).astype(np.uint8);original[f['source_id']]=pixels.transpose(1,2,0)
  rows.append(dict(source_id=f['source_id'],codec=f['codec'],endpoint=f['endpoint'],source_input_key=f['source_input_key'],image=x['image'],array_key=x['array_key'],image_float_sha256=x['float_sha256'],origin=x['origin'],actual_truth_audit_after_RX_exit=True))
 assert len(rows)==6
 out=O/'qualification';put(out/'native_outputs.json',rows)
 canvas=Image.new('RGB',(3*256,3*286+24),'white');draw=ImageDraw.Draw(canvas)
 for c,t in enumerate(('Source','NeST actual prefix','ARPC actual prefix')):draw.text((c*256+8,5),t,fill='black')
 for i,sid in enumerate(original):
  for c,a in enumerate((original[sid],pictures[sid,0],pictures[sid,1])):canvas.paste(Image.fromarray(a),(c*256,24+i*286))
  draw.text((5,24+i*286+258),sid.split('/')[-1],fill='black')
 path=out/'qualification_contact_sheet.png';canvas.save(path)
 put(out/'native_audit.json',dict(status='NUMERICAL_PASS_VISUAL_REVIEW_PENDING',original_RX_failed_metadata_finalization_preserved=True,new_neural_RX=2,cache_reuse=4,new_TX=0,additional_PHY=0,received_tokens_new2_exact=True,native_wait=pin(L/'controls/qualification_native/actual_wait.json'),native_identity=pin(L/'qualification/native/rx/actual_native_identity.json'),outputs=pin(out/'native_outputs.json'),preview=pin(path)))
 print(json.dumps(dict(preview=pin(path),outputs=pin(out/'native_outputs.json'))))
if __name__=='__main__':main()
