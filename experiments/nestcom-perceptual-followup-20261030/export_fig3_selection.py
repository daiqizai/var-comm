"""Export an explicitly selected illustration; retain full native identities."""
from followup_common import *
from PIL import Image

def main():
 closed('F1_contacts');closed('E1_statistics')
 dst=D/'f1';out=L/'F1_selected';out.mkdir(exist_ok=False)
 sources=read(L/'inventory_core/sources.json');frames=read(L/'E1_join/frames.json')
 chosen=[dict(source_index=84,label='Television',snr_db=7,reason='Inspect cabinet, screen and antenna geometry; NeST preserves more of the source layout than the displayed alternatives.'),dict(source_index=143,label='Patterned leather',snr_db=7,reason='Inspect rectangular buckle and printed motifs; the source geometry and markings distinguish these reconstructions.')]
 selection=dict(selection='Selected illustrative advantage examples, not random or representative',chosen=chosen,repeat_index=0,inspected=dict(cells=[7,10],ranks_per_cell=list(range(1,41)),contact_pages_per_cell=list(range(1,6)),review='All ten top-40 contact sheets were actually opened and visually inspected before this selection.'),ranking_protocol=pin(dst/'selection_protocol.json'),no_seed_search=True,main_statistics_unchanged=True,display_only_quantization=True,new_model_calls=0,new_PHY=0)
 save(dst/'selected_examples.json',selection)
 exported=[]
 for row,c in enumerate(chosen):
  s=sources[c['source_index']];assert s['source_index']==c['source_index']
  with np.load(checked(s['archive']),allow_pickle=False)as z:a=z['pixels'];assert hashlib.sha256(a.tobytes()).hexdigest()==s['preprocessing_id']
  path=out/f'row{row+1}_source.png';Image.fromarray(a.transpose(1,2,0)).save(path)
  exported.append(dict(row=row+1,column=0,label=c['label'],source_id=s['source_id'],source_index=s['source_index'],source_pixels_sha256=s['preprocessing_id'],reference_sha256=s['reference_sha256'],method='Source',N=1024,snr_db=c['snr_db'],channel_repeat=0,display=pin(path),native_source=s['archive']))
  group={m:next(v for v in frames if v['source_id']==s['source_id']and v['N']==1024 and v['snr_db']==c['snr_db']and v['channel_repeat']==0 and v['method']==m)for m in ['WHOLE_final_receiver','Ada_frozen','final_NeST','original_NeST']}
  assert group['WHOLE_final_receiver']['received_key']!=group['final_NeST']['received_key']
  assert group['WHOLE_final_receiver']['pixel_sha256']!=group['final_NeST']['pixel_sha256']
  for column,(m,v)in enumerate(group.items(),1):
   if v['image']is None:im=np.full((3,256,256),.5,np.float32)
   else:
    with np.load(checked(v['image']),allow_pickle=False)as z:im=z['image']
   assert asha(im)==v['pixel_sha256'];path=out/f'row{row+1}_{m}.png';Image.fromarray(np.rint(im.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(path)
   exported.append(dict(v,row=row+1,column=column,label=c['label'],source_pixels_sha256=s['preprocessing_id'],display=pin(path),metrics_are_native_not_PNG=True))
 save(dst/'selected_native_mapping.json',exported);csvout(dst/'display_metrics.csv',[{k:v[k]for k in ['row','column','label','source_id','source_index','method','N','snr_db','channel_repeat',*METRICS,'noise_seed','pixel_sha256','display']if k in v}for v in exported])
 save(out/'completion.json',dict(status='COMPLETE',source_count=2,PNG_count=10,new_model_calls=0,new_PHY=0,new_metric_calls=0,main_columns=['Source','WHOLE','Ada (adapted)','NeST-Com'],full_frame_256=True))
 print('F1_SELECTED',[(x['source_index'],x['label'])for x in chosen])
if __name__=='__main__':main()
