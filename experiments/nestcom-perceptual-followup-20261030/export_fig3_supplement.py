"""Display-only export of the already selected sources, no new scientific calls."""
from followup_common import *
from PIL import Image

def main():
 closed('E1_statistics'); selection=read(D/'f1/selected_examples.json')
 frames=read(L/'E1_join/frames.json'); out=L/'F1_selected_external';out.mkdir(exist_ok=False)
 exported=[]
 methods=['Latent_JSCC','SwinJSCC','BPG_LDPC','ARPC']
 for row,c in enumerate(selection['chosen'],1):
  for method in methods:
   vs=[v for v in frames if v['source_index']==c['source_index'] and v['N']==1024 and v['snr_db']==c['snr_db'] and v['channel_repeat']==0 and v['method']==method]
   assert len(vs)==1;v=vs[0]
   if v['image'] is None:im=np.full((3,256,256),.5,np.float32)
   else:
    with np.load(checked(v['image']),allow_pickle=False)as z:im=z['image']
   assert asha(im)==v['pixel_sha256'] and im.shape==(3,256,256)
   p=out/f'row{row}_{method}.png';Image.fromarray(np.rint(im.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(p)
   exported.append(dict(v,row=row,label=c['label'],display=pin(p),metrics_are_native_not_PNG=True))
 save(D/'f1/external_native_mapping.json',exported)
 csvout(D/'f1/external_display_metrics.csv',[{k:v[k]for k in ['row','label','source_id','source_index','method','N','snr_db','channel_repeat',*METRICS,'noise_seed','pixel_sha256','display']if k in v}for v in exported])
 allrows=read(D/'f1/selected_native_mapping.json')+exported
 csvout(D/'f1/fig3_selected_sources.csv',[{k:v[k]for k in ['row','label','source_id','source_index','method','N','snr_db','channel_repeat','noise_seed','received_key','receiver_status','source_bits','capacity_bits','pixel_sha256','reference_sha256','image','display']if k in v}for v in allrows])
 csvout(D/'f1/fig3_metrics.csv',[{k:v[k]for k in ['row','label','source_id','source_index','method','N','snr_db','channel_repeat',*METRICS,'pixel_sha256']if k in v}for v in allrows])
 notes='Selected illustrative advantages, not random or representative. Fixed repeat index 0; historical method-specific noise identities retained. No new model, PHY, or scoring calls.\n\n'+ '\n\n'.join(f"Source {c['source_index']} ({c['label']}), N1024/{c['snr_db']} dB: {c['reason']}" for c in selection['chosen'])
 (D/'f1/fig3_selection_notes.txt').write_text(notes+'\n')
 save(out/'completion.json',dict(status='COMPLETE',source_count=2,PNG_count=8,new_model_calls=0,new_PHY=0,new_metric_calls=0,selected_mapping=pin(D/'f1/selected_native_mapping.json')))
 print('EXPORTED',len(exported));print('FRAME_KEYS',sorted(frames[0]))
if __name__=='__main__':main()
