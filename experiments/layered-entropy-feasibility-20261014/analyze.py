"""LE0: all actually evaluated raw candidates at true7/10; no reselection."""
from common import *
from collections import Counter
def run():
 out=O/'LE0';assert not out.exists();out.mkdir()
 master=read(OLD/'r1_execution_v1/registered/registration.json')
 nominations=read(checked(master['pins']['nominations']))['rows']
 catalogue=read(checked(master['pins']['grouped_catalogue']))['profiles'];byid={p['profile_id']:p for p in catalogue}
 data=list(csv.DictReader((OLD/'r1_cache_recovery_delivery_v1/candidate_summary.csv').open()));assert len(data)==216
 values={(int(x['snr_hat']),x['candidate_id'],x['receiver_rule'],int(x['snr_true']),x['metric']):float(x['mean']) for x in data}
 points={}
 for a in nominations:
  p=byid[a['profile_id']]
  for rule in a['receiver_variants']:
   key=(a['snr_hat'],a['candidate_id'],rule)
   if key in points:
    points[key]['nomination_labels']+=';'+a['arm'];continue
   row=dict(snr_hat=a['snr_hat'],candidate_id=a['candidate_id'],receiver_rule=rule,nomination_labels=a['arm'],profile_id=a['profile_id'],G=p['G'],j=p.get('j'),m=p['m'],K=p['K'],source_count=200,noise_count=3,groups_json=canonical(p['groups']))
   for metric in ('dinov2_vitl14_cosine','psnr_db','lpips_alex'):
    for snr in (7,10):row[f'{metric}_true{snr}']=values[(*key,snr,metric)]
   points[key]=row
 assert len(points)==24
 rows=list(points.values())
 for r in rows:
  for metric in ('dinov2_vitl14_cosine','psnr_db'):
   x,y=r[f'{metric}_true7'],r[f'{metric}_true10']
   dom=[s for s in rows if s['snr_hat']==r['snr_hat'] and s['G']==1 and s[f'{metric}_true7']>=x and s[f'{metric}_true10']>=y and (s[f'{metric}_true7']>x or s[f'{metric}_true10']>y)]
   r[metric+'_single_dominators']=canonical([dict(candidate_id=s['candidate_id'],receiver_rule=s['receiver_rule']) for s in dom])
 csvsave(out/'all_actual_candidates_true7_true10.csv',rows)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':['DejaVu Sans','sans-serif'],'font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.7,'legend.frameon':False,'svg.fonttype':'none'})
 for metric,label in (('dinov2_vitl14_cosine','DINOv2-L cosine'),('psnr_db','PSNR (dB)')):
  fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
  for ax,hat in zip(axes,(7,10)):
   groups=[('Single codeword','#0F4D92','o',lambda r:r['G']==1),('Two-group EEP','#B64342','^',lambda r:r['G']==2 and 'EEP' in r['nomination_labels']),('Two-group UEP','#42949E','D',lambda r:r['G']==2 and 'U' in r['nomination_labels'])]
   for title,color,marker,accept in groups:
    q=[r for r in rows if r['snr_hat']==hat and accept(r)]
    ax.scatter([r[f'{metric}_true7'] for r in q],[r[f'{metric}_true10'] for r in q],c=color,marker=marker,s=75,alpha=.85,edgecolors='white',linewidths=.7,label=title)
   ax.set(xlabel=f'{label}, true SNR 7 dB',ylabel=f'{label}, true SNR 10 dB',title=f'Frozen raw candidates: TX estimate {hat} dB')
   ax.legend(loc='best',fontsize=10);ax.grid(alpha=.15)
  fig.suptitle('Original cal200: three noises/source; all evaluated receive variants',fontsize=14)
  for ext in ('pdf','svg','png'):fig.savefig(out/f'raw_R1_true7_true10_{metric}.{ext}',dpi=600)
  plt.close(fig)
 gate=read(OLD/'r1_cache_recovery_delivery_v1/selection.json')['gate'];assert gate['status']=='NO_GO_STOP' and gate['r2_authorized'] is False
 report=dict(status='LE0_COMPLETE_READ_ONLY',candidate_RX_variants=24,physical_candidates_per_hat=9,metrics_source=pin(OLD/'r1_cache_recovery_delivery_v1/candidate_summary.csv'),old_gate=gate,source_count=200,noise_count=3,selection_uses_reweighted_p=False,new_model_calls=0,new_wireless_frames=0,
  domination_counts={f'hat{hat}_{metric}':dict(two_group_candidates=sum(r['G']==2 and r['snr_hat']==hat for r in rows),dominated_by_single=sum(r['G']==2 and r['snr_hat']==hat and r[metric+'_single_dominators']!='[]' for r in rows)) for hat in (7,10) for metric in ('dinov2_vitl14_cosine','psnr_db')})
 save(out/'report.json',report);print(canonical(report),flush=True)
if __name__=='__main__':run()
