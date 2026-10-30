"""Primary C vs independently selected matched-capacity control; no seed choice."""
from opt_common import *
from route_statistics_v2 import bank,interval,CORE,csvout
import csv

def main():
 folder=D/'fit_C_delivery';g=read(folder/'gate.json')['gates'];rule=read(D/'C_checkpoint_screen_rule.json')
 assert rule['primary_seed']==2026103001
 def choose(names):
  if not names:return None
  best=min(g[n]['score']for n in names)
  return min((n for n in names if g[n]['score']-best<.001),key=lambda n:int(n.rsplit('_',1)[1]))
 content=choose([n for n,v in g.items()if n.startswith('C_content_')and v['fit_pass']])
 control=choose([n for n in g if n.startswith('C_no_content_')])
 increment=content is not None and g[content]['score']<g[control]['score']and any(g[content]['core_normalized_means'][m]<g[control]['core_normalized_means'][m]for m in ('lpips_alex','dreamsim'))
 result=dict(route='C',status='PREPARED'if increment else'NO_GO',selected=content if increment else None,best_eligible_content=content,selected_no_content=control,content_increment=bool(increment),checkpoint_rule=pin(D/'C_checkpoint_screen_rule.json'),image_gate=pin(folder/'gate.json'),second_seed='NOT_RUN_CONDITIONAL',guards='NOT_RUN_CONDITIONAL',confirmation='NOT_RUN_CONDITIONAL',no_control_tuning=True)
 if content is not None:
  ids,b,bp=bank();rows=list(csv.DictReader((folder/'per_source_metrics.csv').open()));d={(v['variant'],int(v['N']),int(v['snr_db']),v['source_id']):v for v in rows if v['method']=='NEST'};fixed={(v['N'],v['snr_db']):v for v in read(D/'fit_baseline_denominators.json')['values']};paired=[]
  for m in ('psnr_db','lpips_alex','dreamsim','clip_image_cosine'):
   for c in CORE:
    x=np.asarray([float(d[(content,*c,s)][m])-float(d[(control,*c,s)][m])for s in ids]);mean,lo,hi=interval(x,b)
    paired.append(dict(candidate=content,reference=control,N=c[0],snr_db=c[1],metric=m,mean_delta=mean,ci95_low=lo,ci95_high=hi))
   if m in ('lpips_alex','dreamsim'):
    x=np.mean([np.asarray([float(d[(content,*c,s)][m])-float(d[(control,*c,s)][m])for s in ids])/fixed[c][m]for c in CORE],axis=0);mean,lo,hi=interval(x,b)
    paired.append(dict(candidate=content,reference=control,N=1024,snr_db='core_mean',metric='FIT_normalized_'+m,mean_delta=mean,ci95_low=lo,ci95_high=hi))
  csvout(folder/'content_minus_selected_control.csv',paired);result['content_paired_diagnostic']=pin(folder/'content_minus_selected_control.csv')
 save(folder/'decision.json',result)
 lines=['C 主种子匹配训练与六个检查点图像评估已完成。','完整模块与无内容控制各自按冻结规则选点，不挑弱控制。',json.dumps(result,ensure_ascii=False,indent=2),'训练 loss 或 token CE 改善不替代图像指标；未通过条件则不启动第二种子。']
 (folder/'REPORT_zh.txt').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
