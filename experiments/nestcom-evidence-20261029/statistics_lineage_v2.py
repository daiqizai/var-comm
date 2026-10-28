"""Equal-source estimates using the immutable original 100-source bootstrap bank."""
from evidence_common import *
import csv,collections,traceback
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
def interval(v,bank):
 v=np.asarray(v,np.float64);assert v.shape==(bank.shape[1],)and np.isfinite(v).all()
 q=np.quantile(v[bank].mean(1,dtype=np.float64),[.025,.975],method='linear')
 return dict(mean=float(v.mean()),ci_low=float(q[0]),ci_high=float(q[1]))
def csvwrite(p,rows):
 assert rows;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='',encoding='utf8')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 out=L/'statistics';out.mkdir(exist_ok=False);beg=time.monotonic()
 try:
  manifest=read(O/'source_manifest.json');order=manifest['source_order'];assert len(order)==len(set(order))==100 and order==manifest['source_bank']['source_ids']
  bank=np.load(checked(manifest['source_bank']['indices']),allow_pickle=False);assert bank.shape==(10000,100)and bank.dtype==np.int64 and bank.min()==0 and bank.max()==99
  allscores={};closure=[];metric_origins={}
  for s in read(O/'metric_stage_manifest_v2.json'):
   w=read(L/(s['role']+'_control')/'actual_wait.json');assert w['actual_wait']and w['returncode']==0 and gone(w['process'])and gone(w['owner'])
   d=L/s['role'];assert not(d/'failure.json').exists();c=read(d/'completion.json');assert c['ledger']['unresolved']==0
   score=cr(c['scores']);assert len(score)==c['count']==s['count']
   metric_origins[s['role']]={k:v['origin']for k,v in score.items()}
   for k,v in score.items():allscores.setdefault(k,{}).update({m:float(v[m])for m in METRICS if m in v})
   closure.append(dict(stage=s,wait=pin(L/(s['role']+'_control')/'actual_wait.json'),completion=pin(d/'completion.json')))

  # The original CLIP path failure is retained and is not a scientific result.
  failure_wait=read(L/'clip_metrics_control/actual_wait.json');assert failure_wait['actual_wait']and failure_wait['returncode']==1 and gone(failure_wait['process'])and gone(failure_wait['owner'])
  revision=read(O/'clip_path_revision_v2.json');assert revision['old_calls']==[['load_clip','once','FAILED']] and revision['numerical_or_method_changes']is False
  assert revision['old_wait']==pin(L/'clip_metrics_control/actual_wait.json') and revision['old_failure']==pin(L/'clip_metrics/failure.json')
  assert not list((L/'clip_metrics/features').glob('*.npy'))and not list((L/'clip_metrics/pairs').glob('*.json'))
  logical=read(L/'metric_preparation/resolved_logical.json');assert len(logical)==3900
  sources={s['source_id']:s for s in manifest['sources']};rows=[];ids=set()
  for r in logical:
   s=sources[r['source_id']];assert r['source_index']==s['source_index']and r['reference_sha256']==s['reference_sha256']
   key=(r['experiment'],r['arm'],r['N'],r['snr_db'],r['source_id'],r['noise_seed']);assert key not in ids;ids.add(key)
   m=allscores[r['pair_key']]if 'new_key'in r else r['metrics'];assert set(m)==set(METRICS)and all(np.isfinite(m[x])for x in METRICS)
   image=r['image'];wire=r.get('wire');rows.append(dict(experiment=r['experiment'],arm=r['arm'],N=r['N'],snr_db=r['snr_db'],source_index=r['source_index'],source_id=r['source_id'],noise_seed=r['noise_seed'],source_status=r['source_status'],fixed_gray=r['fixed_gray'],preprocessing_id=s['preprocessing_id'],reference_sha256=r['reference_sha256'],pixel_sha256=r['pixel_sha256'],image_path=image['path']if image else'',image_sha256=image['sha256']if image else'',received_key=r.get('received_key',''),wire_path=wire['path']if wire else'',wire_sha256=wire['sha256']if wire else'',receiver_origin='NEW_COUNTERFACTUAL'if'new_key'in r else r.get('reuse_kind','ORIGINAL_NATIVE_REUSE'),new_key=r.get('new_key',''),**m))
  group=collections.defaultdict(list)
  for r in rows:group[(r['experiment'],r['arm'],r['N'],r['snr_db'],r['source_id'])].append(r)
  means=[];values={}
  for (exp,arm,N,snr,sid),v in group.items():
   assert len(v)==3 and len({r['noise_seed']for r in v})==3
   m={metric:float(np.mean([r[metric]for r in v],dtype=np.float64))for metric in METRICS}
   means.append(dict(experiment=exp,arm=arm,N=N,snr_db=snr,source_index=sources[sid]['source_index'],source_id=sid,repeats=3,fixed_gray_count=sum(r['fixed_gray']for r in v),**m));values[(exp,arm,N,snr,sid)]=m
  summary=[];paired=[];conditions=sorted(set(k[:4]for k in group))
  for exp,arm,N,snr in conditions:
   vs=[values[(exp,arm,N,snr,sid)]for sid in order];rr=[r for r in rows if(r['experiment'],r['arm'],r['N'],r['snr_db'])==(exp,arm,N,snr)];assert len(rr)==300
   for metric in METRICS:summary.append(dict(experiment=exp,arm=arm,N=N,snr_db=snr,metric=metric,sources=100,logical_results=300,repeats_per_source=3,fixed_gray_count=sum(r['fixed_gray']for r in rr),**interval([x[metric]for x in vs],bank)))
  for exp,points,comparisons in [('e1',[(512,4),(512,10),(1024,10)],[('Ada-argmax','Ada-sampling'),('NeST','Ada-argmax'),('NeST','Ada-sampling')]),('e2',[(512,4),(512,10)],[('R1-VAR-completed','R0-coarse-only')])]:
   for N,snr in points:
    for left,right in comparisons:
     for metric in METRICS:
      delta=np.array([values[(exp,left,N,snr,sid)][metric]-values[(exp,right,N,snr,sid)][metric]for sid in order],np.float64)
      paired.append(dict(experiment=exp,N=N,snr_db=snr,left=left,right=right,comparison=left+' minus '+right,metric=metric,sources=100,logical_results_per_arm=300,**interval(delta,bank)))
  assert len(summary)==52 and len(paired)==44 and len(means)==1300
  receiver=read(O/'receiver_closure.json');allcounts={}
  for exp in ['e1','e2']:
   rr=[r for r in rows if r['experiment']==exp];delivery=O/exp;delivery.mkdir(exist_ok=False)
   csvwrite(delivery/'frame_metrics.csv',rr);csvwrite(delivery/'source_means.csv',[r for r in means if r['experiment']==exp]);csvwrite(delivery/'summary.csv',[r for r in summary if r['experiment']==exp]);csvwrite(delivery/'paired_deltas.csv',[r for r in paired if r['experiment']==exp])
   rc=[c for c in receiver if c['stage']['role'].startswith(exp+'_')];outputs=[j for c in rc for j in cr(read(checked(c['completion']))['outputs'])];quals=[q for c in rc for q in c['qualifications']]
   assert all(q['matched']for q in quals)
   count=dict(logical_results=len(rr),logical_results_by_arm=dict(collections.Counter(r['arm']for r in rr)),unique_received_inputs=len({r['received_key']for r in rr}),new_unique_receivers=len(outputs),qualification_replays=len(quals),new_render_calls=len(outputs)+len(quals),new_prior_forwards=sum(j['proof']['prior_forwards']for j in outputs)+sum(q['prior_forwards']for q in quals),new_decoder_forwards=sum(j['proof']['decoder_forwards']for j in outputs)+sum(q['decoder_forwards']for q in quals),new_model_loads=len(rc),counterfactual_logical_results=sum(bool(r['new_key'])for r in rr),reused_original_logical_results=sum(not bool(r['new_key'])for r in rr),new_metric_pairs=len({(r['reference_sha256'],r['pixel_sha256'])for r in rr if r['new_key']}),new_wireless_frames=0,new_training=0,new_source_encoding=0,figure_restore=0,receiver_device_seconds=sum(read(checked(c['wait'] if 'wait'in c else c['actual_wait']))['seconds']for c in rc),all_qualification_outputs_matched=True)
   pairkeys={digest([r['reference_sha256'],r['pixel_sha256']])for r in rr if r['new_key']}
   count['metric_pair_origins_by_stage']={role:dict(collections.Counter(origins[k]for k in pairkeys))for role,origins in metric_origins.items()}
   allcounts[exp]=count;save(delivery/'actual_compute_and_reuse.json',count)
   text=['# '+exp.upper()+' 实测结果','',f"固定 external_common100_v1 的100源，源内平均3次原信道重复。{len(rr)}个逻辑结果；新增{len(outputs)}个唯一接收端反事实重建，{len(quals)}个原路径资格回放。新增无线帧、训练均为0。",'', '原10000×100来源重采样索引完整复用；当前比较定义见 paired_deltas.csv。非独立盲测，不混用500源均值；所有灰图和分母保留。','', '| N | SNR | 比较 | 指标 | 均值差 | 95%来源配对区间 |','|---:|---:|---|---|---:|---|']
   for p in paired:
    if p['experiment']==exp:text.append(f"| {p['N']} | {p['snr_db']} | {p['comparison']} | {p['metric']} | {p['mean']:+.6f} | [{p['ci_low']:+.6f}, {p['ci_high']:+.6f}] |")
   text+=['','所有均值、单源值、逐重复值及原生文件身份分别见 summary.csv、source_means.csv、frame_metrics.csv。区间为逐点95%，未作多重比较校正。', 'E1只改变未收尾部的选择；原sampling主基线不替换，不能排除tokenizer/prior/decoder差异。'if exp=='e1'else'E2限定于同一适配后冻结接收机、相同实际WHOLE前缀的补全作用；并非A0/B0/C0，也不排除decoder适配因素。']
   (delivery/'REPORT_zh.md').write_text('\n'.join(text)+'\n',encoding='utf8')
  metriccounts={s['stage']['role']:read(checked(s['completion']))['ledger']for s in closure}
  device_seconds=sum(read(checked(c['actual_wait']))['seconds']for c in receiver)+sum(read(checked(c['wait']))['seconds']for c in closure)
  device_seconds+=failure_wait['seconds'];assert device_seconds/3600<12
  save(O/'actual_compute_and_reuse.json',dict(experiments=allcounts,metric_ledger=metriccounts,total_allocated_GPU_device_hours=device_seconds/3600,source_bootstrap_draws_new=0,wireless_frames_new=0,figure_restore=0))
  save(O/'statistics_identity.json',dict(source_manifest=pin(O/'source_manifest.json'),bank=manifest['source_bank'],source_order=order,comparison_definitions=read(O/'protocol.json')['statistics'],bootstrap='10000 fixed source resamples; percentile with linear quantiles; directly paired source differences',mean_CIs=52,paired_CIs=44,old_descriptive_metadata_not_adopted=True,known_preinference_failure=pin(O/'clip_path_revision_v2.json'),metric_closure=closure,code=pin(Path(__file__))))
  save(out/'completion.json',dict(status='COMPLETE',seconds=time.monotonic()-beg,mean_CIs=52,paired_CIs=44,rows=len(rows),source_means=len(means),result_identity=pin(O/'statistics_identity.json')))
  print(json.dumps(dict(status='COMPLETE',mean_CIs=52,paired_CIs=44,allocated_GPU_hours=device_seconds/3600,paired=paired)))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
