"""Own remaining low-budget stages and source-paired reporting, without retries."""
from study_common import *
import low_budget,traceback

def wait(role):
 while True:
  if (L/role/'failure.json').exists():raise RuntimeError('Preserved failure: '+role)
  p=L/(role+'_control')/'actual_wait.json'
  if p.exists():
   w=read(p)
   if w['returncode']!=0:raise RuntimeError(('actual child failure',role,w))
   n=ident(w['owner']['pid'])
   if n is None or n['start_ticks']!=w['owner']['start_ticks']or n['state']=='Z':return closed(role)
  assert time.time()<read(D/'protocol.json')['deadline_unix']and not(L/'STOP').exists()
  time.sleep(5)

def summarize():
 frames=read(L/'low_metric_plan/logical.json');ss=read(L/'low_plan/sources.json');sets=read(F/'source_sets.json')
 assert [s['source_id']for s in ss]==sets['control100']
 bank=np.load(checked(sets['bootstrap100']),allow_pickle=False);assert bank.shape==(10000,100)and bank.min()>=0 and bank.max()<100
 scores={}
 for mode in ['three','clip','dists']:
  c=closed('low_'+mode)if (L/('low_'+mode)).exists()else None
  if c:
   for key,v in cr(c['scores']).items():scores.setdefault(key,{}).update(v)
 for v in frames:
  if v['metric_origin']=='NEW_FIVE_METRIC_GAP':v.update({m:scores[v['pair_key']][m]for m in METRICS})
  for m in METRICS:v[m]=float(v[m]);assert np.isfinite(v[m])
 assert len(frames)==1200 and len({(v['source_id'],v['method'],v['channel_repeat'])for v in frames})==1200
 sources=[];vectors={}
 for method in ['whole','quarter','eighth','token']:
  for s in ss:
   rows=[v for v in frames if v['source_id']==s['source_id']and v['method']==method];assert sorted(v['noise_seed']for v in rows)==[6201,6202,6203]
   sources.append(dict(source_id=s['source_id'],source_index=s['source_index'],method=method,N=512,snr_db=10,repeats=3,rejected=sum(v['receiver_status']!='ACCEPTED_SOURCE'for v in rows),**{m:float(np.mean([v[m]for v in rows]))for m in (*METRICS,'utilization','source_bits','actual_length_comparisons')}))
  for m in (*METRICS,'utilization','source_bits','actual_length_comparisons'):vectors[method,m]=np.asarray([v[m]for v in sources if v['method']==method])
 def ci(a):return dict(mean=float(a.mean()),ci_low=float(np.quantile(a[bank].mean(1),.025)),ci_high=float(np.quantile(a[bank].mean(1),.975)),sources=100,repeats_per_source=3)
 summary=[dict(method=method,metric=m,**ci(a))for(method,m),a in vectors.items()]
 paired=[dict(comparison=left+'-'+right,metric=m,**ci(vectors[left,m]-vectors[right,m]))for left,right in [('token','whole'),('eighth','whole'),('token','eighth')]for m in METRICS]
 out=D/'low_budget';out.mkdir(exist_ok=False)
 csvout(out/'frame_metrics.csv',frames);csvout(out/'source_means.csv',sources);csvout(out/'summary.csv',summary);csvout(out/'paired_deltas.csv',paired)
 save(out/'statistics_identity.json',dict(bank=sets['bootstrap100'],source_order=[s['source_id']for s in ss],source_order_sha256=digest([s['source_id']for s in ss]),replicates=10000,confidence=.95,unit='source; average three received outcomes first',direct_paired=True))
 c={role:closed(role)for role in ['low_phy','low_rx']+['low_'+x for x in ['three','clip','dists']if (L/('low_'+x)).exists()]}
 save(out/'actual_compute_and_reuse.json',c)
 lines=['# N=512, 10 dB：完全匹配粒度控制','', '同一 control100、同一 16-QAM 2/3、共同上限 (7,0)、同一模型/实际接收规则与 6201/6202/6203 噪声。仅改变合法终止位置；所有失败保留分母。','', '| 对比 | 指标 | 源均值差 | 来源配对 95% 区间 |','|---|---|---:|---|']
 lines += [f"| {v['comparison']} | {v['metric']} | {v['mean']:.6f} | [{v['ci_low']:.6f}, {v['ci_high']:.6f}] |"for v in paired]
 lines += ['', '源码容量利用率不是总空口效率。码长检查次数按完整扫描计数，不是额外模型前向或计时。正负结果均保留；本结果只解释此固定工作点。']
 (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 save(L/'low_finish/completion.json',dict(status='COMPLETE',logical=1200,source_rows=400,quality_mean_CIs=20,quality_paired_CIs=15,report=pin(out/'REPORT_zh.md')))

def main():
 out=L/'low_finish';out.mkdir(exist_ok=False);save(out/'owner.json',dict(owner=ident(os.getpid()),created=time.time(),code=[pin(E/x)for x in ['low_finish.py','low_budget.py','study_common.py']]))
 try:
  wait('low_phy');low_budget.receiver();wait('low_rx');low_budget.metric_plan()
  if read(L/'low_metric_plan/completion.json')['missing_pairs']:
   for mode in ['three','clip','dists']:low_budget.score(mode);wait('low_'+mode)
  summarize()
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
