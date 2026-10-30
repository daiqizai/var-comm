"""One source-bootstrap pass using the original fixed10000x500 index bank."""
from g import *
import csv,collections
def csvsave(path,rows):
 with Path(path).open('x',newline='',encoding='utf8')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 done=closed('metrics');assert done['ledger']['unresolved']==0;closed('chain')
 out=D/'delivery';out.mkdir();protocol=read(D/'protocol.json');ix=np.load(checked(protocol['statistics']['bootstrap']),allow_pickle=False);scores=cr(done['scores']);mapping=read(L/'logical_pair_map.json');sources=read(D/'source_manifest.json');metrics=protocol['metrics'];frames=[]
 for f in mapping:
  v=scores[f['pair_key']];assert v['pixel_sha256']==f['pixel_sha256']and v['reference_sha256']==sources[f['source_index']]['reference_sha256']
  record={k:f[k]for k in ['source_id','source_index','method','N','snr_db','repeat','noise_seed','receive_status','pixel_sha256','pair_key']};record.update({k:v[k]for k in metrics});record['image_path']=f['image']['path']if f['image']else'FIXED_GRAY';record['image_file_sha256']=f['image']['sha256']if f['image']else'';frames.append(record)
 assert len(frames)==9000
 means=[];summary=[];arrays={};paired=[];aliases=dict(NEST='NEST',ADA='ADA_PREDCLASS_FIXEDFRAME',ARPC='ARPC_ADAPTED',BPG='BPG',P100000='P100000',Swin200000='Swin200000')
 for method in METHODS:
  rows=[f for f in frames if f['method']==method];assert len(rows)==1500
  for metric in metrics:
   values=[]
   for s in sources:
    f=[r for r in rows if r['source_id']==s['source_id']];assert len(f)==3 and {r['repeat']for r in f}=={1,2,3}
    value=float(np.mean([r[metric]for r in f]));values.append(value);means.append(dict(population='common500',source_id=s['source_id'],source_index=s['source_index'],method=aliases[method],N=1024,snr_db=7,metric=metric,repeats=3,value=value))
   a=np.array(values,np.float64);assert a.shape==(500,) and np.isfinite(a).all();arrays[method,metric]=a
   ci=np.quantile(a[ix].mean(1),[.025,.975]);summary.append(dict(population='common500',method=aliases[method],snr_db=7,metric=metric,direction='higher'if metric=='psnr_db'else'lower',source_count=500,logical_frames=1500,mean=float(a.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),statistics_origin='FROZEN7DB_COMMON500_ORIGINAL_SOURCE_BANK'))
 for method in METHODS:
  if method=='NEST':continue
  for metric in metrics:
   a=arrays['NEST',metric]-arrays[method,metric];ci=np.quantile(a[ix].mean(1),[.025,.975]);paired.append(dict(population='common500',comparison='NEST minus '+aliases[method],N=1024,snr_db=7,metric=metric,source_count=500,mean=float(a.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
 csvsave(out/'frame_metrics.csv',frames);csvsave(out/'source_means.csv',means);csvsave(out/'summary.csv',summary);csvsave(out/'paired_deltas.csv',paired)
 roles=['phy_nest','phy_ada','phy_arpc','rx_nest','rx_ada','rx_arpc','bpg_tx','bpg_phy','bpg_rx','p100000_tx','p100000_rx','swin200000_tx','swin200000_rx','metrics'];receipts={};gpu_seconds=0
 for role in roles:
  reg=read(D/(role+'_registration.json'));c=closed(role);w=read(L/(role+'_control/actual_wait.json'));receipts[role]=dict(registration=pin(D/(role+'_registration.json')),wait=pin(L/(role+'_control/actual_wait.json')),completion=pin(L/role/'completion.json'),seconds=w['seconds'],gpu=reg['gpu'],ledger=c['ledger'])
  if reg['gpu']:gpu_seconds+=w['seconds']
 assert gpu_seconds<=43200
 failure_counts={m:dict(collections.Counter(r['receive_status']for r in frames if r['method']==m))for m in METHODS}
 reuse=dict(logical_results=9000,source_count=500,methods=6,repetitions=3,new_calibration=0,new_training=0,new_digital_tokenizer=0,new_digital_source_probability_pass=0,reused_digital_source_caches=1500,new_continuous_source_encodes=1000,new_bpg_source_searches=500,new_digital_frames=6000,new_continuous_channel_events=3000,unique_metric_pairs=len(scores),logical_metric_pair_reuses=9000-len(scores),new_metric_definitions=False,GPU_actual_seconds=gpu_seconds,GPU_limit_seconds=43200,receipts=receipts,receive_status_counts=failure_counts)
 save(out/'actual_compute_and_reuse.json',reuse);save(out/'statistics_identity.json',dict(protocol=pin(D/'protocol.json'),source_order=protocol['statistics']['source_order'],bank=protocol['statistics']['bootstrap'],new_index_generation=0,source_means=18*500,mean_intervals=18,direct_paired_intervals=15,failures_kept_in_denominator=True))
 report=['# common500 的7 dB补点结果','', '六方法、原500源、每源三次信道实现，共9000个逻辑结果。沿用两图实验前已经冻结的7 dB策略；没有用本次测试结果重选MCS、端点或生成参数。7 dB的NEST/Ada/ARPC策略使用100张校准图、一次信道实现；其校准规模与原4/10/19 dB不同。BPG复用历史100图三次信道、PSNR目标的7 dB策略。','', '| 方法 | LPIPS-Alex↓ | DreamSim↓ | PSNR↑ |','|---|---:|---:|---:|']
 for m in METHODS:
  cells=[]
  for metric in ['lpips_alex','dreamsim','psnr_db']:
   s=next(s for s in summary if s['method']==aliases[m]and s['metric']==metric);cells.append('%.6f [%.6f, %.6f]'%(s['mean'],s['ci_low'],s['ci_high']))
  report.append('| '+aliases[m]+' | '+' | '.join(cells)+' |')
 report+=['','每源先平均三个重复，再以原10000×500来源重采样索引计算逐点95%区间。配对差值直接对同源差计算；两个开发案例的结果没有混入。灰图和所有失败保持原分母。','', '状态计数：`'+json.dumps(failure_counts,ensure_ascii=False)+'`。','', '实际GPU设备小时：%.6f；上限12小时。未补测CLIP-I，原表格仍保留其原4/10/19 dB范围。'%(gpu_seconds/3600),'', '大数组、实际RX输入与完整逐帧日志保存在PFS；仓库交付指标、身份、控制回执及绘图数据。']
 (out/'REPORT_zh.md').write_text('\n'.join(report)+'\n',encoding='utf8');save(out/'completion.json',dict(status='COMPLETE',rows=9000,summary=pin(out/'summary.csv'),paired=pin(out/'paired_deltas.csv'),compute=pin(out/'actual_compute_and_reuse.json')))
 print(json.dumps(read(out/'completion.json')))
if __name__=='__main__':main()
