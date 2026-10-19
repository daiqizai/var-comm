"""Finite closed-run accounting and report; never imports a scientific model."""
from shared import *
from collections import Counter,defaultdict
import csv,statistics
from analysis_data import METRICS,METHODS

def dbrows(p):
 conn=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
 try:return conn.execute('select kind,key,status from calls').fetchall()
 finally:conn.close()

def main():
 r=read(REG);done=read(O/'metrics_v1/completion.json');assert done['status']=='N2048_TWO_METHOD_COMPARISON_COMPLETE'
 for d in done['outputs']:verify(d)
 for stage in ['formal_tx','formal_rx','formal_audit','formal_nest','metrics_score','metrics_statistics']:
  c=L/'controls'/stage;w=read(c/'actual_wait.json');assert type(w['returncode'])is int and w['returncode']==0 and gone(w['process']) and gone(read(c/'supervisor.json')['process']) and not(c/'failure.json').exists()
 assert read(O/'formal/arpc_process_closure.json')['status']=='GONE' and read(O/'formal/native_audit.json')['status']=='PASS_200_ACTUAL_FULL_REPRESENTATIONS'
 tx=[(phase,*q)for phase in ['qualification','formal']for q in dbrows(L/phase/'tx/calls.sqlite')];assert len(tx)==3648 and all(q[-1]=='COMPLETE'for q in tx)
 old=dbrows(L/'qualification/rx/calls.sqlite');new=dbrows(L/'qualification/rx_import_recovery_v1/calls.sqlite');formal=dbrows(L/'formal/rx/calls.sqlite')
 assert len(old)==12 and len(new)==88 and all(q[-1]=='COMPLETE'for q in old+new+formal)
 duplicated=Counter((q[0],q[1])for q in old+new);repeat=[dict(kind=k[0],key=k[1],attempts=v)for k,v in duplicated.items()if v>1]
 assert len(repeat)==4 and all(x['attempts']==2 and x['key'].startswith('q0002_')for x in repeat)
 assert len(old+new+formal)<=7300
 frames=read(O/'formal/logical_frames.json');jobs={j['key']:j for j in r['formal_jobs']};txs=read(L/'formal/tx/records.json');groups=defaultdict(list);energy=defaultdict(list)
 for t in txs:
  j=jobs[t['key']];groups[j['source_id'],j['snr_db'],j['noise_seed']].append(t)
  assert len(t['transmissions'])==2 and all(x['N']==1024 and x['Es']==2 for x in t['transmissions'])
  assert len({x['counter']for x in t['transmissions']})==2
  energy[t['codec']].append(sum(x['energy']for x in t['transmissions']))
 assert len(groups)==900 and all(len(a)==2 and a[0]['codec']!=a[1]['codec'] and a[0]['noise_sha256']==a[1]['noise_sha256']for a in groups.values())
 link=[]
 for codec in [0,1]:
  for snr in [4,10,19]:
   a=[f for f in frames if f['codec']==codec and jobs[f['key']]['snr_db']==snr];assert len(a)==300
   link.append(dict(method=METHODS[codec],snr_db=snr,logical_objects=300,accepted=sum(x['accepted']for x in a),accepted_wrong=sum(x['accepted_wrong']for x in a),rejected=sum(not x['accepted']for x in a)))
 metrics=read(O/'metrics_v1/score_completion.json');out=O/'delivery';out.mkdir()
 def savecsv(name,rows):
  with (out/name).open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 savecsv('link_delivery.csv',link)
 layout=[]
 for codec in [0,1]:
  bysource={j['source_id']:j['asset']['payload_bits']for j in r['formal_jobs']if j['codec']==codec};v=list(bysource.values());assert len(v)==100 and max(v)<=9296
  layout.append(dict(method=METHODS[codec],source_count=100,source_payload_mean_bits=statistics.mean(v),source_payload_min_bits=min(v),source_payload_max_bits=max(v),net_source_capacity_bits=9296,fit_count=100,total_channel_uses=2048,header_symbols=136,body_symbols=1912,total_body_information_bits=9560,length_fields_bits=26,outer_crc_bits=32,reassembly_bits=192,capacity_byte_padding_bits=14,nominal_energy=4096,actual_energy_mean=statistics.mean(energy[codec]),actual_energy_min=min(energy[codec]),actual_energy_max=max(energy[codec])))
 savecsv('layout_and_payload.csv',layout)
 counts=dict(qualification_transmissions=48,formal_transmissions=3600,total_transmissions=3648,qualification_decoder_attempts=100,formal_decoder_attempts=len(formal),total_decoder_attempts=len(old+new+formal),authorized_decoder_cap=7300,explicit_repeated_decodes=repeat,qualification_independent_native=12,formal_independent_native=200,new_TX_probability_passes=0,new_tokenizer_calls=0,training=0,formal_logical_metric_values=1800*7,unique_scored_pairs=metrics['unique_pairs'],new_seven_metric_pairs=metrics['new_pairs'],reused_seven_metric_pairs=metrics['reused_pairs'],bootstrap_draw_calls=1,bootstrap_replicates=10000)
 put(out/'actual_counts.json',counts)
 with (O/'metrics_v1/summary.csv').open()as f:summary=list(csv.DictReader(f))
 with (O/'metrics_v1/paired_differences.csv').open()as f:paired=list(csv.DictReader(f))
 def value(method,k):return next(float(x['mean'])for x in summary if x['method']==method and x['metric']==k and int(x['snr_db'])==19)
 lines=['# N2048 等保护双子帧：固定100源实测交付','','本轮已完成既定双方法验证。旧 N1024 结果不变；没有训练、重新生成发送端概率或扩展五方法扫描。','',
 '## 核心结果','','两方法在 4 和 10 dB 各300个对象均拒收；19 dB 各300个对象全部正确重组。全部失败保留在分母并使用固定灰图。固定64-QAM、5/6码率的完整表示在容量上可装入，但低 SNR 可靠性不足；本结果不代表已优化的全 SNR 策略。','',
 '19 dB 的完整表示质量如下。差值统一为 NeST-Com 减 ARPC；↓指标负差更有利于NeST，↑指标正差更有利。区间由源级配对差直接得到。','',
 '| 指标 | NeST-Com full m10 | ARPC full e7 | 配对差 [95% CI] |','|---|---:|---:|---:|']
 for k in METRICS:
  p=next(x for x in paired if x['metric']==k and int(x['snr_db'])==19);lines.append('| %s | %.6g | %.6g | %.6g [%.6g, %.6g] |'%(k,value(METHODS[0],k),value(METHODS[1],k),float(p['mean']),float(p['ci_low']),float(p['ci_high'])))
 lines+=['','## 布局与公平性','','每对象2×(68头符号+956正文符号)=2048信道使用；正文使用相同k4780/n5736/q6。每个子帧都付13bit长度、16bit外CRC和96bit重组记录（含完整对象CRC32副本）。总信息开销250bit，整字节容量再留14bit填充，净源码容量9296bit。NeST另保留0–7bit字节末端填充；原源码终止均保留。',
 '两方法使用同一100源、同SNR、每源相同三次2048符号标准噪声；两个子帧分别取不重叠连续切片。标称Es=2、总能量4096；实际调制帧能量逐帧记录，无逐帧归一化。双方完整表示都必须两片全部通过；没有短前缀回退、KEEP或TX真值修复。',
 '这保证了本次信道预算与传输规则对称，并不声称两模型的训练规模、结构或推理成本相同。各自使用其冻结模型的合法完整表示。','',
 '## 实际调用与异常披露','',
 f"发射{counts['total_transmissions']}个子帧（资格48、正式3600），实际头/正文译码{counts['total_decoder_attempts']}次，上限7300。首次导入故障保留；按用户明确授权仅重复q0002的4次译码，已保存的两对和全部发射不重复。ARPC首次GPU4显存准入失败在模型/ledger创建前发生；GPU1资源续接仅执行尚未开始的科学调用。", 
 f"资格独立源码往返12次，正式完整源码解码/重建200次；相同实际接收对象在三噪声逻辑帧间复用。七指标形成{metrics['unique_pairs']}个不同原图—像素对，其中{metrics['reused_pairs']}个经源码/像素/七指标数值身份核验后复用，新增评分{metrics['new_pairs']}对。逻辑指标数12600，失败未删除。", 
 '规划时发现的1个历史完整图像候选来自不同GPU/运行身份，未获跨运行复用资格；本次独立重建后核实像素相同，该次调用仍如实计为新增。旧T1-C七指标运行身份也与本次不同，故没有将其当作完全相同身份的评分缓存。详见metrics_v1/existing_cache_eligibility_audit.json。',
 '每源先平均三次噪声；仅一次10000×100源抽样矩阵，seed2026101901。所有均值与直接配对95%百分位区间共享该矩阵；区间为逐点区间，未做多重比较校正。',
 '所有接收图像固定并退出后才读取源token/CDF进行审计。正式NeST100源680token、ARPC100源七尺度token与CDF均逐一核对。所有成功调用不重跑；大备份仍按用户要求延后。','',
 '## 交付文件','','`link_delivery.csv`、`layout_and_payload.csv`、`actual_counts.json`；上级`metrics_v1/`含1800逻辑帧评分、逐源均值、42个汇总与21个配对差区间；`figures_v1/`提供矢量PDF/SVG及600dpi PNG。旧N1024、校准载荷—质量曲线和规划证据保持原样。','']
 (out/'REPORT_zh.md').write_text('\n'.join(lines))
 pins=[pin(out/n)for n in ['link_delivery.csv','layout_and_payload.csv','actual_counts.json','REPORT_zh.md']]
 put(out/'completion.json',dict(status='N2048_FINITE_TWO_METHOD_DELIVERY_COMPLETE',registration=pin(REG),qualification=pin(O/'qualification_review.json'),formal_authorization=pin(O/'formal_entry_authorization.json'),source_audit=pin(O/'formal/native_audit.json'),metrics=pin(O/'metrics_v1/completion.json'),outputs=pins,publication_pending=True,backup='DEFERRED',no_further_scientific_launch=True))
 print(json.dumps(counts))
if __name__=='__main__':main()
