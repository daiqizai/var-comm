"""Closed LE0/LE1 arithmetic/quality/capacity reports; zero new science."""
from common import *
import statistics,sys
def wait0(role):
 C=T/(role+'_control_v1');w=read(C/'actual_wait.json')
 assert type(w['returncode']) is int and w['returncode']==0 and w['actual_wait'] and not any(w[x] for x in ('timed_out','terminated','killed')) and w['metadata_error'] is None
 assert gone(w['process']) and gone(w['parent_process']);checked(w['stdout']);checked(w['stderr']);return pin(C/'actual_wait.json')
def stats(values):
 import numpy as np
 a=np.asarray(values,dtype=np.float64);assert np.isfinite(a).all()
 return dict(n=len(a),mean=float(a.mean()),min=float(a.min()),p10=float(np.quantile(a,.1)),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))
def run():
 waits={role:wait0(role) for role in ('score_completion','candidate_plan')}
 assert not (T/'score_completion_v1/failure.json').exists()
 reg=read(O/'registration.json');assert reg['authorized_stages']==['LE0','LE1'];[checked(p) for p in reg['source_bindings']]
 score=read(T/'score_completion_v1/completion.json');assert score['TX_sources']==50 and score['source_roundtrips']==24 and score['clean_quality_rows']==1200
 assert score['historical_failed_source_embedding_argument_binding_attempts']==1 and score['active_reserved_calls']==0
 for p in score['outputs']:checked(p)
 plan=read(O/'feasibility/summary.json');assert plan['wireless_image_frames']==0 and plan['encoder_decoder_forward_calls']==0
 for p in plan['data']:checked(p)
 assert read(O/'LE0/actual_visual_review.json')['actual_opened_both_PNGs'] is True
 out=O/'final';out.mkdir();lengths=list(csv.DictReader((O/'source_lengths.csv').open()));quality=list(csv.DictReader((O/'clean_quality.csv').open()))
 assert len(lengths)==3400 and len(quality)==1200
 for r in lengths:
  for k in ('pilot_slot','j','m','K','total_tokens','base_tokens','b_single','b_base','b_enhancement','extra_arithmetic_bits','extra_information_bits'):r[k]=int(r[k])
  assert r['extra_information_bits']==r['b_base']+r['b_enhancement']-r['b_single']+29
 for r in quality:
  for k in ('pilot_slot','m','K','tokens'):r[k]=int(r[k])
  for k in ('psnr_db','dinov2_vitl14_cosine'):r[k]=float(r[k])
 qby={(r['pilot_slot'],r['m'],r['K']):r for r in quality};assert len(qby)==1200
 lsummary=[]
 for j in (4,5,6,7):
  for m,K in ENDS:
   if OFFSETS[m]+K<=OFFSETS[j]:continue
   rows=[r for r in lengths if (r['j'],r['m'],r['K'])==(j,m,K)];assert len(rows)==50
   row=dict(j=j,m=m,K=K,base_tokens=OFFSETS[j],total_tokens=OFFSETS[m]+K,raw_base_bits=12*OFFSETS[j],raw_enhancement_bits=12*(OFFSETS[m]+K-OFFSETS[j]),source_count=50,base_length_bits=13,base_CRC_bits=16,enhancement_length_bits=13,enhancement_CRC_bits=16)
   for key in ('b_single','b_base','b_enhancement','extra_arithmetic_bits','extra_information_bits'):
    row.update({key+'_'+k:v for k,v in stats([x[key] for x in rows]).items() if k!='n'})
   row['median_base_to_raw_ratio']=row['b_base_median']/row['raw_base_bits'];row['median_enhancement_to_raw_ratio']=row['b_enhancement_median']/row['raw_enhancement_bits'];lsummary.append(row)
 csvsave(out/'source_length_summary.csv',lsummary)
 qsummary=[]
 for m,K in ENDS:
  rows=[r for r in quality if (r['m'],r['K'])==(m,K)];assert len(rows)==50
  row=dict(m=m,K=K,tokens=OFFSETS[m]+K,source_count=50)
  for key in ('psnr_db','dinov2_vitl14_cosine'):row.update({key+'_'+k:v for k,v in stats([r[key] for r in rows]).items() if k!='n'})
  qsummary.append(row)
 csvsave(out/'clean_quality_summary.csv',qsummary)
 delta=[]
 for j in (6,7):
  for m,K in ENDS:
   if OFFSETS[m]+K<=OFFSETS[j]:continue
   row=dict(j=j,m=m,K=K,source_count=50)
   for key in ('psnr_db','dinov2_vitl14_cosine'):
    values=[qby[s,m,K][key]-qby[s,j,0][key] for s in range(50)]
    row.update({key+'_delta_'+k:v for k,v in stats(values).items() if k!='n'});row[key+'_delta_positive_sources']=sum(v>0 for v in values);row[key+'_delta_negative_sources']=sum(v<0 for v in values)
   delta.append(row)
 csvsave(out/'clean_enhancement_vs_base.csv',delta)
 examples=[]
 desired={(6,8,0,704,4,'1/3',6,'5/6'),(7,8,0,608,4,'2/3',6,'5/6'),(6,8,0,416,6,'2/3',6,'2/3'),(7,8,0,416,6,'2/3',6,'2/3')}
 with (T/'feasibility/capacity_grid.csv').open() as f:
  for r in csv.DictReader(f):
   key=(int(r['target_j']),int(r['target_m']),int(r['target_K']),int(r['N_B']),int(r['q_B']),r['rate_B'],int(r['q_E']),r['rate_E'])
   if key in desired:examples.append(r)
 assert len(examples)==4;csvsave(out/'predefined_capacity_examples.csv',examples)
 base6=next(r for r in qsummary if (r['m'],r['K'])==(6,0));base7=next(r for r in qsummary if (r['m'],r['K'])==(7,0));end8=next(r for r in qsummary if (r['m'],r['K'])==(8,0))
 cost=stats([r['extra_information_bits'] for r in lengths]);arithmetic=stats([r['extra_arithmetic_bits'] for r in lengths]);old=read(O/'LE0/report.json')
 result=dict(status='COMPLETE_LE0_LE1_ONLY_WITH_DISCLOSED_HISTORICAL_FAILURES',old_raw_NO_GO_preserved=old['old_gate'],pilot50_hash_fixed=True,TX_probability_traversals=50,independent_source_roundtrips=24,reused_clean_states=750,new_clean_states=450,clean_quality_rows=1200,wireless_image_frames=0,packet_trials=0,tokenizer_calls=0,training_updates=0,source_model_loads=2,dino_model_loads=2,source_embedding_successes=50,failed_initial_metric_argument_attempts=1,successful_scientific_calls_repeated=0,metadata_only_asset_status_failure_preserved=True,original_resource_floor_failure_preserved=True,original_metric_API_failure_preserved=True,finish_increment_missing_rows=2584,finish_increment_persisted_rows=816,full_terminated_lengths_all_exact=True,segmentation_extra_arithmetic_bits=arithmetic,segmentation_extra_information_bits_including_second_length_and_CRC=cost,base6_quality=base6,base7_quality=base7,end8_quality=end8,capacity_examples=examples,capacity_plan=pin(O/'feasibility/summary.json'),LE2_packet_trial_cap_if_separately_authorized=21504,LE2_LE3_LE4='NOT_AUTHORIZED_NOT_RUN',hardware='Actual source/RX segment GPU3; remaining TX and clean segment GPU1, same H800 host; final DINO scores GPU1. PRO6000 inventory inspected but no scientific invocation.',actual_waits=waits)
 save(out/'summary.json',result)
 # Human report is descriptive; clean quality and capacity never assert radio gain.
 lines=['# LE0 / LE1 实际交付','',
 '参考快照078ff0d；开始HEAD为ccf9497。差异仅为已完成HiFi/common500补充指标发布；没有重复旧raw、训练、common500或六方法计时。新计划压缩包scope已核对，当前仅LE0/LE1。','',
 '## 直接回答','',
 f'1. **压缩后有可用增强空间，但不能直接称为强保护下的无线收益。** 基础j=6、j=7的实际终止码长中位数分别为{next(r for r in lsummary if (r["j"],r["m"],r["K"])==(6,8,0))["b_base_median"]:g}、{next(r for r in lsummary if (r["j"],r["m"],r["K"])==(7,8,0))["b_base_median"]:g} bit；对应raw为1092、1860 bit。到总终点(m=8,K=0)的增强流中位数分别为1602.5、944 bit。精确容量例子与完整可行域见CSV；容量可行不等于7/10 dB可靠。',
 f'2. **基础层有可描述的干净质量，是否值得无线保护仍需LE2包可靠性。** j=6：平均PSNR {base6["psnr_db_mean"]:.6f} dB、DINO-L {base6["dinov2_vitl14_cosine_mean"]:.6f}；j=7：{base7["psnr_db_mean"]:.6f} dB、{base7["dinov2_vitl14_cosine_mean"]:.6f}。总终点(8,0)：{end8["psnr_db_mean"]:.6f} dB、{end8["dinov2_vitl14_cosine_mean"]:.6f}。干净指标保留逐源正负增益，不假定更深必然更好。',
 f'3. **实际分段新增成本：** 两组独立finish相对单段的算术差中位数{arithmetic["median"]:g} bit、范围[{arithmetic["min"]:g},{arithmetic["max"]:g}]；另加第二个13 bit长度字段和16 bit CRC，总新增中位数{cost["median"]:g} bit、范围[{cost["min"]:g},{cost["max"]:g}]。两个长度字段、两次CRC和两次finish均保留。',
 f'4. **下一步原预算上限为21504个新包试验：** 96单元×128，加最多24单元×384额外试验，头也计入。精确完整缺口{plan["full_exact_missing_cell_count"]}单元，下一批96建议清单单独给出；不能声称这个有限预算覆盖全部离线可行域。LE2/3/4没有执行或自动启动。','',
 '## LE0旧raw结论','',
 '旧R1真实7/10 dB图包含24个已评RX变体（每hat9个独特物理候选），每个hat的5个两组候选在DINO二维面和PSNR二维面都被某个已评单组候选支配。二维面的支配者未必是同一个，不等价于一条单组同时支配全部四坐标。旧NO_GO_STOP及r2_authorized=false原字节保留；未改p、未重新选择。','',
 '## 范围、身份和故障披露','',
 '50源按给定哈希从旧cal200固定。原像素/token/模型全核对；tokenizer缺口0。1200个干净终点中750个精确复用，450个仅补缺口。独立源码往返24/24实际通过，RX输出先保存，再审计token/CDF；没有TX真值进入RX，图像无线帧0。',
 '原GPU3资源段在38/50 TX后触发2 GiB空闲余量线。保存全部38流与24证明；新资源登记转同主机GPU1，仅补12 TX与450恢复，原deadline及科学预算不重置。必要VAR/Dc模型重载1次单独登记。原始失败及全部actual_wait保留。',
 '原评分API入口参数错误在第一条prepare_reference参数绑定时停止，未进入源特征计算。独立评分入口核对原函数签名和reference/reconstruction顺序后，仅完成50源embedding与450新DINO分数；旧750分数不重算。DINO模型必要重载1次、失败源特征reservation1次均披露，最终没有活跃RESERVED。',
 '前38源的各次finish独立增量未在中止前落盘，因此CSV这些辅助列留空并标NOT_PERSISTED；所有最终终止bitvector和其完整码长均精确保存，B+E−single的实际总分段成本也精确。没有为补这些辅助列重复成功TX。',
 '分组后端提交独立构造/容量适配设计，原ep_phy.Backend.plan及n=956*q限制原字节不改。实际Sionna构造只是合法性证据，编码器/解码器forward与包试验0。旧iid raw几何的匹配只供溯源，不把旧CRC/长度/新profile域未验证的BLER当新协议结果。',
 '全帧固定68头+956正文=1024，名义Es=2；每组13长度+16CRC；j/m/K/布局必须由公共付费profile确定。空增强只属于显式新profile，空流仍付字段/CRC；基础无法容纳时null-profile保留失败分母。','',
 '源码往返验证不等同无线性能验证；干净前缀质量不用于改变主配置、检查点、旧raw结论。完成LE0/LE1后停止，不自动推进后续阶段。','']
 with (O/'REPORT_zh.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines));f.flush();os.fsync(f.fileno())
 save(out/'output_manifest.json',dict(status='CLOSED_CPU_REPORT_OUTPUTS',files=[pin(p) for p in sorted(out.iterdir()) if p.is_file()]+[pin(O/'REPORT_zh.md')]))
 print(canonical({k:result[k] for k in ('status','TX_probability_traversals','independent_source_roundtrips','new_clean_states','wireless_image_frames','segmentation_extra_information_bits_including_second_length_and_CRC','base6_quality','base7_quality')}),flush=True)
if __name__=='__main__':run()
