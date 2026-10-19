"""Only the package-required uncertainty-stop delivery, never a scientific rerun."""
from shared import *
from adaptive_plan import layouts,NEST_ENDPOINTS,choose
import csv
def table(path,rows):
 with path.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 w=read(L/'controls/cal100_prescreen/actual_wait.json');assert w['returncode']==0 and gone(w['process'])
 nom=read(O/'cal100_prescreen_v1/nomination.json');assert nom['status']=='PENDING_UNCOVERED_CANDIDATE_UNCERTAINTY'
 out=O/'B0_B1_delivery';out.mkdir()
 with (O/'cal100_prescreen_v1/strategy_proxy.csv').open()as f:rows=list(csv.DictReader(f))
 lookup={(int(x['codec']),int(x['snr_db']),x['policy_id']):x for x in rows};costs=[];choices=[]
 for condition,d in nom['decisions'].items():
  codec,snr=map(int,condition.split('_'));name='NeST-Com (entropy-coded)' if codec==0 else 'ARPC + LDPC-QAM (adapted)'
  for policy in d['within_MCS_leaders']:
   r=lookup[codec,snr,policy];choices.append(dict(method=name,snr_db=snr,policy_id=policy,proposal=policy in d['selected_proposals'],possible_best=policy in d['possible_best'],uncovered=policy in d['uncovered'],clean_dino_mean=r['clean_dino_mean'],proxy_dino=r['proxy_dino'],proxy_lower=r['proxy_lower'],proxy_upper=r['proxy_upper'],target=r['target'],layout_id=r['layout_id']))
  for policy in d['uncovered']:
   r=lookup[codec,snr,policy]
   costs.append(dict(method=name,snr_db=snr,policy_id=policy,additional_cal1000_logical_objects=3000,additional_physical_subframes=6000,additional_header_attempts=6000,additional_body_attempts_upper=6000,additional_decoder_attempts_upper=12000,correct_object_native_RX_upper=1000,correct_object_DINO_upper=1000,all_accepted_objects_native_RX_strict_upper=3000,all_accepted_objects_DINO_strict_upper=3000,new_TX_probability=0,new_training=0,reason='Different MCS from the two proposed leaders; cannot reuse their actual transmitted frames. Exact caches must be checked. Correctly received objects have at most1000 source/representation pairs; rare accepted-wrong objects can differ across noise, so the strict upper bound is3000.'))
 assert costs
 table(out/'candidate_status.csv',choices);table(out/'additional_costs.csv',costs)
 count=len(costs);score=read(O/'cal100_score_registration.json');native=read(L/'cal100_clean_v1/rx/completion.json')
 put(out/'actual_counts.json',dict(B0_source_objects=35200,source_TX_new=0,tokenizer_new=0,qualification_logical=24,qualification_subframes=48,qualification_decoder_attempts=96,qualification_repeated_scientific_calls=0,qualification_native_new=2,qualification_native_reused=4,cal100_native_additional=1791,cal100_native_total_including_qualification=1793,clean_states=3200,existing_image_score_reuse=1407,DINO_new_pairs=score['new_DINO_pairs'],DINO_source_features_reused=score['source_reference_reuse'],DINO_new_source_forward=0,B2_objects=0,B3_objects=0,bootstrap=0,training=0,original_RX_failed_exit_preserved=1))
 lines=['# N2048 自适应补测：B0/B1 已完成，候选覆盖待定','','本轮按附件执行到其明确的不确定性门槛。没有新增 B2/cal1000 或 B3/test100 无线成绩；不得把干净前缀质量当无线结果。','','## 已实际完成','','- 核对35,200个既有完整终止源码对象：发送端遍历与tokenizer新增均为0。','- 新双片协议通过14项单元测试和12种原LDPC布局构造。每片仍付68符号头部及13+16+96 bit正文开销。','- 有限资格24对象/48子帧/96次头正文译码：12正常或带噪对象正确重组，12主动损坏或端点不匹配对象拒收。','- 原RX最后因可选旧Python字节码缓存被隔离器拒读而退出1。所有译码结果此前已落盘，独立元数据闭合没有重复任何发送/译码；原失败原样保留。','- 实际打开六个资格恢复图；四个精确复用，两个独立补齐。全部新NeST收到token仅在输出和退出后对照真值核验。','- cal100复用1,407个已有图像/评分状态；补齐1,793个NeST前缀（含上述2个资格恢复，后续1,791个），得到3,200个完整干净状态。',f'- 新DINO评分{score["new_DINO_pairs"]}个实际缺失像素对；100源特征复用，无新源特征前向，无bootstrap。','','## 为什么停止在候选门槛','','冻结的预筛将每张图的实际长度回退质量纳入DINO均值。单包历史联合接收率的平方只作条件独立近似，并传播原Bonferroni区间；它不是新的双片实测可靠性。','同一MCS内只按共同非负成功概率代理选择干净均值最大策略，原384个完整策略全部保留。若第三种MCS策略的上界仍可超过最好的下界，不能仅为满足两个名额把它删掉。','本轮有以下未覆盖候选：','']
 for c in costs:lines.append(f'- {c["method"]}，{c["snr_db"]} dB：`{c["policy_id"]}`。')
 lines+=['',f'每个额外完整策略在cal1000×三噪声上需要3,000逻辑对象、6,000子帧、最多12,000次头/正文译码；当前未覆盖{count}项，合计额外{3000*count:,}逻辑对象、{6000*count:,}子帧、最多{12000*count:,}次译码。此为成本披露，不是已获准新增预算或实际执行。','恢复与DINO新增量只在实际合法收包后、核验精确缓存时确定。正确收包每个策略至多1,000个源/表示；若出现合法误收且三个噪声产生不同对象，严格上限是3,000。精确相同像素对仍复用。','','## 保留的边界','','若ARPC两个SNR都仅保留唯一仍可能最优的候选，可少跑两个候选。最小覆盖为9个方法/SNR/策略单元：27,000逻辑对象、54,000子帧、最多108,000次译码；相比原全局上限仍多3,000逻辑对象/6,000子帧/12,000次译码，并且NeST10dB仍需5个候选、超过每条件2个。未执行这一扩展。\n\n原N1024和旧固定full N2048不变。19dB旧600逻辑对象仅可单列历史固定完整参照；本轮没有19dB重新选策。未训练、未扩功率/UEP/反馈/KEEP，也未增加外部方法。','此次尚不能回答低SNR自适应后的实际无线交付率或最终七指标，必须完成候选覆盖问题再运行真实链路。附件规定该情况先交付待定项与成本，因此B2/B3均未启动。','','## 文件','','- candidate_status.csv：各条件12种MCS领导策略、提名与未覆盖标记。','- additional_costs.csv：逐未覆盖策略的新计算上限。','- ../cal100_prescreen_v1/actual_fallback_quality.csv、strategy_proxy.csv、nomination.json：所有实际回退与预筛证据。','- ../layouts.csv、arpc_source_cost_breakdown_detailed.csv：完整计费与源码封装开销。','- ../qualification/：保存的物理结果审计、真实图与根审查。','- actual_counts.json：本轮实际调用及零执行阶段。','','发布状态另以正常Git检查、commit/push与remoteSHA收据为准。']
 with (out/'REPORT_zh.md').open('x',encoding='utf8')as f:f.write('\n'.join(lines)+'\n')
 put(out/'completion.json',dict(status='PACKAGE_REQUIRED_PENDING_RESULT_DELIVERY',B2=False,B3=False,new_budget_authorized=False,nomination=pin(O/'cal100_prescreen_v1/nomination.json'),outputs=[pin(out/n)for n in ('REPORT_zh.md','candidate_status.csv','additional_costs.csv','actual_counts.json')],publication_pending=True))
 print(json.dumps(dict(status='PENDING_DELIVERY_READY',uncovered=count,extra_logical=3000*count,extra_subframes=6000*count,extra_decoder_upper=12000*count)))
if __name__=='__main__':main()
