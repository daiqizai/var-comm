"""One CPU-only closure and light delivery of already fixed T1-B results."""
from tb_common import *
import csv,collections,sqlite3,traceback,subprocess
import numpy as np

REG='6186a4c9c01daf9101cf0ac8478152a1a7d1c60978a3b4465213b04201307a63'
def complete_wait(p):
 w=read(p);assert w['actual_wait'] and type(w['returncode'])is int and w['returncode']==0
 assert not any(w[k]for k in('timed_out','terminated','killed')) and w['metadata_error']is None and gone(w['process'])
 checked(w['stdout']);checked(w['stderr']);return w
def copy_exact(source,dest):
 b=Path(source).read_bytes()
 with Path(dest).open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 assert pin(source)['sha256']==pin(dest)['sha256']
def read_ledger(p,packet=False):
 db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
 try:
  assert db.execute('pragma integrity_check').fetchone()==('ok',)
  if packet:
   groups=db.execute('select status,count(*) from events group by status').fetchall()
   kinds=collections.Counter(json.loads(v)['kind']for v, in db.execute('select event from events'))
   return dict(groups=groups,total=sum(n for _,n in groups),kinds=dict(kinds),unresolved=sum(n for s,n in groups if s!='COMPLETE'))
  groups=[dict(kind=k,status=s,count=n)for k,s,n in db.execute('select kind,status,count(*) from calls group by kind,status')]
  return dict(groups=groups,unresolved=sum(x['count']for x in groups if x['status']!='COMPLETE'))
 finally:db.close()

def main():
 from recovery_binding import verify_recovery
 recovery=verify_recovery('4f6b2f22b1514c0c21508ef1cf1162d1deda0cb1673daccd5eb67d3b0e2ed057')
 from metric_device_recovery_v2 import verify_metric_recovery
 verify_metric_recovery('567b8e8b39459ec8548db7ca7f2cd0fe3075a11c7d6c03d062a5e7731400036f')
 r=registration(REG);out=T/'delivery';out.mkdir(exist_ok=False)
 save(out/'launch.json',dict(process=pid_identity(os.getpid()),parent_process=pid_identity(os.getppid()),argv=sys.argv,registration=pin(O/'registration.json')))
 try:
  assert not(T/'dispatch_recovery_v2/failure.json').exists() and not(T/'control_recovery_v2/failure.json').exists()
  complete_wait(T/'dispatch_recovery_v2/actual_wait.json')
  assert gone(read(T/'dispatch_recovery_v2/supervisor.json')['process']) and gone(read(T/'control_recovery_v2/supervisor.json')['process'])
  receipt=[];done={}
  for stage in('phy','receiver','metrics','selection'):
   runtime_stage={'receiver':'receiver_recovery_v1','metrics':'metrics_recovery_v2'}.get(stage,stage)
   control=T/('control' if stage=='phy' else 'control_recovery_v1'if stage=='receiver'else'control_recovery_v2')/runtime_stage
   complete_wait(control/'actual_wait.json');assert not(T/runtime_stage/'failure.json').exists()
   done[stage]=read(T/runtime_stage/'completion.json');assert done[stage]['registration_sha256']==REG
   receipt +=[pin(control/n)for n in('prepared.json','launch.json','actual_wait.json','stdout.txt','stderr.txt')]+[pin(T/runtime_stage/'completion.json')]
  phy=done['phy'];visual=done['receiver'];metric=done['metrics'];selection=done['selection']
  logical=checkread(phy['logical_frames']);images=checkread(visual['images']);rows=checkread(metric['metric_rows']);policies=checkread(selection['policies'])
  assert len(logical)==len(rows)==36000 and policies['T1C']is False and policies['automatic_followup']is False
  inh=checkread(r['inheritance']);sources=checkread(r['sources']);manifest=checkread(r['candidate_delta']);oldw=checkread(inh['old_full_winners'])
  oldrows=checkread(inh['old_metric_rows']['full']);assert len(oldrows)==36000
  packets={};pairs={};archives={};newphysical=set();sourcekeys=set();coverage=set();grayhash=asha(np.full((3,256,256),.5,dtype=np.float32));counts=collections.Counter();framecsv=[]
  newcandidateids={x['candidate_id']for x in manifest['candidate_changes']if x['treatment']=='FULL1000_THREE_NOISE'}
  assert len(newcandidateids)==4
  for item,row in zip(logical,rows):
   ev=item['event'];i=ev['source_index'];key=(i,ev['snr_db'],ev['noise_seed'],ev['candidate_id'])
   assert key not in coverage and 0<=i<1000 and ev['snr_db']in SNRS and ev['noise_seed']in SEEDS and ev['candidate_id']in newcandidateids;coverage.add(key)
   assert all(row[k]==ev[k]for k in('source_index','source_id','snr_db','noise_seed','candidate_id','strategy_family'))
   assert row['source_id']==sources[i]['source_id'] and row['source_input_key']==item['source_input_key']
   pp=item['physical_frame'];path=pp['path']
   if path not in packets:
    packet=checkread(pp);assert source_key(packet['actual_RX'])==item['source_input_key'];packets[path]=dict(pin=pp,data=packet)
    if item['origin']=='ACTUAL_NEW':
     assert Path(path).parent==T/'phy/physical';checked(packet['observation']);newphysical.add(path)
   packet=packets[path]['data'];rx=packet['actual_RX'];image=images[item['source_input_key']];sourcekeys.add(item['source_input_key'])
   assert row['image_sha256']==image['image_sha256']
   d=image['image_archive']
   if d['path']not in archives:
    checked(d)
    with np.load(d['path'],allow_pickle=False)as z:a=z['image'].copy()
    assert a.dtype==np.float32 and a.shape==(3,256,256)and np.isfinite(a).all()and asha(a)==image['image_sha256'];archives[d['path']]=d
   pp=row['metric_pair_result']
   if pp['path']not in pairs:pairs[pp['path']]=dict(pin=pp,data=checkread(pp))
   pair=pairs[pp['path']]['data'];assert pair['metric_identity']==inh['old_metric_identity']['identity'] and pair['reference_archive']==sources[i]['archive']
   assert pair['metrics']=={k:row[k]for k in METRICS}and all(math.isfinite(float(row[k]))for k in METRICS)
   counts[ev['candidate_id'],ev['snr_db'],'logical']+=1;counts[ev['candidate_id'],ev['snr_db'],'m10_TX']+=int(ev['m']==10)
   counts[ev['candidate_id'],ev['snr_db'],'fixed_gray']+=int(row['image_sha256']==grayhash)
   counts[ev['candidate_id'],ev['snr_db'],rx['status']]+=1
   framecsv.append(dict(source_index=i,source_id=ev['source_id'],snr_db=ev['snr_db'],noise_seed=ev['noise_seed'],candidate_id=ev['candidate_id'],actual_TX_m=ev['m'],actual_TX_K=ev['K'],actual_RX_status=rx['status'],fixed_gray=row['image_sha256']==grayhash,image_sha256=row['image_sha256'],metric_reuse=row['reuse'],**{k:row[k]for k in METRICS}))
  assert len(newphysical)==22139 and sourcekeys==set(images)
  budgets={k:read_ledger(T/{'receiver':'receiver_recovery_v1','metrics':'metrics_recovery_v2'}.get(k,k)/('packets.sqlite'if k=='phy'else'calls.sqlite'),packet=k=='phy')for k in('phy','receiver','metrics')}
  assert all(x['unresolved']==0 for x in budgets.values()) and budgets['phy']['total']<=44278 and budgets['phy']['kinds']['header']==22139
  assert budgets['phy']['total']==phy['packets']['total']
  for label in('receiver','metrics'):
   declared=done[label]['ledger'];assert budgets[label]['groups']==declared['counts']
   assert all(sum(x['count']for x in budgets[label]['groups']if x['kind']==kind)<=cap for kind,cap in declared['caps'].items())
  # Validate saved means and policy choice; do not call selection again.
  grouped=collections.defaultdict(list)
  for row in oldrows+rows:grouped[row['snr_db'],row['candidate_id'],row['source_index']].append(row)
  means={}
  for snr,cid in {(s,c)for s,c,i in grouped}:
   per=[]
   for i in range(1000):
    group=grouped[snr,cid,i];assert len(group)==3 and sorted(x['noise_seed']for x in group)==list(SEEDS)
    per.append({k:math.fsum(x[k]for x in group)/3 for k in METRICS})
   means[snr,cid]={k:math.fsum(x[k]for x in per)/1000 for k in METRICS}
  with checked(selection['summary']).open(newline='',encoding='utf8')as f:summary=list(csv.DictReader(f))
  assert len(summary)==24
  for row in summary:assert all(float(row[k])==means[int(row['snr_db']),row['candidate_id']][k]for k in METRICS)
  for snr in SNRS:
   for cid,v in oldw['snrs'][str(snr)]['means'].items():assert means[snr,cid]==v
   for family,p in policies['policies'][str(snr)].items():
    assert p['DINO_means']=={c:means[snr,c]['dinov2_vitl14_cosine']for c in p['eligible']}
    assert p['DINO_means'][p['winner']]==max(p['DINO_means'].values()) and p['old_winner']in p['eligible']
  save(out/'audited_payloads.json',dict(physical_packets=[x['pin']for x in packets.values()],image_archives=list(archives.values()),metric_pairs=[x['pin']for x in pairs.values()],stage_receipts=receipt))
  light=O/'delivery';light.mkdir(exist_ok=False)
  for source in(T/'selection/calibration_summary.csv',T/'selection/source_means.csv',T/'selection/frozen_policies.json'):copy_exact(source,light/source.name)
  csvsave(light/'new_logical_frame_scores.csv',framecsv)
  statusrows=[dict(candidate_id=cid,snr_db=s,status=status,count=n)for(cid,s,status),n in sorted(counts.items())];csvsave(light/'reception_counts.csv',statusrows)
  audit=dict(schema='T1B_ACTUAL_CLOSURE_AUDIT_V1',status='T1B_DELIVERED_CALIBRATION_ONLY',registration=pin(O/'registration.json'),successful_inherited_PHY_and_recovery_stage_exits=[0,0,0,0],original_owner_exit=1,original_RX_initialization_exit=1,failed_initialization_model_load_reservations=1,failed_initialization_actual_model_weight_loads=0,recovery_registration=pin(O/'runtime_import_recovery_v1/registration.json'),failure_audit=pin(O/'runtime_import_recovery_v1/failure_audit.json'),all_stage_and_owner_processes_gone=True,actual_new_physical=len(newphysical),actual_packet_calls=budgets['phy']['total'],budgets=budgets,logical_frames=36000,unique_RX_inputs=len(images),unique_metric_pairs=len(pairs),new_metric_pairs=metric['new_pairs'],actual_new_images=visual['new_images'],old_scores_unchanged=True,received_truth_supplied=False,source_truth_audit=metric['post_output_source_audit'],audited_payloads=pin(out/'audited_payloads.json'),new_source_TX=0,new_tokenizer=0,T1C=False,LE3=False,training=False,bootstrap=False,backup_deferred=True,automatic_followup=False)
  failed=read_ledger(T/'receiver/calls.sqlite');assert failed['groups']==[dict(kind='model_load',status='FAILED',count=1)]and failed['unresolved']==1
  audit['preserved_initialization_failure_ledger']=failed
  metric_failure=read_ledger(T/'metrics/calls.sqlite');assert metric_failure==dict(groups=[],unresolved=0)
  audit.update(metric_initialization_failure=pin(O/'metric_device_recovery_v2/failure_audit.json'),metric_recovery_registration=pin(O/'metric_device_recovery_v2/registration.json'),intermediate_owner_exit=1,metric_initialization_exit=1,metric_initialization_model_calls=0,metric_initialization_scores=0)
  save(light/'closure_audit.json',audit)
  comparison=[]
  for snr in SNRS:
   for family,p in policies['policies'][str(snr)].items():comparison.append(dict(snr_db=snr,family=family,old_winner=p['old_winner'],winner=p['winner'],old_DINO=p['DINO_means'][p['old_winner']],new_DINO=p['DINO_means'][p['winner']],delta=p['delta_vs_old'],changed=p['changed']))
  csvsave(light/'policy_comparison.csv',comparison)
  text=['# T1-B：m10 对称候选校准交付','',f'原 cal1000 × 4/10/19 dB × 三次噪声。新增 36,000 个逻辑帧，实际补测 {len(newphysical):,} 个物理帧、{budgets["phy"]["total"]:,} 次包解码。已完成的旧事件、旧像素和旧评分按身份复用；没有新增源码 TX 概率遍历、tokenizer、训练或测试集调用。','', '| SNR | 家族 | 旧赢家 | 本轮冻结赢家 | 原 DINO-L | 本轮 DINO-L | 差值 |','|---:|---|---|---|---:|---:|---:|']
  for x in comparison:text.append(f'| {x["snr_db"]} | {x["family"]} | {x["old_winner"]} | {x["winner"]} | {x["old_DINO"]:.9f} | {x["new_DINO"]:.9f} | {x["delta"]:+.9f} |')
  text+=['','新增的 24 个对称策略中，20 个在原校准集上与旧最大目标候选逐源前缀完全相同；4 个有变化，全部在三个 SNR 完整评分，没有按新质量或测试集表现挑 MCS。旧 144 个 pilot 候选、12 个完整校准候选和旧赢家均保留。表中的分数按每源先平均三次噪声，再对 1000 源平均。完整四指标见 calibration_summary.csv；全部新增逻辑帧（含失败）见 new_logical_frame_scores.csv。','',f'实际新增图像输出 {visual["new_images"]:,} 个，实际新增图像对评分 {metric["new_pairs"]:,} 个。它们与逻辑帧数不同，精确复用没有改变分母。新增 m10 与回退端点均通过付费 profile 传输；仍为 68 头 + 956 正文符号，每正文支付 13 bit 长度和 16 bit CRC，Es=2。RX 仅由实际接收比特、付费配置和共享模型恢复；真值审计在固定像素之后执行。','', '这是候选集合修正后的校准结果；候选增多带来的校准最大值变化不能作为独立测试增益或新颖性结论。保留旧结果，不改写 raw R1 的 NO_GO。T1-C/common500/new100、LE3、功率扩展和新训练均未执行。按用户要求暂不备份，T1-B 正常发布交付后停止。','', 'GPU 为旧缓存同一物理 H800；FP32、TF32off、原模型/采样/指标数学保持一致。本批启动空闲显存 15GiB、私有分配 12GiB、运行空闲底线 2GiB，资源条件另行记录，不声称与所有旧段完全相同。','']
  text+=['原 RX 在导入 Torch 的第一处库文件读取时被边界检查拒绝：保留该初始化失败和一次 FAILED model_load 预留，实际权重加载、源码解码和图像生成均为零。独立恢复只允许三个冻结 Python 运行库目录的只读访问；原已完成 PHY 不重跑，原模型、CDF、恢复与评分数学未改。成功链为旧 PHY 加新 RX/评分/选择，不能将旧 owner 或旧 RX 写成退出成功。','']
  text+=['评分另有一次启动前 UUID 格式断言失败：CUDA UUID 主体和 PCI 地址与登记同卡，仅 Torch 字符串不带 GPU- 前缀。保留零模型构造、零评分的失败现场；独立修正只规范 UUID 拼写，不换设备或数值设置，已完成图像未重复生成。','']
  with(light/'REPORT_zh.md').open('x',encoding='utf8',newline='\n')as f:f.write('\n'.join(text))
  save(light/'manifest.json',dict(files=[pin(p)for p in sorted(light.iterdir())if p.is_file()],scope='T1B_ONLY',registration_sha256=REG))
  save(out/'completion.json',dict(status='CPU_DELIVERY_COMPLETE_PUBLICATION_PENDING',registration_sha256=REG,manifest=pin(light/'manifest.json'),T1C=False,automatic_followup=False))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
