"""One finite LE2 owner: singles/header -> complete policies -> capped topups."""
from common import *
import planning as p,packets as phy
import numpy as np,sys,fcntl,traceback,socket,itertools
class Measurement:
 def __init__(self,runtime,ledger,layouts):self.runtime=runtime;self.ledger=ledger;self.layouts=layouts;self.counts={};self.payload_lengths={};self.initial=set();self.topup=set();self.calls=0
 def table(self):
  return {k:dict(p_correct=v['correct']/v['n'],ci=wilson(v['correct'],v['n']),n=v['n'],n_correct=v['correct'],n_rejected=v['rejected'],n_undetected=v['undetected']) for k,v in self.counts.items()}
 def point(self,gid,snr,lengths,topup=False):
  key=gid,snr
  if not topup and key in self.counts:return
  if topup:
   assert key in self.counts and key not in self.topup;self.topup.add(key);assert len(self.topup)<=24
  else:self.initial.add(key);assert len(self.initial)<=96;self.payload_lengths[key]=list(lengths)
  lengths=self.payload_lengths.get(key,lengths);count=384 if topup else 128
  directory=T/'packets/cells'/f'{gid}_{snr}';directory.mkdir(parents=True,exist_ok=True);self.counts.setdefault(key,dict(n=0,correct=0,rejected=0,undetected=0,energy_sum=0.,exact_hard=0));totals=self.counts[key]
  for start in range(totals['n'],totals['n']+count,32):
   assert not (T/'STOP').exists();keys=[f'{gid}:{snr}:{i}' for i in range(start,start+32)];assert self.calls+32<=CAP
   self.ledger.reserve('packet',keys,CAP);self.calls+=32
   try:
    if gid=='header':
     rows,pids=phy.header_batch(self.runtime,start,32,snr)
     # Raw actual received field/CRC is sealed before comparison to TX pid.
     save(directory/f'rx_{start:04d}.json',dict(received=rows,receiver_truth_supplied=False,domain='ALL4096_RAW_FIELDS_NO_LE3_FALSE_ACCEPT_CLAIM'))
     accepted=np.array([r['header_ok'] for r in rows]);hard_exact=np.array([r['profile_id']==pid for r,pid in zip(rows,pids)]);energy=np.full(32,136.)
     save(directory/f'truth_audit_{start:04d}.json',dict(transmitted_pids=pids,correct=(accepted&hard_exact).tolist(),actual_rx=pin(directory/f'rx_{start:04d}.json')))
    else:
     rec,truth=phy.body_batch(self.runtime,self.layouts[gid],start,32,snr,lengths)
     rx=directory/f'rx_{start:04d}.npz'
     with rx.open('xb') as f:np.savez(f,hard=rec['hard'],accepted=rec['accepted'],declared_lengths=rec['declared_lengths']);f.flush();os.fsync(f.fileno())
     save(directory/f'rx_{start:04d}.json',dict(rx=pin(rx),statuses=rec['statuses'],receiver_truth_supplied=False))
     accepted=rec['accepted'];hard_exact=np.all(rec['hard']==truth,axis=1);energy=rec['actual_energy']
     with (directory/f'tx_audit_{start:04d}.npz').open('xb') as f:np.savez(f,information=truth,lengths=rec['transmitted_lengths'],actual_energy=energy);f.flush();os.fsync(f.fileno())
     save(directory/f'truth_audit_{start:04d}.json',dict(actual_rx=pin(rx),correct=(accepted&hard_exact).tolist(),tx_audit=pin(directory/f'tx_audit_{start:04d}.npz'),truth_read_only_after_RX_fixed=True))
    correct=accepted&hard_exact;totals['n']+=32;totals['correct']+=int(correct.sum());totals['exact_hard']+=int(hard_exact.sum());totals['rejected']+=int((~accepted).sum());totals['undetected']+=int((accepted&~correct).sum());totals['energy_sum']+=float(energy.astype(np.float64).sum())
    assert totals['correct']+totals['rejected']+totals['undetected']==totals['n'] and totals['n']<=512
    save(directory/f'counts_{totals["n"]:04d}.json',dict(gid=gid,snr=snr,counts=dict(totals),payload_lengths=lengths,layout=self.layouts.get(gid),registration=pin(O/'registration.json')))
   except BaseException as e:self.ledger.finish('packet',keys,repr(e));raise
   self.ledger.finish('packet',keys)
  print(f'CELL_COMPLETE {gid[:12]} snr={snr} n={totals["n"]} correct={totals["correct"]} total={self.calls}',flush=True)
 def paired_geometry(self,gid,lengths,topup=False):
  for s in SNRS:self.point(gid,s,lengths,topup)
def rank(rows,table,q0):
 out=[]
 for r in rows:
  if all((g,s) in table for g in (r['B'],r['E']) if g for s in SNRS):
   score=p.proxy(r,table,q0);lo,hi=p.interval(r,table,q0);out.append(dict(r,**score,J_sensitivity_low=lo,J_sensitivity_high=hi))
 return sorted(out,key=lambda r:(-r['J'],r['policy_id']))
def run(expected):
 assert sha(O/'registration.json')==expected;reg=verify_registration();out=T/'packets';out.mkdir()
 save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),argv=sys.argv,host=socket.gethostname(),registration=pin(O/'registration.json')))
 lock=(T/'science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);available=os.sched_getaffinity(0);os.sched_setaffinity(0,set(sorted(available)[:2]))
 assert os.environ['CUDA_VISIBLE_DEVICES']=='';ledger=Ledger(out/'calls.sqlite');runtime=phy.original();backend=runtime[0]
 save(out/'runtime_identity.json',dict(backend=backend.identity,device='cpu',host=socket.gethostname(),cpu_affinity=sorted(os.sched_getaffinity(0)),threads=backend.torch.get_num_threads(),interop=backend.torch.get_num_interop_threads(),CUDA_initialized=backend.torch.cuda.is_initialized(),numeric='FP32 TF32off deterministic original2/1',mapping_sources=reg['mapping_bindings']))
 layouts,props=p.make_layouts(backend);bygid={g['geometry_id']:g for g in layouts.values() if g['layout_status'].startswith('ACTUAL')};data=p.load();d,Q,Q0=data;q0=float(Q0.mean());single=p.singles(layouts,data)
 save(T/'single_policy_grid.json',single);m=Measurement(runtime,ledger,bygid);single_lengths={}
 for q,r in p.MCS:
  layout=layouts[956,q,r];st=p.source_states(d,Q,Q0,None,(9,192),layout['k']-29);lengths=[x['L_B'] for x in st if x['L_B']>=2]
  assert lengths;single_lengths[layout['geometry_id']]=lengths
 save(O/'stage_A_plan.json',dict(cells=[dict(gid=gid,snr=s,layout=bygid[gid],payload_lengths=ls,trials=128) for gid,ls in single_lengths.items() for s in SNRS]+[dict(gid='header',snr=s,trials=128) for s in SNRS],initial_cells=32,maximum_packets=4096,all_single_MCS_covered=True,source_quality=pin(O1/'clean_quality.csv')))
 m.paired_geometry('header',[])
 for gid,ls in single_lengths.items():m.paired_geometry(gid,ls)
 assert m.calls==4096;save(out/'stage_A_completion.json',dict(packet_trials=m.calls,cells=32,counts=ledger.summary(),table=[dict(gid=k[0],snr=k[1],**v) for k,v in m.table().items()]))
 pool=p.candidates(layouts,props,data);save(T/'group_policy_grid.json',pool);chosen=p.nominate(pool,single,layouts,m.table(),q0);assert len(chosen)==16
 save(O/'stage_B_complete_policy_nominations.json',dict(records=chosen,full_group_SNR_coverage=True,quality_transfer_screen_is_NOT_group_BLER=True,capacity_frequency_used=False,pool_size=len(pool),max_new_cells=64,maximum_cumulative_packets=12288))
 detail=[];group_lengths={}
 for row in chosen:
  b,e=bygid[row['B']],bygid[row['E']];states=p.source_states(d,Q,Q0,row['target_j'],(row['target_m'],row['target_K']),b['k']-29,e['k']-29)
  detail.extend(dict(policy_id=row['policy_id'],**s) for s in states)
  for role in ('B','E'):
   gid=row[role];ls=[s['L_'+role] for s in states if s['status']!='NULL'];assert ls
   if gid not in group_lengths:group_lengths[gid]=ls
 save(O/'stage_B_cells.json',dict(cells=[dict(gid=g,snr=s,layout=bygid[g],payload_lengths=ls,trials=128) for g,ls in group_lengths.items() for s in SNRS],public_format='isolated paid geometry and explicit empty flag, no end-to-end profile claim'))
 csvsave(O/'nominated_source_fallbacks.csv',detail)
 for gid,ls in group_lengths.items():m.paired_geometry(gid,ls)
 assert len(m.initial)<=96 and m.calls<=12288
 table=m.table();srank=rank(single,table,q0);grank=rank(pool,table,q0)
 assert grank;robust=srank[0];back=max(srank,key=lambda r:(r['Q7'],r['policy_id']));nom=next(r for r in srank if (r['target_m'],r['target_K'],r['q_B'],r['rate_B'])==(8,126,4,'2/3'))
 refined={'header'};decisions=[dict(reason='shared_paid_header',geometries=['header'],admitted=True)]
 targets=[(robust,'Srobust'),(back,'Sback3'),(nom,'existing_entropy_Snom')]+[(r,'top_U_complete') for r in grank[:3]]+[(r,'top_EEP_complete') for r in [x for x in grank if x['family']=='EEP'][:2]]
 targets+= [(r,'ranking_width_complete') for r in sorted(grank[:30],key=lambda r:-(r['J_sensitivity_high']-r['J_sensitivity_low']))]
 for row,reason in targets:
  required={g for g in (row['B'],row['E']) if g};new=required-refined;yes=len(refined|new)<=12
  decisions.append(dict(policy_id=row['policy_id'],reason=reason,geometries=sorted(required),new=sorted(new),admitted=yes))
  if yes:refined.update(new)
 save(O/'stage_C_topup_plan.json',dict(decisions=decisions,geometries=sorted(refined),cells=[dict(gid=g,snr=s,additional_trials=384) for g in sorted(refined) for s in SNRS],paired_SNR_bundles=True,maximum_total_packets=21504))
 for gid in sorted(refined):m.paired_geometry(gid,group_lengths.get(gid,single_lengths.get(gid,[])),True)
 assert m.calls<=CAP and len(m.topup)<=24 and len(m.initial)<=96
 table=m.table();srank=rank(single,table,q0);grank=rank(pool,table,q0);robust=srank[0];back=max(srank,key=lambda r:(r['Q7'],r['policy_id']));nom=next(r for r in srank if (r['target_m'],r['target_K'],r['q_B'],r['rate_B'])==(8,126,4,'2/3'))
 rows=[]
 for r in srank+grank:
  dl,dh=p.delta_bounds(r,robust,table);bl,bh=p.delta_bounds(r,back,table)
  rows.append(dict(r,delta_J_Srobust=r['J']-robust['J'],delta_J_Sback3=r['J']-back['J'],delta_robust_sensitivity_low=dl,delta_robust_sensitivity_high=dh,delta_back3_sensitivity_low=bl,delta_back3_sensitivity_high=bh,quality_kind='CORRECT_PREFIX_PACKET_PROXY_NOT_WIRELESS_IMAGES',U_domain=True))
 csvsave(O/'measured_policy_results.csv',rows)
 csvsave(O/'packet_cells.csv',[dict(gid=g,snr_db=s,kind='header' if g=='header' else 'body',N=68 if g=='header' else bygid[g]['N'],q=2 if g=='header' else bygid[g]['q'],rate='header' if g=='header' else bygid[g]['nominal_rate'],k=28 if g=='header' else bygid[g]['k'],n=136 if g=='header' else bygid[g]['n'],**{k:value for k,value in v.items() if k!='ci'},ci=canonical(v['ci']),zero_errors_not_zero_BLER=True,protocol='L13_CRC16_FIXED_PADDING' if g!='header' else 'PROFILE12_CRC16_TAIL6') for (g,s),v in table.items()])
 # Preserve exact k/n/q/interleaver old raw lookup as AUXILIARY only. Its raw
 # source parser and payloads differ, so no trials are silently double-counted.
 old_gaps=list(csv.DictReader((O1/'feasibility/LE2_exact_missing_cells.csv').open()));legacy={r['geometry_id']:r['legacy_geometry_points'] for r in old_gaps if r['kind']=='body' and r['legacy_geometry_match']=='True'}
 save(O/'legacy_cell_reuse_audit.json',dict(old_evidence=pin(O1/'feasibility/LE2_exact_missing_cells.csv'),exact_geometry_auxiliary_matches=[dict(gid=g,old=legacy[g]) for g in group_lengths if g in legacy],new_protocol_success_rate_reuse=0,reason='Old raw geometry/IID table is not new L13 parser/pilot-length protocol; source/distribution applicability retained, no old scientific stage rerun'))
 chosen3=[];roles={}
 def add(row,role):
  # Identity includes source target and physical layout; repeated named anchors
  # are one unique candidate with multiple descriptive roles.
  identity=(row['target_j'],row['target_m'],row['target_K'],row['B'],row['E'])
  existing=next((r for r in chosen3 if r['_identity']==identity),None)
  if existing:existing['roles'].append(role);return
  chosen3.append(dict(row,_identity=identity,roles=[role],LE3_execution_authorized=False))
 for r in grank[:3]:add(r,'U')
 for r in srank[:3]:add(r,'robust_single')
 for r in [x for x in grank if x['family']=='EEP'][:2]:add(r,'EEP')
 add(nom,'old_entropy_nominal');add(back,'entropy_backoff3')
 assert len(chosen3)<=10
 for r in chosen3:del r['_identity']
 save(O/'LE3_candidates.json',dict(status='EXPLORATORY_CANDIDATES_ONLY_STOP_BEFORE_LE3',records=chosen3,LE3_authorized=False,existing_nominal_config=reg['entropy_nominal_binding'],no_raw_anchor_substitution=True))
 positive=[r for r in grank if r['J']>robust['J'] and r['J']>back['J']];best=grank[0];bestrow=next(r for r in rows if r['policy_id']==best['policy_id'])
 status='BLOCKED_TABLE_UNCERTAINTY' if positive else 'NO_PROMISING_MEASURED_CANDIDATE'
 ledger.db.commit();counts=ledger.summary();assert sum(x['count'] for x in counts if x['kind']=='packet')==m.calls and all(x['status']=='COMPLETE' for x in counts)
 summary=dict(status=status,LE2_packet_execution='COMPLETE_FINITE_CAPPED',packet_trials=m.calls,initial_cells=len(m.initial),topup_cells=len(m.topup),packet_cap=CAP,measured_single_policies=len(srank),measured_group_policies=len(grank),unmeasured_group_policies=len(pool)-len(grank),measured_positive_group_estimates=len(positive),best_group=bestrow,Srobust=robust,Sback3=back,Snom=nom,gray_sources50=True,gray_new_pairs33=True,LE0_LE1_preserved=True,source_VAR_TX_calls=0,clean_render_calls=0,wireless_image_frames=0,train_updates=0,LE3_executed=False,power_expansion_executed=False,global_negative_or_GO_claim=False,ledger=pin(out/'calls.sqlite'),counts=counts,unresolved=0,source_runtime=pin(out/'runtime_identity.json'),interval_warning='Wilson cellwise95%; sensitivity rectangles not simultaneous/paired confidence intervals',proxy_limitations=['iid scrambled codeblock approximation','wrong header/catalogue-domain effects','undetected arithmetic/parser corruption','source and conditional packet dependence','no actual wireless image quality'])
 save(O/'summary.json',summary)
 report=f'''# LE2 有限包试验结果\n\nLE0／LE1 和旧 raw R1 NO_GO 保留。新版比例分配已为每组计入13 bit长度和16 bit CRC；完整源码终止长度来自原LE1。没有新训练、VAR遍历、干净恢复或无线图像帧。\n\n实际完成 {m.calls:,}／21,504 包，初测 {len(m.initial)} 单元，{len(m.topup)} 单元追加至512包。15个956符号单码字MCS和所有提名两组在7／10 dB均实测。提名按每源实际回退质量和明确标记的单码字MCS迁移筛选启发式，未使用容量出现次数。短组最终评分仅使用其真实包结果，未测布局保持未知。\n\n状态：**{status}**。最佳实测两组的代理J={best['J']:.9f}；单组稳健对照J={robust['J']:.9f}，差值={best['J']-robust['J']:+.9f}。相对back3差值={best['J']-back['J']:+.9f}。实测两组策略共{len(grank)}个（包括目标/回退别名），其余{len(pool)-len(grank)}个未测，不能据此声称完整网格全局最优或全局NO_GO。\n\n每源的实际基础前缀、增强前缀、空增强和null均保留；17个灰图对精确复用，33个固定灰图对补评。压缩确实为增强组留下容量，但LE1的干净前缀质量和容量不是无线性能。此处质量是正确前缀包概率代理，依赖iid码块近似，未计错误头、未检出算术流损坏和真实接收条件依赖。两侧95% Wilson单元区间及共享头变量传播的敏感性范围，不是同时置信区间，也不是来源级无线图像配对区间。零观察错误不代表真实BLER为零。\n\nLE3_candidates.json保留不超过10个去重候选及原熵编码nominal/back3锚点；只供审查，LE3未授权、未执行。本轮在此停止，不扩功率、不训练。科学备份按用户要求暂缓。\n'''
 with (O/'REPORT_zh.md').open('x',encoding='utf8',newline='\n') as f:f.write(report)
 save(out/'completion.json',dict(status='COMPLETE_LE2_STOP',summary=pin(O/'summary.json'),packet_trials=m.calls,output_files=[pin(q) for q in O.glob('*') if q.is_file()],registration=pin(O/'registration.json')))
 print('LE2_FINITE_COMPLETE_STOP',m.calls,status,flush=True)
if __name__=='__main__':
 try:run(sys.argv[1])
 except BaseException:
  if (T/'packets').exists() and not (T/'packets/failure.json').exists():save(T/'packets/failure.json',dict(error=traceback.format_exc(),automatic_retry=False,successful_calls_preserved=True))
  raise
