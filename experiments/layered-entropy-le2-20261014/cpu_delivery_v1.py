"""Recover only CSV/report delivery after closed scientific packet measurements.

No encode/decode/channel/model import is allowed. Original failed owner and
partial CSV remain immutable. All probabilities use saved hard RX evidence.
"""
from common import *
import planning as p,numpy as np,sys,itertools,subprocess,signal,traceback
D=O/'delivery_v1';C=T/'cpu_delivery_control_v1';AUTH=O/'cpu_delivery_authorization.json'
def csv_union(path,rows):
 names=list(dict.fromkeys(k for row in rows for k in row))
 with Path(path).open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
def audit_closed():
 verify_registration();w=read(T/'packets_control_v1/actual_wait.json');assert w['actual_wait'] and type(w['returncode'])is int and w['returncode']==1 and gone(w['process']) and gone(w['parent_process']) and not any(w[k] for k in ('timed_out','terminated','killed'))
 failure=read(T/'packets/failure.json');assert 'ValueError: dict contains fields not in fieldnames:' in failure['error'] and "csvsave(O/'measured_policy_results.csv',rows)" in failure['error']
 db=sqlite3.connect('file:'+str(T/'packets/calls.sqlite')+'?mode=ro',uri=True);counts=db.execute('SELECT kind,status,COUNT(*) FROM calls GROUP BY kind,status').fetchall();db.close();assert counts==[('packet','COMPLETE',18688)]
 checked(read(O/'h100_closed_preparation_processes.json')['records'][0]['actual_wait'])
 return w,failure,counts
def authorize():
 assert not AUTH.exists() and not D.exists() and not C.exists();w,failure,counts=audit_closed()
 # Synthetic regression for the ONLY formatting defect: mixed field sets.
 import tempfile
 with tempfile.TemporaryDirectory(prefix='LE2_METADATA_CSV_TEST_') as td:
  q=Path(td)/'union.csv';csv_union(q,[dict(a=1,b=2),dict(a=3,c=4)]);rows=list(csv.DictReader(q.open()));assert len(rows)==2 and rows[1]['c']=='4' and rows[0]['b']=='2' and rows[0]['c']==''
 save(AUTH,dict(status='CPU_DELIVERY_ONLY_ACTUAL_PACKET_ALL_COMPLETE',reviewer='/root',source=pin(E/'cpu_delivery_v1.py'),old_registration=pin(O/'registration.json'),original_failed_owner_wait=pin(T/'packets_control_v1/actual_wait.json'),original_failed_source=pin(E/'runner.py'),original_failure=pin(T/'packets/failure.json'),partial_CSV=pin(O/'measured_policy_results.csv'),ledger=pin(T/'packets/calls.sqlite'),preparation_host_closure=pin(O/'h100_closed_preparation_processes.json'),actual_completed_packets=18688,calls_to_REPEAT=0,packet_budget_unchanged=CAP,CSV_union_synthetic_regression_PASS=True,changes='Independent postprocess only; union CSV fieldnames; rank distinct actual fallback strategies; prospective paid catalogue; no source/model/PHY mutation',no_encode_decode_channel_or_model_imports=True,automatic_retry=False))
 print(canonical(pin(AUTH)))
def run(expected):
 assert sha(AUTH)==expected;auth=read(AUTH);checked(auth['source']);audit_closed();D.mkdir();sourcepins=read(O/'registration.json')['bindings'];
 def guard(event,args):
  if event=='import' and args and str(args[0]).split('.')[0] in ('torch','sionna','tensorflow'):raise RuntimeError('CPU delivery forbids model/PHY imports')
 sys.addaudithook(guard);assert not any(n.split('.')[0] in ('torch','sionna','tensorflow') for n in sys.modules)
 save(T/'cpu_delivery_launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),argv=sys.argv,authorization=pin(AUTH),zero_scientific_calls=True))
 layouts={r['geometry_id']:r for r in read(T/'layouts.json')['layouts'] if r['layout_status'].startswith('ACTUAL')};table={};audit=[];total=0;topups=0
 initial_cells=read(O/'stage_A_plan.json')['cells']+read(O/'stage_B_cells.json')['cells'];expected_cells={(r['gid'],r['snr']) for r in initial_cells};refined=set(read(O/'stage_C_topup_plan.json')['geometries']);initials={}
 for r in initial_cells:initials.setdefault((r['gid'],r['snr']),r)
 assert len(expected_cells)==92 and len(refined)==9
 for gid,snr in sorted(expected_cells):
  folder=T/'packets/cells'/f'{gid}_{snr}';snapshot=max(folder.glob('counts_*.json'),key=lambda q:q.name);v=read(snapshot);n=v['counts']['n'];assert n==(512 if gid in refined else 128);topups+=n==512
  sums=dict(correct=0,rejected=0,undetected=0,exact_hard=0);rxpins=[]
  for start in range(0,n,32):
   fixed=read(folder/f'rx_{start:04d}.json');truth=read(folder/f'truth_audit_{start:04d}.json')
   if gid=='header':
    accepted=np.asarray([r['header_ok'] for r in fixed['received']],bool);exact=np.asarray([r['profile_id']==pid for r,pid in zip(fixed['received'],truth['transmitted_pids'])]);checked(truth['actual_rx']);rxpins.append(pin(folder/f'rx_{start:04d}.json'))
   else:
    assert fixed['receiver_truth_supplied'] is False and truth['truth_read_only_after_RX_fixed'] is True
    rx=checked(fixed['rx']);checked(truth['actual_rx']);tx=checked(truth['tx_audit']);assert tx.stat().st_mtime_ns>=rx.stat().st_mtime_ns
    with np.load(rx,allow_pickle=False) as z:hard=z['hard'];accepted=z['accepted'];assert hard.shape==(32,layouts[gid]['k']) and hard.dtype==np.uint8 and accepted.shape==(32,) and accepted.dtype==np.bool_
    with np.load(tx,allow_pickle=False) as z:info=z['information'];assert info.shape==hard.shape;exact=np.all(hard==info,axis=1)
    rxpins.append(pin(rx))
   correct=accepted&exact;assert correct.tolist()==truth['correct'];sums['correct']+=int(correct.sum());sums['rejected']+=int((~accepted).sum());sums['undetected']+=int((accepted&~correct).sum());sums['exact_hard']+=int(exact.sum())
  assert all(v['counts'][k]==value for k,value in sums.items());assert sum(sums[k] for k in ('correct','rejected','undetected'))==n
  table[gid,snr]=dict(n=n,n_correct=sums['correct'],n_rejected=sums['rejected'],n_undetected=sums['undetected'],p_correct=sums['correct']/n,ci=wilson(sums['correct'],n));total+=n;audit.append(dict(gid=gid,snr=snr,latest_counts=pin(snapshot),actual_hard_RX=rxpins))
 assert total==18688 and topups==18
 data=p.load();d,Q,Q0=data;q0=float(Q0.mean());single=read(T/'single_policy_grid.json');pool=read(T/'group_policy_grid.json')
 def rank(rows):
  out=[]
  for row in rows:
   if all((g,s) in table for g in (row['B'],row['E']) if g for s in SNRS):
    score=p.proxy(row,table,q0);lo,hi=p.interval(row,table,q0);out.append(dict(row,**score,J_sensitivity_low=lo,J_sensitivity_high=hi))
  return sorted(out,key=lambda r:(-r['J'],r['policy_id']))
 srank=rank(single);grank=rank(pool);robust=srank[0];back=max(srank,key=lambda r:(r['Q7'],r['policy_id']));nom=next(r for r in srank if (r['target_m'],r['target_K'],r['q_B'],r['rate_B'])==(8,126,4,'2/3'))
 result=[]
 for row in srank+grank:
  lo,hi=p.delta_bounds(row,robust,table);bl,bh=p.delta_bounds(row,back,table)
  result.append(dict(row,delta_J_Srobust=row['J']-robust['J'],delta_J_Sback3=row['J']-back['J'],delta_robust_sensitivity_low=lo,delta_robust_sensitivity_high=hi,delta_back_sensitivity_low=bl,delta_back_sensitivity_high=bh,quality_kind='CORRECT_PREFIX_PACKET_PROXY_NOT_WIRELESS_IMAGES',U_domain=True))
 csv_union(D/'measured_policy_results.csv',result)
 cells=[dict(gid=g,snr_db=s,kind='header' if g=='header' else 'body',N=68 if g=='header' else layouts[g]['N'],q=2 if g=='header' else layouts[g]['q'],rate='header' if g=='header' else layouts[g]['nominal_rate'],k=28 if g=='header' else layouts[g]['k'],n=136 if g=='header' else layouts[g]['n'],**{k:x for k,x in v.items() if k!='ci'},p_correct_Wilson_low=v['ci'][0],p_correct_Wilson_high=v['ci'][1],zero_errors_not_zero_BLER=True) for (g,s),v in table.items()];csv_union(D/'packet_cells.csv',cells)
 # Distinct Pilot50 source-fallback fingerprints avoid three aliases of the
 # exact same tested strategy filling the exploratory LE3 slots.
 chosen=[];states_by_id={};fingerprints={}
 def states(row):return p.source_states(d,Q,Q0,row['target_j'],(row['target_m'],row['target_K']),layouts[row['B']]['k']-29,layouts[row['E']]['k']-29 if row['E'] else None)
 def add(rows,role,limit):
  added=0
  for row in rows:
   st=states(row);fp=digest(dict(B=row['B'],E=row['E'],fallback=[{k:s[k] for k in ('status','actual_j','actual_m','actual_K','L_B','L_E')} for s in st]))
   if fp in fingerprints:
    old=fingerprints[fp]
    if role not in old['roles']:old['roles'].append(role)
    if limit==1:return
    continue
   record=dict(row,roles=[role],actual_fallback_fingerprint=fp,LE3_execution_authorized=False);chosen.append(record);fingerprints[fp]=record;states_by_id[row['policy_id']]=st;added+=1
   if added>=limit:return
 add(grank,'U',3);add(srank,'robust_single',3);add([r for r in grank if r['family']=='EEP'],'EEP',2);add([nom],'old_entropy_nominal',1);add([back],'entropy_backoff3',1);assert len(chosen)<=10
 source_ids={s['pilot_slot']:s['source_id'] for s in read(O1/'assets/pilot50.json')['records']}
 source_rows=[dict(policy_id=row['policy_id'],source_id=source_ids[s['pilot_slot']],**s) for row in chosen for s in states_by_id[row['policy_id']]];csv_union(D/'LE3_candidate_source_fallbacks.csv',source_rows)
 # Prospective immutable directory; these IDs ARE NOT radio-tested here.
 profiles={'0':dict(profile_id=0,null_profile=True,header_symbols=68,body_symbols=956,N=1024,nominal_Es=2,radio_validation='NOT_RUN_LE3')};pid=1
 for row in chosen:
  endpoints=[(None,*end) for end in ENDS] if row['E'] is None else [(j,*end) for j in (4,5,6,7) for end in ((j,0),*(x for x in ENDS if OFFSETS[x[0]]+x[1]>OFFSETS[j]))]
  for j,m,K in endpoints:
   profiles[str(pid)]=dict(profile_id=pid,policy_id=row['policy_id'],family='EC_VAR_PARTIAL' if j is None else 'EC_VAR_TWO_INDEPENDENT_TERMINATED',actual_j=j,m=m,K=K,explicit_empty_enhancement=(j is not None and (m,K)==(j,0)),B_layout=layouts[row['B']],E_layout=layouts[row['E']] if row['E'] else None,length13_CRC16_per_group=True,header_symbols=68,body_symbols=956,N=1024,nominal_Es=2,radio_validation='PROSPECTIVE_NOT_TRANSMITTED_LE3_NOT_RUN');pid+=1
 assert pid<=4096;save(D/'LE3_prospective_paid_profiles.json',dict(profiles=profiles,paid_fields='12bit ID binds actual j/m/K, complete layouts and empty flag',LE3_radio_execution=False))
 save(D/'LE3_candidates.json',dict(status='EXPLORATORY_CANDIDATES_ONLY_STOP_BEFORE_LE3',records=chosen,LE3_authorized=False,candidate_count=len(chosen),deduplication='exact physical layouts plus all50 actual fallback states',public_directory=pin(D/'LE3_prospective_paid_profiles.json'),original_nominal_anchor=read(O/'registration.json')['entropy_nominal_binding']))
 positive=[r for r in grank if r['J']>robust['J'] and r['J']>back['J']];best=next(r for r in result if r['policy_id']==grank[0]['policy_id']);status='BLOCKED_TABLE_UNCERTAINTY' if positive else 'NO_PROMISING_MEASURED_CANDIDATE'
 summary=dict(status=status,packet_measurements='ALL18688_COMPLETE',delivery='CPU_POSTPROCESS_COMPLETE_WITH_ORIGINAL_CSV_FAILURE_DISCLOSED',packet_trials=total,initial_cells=92,topup_cells=18,packet_cap=CAP,unused_packets=CAP-total,unused_reason='Deduplicated complete-policy bundles; frozen topup ranking selected9 geometries, no extra post-hoc packet hypotheses',nominated_complete_physical_strategies=16,measured_single_policies=len(srank),measured_group_policies=len(grank),unmeasured_group_policies=len(pool)-len(grank),measured_positive_group_point_estimates=len(positive),best_group=best,Srobust=robust,Sback3=back,Snom=nom,LE3_candidate_count=len(chosen),LE3_executed=False,LE0_LE1_bytes_preserved=True,original_packet_owner_returncode=1,original_packet_owner_failure_stage='CSV serialization after ALL science',postprocess_additional_packet_model_or_quality_calls=0,source_TX_VAR_calls=0,clean_render_calls=0,wireless_image_frames=0,training_updates=0,power_expansion=False,backup_deferred=True,old_raw_R1_NO_GO_preserved=True,gray_fixed_pairs_reused=17,gray_fixed_pairs_new=33,initial_pixel_metadata_failure_preserved=True,additional_DINO_model_reload_disclosed=1,undetected_packets=sum(r['n_undetected'] for r in cells),interval_kind='Cellwise two-sided95% Wilson, shared-header sensitivity rectangles; not simultaneous/paired image confidence intervals',no_GO_or_global_negative_claim=True)
 save(D/'summary.json',summary);save(D/'closure_audit.json',dict(actual_packets=total,unresolved=0,cells=audit,ledger=pin(T/'packets/calls.sqlite'),original_packet_failure=pin(T/'packets/failure.json'),original_packet_wait=pin(T/'packets_control_v1/actual_wait.json'),original_partial_CSV=pin(O/'measured_policy_results.csv'),source_bindings=sourcepins,source_math_models_unchanged=True,zero_new_scientific_calls=True))
 report=f'''# LE2 结果与 LE3 候选（在此停止）\n\nLE0／LE1 和旧 raw R1 NO_GO 保留。比例分配改为两组各计13 bit长度＋16 bit CRC，使用原实码长及各自finish。每源实际回退后的基础／最终前缀、空增强和null均参与预筛，没有容量出现次数提名。头部、全部15个956符号单码字MCS和16套两组提名在7／10 dB均实测。单码字MCS可靠性的迁移值只用于明确标记的提名启发式，最终短组评分仅用自己的实测数据；未测布局不填成功率。\n\n完成 **{total:,}／21,504 包**：92单元各128包，其中18单元增至512包；2,816预算未使用。16套完整提名覆盖所有必要组×SNR，另有物理单元共享产生的目标别名可完整评分。共有{len(srank)}个单组和{len(grank)}个两组目标策略可评分，{len(pool)-len(grank)}个两组布局组合仍未覆盖；不能宣称完整网格全局最优或全局NO_GO。\n\n**{status}**。最佳已测两组代理：7 dB={best['Q7']:.9f}，10 dB={best['Q10']:.9f}，J={best['J']:.9f}。单组Srobust：7={robust['Q7']:.9f}，10={robust['Q10']:.9f}，J={robust['J']:.9f}。相对Srobust的ΔJ={best['delta_J_Srobust']:+.9f}，共享头变量敏感性范围[{best['delta_robust_sensitivity_low']:+.9f}, {best['delta_robust_sensitivity_high']:+.9f}]；相对Sback3 ΔJ={best['delta_J_Sback3']:+.9f}。\n\n质量是**正确前缀包概率代理**，没有新的无线图像。随机码块／长度混合、错误头／错误profile、未检出损坏和源及条件依赖仍需以后真实RX验证。观察未检出包={summary['undetected_packets']}。零观察错误不是零BLER。单元Wilson区间传播只是敏感性范围，不是同时置信区间，也不是来源级无线图像配对区间。\n\nLE3_candidates.json交付{len(chosen)}个去重候选，附全部50源实际回退状态及不超过4096个12bit付费ID的**待执行**目录；候选角色可重合。保留原熵编码nominal锚点和本pilot的back3，不拿raw策略代替。LE3、功率扩展、新训练均未运行，不自动开启。\n\n故障如实保留：固定灰图准备首次因数组哈希与原始像素字节哈希约定不同而停止，零图像评分后由独立元数据修订补齐33对（另17对完全复用），必要的DINO模型重载一次披露；包owner在全部18,688包完成后因混合CSV字段导出exit1，原部分CSV／失败／实际退出不改，独立CPU交付只重建表格，没有重跑任何包或评分。科学备份按用户要求暂缓。\n'''
 with (D/'REPORT_zh.md').open('x',encoding='utf8',newline='\n') as f:f.write(report)
 for v in sourcepins:checked(v)
 assert not any(n.split('.')[0] in ('torch','sionna','tensorflow') for n in sys.modules)
 save(D/'output_manifest.json',dict(files=[pin(q) for q in sorted(D.iterdir()) if q.is_file()],CPU_only=True));save(T/'cpu_delivery_completion.json',dict(status='COMPLETE_LE2_CPU_DELIVERY_STOP',manifest=pin(D/'output_manifest.json'),summary=pin(D/'summary.json'),additional_scientific_calls=0))
 print(canonical(dict(status=status,packet_trials=total,best_J=best['J'],robust_J=robust['J'],delta=best['delta_J_Srobust'],LE3_candidates=len(chosen))))
def control(expected):
 assert sha(AUTH)==expected;C.mkdir();argv=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'cpu_delivery_v1.py'),'run',expected];start=time.monotonic();parent=process(os.getpid());timed=terminated=killed=False
 with (C/'stdout.txt').open('xb') as out,(C/'stderr.txt').open('xb') as err:
  child=subprocess.Popen(argv,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=err,start_new_session=True);identity=process(child.pid);assert identity;save(C/'launch.json',dict(argv=argv,process=identity,parent_process=parent,authorization=pin(AUTH)))
  try:rc=child.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   timed=terminated=True;os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=20)
 save(C/'actual_wait.json',dict(actual_wait=True,returncode=rc,elapsed_seconds=time.monotonic()-start,process=identity,parent_process=parent,argv=argv,timed_out=timed,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(C/'stdout.txt'),stderr=pin(C/'stderr.txt')));return rc
if __name__=='__main__':
 if sys.argv[1]=='authorize':authorize()
 elif sys.argv[1]=='control':raise SystemExit(control(sys.argv[2]))
 else:
  try:run(sys.argv[2])
  except BaseException:
   save(T/'cpu_delivery_failure.json',dict(error=traceback.format_exc(),automatic_retry=False));raise
