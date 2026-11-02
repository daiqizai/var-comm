"""Read-only real-stream audit plus exact original integer-encoder CPU checks."""
from study_common import *
import copy,ast,traceback
from grid_codec import endpoints

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']]
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));beg=time.monotonic()
 ledger=Ledger(out/'calls.sqlite',dict(CPU_codec_cases=4,stream_archive_audit=1500))
 try:
  sys.path.insert(0,str(R/'experiments/wcl-evidence-closure-20261009/scripts'))
  from t1_entropy_core import load_integer_codec,CODE_SHA
  entropy,Encoder,Decoder=load_integer_codec(R);rng=np.random.default_rng(2026110301);tests=[];counterexample=None
  for case in range(4):
   def check():
    nonlocal counterexample
    count=128;lp=rng.normal(0,3,(count,4096))
    if case==0:lp[:]=0
    if case==1:lp[:,0]=35
    if case==2:lp[:,::2]+=12
    cdf=entropy.probability_cdf(lp);tokens=np.array([0]*count if case==1 else[rng.choice(4096,p=np.diff(a)/entropy.TOTAL)for a in cdf],np.int64)
    encoder=Encoder();previous=encoder.finish();ties=0;pending_seen=False;max_pending=0
    for i in range(count):
     encoder.encode(tokens[i:i+1],cdf[i:i+1]);before=copy.deepcopy(encoder.__dict__);bits=encoder.finish()
     assert len(bits)==len(encoder.bits)+encoder.pending+2 and encoder.__dict__==before
     assert np.array_equal(bits,encoder.finish())and encoder.__dict__==before
     assert len(bits)>=len(previous);ties+=int(len(bits)==len(previous));pending_seen|=encoder.pending>0;max_pending=max(max_pending,encoder.pending)
     decoded=Decoder(bits).decode(cdf[:i+1]);assert np.array_equal(decoded,tokens[:i+1]);canonical=Encoder();canonical.encode(decoded,cdf[:i+1]);assert np.array_equal(canonical.finish(),bits)
     if counterexample is None and not np.array_equal(previous,bits[:len(previous)]):counterexample=dict(case=case,previous_tokens=i,next_tokens=i+1,previous_bits=previous.tolist(),next_head=bits[:len(previous)].tolist(),reason='Independently terminated stream is not generally a truncation of the longer terminated stream')
     previous=bits
    return dict(case=case,prefixes=count,E_plus_P_plus_2=True,nonmutating_finish=True,nondecreasing=True,canonical_roundtrip=True,tied_steps=ties,pending_seen=pending_seen,max_pending=max_pending)
   tests.append(ledger.call('CPU_codec_cases',str(case),check))
  assert counterexample is not None and any(t['pending_seen']for t in tests)and any(t['tied_steps']for t in tests)
  streams=read(L/'source_plan/reuse.json');closed('cal_token_source800')
  for v in read(L/'cal_token_source800/outputs.json'):assert v['source_id']not in streams;streams[v['source_id']]=v
  assert len(streams)==1000
  formal=read(OL/'final_D_source/outputs.json');assert len(formal)==500 and not(set(streams)&{v['source_id']for v in formal})
  ordered=list(streams.values())+formal;ends=endpoints('token');names=['m%d_K%d'%e for e in ends];rows=[]
  for v in ordered:
   assert time.monotonic()-beg<r['seconds']and not(L/'STOP').exists()
   def audit():
    with np.load(checked(v['streams']),allow_pickle=False)as z:
     assert set(z.files)==set(names);lengths=[]
     for name in names:
      b=z[name];assert len(b)==v['lengths'][name]and asha(b)==v['array_sha256'][name];lengths.append(len(b))
    dif=np.diff(lengths);assert(dif>=0).all()
    return dict(source_id=v['source_id'],endpoints=651,adjacent_pairs=650,strict_increases=int((dif>0).sum()),ties=int((dif==0).sum()),decreases=int((dif<0).sum()),stream_sha256=v['streams']['sha256'])
   rows.append(ledger.call('stream_archive_audit',v['source_id'],audit))
   if len(rows)%100==0:print('STREAM_AUDIT',len(rows),1500,flush=True)
  grids={g:[list(e)for e in endpoints(g)]for g in ['quarter','eighth','token']};assert[len(grids[g])for g in grids]==[25,49,651]
  d=D/'encoder_audit';d.mkdir(exist_ok=False);csvout(d/'real_stream_monotonicity.csv',rows);save(d/'integer_codec_checks.json',dict(original_source_SHA=CODE_SHA,tests=tests,counterexample=counterexample,synthetic_CPU_tests_not_quality_evidence=True))
  timing=R/'experiments/nestcom-perceptual-followup-20261030/timing_end_to_end_identity_v5.py';tx=next(n for n in ast.parse(timing.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='tx');txtext=ast.get_source_segment(timing.read_text(),tx)
  assert "fits=[e for e in ends if len(streams[e]['bits'])<=min(cap,8191)]"in txtext
  save(d/'implementation_identity.json',dict(timing_source=pin(timing),source_encoder=pin(R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_source_codec.py'),endpoint_grammar=pin(R/'experiments/nestcom-opt-20261030/grid_codec.py'),full_scan=True,legacy_length_checks_field='First feasible rank; actual list comprehension evaluates all candidate lengths',grid_rounding='K=floor(L*j/d), j=1..d-1; include0, deduplicate; final(10,0)',grids=grids,new_training=0,new_model_forward=0,new_PHY=0,new_timing=0,search_optimization='not-run-optional'))
  proof='''# 终止长度与扫描成本核查

原实现的 finish() 不改变编码状态。令 E 为已输出的位数、P 为待决 underflow 位数，独立终止输出为已有 E 位、一个终止位和 P+1 个补位，故 b=E+P+2 严格成立。

编码一个 token 的区间更新本身不改变 E+P。每轮 E1/E2 renormalization 输出原 P+1 位并将 P 清零，因此 E+P 增加1；E3 使 P 增加1。因此共同增量编码路径上的 b 随前缀增长非递减，允许相等。该结论不意味着终止后的短码流是长终止码流的切片；CPU检查给出实际反例。

已检查原SHA绑定编码器的512个合成前缀，包括非变异finish、完整解码和规范重编码；另逐数组校验cal1000/common500共1500个既有源码archive，每源651个端点、650个相邻长度关系。所有原生数组身份与长度核对，未进行模型、图像生成或无线重跑。

当前实测TX先共享一次增量源码遍历，为每个合法候选独立终止，再用完整列表扫描比较码长。timing脚本中的历史 length_checks 字段记录第一可行候选的排序位置，不是该列表实际执行的比较总数。本文应报告完整候选扫描成本；候选个数不等于先验前向次数。

可利用单调性优化长度检索，或在容量已越界时终止后续源遍历，但这不是当前被计时的实现。本轮不修改搜索，也不生成未经重计时的加速结论。

伪代码（当前路径）：

1. 输入固定MCS、公开目录、合法端点集合和校准目标上限。
2. 取上限以下的合法端点；Quarter/Eighth使用 floor(L*j/d)，去重并包含完整尺度边界。
3. 一次共享增量算术编码遍历，在每个候选复制非变异 finish() 输出，得到独立终止 bits[e]。
4. 按描述token数从长到短排列候选，对全部候选检查 len(bits[e])+29<=k；Ada另付10bit类别。
5. 选第一个可行端点，发送其完整独立终止码流及受保护profile；本轮统一协议无可行项时发送付费NULL。
6. 接收端从收到的profile恢复实际端点及固定MCS，独立解码；失败/NULL按灰图计入统计。

历史N1024旧版本无可行项抛错，而N512/本轮统一协议含付费NULL；不得将其混写成相同历史协议。
'''
  (d/'REPORT_zh.md').write_text(proof,encoding='utf8')
  save(out/'completion.json',dict(status='COMPLETE',sources=1500,actual_streams=1500*651,adjacent_relations=1500*650,all_nondecreasing=True,ledger=ledger.summary(),seconds=time.monotonic()-beg,report=pin(d/'REPORT_zh.md')))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
