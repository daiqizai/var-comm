"""Read-only configuration, source-decoder and existing-timing evidence audit.
No model imports, packet simulation, metric evaluation or bootstrap generation.
Run in the canonical checkout; output directory must not already exist.
"""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math,platform,re,statistics,subprocess,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
BASE='93205a2e007ac587e28246cd399a531347a4f3f7'
def main(out):
 out.mkdir(parents=True,exist_ok=False);pins={};cache={}
 def read(p,expected=None):
  p=Path(p);s=str(p)
  if s in cache:
   if expected:assert pins[s]['sha256']==expected
   return cache[s]
  b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
  if expected:assert h==expected,(s,h,expected)
  pins[s]=dict(path=s,bytes=len(b),sha256=h);return b
 def js(p,expected=None):
  s=str(p)
  if s not in cache:cache[s]=json.loads(read(p,expected))
  elif expected:assert pins[s]['sha256']==expected
  return cache[s]
 def pin(d):return js(d['path'],d['sha256'])
 def save(name,data):
  (out/name).write_text(json.dumps(data,indent=2)+'\n')
 def csvout(name,data):
  with (out/name).open('x',newline='')as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip();assert head==BASE
 branch=subprocess.check_output(['git','-C',str(R),'branch','--show-current'],text=True).strip();assert branch=='main'
 status=subprocess.check_output(['git','-C',str(R),'status','--porcelain','--untracked-files=no'],text=True)
 stop=B/'nested_residual_rd_20261028_v1/STOP';sb=read(stop);assert hashlib.sha256(sb).hexdigest()=='452e3eb7391fe6bd55b430bea01c1a443deb40be9571c8141d52bebd67c53f6f'
 environment=dict(base_commit=head,branch=branch,tracked_status=status,python=platform.python_version(),platform=platform.platform(),host=platform.node(),unix=time.time(),paused_guard=pins[str(stop)],new_PHY=0,new_model_calls=0,new_metrics=0,new_bootstrap=0)
 # Read the actual source-decoder evidence, including pinned new-source repair outputs.
 closure=js(R/'results/n1024_consistency_20261019_v1/repair_v1/delivery/closure_audit.json');rows=pin(closure['metric_rows'])
 source=[];groups=collections.defaultdict(collections.Counter)
 for r in rows:
  d=r['reconstruction']['evidence'];kind='pinned'if 'path'in d else'inline'
  e=pin(d)['evidence']if kind=='pinned'else d
  assert e['source_decode_called']and e['render_called']
  assert not e.get('receiver_truth_used',False)and not e.get('comparison_truth_used',False)and not e.get('TX_CDF_used',False)and not e.get('transmitted_CDF_used',False)
  assert e['source_status']=='ARITHMETIC_SOURCE_DECODED',e['source_status']
  groups[r['strategy_family'],r['snr_db']][kind]+=1
  source.append(dict(N=1024,method=r['strategy_family'],snr_db=r['snr_db'],source_index=r['source_index'],source_id=r['source_id'],noise_seed=r['noise_seed'],source_status=e['source_status'],evidence_kind=kind,evidence_path=d.get('path',closure['metric_rows']['path']),source_input_key=r['reconstruction'].get('source_input_key','')))
 assert len(source)==9000
 csvout('n1024_source_status.csv',source)
 save('source_decoder_audit.json',dict(logical_rows=9000,source_decode_success=9000,source_reject=0,truth_used=False,groups=[dict(method=k[0],snr_db=k[1],logical=sum(v.values()),inline=v['inline'],pinned=v['pinned'])for k,v in sorted(groups.items())]))
 # Policies are the frozen target, not the fitted endpoint.
 pol512=js(R/'results/nestcom_n512_20261028_v1/frozen_policies.json');pol1024=js(R/'results/t1b_m10_calibration_20261016_v1/delivery/frozen_policies.json','81dfc416ccd82bad538885e8f71f394a6f7f94bb21e403b510721d78582b3e02');ada=js(R/'results/adatoken_response_20261026_v1/c_predclass_cal1000_dino/frozen_policies.json','68d12e9b19006e8509b3ae5a62132e73044b88ad29ca041f96b436144923ae8c')
 pol=[]
 for snr,methods in pol512['policies'].items():
  for method,p in methods.items():pol.append(dict(N=512,method=method,snr_db=int(snr),target=('p'+str(p['target'][0]))if method=='ADA'else'(%s,%s)'%tuple(p['target']),q=p['q'],rate=p['rate'],candidate_id=p['candidate_id']))
 for snr,methods in pol1024['policies'].items():
  for method,p in methods.items():
   c=p['winner'];m=int(re.search(r'm(\d+)',c).group(1));kk=re.search(r'_K(\d+)',c);q=int(re.search(r'_q(\d+)',c).group(1));rate=re.search(r'_r(\d+-\d+)',c).group(1).replace('-','/')
   pol.append(dict(N=1024,method='NEST'if method=='PARTIAL'else method,snr_db=int(snr),target='(%s,%s)'%(m,int(kk.group(1))if kk else 0),q=q,rate=rate,candidate_id=c))
 for snr,c in ada['policies'].items():pol.append(dict(N=1024,method='ADA',snr_db=int(snr),target=re.search(r'p\d+',c).group(0),q=int(re.search(r'_q(\d+)',c).group(1)),rate=re.search(r'_r(\d+-\d+)',c).group(1).replace('-','/'),candidate_id=c))
 assert len(pol)==18
 for method in ['NEST','WHOLE']:assert pol512['policies']['10'][method]['target']==[7,0]and pol512['policies']['10'][method]['q']==4 and pol512['policies']['10'][method]['rate']=='2/3'
 csvout('frozen_policies_compact.csv',sorted(pol,key=lambda r:(r['N'],r['snr_db'],r['method'])))
 # Existing executed software timing logs. Only measured calls, not three warmups.
 timing=js(B/'five_method_timing_20261020_v1/NEST/tx/completion.json');tr=js(R/'results/five_method_timing_20261020_v1/registration.json',timing['registration_sha256'])
 calls=[]
 for r in timing['results']:
  if r['phase']!='measure':continue
  a=r['evidence']['attempts'];assert a and a[-1]['fits']and not any(x['fits']for x in a[:-1])
  calls.append(dict(snr_db=r['snr_db'],source_index=r['source_index'],source_id=r['source_id'],repeat=r['repeat'],key=r['key'],attempt_count=len(a),selected_m=a[-1]['m'],selected_K=a[-1]['K'],attempted_endpoints=';'.join('(%s,%s)'%(x['m'],x['K'])for x in a),source_tx_seconds=r['seconds']['source_tx'],source_and_actual_fallback_seconds=r['seconds']['source_and_actual_fallback'],prior_scale_calls=r['model_calls']['prior_scale']))
 assert len(calls)==144
 def pct95(xs):
  s=sorted(xs);p=.95*(len(s)-1);i=int(p);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(p-i)
 summ=[]
 for snr in [4,10,19]:
  rs=[r for r in calls if r['snr_db']==snr];assert len(rs)==48 and len({r['source_id']for r in rs})==16
  for sid in {r['source_id']for r in rs}:assert len({r['attempted_endpoints']for r in rs if r['source_id']==sid})==1
  v=[r['attempt_count']for r in rs];summ.append(dict(N=1024,method='NEST',snr_db=snr,sources=16,measured_repeats=3,measured_calls=48,warmups_excluded=48,attempt_mean=statistics.fmean(v),attempt_median=statistics.median(v),attempt_p95=pct95(v),attempt_min=min(v),attempt_max=max(v),p95_definition='linear interpolation, position 0.95*(n-1)',interpretation='actual sequential length-fit attempts; not extra prior forward passes'))
 csvout('timing_attempt_calls.csv',calls);csvout('timing_attempt_summary.csv',summ)
 actual=js(B/'nestcom_n512_20261028_v1/formal_rx_nest_00/actual_native_identity.json');timingnative=js(B/'five_method_timing_20261020_v1/NEST/tx/actual_native_identity.json');routes=js(R/'results/layered_entropy_feasibility_20261014_v1/assets/model_routes.json','33ba8b5f9ba0511bb0832e054f1c1b6238bc2a203927789c38bb4c96aeb5a3e5')
 sources={
 'src/var_comm/next_scale_prior.py':r'build_vae_var|depth=16|flash_if_available|cond_drop_rate',
 'src/var_comm/entropy.py':r'TOTAL|FULL|counts =|np.argmax|def probability_cdf',
 'src/var_comm/whole_entropy.py':r'class Integer|FULL|HIGH|TOTAL',
 'experiments/content-real-64qam-20261006/h_source_driver.py':r'IndependentProvider|cdf_function|self.cdf|class1000',
 'experiments/wcl-evidence-closure-20261009/scripts/t1_codec_runtime.py':r'provider_factory|load_integer_codec|precision|cdf|MODEL_STATES',
 'experiments/wcl-evidence-closure-20261009/scripts/t1_entropy_core.py':r'load_integer_codec|IntegerEncoder|IntegerDecoder|entropy.py|whole_entropy.py',
 'experiments/entropy-partial-mismatch-20261010/scripts/ep_source_codec.py':r'cdf_total|precision|partial_cdf|cdf|PartialSourceCodec',
 'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py':r'SCRAMBLING|scramble_mask|scrambled=|logits=logits|no_frame_normalization|after_RV0|mask=',
 'experiments/nestcom-n512-20261028/n512_phy.py':r'scram|mask|logits|modulate|no_frame',
 'experiments/nestcom-n512-20261028/rx_worker_v2.py':r'kn_native|receiver_codec|load|dtype|canonical',
 'experiments/kodak-current-20261026/nest_v1/kn_native.py':r'tb_native|ReceiverCodec|SourceCodec10|load|receiver_codec',
 'experiments/t1b-m10-calibration-20261016/tb_native.py':r'deterministic|tf32|float32|build_model|MODEL|decoder',
 'experiments/var-latent-enhancement-20260917/src/latent_enhancement/training.py':r'branch|interpol|Fb_TX|Fq',
 'experiments/var-latent-enhancement-20260917/src/latent_enhancement/cache.py':r'complete_latent|Fb_TX|base_prefix',
 'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py':r'def complete_latent|patch_nums|prefix|argmax|latent|f_hat',
 'experiments/five-method-timing-20261020/worker.py':r'attempts|fits|source_and_actual_fallback|finish|def nest_tx',
 }
 code=[]
 for rel,pattern in sources.items():
  b=read(R/rel);ls=b.decode().splitlines();code.append(dict(path=rel,sha256=pins[str(R/rel)]['sha256'],matching_lines=[dict(line=i+1,text=l)for i,l in enumerate(ls)if re.search(pattern,l)]))
 numeric=actual['numeric_flags'];assert actual['sampling']=='unconditional_class1000_greedy_no_CFG'and not numeric['matmul_tf32']and not numeric['cudnn_tf32']and numeric['deterministic']
 configuration=dict(schema='READ_ONLY_CONFIGURATION_AUDIT_V1',model=dict(name='VAR-d16',V=4096,Cvae=32,depth=16,patch_nums=[1,2,3,4,5,6,8,10,13,16],flash=False,fused=False,cond_drop_rate=0,state_hashes=actual['models'],state_hash_basis='actual closed N512 formal RX receipt, not newly loaded weights'),actual_formal_native_identity=actual,actual_timing_native_identity=timingnative,CDF=dict(total=2**24,integer_interval_bits=32,softmax_dtype='float64',minimum_frequency=1,formula='floor(p*(2^24 - 4096)) + 1',remainder='entire remaining mass added to maximum-probability token',tie='numpy argmax, first index',probability_route='native FP32 logits to CPU float64 CDF; no CFG in entropy decoding'),decoder_training=dict(branches={'continuous_F':.5,'full_quantized_Fq':.25,'interpolation':.25},interpolation='F_bTX + alpha*(F - F_bTX)',F_bTX='VAR-completed latent through all original 10 scales from the first eight source scales',coarse_only_training_branch=False,training_completion_condition='source ImageNet label passed to complete_latent; current formal NeST RX is unconditional class1000'),scrambling=dict(TX='LDPC/RV0/interleave -> XOR public PCG64 mask -> QAM',RX='soft demap -> multiply LLR by (1 - 2*mask) -> LDPC -> CRC/parser',mask_seed='SHA256(protocol, scramble, session, frame_counter), first 16 bytes little endian',power='constellation average Es=2; no per-frame normalization',audit='raw formal transmission energy fields, consistency E_header+E_body=E_frame; no waveform regeneration'),code=code)
 save('configuration_audit.json',configuration);save('environment_and_scope_guard.json',environment);save('configuration_source_identity.json',dict(input_files=list(pins.values()),new_PHY=0,new_model_calls=0,new_metrics=0,new_bootstrap=0))
 print(json.dumps(dict(output=str(out),source_rows=9000,policies=18,timing=summ,HEAD=head,guard=environment['paused_guard']['sha256'])))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);main(ap.parse_args().output)
