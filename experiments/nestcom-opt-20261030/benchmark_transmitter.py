"""Fresh RGB-to-terminated-bitstream deployment timing, including true E scoring."""
from opt_common import *
from grid_codec import codec,endpoints
from frame_plan import nt
from receiver_variants import completed_latent,decode_image
from metrics import module
import tempfile,traceback,subprocess

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];cases=cr(r['cases']);routes=cr(r['model_routes']);out=Path(r['output']);out.mkdir(exist_ok=False);(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp');os.sched_setaffinity(0,set(r['cpu_affinity']))
 os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 def boundary():
  if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('TX timing cap')
 try:
  sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'));from kn_native import load,tokenizer
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));t=n.torch;c=codec(n)
  from h800_ep48_link_core_v1 import bind_static
  bind_static(c,r['static']);rows=[];outputs=[];fingerprints={}
  for variant in r['variants']:
   name=variant['name'];network=None
   if name=='E':
    old=cr(r['metric_registration']);cfg=old['config'];met=module(checked(cfg['clip']['loader']),'_BENCH_TX_METRIC');sys.path.insert(0,str(Path(old['original_four']['path']).parent));four=module(checked(old['original_four']),'_BENCH_FOUR');mc=cfg['config']['metric_config'];sys.path.insert(0,str(R/'tools'));import checkpoint_relocation_20261028 as relocation
    assets=relocation.metric_assets(four,mc);import lpips
    assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==four.WEIGHT_SHA['lpips']
    lp_load,lp_pair,proof=four.lpips_projection(mc['sources']['quality.py']['path'],dict(Path=Path,lpips=lpips,torch=t))
    with met.no_network():network=ledger.call('load_lpips','once',lambda:lp_load({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0'))
   t.cuda.reset_peak_memory_stats();snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
   for case in cases:
    with np.load(checked(case['archive']),allow_pickle=False)as z:pixels=z['pixels'].copy()
    assert asha(pixels)==case['source_pixels_sha256'];x=pixels.astype(np.float32)/np.float32(255);key=case['case_id'];es=tuple(sorted({e for e in endpoints(variant['grid'])if nt(e)<=nt(case['target'])}|{tuple(case['target'])},key=nt,reverse=True))
    for phase,count in [('warmup',2),('measured',3)]:
     for repeat in range(count):
      boundary();event=f'{name}:{key}:{phase}:{repeat}'
      @t.no_grad()
      def trial():
       t.cuda.synchronize();beg=time.perf_counter();tokens=tokenizer(n,pixels);t.cuda.synchronize();token_seconds=time.perf_counter()-beg
       streams=c.encode_endpoints(tokens,es);feasible=[e for e in es if len(streams[e]['bits'])<=min(case['capacity_bits'],8191)];assert feasible
       e=feasible[0];candidates=feasible[:3]if name=='E'else[e];scores=[]
       if name=='E':
        xt=t.from_numpy(x[None]).cuda()
        for candidate in candidates:
         bits=streams[candidate]['bits'];parsed=c.decode('EC_VAR_PARTIAL'if candidate[1]else'EC_VAR_WHOLE',bits,*candidate);assert parsed['canonical']and not parsed['receiver_truth_used']
         lat,_=completed_latent(n,parsed['received_tokens'],*candidate);y=decode_image(n,lat)
         value=float(np.square(x.astype(np.float64)-y.astype(np.float64)).mean())+.1*float(lp_pair(network,t.from_numpy(y[None]).cuda(),xt)[0].item());scores.append(value)
        best=0
        for i in range(1,len(scores)):
         if scores[i]<scores[best]-1e-12:best=i
        e=candidates[best]
       t.cuda.synchronize();total=time.perf_counter()-beg
       return dict(tokenizer_seconds=token_seconds,entropy_fitting_and_selection_seconds=total-token_seconds,tx_source_seconds=total,endpoint=list(e),payload=streams[e]['bits'].tolist(),payload_sha256=asha(streams[e]['bits']),independent_termination_attempts=len(es),length_checks=len(es),candidate_reconstructions=len(candidates)if name=='E'else 0,candidate_LPIPS_calls=len(scores),objective_scores=scores)
      v=ledger.call('timing_trial',event,trial);assert v['payload_sha256']==case['expected_payloads'][variant.get('input_grid',variant['grid'])];fk=(name,key)
      if fk in fingerprints:assert fingerprints[fk]==v['payload_sha256']
      else:fingerprints[fk]=v['payload_sha256']
      rows.append(dict(variant=name,case_id=key,source_id=case['source_id'],N=case['N'],snr_db=case['snr_db'],phase=phase,repeat=repeat,**{k:x for k,x in v.items()if k!='payload'}))
      if phase=='measured'and repeat==2:outputs.append(dict(variant=name,case_id=key,source_id=case['source_id'],N=case['N'],snr_db=case['snr_db'],endpoint=v['endpoint'],payload=v['payload'],payload_sha256=v['payload_sha256']))
   save(out/(name+'_memory.json'),dict(peak_allocated=t.cuda.max_memory_allocated(),peak_reserved=t.cuda.max_memory_reserved(),shared_GPU_snapshot=snapshot));print('TIMING_TX',name,len(rows),flush=True)
  save(out/'rows.json',rows);save(out/'transmitted_inputs.json',outputs);save(out/'completion.json',dict(status='COMPLETE',rows=len(rows),warmups_per_case=2,measured_per_case=3,batch=1,ledger=ledger.summary(),fresh_RGB_tokenizer=True,no_precomputed_token_or_image_cache=True,E_includes_source_decoding_completion_Dc_and_LPIPS=True,excludes_PHY_to_be_added_from_matched_CPU_measurements=True,seconds=time.monotonic()-start))
 except BaseException as e:save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
