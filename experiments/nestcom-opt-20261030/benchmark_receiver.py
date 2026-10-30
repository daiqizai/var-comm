"""Source-blind batch-one receiver timing; two warmups and three measured runs."""
from opt_common import *
from receiver_variants import completed_latent,decode_image,load_anchor
import tempfile,traceback,gc,subprocess

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 cases=cr(r['cases']);routes=cr(r['model_routes']);static=r['static'];out=Path(r['output']);out.mkdir(exist_ok=False);(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.sched_setaffinity(0,set(r['cpu_affinity']));os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 def boundary():
  if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('Receiver benchmark budget')
 try:
  sys.path[:0]=[str(R/'src'),str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/kodak-current-20261026/nest_v1')]
  from runtime_boundary import firewall
  allowed=[v['path']for v in routes['files']+routes['author_files']]+[routes['dist']['path'],static['table']['path'],static['completion']['path']]+[v['checkpoint']['path']for v in r['variants']if'checkpoint'in v]
  firewall(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  from kn_native import load
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));t=n.torch
  from grid_codec import codec as make_codec
  codec=make_codec(n)
  from h800_ep48_link_core_v1 import bind_static
  from t1_codec_runtime import InvalidSourceStream
  bind_static(codec,static);rows=[];fingerprints={}
  for variant in r['variants']:
   boundary();name=variant['name'];anchor=corrector=None
   if variant['kind']=='A':anchor=ledger.call('anchor_load',name,lambda:load_anchor(n,variant['checkpoint'],2026102201))
   elif variant['kind']=='C':
    from c_model import Corrector
    t.backends.mha.set_fastpath_enabled(False)
    class MathCorrector(t.nn.Module):
     def __init__(self,inner):super().__init__();self.inner=inner
     def forward(self,*a,**kw):
      with t.nn.attention.sdpa_kernel(t.nn.attention.SDPBackend.MATH):return self.inner(*a,**kw)
    def build():
     cp=t.load(checked(variant['checkpoint']),weights_only=True,map_location='cpu');assert cp['step']==1000 and cp['seed']==2026103001 and cp['arm']==variant['arm']
     m=Corrector().to(n.loaded['device']).eval().requires_grad_(False);m.load_state_dict(cp['model'],strict=True);return MathCorrector(m).eval()
    corrector=ledger.call('corrector_load',name,build)
   t.cuda.reset_peak_memory_stats();snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
   for case in cases:
    p=case['projections'][variant.get('input_grid','quarter')];key=case['case_id']
    for phase,count in [('warmup',2),('measured',3)]:
     for repeat in range(count):
      boundary();event=f'{name}:{key}:{phase}:{repeat}'
      @t.no_grad()
      def trial():
       parsed=None;status='FIXED_GRAY';tokens=np.empty(0,np.int64)
       t.cuda.synchronize();beg=time.perf_counter()
       if p['status']=='SOURCE_INPUT':
        try:parsed=codec.decode(p['family'],np.asarray(p['bits'],np.uint8),p['m'],p['K'])
        except InvalidSourceStream:parsed=None
       t.cuda.synchronize();decode_seconds=time.perf_counter()-beg
       if parsed is None:image=np.full((3,256,256),.5,np.float32)
       else:
        assert parsed['canonical']and not parsed['receiver_truth_used']and not parsed['transmitter_probability_table_used'];tokens=np.asarray(parsed['received_tokens'],np.int64)
        lat,_=completed_latent(n,tokens,p['m'],p['K'],temperature=variant.get('temperature',0),seed=2026103001,corrector=corrector,no_content=variant.get('arm')=='no_content')
        use_anchor=anchor if(p['m'],p['K'])in((7,0),(7,25))else None
        image=decode_image(n,lat,use_anchor);status='ACCEPTED_SOURCE'
       t.cuda.synchronize();total=time.perf_counter()-beg
       return dict(source_decode_seconds=decode_seconds,completion_and_decoder_seconds=total-decode_seconds,rx_source_image_seconds=total,pixel_sha256=asha(image),received_tokens_sha256=asha(tokens),status=status)
      v=ledger.call('timing_trial',event,trial);fk=(name,key)
      if fk in fingerprints:assert fingerprints[fk]==(v['pixel_sha256'],v['received_tokens_sha256'])
      else:fingerprints[fk]=(v['pixel_sha256'],v['received_tokens_sha256'])
      rows.append(dict(variant=name,kind=variant['kind'],case_id=key,source_id=case['source_id'],N=case['N'],snr_db=case['snr_db'],phase=phase,repeat=repeat,**v))
   save(out/(name+'_memory.json'),dict(peak_allocated=t.cuda.max_memory_allocated(),peak_reserved=t.cuda.max_memory_reserved(),shared_GPU_snapshot=snapshot))
   print('TIMING_RX',name,len(rows),flush=True);anchor=corrector=None;gc.collect();t.cuda.empty_cache()
  save(out/'rows.json',rows);save(out/'completion.json',dict(status='COMPLETE',rows=len(rows),variants=[v['name']for v in r['variants']],warmups_per_case=2,measured_per_case=3,batch=1,ledger=ledger.summary(),source_blind_RX=True,includes_source_decode_and_actual_uncached_completion=True,excludes_PHY_to_be_added_from_matched_CPU_measurements=True,new_quality_scores=0,seconds=time.monotonic()-start))
 except BaseException as e:save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
