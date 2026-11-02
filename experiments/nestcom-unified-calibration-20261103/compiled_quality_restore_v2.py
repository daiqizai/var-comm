"""Recover the timed implementation plus a separately labelled seed control."""
from study_common import *
import traceback,tempfile

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(p)for p in r['code']]
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'images').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'),TRITON_CACHE_DIR=str(out/'triton'),TORCHINDUCTOR_COMPILE_THREADS='8',MAX_JOBS='8',TMPDIR=str(out/'tmp'))
 sys.path.insert(0,str(R/'experiments/nestcom-perceptual-followup-20261030'))
 from stdlib_statistics_v1 import bind_stdlib_statistics
 bind_stdlib_statistics();sys.path.insert(0,str(R/'experiments/nestcom-perceptual-followup-20261030/arpc_child_stdlib_v1'))
 sys.path.insert(0,str(R/'experiments/nestcom-timing-fairness-20261102'))
 from ada_paired_prefill_headers_v2 import PairedPrefill
 from timing_dependency_v4 import activate
 activate(r['sionna_package'])
 import timing_end_to_end_identity_v5 as original
 sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'))
 from n512_phy import received_projection
 from c_predclass_protocol import generation_seed
 inputs=cr(r['inputs']);assert len(inputs)==32
 ledger=Ledger(out/'calls.sqlite',dict(compile=2,receiver_restore=32,seed_control_render=32));start=time.monotonic()
 try:
  cfg=cr(r['config']);sources=cr(r['sources']);denied={str(Path(s['archive']['path']).resolve())for s in sources}
  def firewall(event,args):
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    p=str(Path(os.fsdecode(args[0])).resolve())
    if p in denied or '/witnesses/'in p:raise PermissionError('Receiver cannot read TX/source truth: '+p)
  sys.addaudithook(firewall)
  os.sched_setaffinity(0,{38,40});ctx=original.setup(cfg,r['group'],out,'rx',r['pci']);n=ctx['n'];t=n.torch;t.set_num_threads(2)
  assert str(t.cuda.get_device_properties(0).uuid)==r['CUDA_UUID'];route=PairedPrefill(n);Encoder,Decoder,cdf=ctx['c'];rows=[]
  with t.inference_mode():
   os.sched_setaffinity(0,{38,40,42,44,46,48,50,52})
   for branches in [1,2]:ledger.call('compile',str(branches),lambda:route.logits([],0,branches));t.cuda.synchronize()
   os.sched_setaffinity(0,{38,40})
   for inp in inputs:
    assert time.monotonic()-start<r['seconds']and time.time()<r['deadline_unix']and not(L/'STOP').exists()
    with np.load(checked(inp['symbols']),allow_pickle=False)as z:observed=z['observed'].copy()
    assert asha(observed)==inp['observation_sha256']
    def restore():
     # Decoder replay of an existing saved channel event, no new TX/noise/frame.
     rx=ctx['rt'].receive(observed,10,inp['metadata']['counter'],ctx['rt'].profiles,ctx['calls'],inp['event'],phase='QUALITY_RESTORE')
     p=received_projection(rx);assert p['status']=='SOURCE_INPUT'
     b=np.asarray(p['bits'],np.uint8);tokens,_=route.decode(b,p['m'],p['received_class'],Encoder,Decoder,cdf);seed=generation_seed(b,p['m'],p['received_class']);im,full=route.render(tokens,p['received_class'],seed)
     expected=inp['expected'];assert asha(tokens)==expected['received_tokens_sha256']==inp['old_received_tokens_sha256'];assert asha(full)==expected['completed_tokens_sha256'];assert asha(im)==expected['output_sha256']
     return im,tokens,p,seed
    im,tokens,p,seed=ledger.call('receiver_restore',inp['source_id'],restore)
    oldseed=inp['old_generation_seed'];oldcount=p['m']
    if oldseed==seed or oldcount==256:aligned=im;origin='EXACT_RENDER_REUSE'
    else:
     aligned,_=ledger.call('seed_control_render',inp['source_id'],lambda:route.render(tokens,p['received_class'],oldseed));origin='NEW_FIXED_OLD_SEED_RENDER'
    for arm,image,usedseed,how in [('compiled_timed',im,seed,'TIMED_HASH_EXACT'),('compiled_fixed_old_seed',aligned,oldseed,origin)]:
     path=out/'images'/(str(inp['source_index'])+'_'+arm+'.npz')
     with path.open('xb')as f:np.savez(f,image=image,received_tokens=tokens)
     rows.append(dict(source_id=inp['source_id'],source_index=inp['source_index'],arm=arm,image=pin(path),pixel_sha256=asha(image),generation_seed=usedseed,received_tokens_sha256=asha(tokens),prefix=oldcount,received_class=p['received_class'],status='ACCEPTED_SOURCE',origin=how,timing_event=inp['event'],observation_sha256=inp['observation_sha256']))
    print('COMPILED_QUALITY_RESTORE',len(rows)//2,32,flush=True)
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='COMPLETE',sources=32,outputs=64,ledger=ledger.summary(),new_wireless_events=0,saved_event_PHY_replays=32,new_TX=0,retiming=0,all_timed_image_hashes_exact=True,all_old_new_received_token_hashes_exact=True,seconds=time.monotonic()-start))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
