"""Verify the fixed 600 test sources and old endpoint assets. No model imports."""
from tc_common import *
import subprocess,numpy as np,traceback
def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==HEAD
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=R,text=True).strip()
 assert sha(FROZEN)=='81dfc416ccd82bad538885e8f71f394a6f7f94bb21e403b510721d78582b3e02'
 pol=read(FROZEN)
 for fam in ('WHOLE','PARTIAL'):assert pol['policies']['19'][fam]['winner']==f'm10_{fam.lower()}_q6_r5-6'
 T.mkdir();O.mkdir();out=T/'source_preparation';out.mkdir()
 try:
  em=B/'entropy_main_common500_20261013_v1';regs=read(R/'results/entropy_main_common500_20261013/registration_v4.json')
  wait=read(em/'source/actual_wait.json');assert type(wait['returncode']) is int and wait['returncode']==0
  done=read(em/'source/completion.json');streams=checkread(done['streams']);assert len(streams)==500
  q=Q/'h800_ep_new100_source_v1_attempt1';nw=read(q/'run_owner_actual_wait.json');assert nw['actual_wait'] and type(nw['returncode']) is int and nw['returncode']==0
  nd=read(q/'registered/run/worker_completion.json');nr=read(q/'registered/request.json');assert nd['request_sha256']==sha(q/'registered/request.json') and len(nd['sources'])==100
  controls=[pin(em/'source/actual_wait.json'),pin(em/'source/completion.json'),pin(R/'results/entropy_main_common500_20261013/registration_v4.json'),pin(q/'run_owner_actual_wait.json'),pin(q/'registered/run/worker_completion.json'),pin(q/'registered/request.json')]
  raw=[]
  for s,st in zip(regs['sources'],streams):
   assert s['source_id']==st['source_id'] and s['source_index']==st['source_index'];raw.append(dict(s,population='common500',old_streams=st['archive'],old_metadata=st['metadata']))
  for s in nd['sources']:raw.append(dict(s,population='new100',archive=s['source_assets'],old_streams=s['archive'],old_metadata=s['metadata']))
  assert len({s['source_id'] for s in raw})==600
  cal=read(R/'results/submission_closure_20261016_v1/source_registration.json')['sources'];assert not {s['source_id'] for s in raw}&{s['source_id'] for s in cal}
  sources=[];existing=0
  for j,s in enumerate(raw):
   checked(s['archive']);checked(s['old_streams']);md=checkread(s['old_metadata'])
   with np.load(s['archive']['path'],allow_pickle=False) as z:tokens=z['tokens'];pixels=z['pixels']
   assert tokens.dtype==np.int64 and tokens.shape==(680,) and hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==s['tokens_sha256']
   assert pixels.dtype==np.uint8 and pixels.shape==(3,256,256) and hashlib.sha256(pixels.tobytes()).hexdigest()==s['preprocessing_id']
   with np.load(s['old_streams']['path'],allow_pickle=False) as z:
    assert set(z.files)=={f'm{m}_K{k}' for m,k in ENDS};endpoints=[dict(m=m,K=k,bits=len(z[f'm{m}_K{k}']),payload_sha256=asha(z[f'm{m}_K{k}'])) for m,k in ENDS]
   sources.append(dict(source_global_index=j,source_index=s['source_index'],source_id=s['source_id'],population=s['population'],archive=s['archive'],tokens_sha256=s['tokens_sha256'],preprocessing_id=s['preprocessing_id'],old_streams=s['old_streams'],old_metadata=s['old_metadata'],old_endpoints=endpoints))
  save(out/'sources.json',sources)
  gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=pci.bus_id,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
  assert int(gpu[2])>=15360 and int(gpu[3])<=25
  old=read(R/'results/t1b_m10_calibration_20261016_v1/registration.json')
  files=[pin(E/n) for n in ('tc_common.py','tc_source.py','tc_control.py','prepare_source.py')]
  files+=old['code']
  for d in files:checked(d)
  r=dict(schema='T1C_SOURCE_REGISTRATION_V1',scope='T1C_19dB_ONLY_FROZEN_TWO_ENTROPY_FAMILIES',basis_HEAD=HEAD,frozen_policies=pin(FROZEN),code=files,sources=pin(out/'sources.json'),old_controls=controls,model_routes=old['model_routes'],gpu_index=2,gpu_pci=gpu[0].strip(),gpu_uuid=gpu[1].strip(),caps=dict(model_load=1,source_tx=600,source_rx=0,prior_scale=6000,decoder_forward=0),max_seconds=7200,tokenizer_calls=0,new_wireless_frames=0,new_quality_scores=0,automatic_followup=False,qualification='Existing m10 first8 independent RX and 24000 legacy regressions retained; no duplicate qualification')
  save(O/'source_registration.json',r);save(out/'completion.json',dict(status='VERIFIED600_EXISTING_TOKENS_AND_24_STREAMS',sources=600,missing_m10=600,tokenizer_missing=0,source_id_intersection_with_cal1000=0,new_scientific_calls=0,registration=pin(O/'source_registration.json')))
  print(json.dumps(pin(O/'source_registration.json')))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),scientific_calls=0));raise
if __name__=='__main__':main()
