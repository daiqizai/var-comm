"""Verify user-paused checkpoints on CPU, without resuming or model execution."""
from pathlib import Path
import argparse,hashlib,json,time
import torch
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def live(i):
 p=Path('/proc',str(i['pid']),'stat');return p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==i['start_ticks']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 L=Path(a.root);O=Path(a.output);assert O.is_dir() and (L/'STOP_P2').exists();request=read(L/'user_pause_request_20261105_v1.json')
 result=[]
 for arm in request['active']:
  assert not live(arm['owner'])
  phase=arm['training'][-1];assert not live(phase['child.json'])
  d=L/arm['name']/phase['phase'];wait=read(d/'actual_wait.json');assert wait['returncode']==0 and wait['child_gone']
  p=read(d/'paused.json');c=p['checkpoint'];assert sha(c['path'])==c['sha256']
  state=torch.load(c['path'],map_location='meta',weights_only=False)
  assert {'model','optimizer','rng','step','cursor','request_sha256','experiment_sha256','arm','seed'}<=state.keys()
  assert state['step']==c['step'] and state['cursor']==c['cursor']==16*c['step']
  assert state['request_sha256']==c['request_sha256']
  assert {'python','numpy','torch','cuda','endpoint_generator'}<=state['rng'].keys()
  assert state['optimizer']['state'] and state['optimizer']['param_groups']
  result.append(dict(name=arm['name'],arm=state['arm'],seed=state['seed'],step=c['step'],cursor=c['cursor'],
    checkpoint=c,optimizer_parameter_states=len(state['optimizer']['state']),model_tensor_keys=len(state['model']),
    RNG_components=sorted(state['rng']),actual_wait=wait,owner_and_child_gone=True,sha256_verified=True))
  del state
 assert all(not (L/n).exists() for n in request['pending'])
 outer=read(L/'p2_owner_control_v1/study/actual_wait.json');assert outer['returncode']==0 and outer['child_gone']
 continuation=read(L/'first_batch_continuation_registration_v1/owner_control/actual_wait.json');assert continuation['child_gone']
 assert not torch.cuda.is_initialized()
 out=dict(passed=True,created=time.time(),reason='Explicit user pause; no automatic resume',arms=result,not_started=request['pending'],
    P2_charged_GPU_device_hours=sum(r['actual_wait']['charged_gpu_device_hours'] for r in result),
    P2_training_CPU_process_seconds=sum(r['actual_wait']['cpu_process_seconds'] for r in result),
    resume_or_inference_calls=0,CUDA_initialized=False,
    analysis_scope='Registered continuation closed with zero trained arms; any 48 means are R0-only, not P2 effects.',
    P2_status='PAUSED_BY_USER_INCOMPLETE',checkpoint_restore_scope='Native save receipt, complete SHA and state structure; no post-pause training replay.')
 (O/'complete.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
