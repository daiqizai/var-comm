"""Read-only inventory first, then exclusive registration and RX-only copies."""
from common import *
import subprocess,collections,numpy as np

def main():
 assert not O.exists() and not L.exists()
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='a5f8ceb0233b0e1f599ee8bb902c1a094d47119d'
 oldreg=read(OLD/'a3_rx_v1/registration.json')
 for p in ['a3_rx_v1/rx.actual_wait.json','a3_rx_v1/audit.actual_wait.json']:
  w=read(OLDL/p);assert w['returncode']==0 and gone(w['process'])
 states=read(OLD/'a3_rx_v1/actual_states.json');logical=read(OLD/'a3_rx_v1/logical_receivers.json')
 assert len(states)==296 and len(logical)==900 and len({r['source_id'] for r in logical})==100
 assert all(s['status']=='IMAGE_READY' and s['endpoint']<7 for s in states)
 assert all(r['receiver_status']=='IMAGE_READY' for r in logical)
 assert collections.Counter((r['snr_db'],r['noise_seed']) for r in logical)==collections.Counter({(s,n):100 for s in [4,10,19] for n in [6201,6202,6203]})
 completion=read(OLDL/'a3_rx_v1/rx/completion.json')
 oldoutputs={r['key']:r for r in completion['outputs']}
 prefixes=[]
 for s in states:
  checked(s['input']);checked(s['image'])
  p=OLDL/'a3_rx_v1/rx'/(s['key']+'_decoded.npz')
  assert p.is_file() and p.stat().st_mtime<=completion['ended_unix']
  with np.load(p,allow_pickle=False) as z:
   assert len(z.files)==s['endpoint']
   for i,side in enumerate([1,2,4,6,8,12,16][:s['endpoint']]):
    a=z[f's{i+1}'];assert a.shape==(1,side**2,16) and a.dtype==np.int32 and np.isin(a,[0,1]).all()
  assert oldoutputs[s['key']]['float_sha256']==s['float_sha256']
  assert len(oldoutputs[s['key']]['cdf_hashes'])==s['endpoint']
  prefixes.append(dict(key=s['key'],paid_profile=s['endpoint'],generation_seed=0,wire=s['input'],prefix=desc(p)))
 # Bind all original model and source bytes without mutating historical registration.
 for d in oldreg['weights']+oldreg['source_files']:checked(d)
 O.mkdir();L.mkdir();wire=L/'wire_inputs';wire.mkdir()
 jobs=[]
 for p in prefixes:
  j={k:v for k,v in p.items() if k not in ['wire','prefix']}
  for key,suffix in [('wire','.arc'),('prefix','_decoded.npz')]:
   target=wire/(p['key']+suffix)
   with target.open('xb') as f:f.write(checked(p[key]).read_bytes())
   j[key]=desc(target);assert j[key]['sha256']==p[key]['sha256']
  jobs.append(j)
 save(wire/'jobs.json',jobs)
 cfg=read(OLDL/'a3_rx_v1/public.json')
 cfg.update(run=str(L),output=str(O),sensitivity_code=str(E),unique_prefixes=296,render_cap=592,
   receiver_source_sha256=sha(Path(cfg['upstream'])/'compression/reciever.py'))
 cfg['resource']['gpu']=dict(index=1,uuid='GPU-f5047136-08e5-9416-3c86-87786d0b355e',pci_bus_id='0000:62:00.0')
 cfg['cuda_admission']=dict(visible_index=1,nvml_uuid='GPU-f5047136-08e5-9416-3c86-87786d0b355e',pci_bus_id='0000:62:00.0')
 save(L/'public.json',cfg);save(O/'prefix_cache_inventory.json',prefixes)
 save(O/'registration.json',dict(schema='ARPC_GENERATION_ONLY_SENSITIVITY_V1',
  basis_commit='a5f8ceb0233b0e1f599ee8bb902c1a094d47119d',created_unix=time.time(),
  population='external_common100_v1 matched100',logical_per_setting=900,source_count=100,
  snr_db=[4,10,19],noise=[6201,6202,6203],N=1024,
  settings=[dict(id='original',tau=.5,source='closed A3 pixels and scores; no rerun'),
   dict(id='greedy',tau=.5,rule='explicit argmax on each untransmitted binary token logit'),
   dict(id='low_tau025',tau=.25,rule='native top-k900 top-p0.97 multinomial after logits/tau, missing scales only')],
  cfg=3,generation_seed=0,received_scales='Original tau .5 sampler and overwrite unchanged; stored independent RX tokens only',
  limits=dict(model_load=1,new_unique_render=592,logical_new_settings=1800,wireless=0,source_decodes=0,caption=0,tokenizer=0,training=0),
  source_files=[desc(p) for p in E.iterdir() if p.is_file()],old_source_files=oldreg['source_files'],
  weights=oldreg['weights'],public=desc(L/'public.json'),jobs=desc(wire/'jobs.json'),
  inputs=[desc(OLD/p) for p in ['a3_rx_v1/registration.json','a3_rx_v1/actual_states.json','a3_rx_v1/logical_receivers.json','a3_metrics_v1/bootstrap_identity.json']],
  original_prefix_files=prefixes,bootstrap=read(OLD/'a3_metrics_v1/bootstrap_identity.json'),
  host='aibox-r59688baa77c-57c794f5c6-44lm7',resource=cfg['resource'],max_rx_seconds=7200,
  no_selection_feedback=True,no_automatic_retry=True,N2048_CLIP='NOT_STARTED_OPTIONAL'))
 print(json.dumps(desc(O/'registration.json')))
if __name__=='__main__':main()
