"""Post-packet immutable join and exact image reuse, new inputs contain RX only."""
from shared import *
import pair_protocol as p
import numpy as np
def main():
 r=registration('4d07344ecd2008f3663558a596afa3a879eb13797bccb4c2a41b3bff5c683ae9')
 for role in ('qualification_close_saved','qualification_audit_saved'):
  w=read(L/'controls'/role/'actual_wait.json');assert w['returncode']==0 and gone(w['process'])
 jobs=read(L/'qualification_rx_closure_v1/source_jobs.json');frames=read(O/'qualification/logical_frames.json')
 cache={(x['source_id'],x['codec'],x['endpoint']):x for x in read(L/'b0/cache_catalogue.json')}
 root=L/'qualification/native';root.mkdir();inputs=root/'wire_inputs';inputs.mkdir();new=[];reused=[]
 for j in jobs:
  data=verify(j['input']).read_bytes();assert p.input_key(j['codec'],j['endpoint'],j['bit_length'],data)==j['key']
  f=next(x for x in frames if x['source_input_key']==j['key']);c=cache[f['source_id'],j['codec'],j['endpoint']];im=c['image']
  if im:
   archive=verify(im['image_archive']);key=im.get('image_key','image')
   with np.load(archive,allow_pickle=False)as z:a=z[key].copy()
   assert a.dtype==np.float32 and a.shape==(3,256,256) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
   raw=hashlib.sha256(a.tobytes()).hexdigest();structured=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
   if im.get('image_sha256'):assert im['image_sha256']==(structured if j['codec']==0 and c['origin']=='ORIGINAL_ENTROPY_RX' else raw)
   if j['codec']==1:
    source=im['evidence'].get('input',im['evidence'].get('wire'));assert verify(source).read_bytes()==data
   elif im.get('reconstruction_result'):verify(im['reconstruction_result']);verify(im['physical_frame'])
   reused.append(dict(**j,source_id=f['source_id'],image=pin(archive),array_key=key,float_sha256=raw,provenance=im,status='EXACT_CLOSED_SOURCE_RX_REUSE'))
  else:
   assert j['codec']==0
   ip=inputs/(j['key']+'.bin')
   with ip.open('xb')as file:file.write(data)
   new.append({k:j[k] for k in ('key','codec','endpoint','bit_length','generation_seed')}|dict(input=pin(ip)))
 assert len(jobs)==6 and len(new)==2 and len(reused)==4
 put(inputs/'jobs.json',new);put(root/'reuse.json',reused)
 gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=pci.bus_id,uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',');assert int(gpu[2])>=15360
 cfg=dict(run=str(root),model_routes=r['model_routes'],static=r['static'],jobs=pin(inputs/'jobs.json'),gpu_index=0,gpu_pci=gpu[0].strip(),gpu_uuid=gpu[1].strip(),new_jobs_cap=2,max_seconds=600)
 put(root/'public.json',cfg)
 put(O/'qualification/native_authorization.json',dict(status='PREPARED_EXACT_RX_ONLY',registration=pin(REG),source=pin(E/'nest_prefix_rx.py'),public=pin(root/'public.json'),jobs=pin(inputs/'jobs.json'),reuse=pin(root/'reuse.json'),actual_source_RX_and_render_new=2,new_ARPC_model_calls=0,original_packet_failure_preserved=True,source_model_mathematics='Unchanged frozen ReceiverCodec/tb_native/render_received',user_scope='B1 finite qualification under 按照这个推进到底'))
 print(json.dumps(dict(new=2,reused=4,authorization=pin(O/'qualification/native_authorization.json'))))
if __name__=='__main__':main()
