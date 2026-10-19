"""Finite engineering checks and separately invoked actual B1 transport stages."""
from shared import *
import argparse,fcntl
def engineering():
 root=L/'engineering';root.mkdir();cache(root);(L/'controls').mkdir()
 put(root/'supervisor.json',dict(process=identity(os.getpid()),parent_process=identity(os.getppid()),argv=sys.argv))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 for role,argv in [('protocol',[str(CPY),'-B',str(E/'test_protocol.py')]),('backend',[str(CPY),'-B',str(E/'b1_control.py'),'backend'])]:
  out=root/role;out.mkdir();wait_child(out,argv,env,300)
 put(root/'completion.json',dict(status='ENGINEERING_PASS_NOT_SCIENTIFIC_COMPLETION',source=[pin(E/n) for n in ('adaptive_plan.py','pair_protocol.py','pair_link.py','shared.py','phy_worker.py','test_protocol.py','b1_control.py')],model_calls=0,packet_calls=0))
def backend():
 from pair_link import Runtime,noise,ep
 rt=Runtime()
 import importlib.util
 oldpath=R/'experiments/arpc-n2048-validation-20261019/pair_link.py'
 # The original noise formula is explicitly evaluated with its unchanged namespace.
 import numpy as np
 for sid in ('synthetic-a','synthetic-b'):
  for seed in (4101,4102,4103,6201,6202,6203):
   h=hashlib.sha256(('NEST_ARPC_N2048_EQUAL_PROTECTION_V1|'+sid+'|'+str(seed)).encode()).digest()
   prior=np.random.Generator(np.random.PCG64(int.from_bytes(h[:16],'big'))).standard_normal((2048,2)).astype(np.float64)
   assert np.array_equal(noise(sid,seed),prior)
 # Only constructors and deterministic metadata, no encode, transmit or decode.
 put(L/'engineering/backend/catalogue.json',rt.catalogue)
 print(json.dumps(dict(layouts=rt.catalogue['layouts'],backend=rt.catalogue['backend_identity'],model_calls=0,packet_calls=0)))
def register():
 eng=read(L/'engineering/completion.json');assert eng['status']=='ENGINEERING_PASS_NOT_SCIENTIFIC_COMPLETION'
 for d in eng['source']:verify(d)
 for role in ('protocol','backend'):
  w=read(L/'engineering'/role/'actual_wait.json');assert w['returncode']==0 and gone(w['process'])
 assert not REG.exists() and not(L/'qualification').exists()
 src=read(OLD_O/'sources.json')['cal100'];assets=read(L/'b0/source_catalogue.json');lookup={(x['source_id'],x['codec'],x['endpoint']):x for x in assets if x['cal100']}
 jobs=[]
 for codec in (0,1):
  for qi,(layout,snr,ne,ae) in enumerate([('q2_r2-3',4,11,4),('q4_r2-3',10,18,6),('q6_r5-6',19,22,7)]):
   s=src[qi];end=ne if codec==0 else ae;asset=lookup[s['source_id'],codec,end]
   for case in ('clean','awgn','whole_corrupt','profile_mismatch'):
    i=len(jobs);jobs.append(dict(key=f'q{i:04d}',pair_id=300000+i,source_id=s['source_id'],source_index=s['source_index'],codec=codec,layout_id=layout,snr_db=snr,noise_seed=4101,case=case,asset=asset))
 oldreg=read(R/'results/arpc_n2048_validation_20261019_v1/registration.json')
 catalogue=read(L/'engineering/backend/catalogue.json');assert catalogue['backend_identity']==oldreg['backend_identity']
 codes={d['path']:d for d in oldreg['code']}
 for d in eng['source']:codes[d['path']]=d
 for d in codes.values():verify(d)
 (L/'qualification').mkdir()
 put(REG,dict(schema='N2048_ADAPTIVE_B1_TRANSPORT_V1',basis_HEAD=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),code=list(codes.values()),qualification_jobs=jobs,backend_identity=catalogue['backend_identity'],model_routes=oldreg['model_routes'],static=oldreg['static'],qualification_cap=dict(objects=24,subframes=48,header=48,body=48),new_TX_source_calls=0,automatic_B2=False,qualification_native_review_required=True,stage_seconds=dict(qualification_tx=900,qualification_rx=900,qualification_audit=300),prescreen_gap=dict(NeST_source_RX_and_render=1793,NeST_DINO=1793,ARPC_render=0,ARPC_DINO=0),scope=pin(O/'scope_proposal.json')))
 print(json.dumps(pin(REG)))
def stage(name,expected):
 r=registration(expected);assert name in r['stage_seconds'];phase,role=name.split('_',1)
 previous={'rx':'tx','audit':'rx'}.get(role)
 if previous:
  w=read(L/'controls'/(phase+'_'+previous)/'actual_wait.json');assert w['returncode']==0 and gone(w['process'])
 root=L/'controls'/name;root.mkdir();put(root/'supervisor.json',dict(process=identity(os.getpid()),parent_process=identity(os.getppid()),argv=sys.argv))
 lock=(L/'execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 argv=[str(CPY),'-B',str(E/'phy_worker.py'),role,phase]+([] if role=='rx' else [expected])
 try:wait_child(root,argv,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),r['stage_seconds'][name]);put(root/'completion.json',dict(status='COMPLETE',actual_wait=pin(root/'actual_wait.json')))
 except BaseException:put(root/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 role=sys.argv[1]
 if role in ('engineering','backend','register'):globals()[role]()
 else:stage(role,sys.argv[2])
