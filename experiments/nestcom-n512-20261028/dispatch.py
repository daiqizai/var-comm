"""Exclusive bounded child owner; callers must seal a fresh stage registration."""
from n512_common import *
import subprocess

def launch(role,rp,entry,gpu=None,seconds=None):
 assert role.isidentifier() and not (L/'controls'/role).exists() and not (L/(role+'_dispatch.json')).exists()
 r=read(rp);[checked(d)for d in r['code']];environment={}
 if gpu is not None:
  line=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();v=[x.strip()for x in line.split(',')]
  assert int(v[2])>=60000 and int(v[3])<=25
  assert v[1].lower().lstrip('0')==r['pci'].lower().lstrip('0');environment['CUDA_VISIBLE_DEVICES']=str(gpu)
 else:line=None
 python=RT/'envs'/('frozen_visual_py310'if gpu is not None else'frozen_ldpc_py311')/'bin/python'
 rq=L/'control_requests'/(role+'.json');save(rq,dict(control=str(L/'controls'/role),argv=[str(python),'-B',str(entry),str(rp),sha(rp)],environment=environment,seconds=seconds or r['max_seconds']+60,code=r['code']+[pin(E/'control.py')]))
 with (L/(role+'_owner.log')).open('xb')as f:p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control.py'),str(rq),sha(rq)],cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
 save(L/(role+'_dispatch.json'),dict(process=ident(p.pid),request=pin(rq),host=os.uname().nodename,GPU_observation=line));return read(L/(role+'_dispatch.json'))
