"""One explicitly selected metric stage; never loops or retries a stage."""
from opt_common import *
from launch_registered import launch

def main(role,mode,route,cache_roles=('fit_A',)):
 assert mode in ('three','clip')
 oldpin=read(D/'protocol_resolved_p0.json')['metrics'if mode=='three'else'clip'];old=cr(oldpin);plan=pin(L/(role+'_metric_plan')/'plan.json');p=cr(plan);count=len(p['pairs'])
 code=[v for v in old['code']if Path(v['path']).exists()]+[pin(E/x)for x in ['opt_common.py','metrics.py','control_once.py','launch_registered.py','launch_metric_stage_v2.py']]
 caches=[]
 for name in cache_roles:
  root=L/(name+'_'+mode);w=read(root.with_name(root.name+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
  caches.append(pin(root/'completion.json'))
 r=dict(code=code,plan=plan,metric_registration=oldpin,mode=mode,cpu_affinity=[22,23],caches=caches,caps=dict(load_lpips=1,load_dreamsim=1,embedding=count+len(p['sources']),pair=count)if mode=='three'else dict(load_clip=1,embedding=count+len(p['sources']),pair=count))
 gpu='GPU-f5047136-08e5-9416-3c86-87786d0b355e'
 launch(role+'_'+mode,r,E/'metrics.py',RT/'envs/frozen_visual_py310/bin/python'if mode=='three'else Path('/usr/bin/python'),5400,route,required_gpu_uuid=gpu)
if __name__=='__main__':main(*sys.argv[1:])
