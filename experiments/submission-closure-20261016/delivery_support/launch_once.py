import sys,subprocess,os
from pathlib import Path
E=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/submission-closure-20261016');sys.path.insert(0,str(E));from common import *
for role,regname,expected in [('source_audit','source_registration.json','13c9057b42cdc9d36a8934b2479e8a9385073d5205402852a355e32803c60845'),('fixed16_score','metric_registration.json','16b31b8d62bdc91170ca7b5e5c3057b70db40cba86cfab7dd9c1a0f91f980c04')]:
 assert sha(O/regname)==expected and not(T/(role+'_control_v1')).exists()
 initial=T/(role+'_initial_v1');initial.mkdir(exist_ok=False)
 argv=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control.py'),role,expected]
 save(initial/'prepared.json',dict(argv=argv,registration=pin(O/regname),control=pin(E/'control.py')))
 with(initial/'stdout.txt').open('xb')as so,(initial/'stderr.txt').open('xb')as se:
  p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=so,stderr=se,cwd=R,start_new_session=True);identity=pid_identity(p.pid);assert identity
  save(initial/'launch.json',dict(process=identity,parent_process=pid_identity(os.getpid()),argv=argv));print(role,identity)
