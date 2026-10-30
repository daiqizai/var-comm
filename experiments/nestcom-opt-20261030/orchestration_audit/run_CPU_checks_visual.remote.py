import sys,subprocess,time
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
os.sched_setaffinity(0,{26,27});out=L/'repository_checks_visual_env';out.mkdir(exist_ok=False);args=[str(RT/'envs/frozen_visual_py310/bin/python'),'tools/run_cpu_checks.py'];start=time.time()
p=subprocess.run(args,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=240)
(out/'stdout.txt').write_text(p.stdout);(out/'stderr.txt').write_text(p.stderr);save(D/'CPU_checks_visual_environment.json',dict(command=args,returncode=p.returncode,seconds=time.time()-start,stdout=pin(out/'stdout.txt'),stderr=pin(out/'stderr.txt'),prior_missing_torchvision_failure_preserved=pin(D/'repository_checks.json'),reason='Use existing visual environment with required torchvision; no installation or scientific run'))
print(p.returncode,p.stdout[-1500:],p.stderr[-700:])