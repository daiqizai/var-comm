import sys,subprocess,time
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
# Pure CPU engineering/repository checks, never scientific launchers.
os.sched_setaffinity(0,{26,27});out=L/'repository_checks';out.mkdir(exist_ok=False);results=[]
for name,args in [('verify_repository',[sys.executable,'tools/verify_repository.py']),('run_cpu_checks',[sys.executable,'tools/run_cpu_checks.py']),('new_C_contracts',[sys.executable,str(E/'test_c_contracts.py')])]:
 start=time.time();p=subprocess.run(args,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=240)
 (out/(name+'.stdout.txt')).write_text(p.stdout);(out/(name+'.stderr.txt')).write_text(p.stderr)
 results.append(dict(name=name,command=args,returncode=p.returncode,seconds=time.time()-start,stdout=pin(out/(name+'.stdout.txt')),stderr=pin(out/(name+'.stderr.txt'))));print(name,p.returncode,p.stdout[-500:],p.stderr[-350:])
save(D/'repository_checks.json',dict(results=results,purpose='CPU engineering checks only; not real image quality',HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),Git_write_actions=False))