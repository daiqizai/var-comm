"""Durable final publication checks and normal commit; no scientific work."""
import os,pathlib,sys,time,traceback
E=pathlib.Path(__file__).parent.parent;sys.path.insert(0,str(E))
import a1_owner as r
PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
root=r.L/'final_publication_format_control_v1';assert not root.exists();root.mkdir()
r.put(root/'supervisor.json',{'process':r.identity(os.getpid()),'parent_process':r.identity(os.getppid()),'argv':sys.argv,'source':r.desc(__file__),'started_unix':time.time()});r.RUN=root
try:
    r.run_stage('child',[PY,'-B',str(pathlib.Path(__file__).parent/'continue.py')],dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),2400)
    r.put(root/'completion.json',{'status':'PASS','wait':r.desc(root/'child.actual_wait.json')})
except BaseException:
    r.put(root/'failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise
