
import sys,os,json,pathlib,unittest,tempfile,gzip,tarfile,hashlib,importlib.util,inspect
root=pathlib.Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert sys.stdin.readline()=='START_QUALIFICATION\n'
blocked=[];imports=[]
protected=('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM','/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm','/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/assets')
models={'torch','torchvision','tensorflow','jax','transformers','timm','lpips'}
def deny(s):blocked.append(s);raise AssertionError(s)
def audit(event,args):
    if event=='import' and args and str(args[0]).split('.')[0] in models:
        imports.append(str(args[0]));deny('model import '+str(args[0]))
    if event in ('subprocess.Popen','os.system','os.exec','os.fork','socket.connect','os.symlink'):deny('execution/network/symlink '+event)
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=pathlib.Path(os.fsdecode(args[0])).resolve();inside=p==root or root in p.parents
        if not inside and any(str(p)==x or str(p).startswith(x+'/') for x in protected):deny('real storage '+str(p))
        mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and not inside:deny('outside qualification write '+str(p))
sys.addaudithook(audit)
sys.path.insert(0,str(root/'source_snapshot'))
suite=unittest.defaultTestLoader.discover(str(root/'source_snapshot'),pattern='test_*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
loaded=sorted(n for n in sys.modules if n.split('.')[0] in models)
print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,blocked=blocked,model_import_requests=imports,loaded_model_modules=loaded,scientific_calls=0,production_preflight_called=False)))
assert result.testsRun>0 and result.wasSuccessful() and not blocked and not imports and not loaded
