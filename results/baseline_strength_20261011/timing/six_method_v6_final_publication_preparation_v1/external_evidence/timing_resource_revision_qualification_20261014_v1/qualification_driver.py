
import sys, os, json, pathlib, unittest, time
root = pathlib.Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert sys.stdin.readline() == 'START_QUALIFICATION\n'
blocked = []
model_imports = []
protected = [
    '/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM',
    '/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm',
    '/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/assets',
]
model_roots = {'torch','torchvision','tensorflow','jax','transformers','timm','lpips'}
def denied(message):
    blocked.append(message)
    raise AssertionError(message)
def audit(event, args):
    if event == 'import' and args and str(args[0]).split('.')[0] in model_roots:
        model_imports.append(str(args[0])); denied('model import: '+str(args[0]))
    if event in ('subprocess.Popen','os.system','os.exec','os.fork','socket.connect'):
        denied('execution/network: '+event)
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = pathlib.Path(os.fsdecode(args[0])).resolve()
        inside = p == root or root in p.parents
        if not inside and any(str(p) == q or str(p).startswith(q+'/') for q in protected):
            denied('real scientific storage: '+str(p))
        mode = args[1] if len(args)>1 else None
        flags = args[2] if len(args)>2 else 0
        writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and not inside:
            denied('write outside qualification: '+str(p))
sys.addaudithook(audit)
sys.path.insert(0,str(root/'source_snapshot'))
import test_resource_adapter
suite = unittest.defaultTestLoader.loadTestsFromModule(test_resource_adapter)
result = unittest.TextTestRunner(verbosity=2).run(suite)
loaded = sorted(n for n in sys.modules if n.split('.')[0] in model_roots)
print(json.dumps(dict(schema='TIMING_RESOURCE_NEW_SYNTHETIC_QUALIFICATION_V1', tests_run=result.testsRun,
    successful=result.wasSuccessful(), blocked=blocked, model_import_requests=model_imports,
    loaded_model_modules=loaded, real_preflight_called=False, scientific_calls=0,
    real_registration_written=False, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')),sort_keys=True))
assert result.wasSuccessful() and not blocked and not model_imports and not loaded
