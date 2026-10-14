from pathlib import Path
import sys,os,importlib.util,importlib.machinery,unittest,tempfile,json,ast,hashlib,tarfile,gzip,sqlite3,subprocess,contextlib,sysconfig
from unittest.mock import patch
sys.dont_write_bytecode=True
q=Path(__file__).resolve().parent
fixture=q/'synthetic_tmp';fixture.mkdir(exist_ok=False)
tempfile.tempdir=str(fixture)
os.environ['R1_PROBE_LINEAGE_SOURCE_FIXTURES']=str(q/'fixtures')
def norm(x):return os.path.abspath(os.fspath(x))
def within(p,r):return p==r or p.startswith(r+os.sep)
qroot=norm(q)
libroots={norm(p) for p in (sys.base_prefix,sys.prefix,sysconfig.get_path('stdlib'),sysconfig.get_path('platstdlib'),sysconfig.get_path('purelib'),sysconfig.get_path('platlib')) if p}
blocked=[];counters={'read_opens':0,'write_opens':0,'directory_reads':0}
def reject(event,path):
    blocked.append({'event':event,'path':str(path)})
    raise RuntimeError('Original CPY synthetic boundary rejected '+event+': '+str(path))
def guard(event,args):
    if event=='open' and not isinstance(args[0],int):
        p=norm(args[0]);mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing:
            counters['write_opens']+=1
            if not within(p,qroot):reject(event,p)
        else:
            counters['read_opens']+=1
            if not within(p,qroot) and not any(within(p,r) for r in libroots):reject(event,p)
    elif event in ('os.listdir','os.scandir') and not isinstance(args[0],int):
        p=norm(args[0]);counters['directory_reads']+=1
        if not within(p,qroot) and not any(within(p,r) for r in libroots):reject(event,p)
    elif event in ('os.remove','os.rmdir','os.mkdir','os.rename','os.replace','os.symlink','os.link'):
        paths=args[:2] if event in ('os.rename','os.replace','os.symlink','os.link') else args[:1]
        for raw in paths:
            if isinstance(raw,(str,bytes,os.PathLike)):
                p=norm(raw)
                if not within(p,qroot):reject(event,p)
    elif event=='import' and args[0].split('.')[0] in {'torch','tensorflow','numpy','sionna','lpips','transformers','timm','triton','matplotlib'}:reject(event,args[0])
    elif event in ('subprocess.Popen','os.system','socket.connect','socket.bind','sqlite3.connect'):reject(event,args[0] if args else '')
# Source-only fixture import avoids consulting cached executable code.
importlib.machinery.SourceFileLoader.get_code=lambda loader,fullname: loader.source_to_code(loader.get_data(loader.path),loader.path)
sys.path[:]=[p for p in sys.path if p and (within(norm(p),qroot) or any(within(norm(p),r) for r in libroots))]
sys.addaudithook(guard)
spec=importlib.util.spec_from_file_location('original_CPY_probe_lineage_tests',q/'source_snapshot/test_adapter.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
print(json.dumps(dict(schema='R1_PROBE_LINEAGE_ORIGINAL_CPY_SYNTHETIC_BOUNDARY_V1',testsRun=result.testsRun,failures=len(result.failures),errors=len(result.errors),blocked=blocked,counters=counters,scientific_calls=0,models_imported=[x for x in ('torch','tensorflow','numpy','sionna','lpips','transformers','timm','triton','matplotlib') if x in sys.modules],real_preflight_invoked=False,real_grid_audit_invoked=False,allowed_qualification_root=str(q),interpreter_roots=sorted(libroots),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')),sort_keys=True),flush=True)
sys.exit(0 if result.wasSuccessful() and result.testsRun==34 and not blocked else 1)
