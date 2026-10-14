"""One metadata-only repair, preserving failed entry and all original math."""
from common import *
import ast,sys,subprocess,signal,numpy as np
OUT=T/'gray_pixel_identity_revision_v1';C=T/'gray_pixel_identity_revision_control_v1';AUTH=O/'gray_pixel_identity_revision_authorization.json'
OLD_SHA='3d46de0f15c61de97ba5ada8e6ede79cdb605b840d3b3bc1414bd62eb880c662'
def audit():
 assert sha(E/'gray_prepare.py')==OLD_SHA
 wait=read(T/'gray_control_v1/actual_wait.json');assert wait['returncode']==1 and gone(wait['process']) and gone(wait['parent_process'])
 db=sqlite3.connect('file:'+str(T/'gray/calls.sqlite')+'?mode=ro',uri=True);calls=db.execute('SELECT kind,status,count(*) FROM calls GROUP BY kind,status').fetchall();db.close();assert calls==[('dino_load','COMPLETE',1)]
 rows=[]
 for s in read(O1/'assets/pilot50.json')['records']:
  with np.load(checked(s['archive']),allow_pickle=False) as z:
   a=z['pixels'];assert a.shape==(3,256,256) and a.dtype==np.uint8 and hashlib.sha256(a.tobytes()).hexdigest()==s['preprocessing_id'];rows.append(dict(source_id=s['source_id'],raw_byte_pixel_identity_matches=True))
 return rows
def assembly():
 tree=ast.parse((E/'gray_prepare.py').read_text());oldassert=ast.parse("assert asha(z['pixels'])==s['preprocessing_id']").body[0];newassert=ast.parse("assert hashlib.sha256(z['pixels'].tobytes()).hexdigest()==s['preprocessing_id']").body[0]
 oldout=ast.parse("out=T/'gray'").body[0];newout=ast.parse("out=T/'gray_pixel_identity_revision_v1'").body[0];matches=[0,0]
 class Patch(ast.NodeTransformer):
  def visit_Assert(self,node):
   if ast.dump(node)==ast.dump(oldassert):matches[0]+=1;return ast.copy_location(newassert,node)
   return self.generic_visit(node)
  def visit_Assign(self,node):
   if ast.dump(node)==ast.dump(oldout):matches[1]+=1;return ast.copy_location(newout,node)
   return self.generic_visit(node)
 tree=Patch().visit(tree);assert matches==[1,1];nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))];ns={'__name__':'_LE2_exact_gray_metadata_revision'};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(E/'gray_prepare.py'),'exec'),ns);return ns['run']
def authorize():
 assert not AUTH.exists() and not OUT.exists() and not C.exists();sources=audit();assembly()
 save(AUTH,dict(status='METADATA_CONVENTION_ONLY_ACTUAL_ASSETS_MATCH',reviewer='/root',failure_wait=pin(T/'gray_control_v1/actual_wait.json'),failure_stderr=pin(T/'gray_control_v1/stderr.txt'),failure_ledger=pin(T/'gray/calls.sqlite'),original_entry=pin(E/'gray_prepare.py'),revision_source=pin(E/'gray_pixel_identity_revision_v1.py'),original_pixel_rule=pin(E1/'asset_audit.py'),source_checks=sources,changes=['one pixel metadata assertion: raw byte SHA per original rule','private output directory only'],math_model_preprocess_unchanged=True,previous_source_or_gray_forward_calls=0,necessary_model_reload_disclosed=1,new_source_forward_cap=33,new_gray_pair_cap=33,packet_calls=0,automatic_retry=False))
 print(canonical(pin(AUTH)))
def worker(authsha):
 assert sha(AUTH)==authsha;v=read(AUTH);checked(v['revision_source']);audit()
 try:assembly()(sha(O/'gray_registration.json'))
 except BaseException as e:
  if OUT.exists():save(OUT/'failure.json',dict(error=repr(e),automatic_retry=False))
  raise
def control(authsha):
 checked(read(AUTH)['revision_source']);assert sha(AUTH)==authsha;C.mkdir();argv=[str(RT/'envs/frozen_visual_py310/bin/python'),'-B',str(E/'gray_pixel_identity_revision_v1.py'),'worker',authsha];env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(T/'runtime_cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(T/'runtime_cache/inductor'),TRITON_CACHE_DIR=str(T/'runtime_cache/triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8');start=time.monotonic();timed=terminated=killed=False;parent=process(os.getpid())
 with (C/'stdout.txt').open('xb') as out,(C/'stderr.txt').open('xb') as err:
  child=subprocess.Popen(argv,env=env,cwd=R,stdout=out,stderr=err,start_new_session=True);identity=process(child.pid);assert identity;save(C/'launch.json',dict(argv=argv,process=identity,parent_process=parent,authorization=pin(AUTH)))
  try:rc=child.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   timed=terminated=True;os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=20)
 save(C/'actual_wait.json',dict(actual_wait=True,returncode=rc,elapsed_seconds=time.monotonic()-start,process=identity,parent_process=parent,argv=argv,timed_out=timed,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(C/'stdout.txt'),stderr=pin(C/'stderr.txt')));return rc
if __name__=='__main__':
 if sys.argv[1]=='authorize':authorize()
 elif sys.argv[1]=='worker':worker(sys.argv[2])
 else:raise SystemExit(control(sys.argv[2]))
