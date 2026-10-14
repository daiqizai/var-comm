"""Metadata-only field-name revision; no scientific retry or hot edit."""
from common import *
import ast,sys,io,shutil,subprocess,signal,traceback
AUTH=O/'cpu_delivery_metadata_revision_authorization.json';C=T/'cpu_delivery_metadata_revision_control_v2';D=O/'delivery_v2'
OLD_SHA='e413d9f8c8a0114b8d34998c023fe6b91f81c4ae619e74d21c09e753f2052404'
def assembly():
 path=E/'cpu_delivery_v1.py';assert sha(path)==OLD_SHA;tree=ast.parse(path.read_text());matches=[0,0]
 class Patch(ast.NodeTransformer):
  def visit_Call(self,node):
   if isinstance(node.func,ast.Name) and node.func.id=='dict' and {'gid','snr_db','n','N'}<={k.arg for k in node.keywords if k.arg}:
    for kw in node.keywords:
     if kw.arg=='n':kw.arg='n_coded_bits';matches[0]+=1
   return self.generic_visit(node)
  def visit_Constant(self,node):
   if node.value=='cpu_delivery_launch.json':node.value='cpu_delivery_metadata_revision_launch.json';matches[1]+=1
   return node
 tree=Patch().visit(tree);assert matches==[1,1];ns={'__name__':'_LE2_CPU_metadata_revision'};exec(compile(ast.fix_missing_locations(ast.Module(body=tree.body[:-1],type_ignores=[])),str(path),'exec'),ns);ns['D']=D
 old_csv=ns['csv_union']
 def csv_route(path,rows):
  if Path(path).name=='measured_policy_results.csv':
   buf=io.StringIO(newline='');names=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(buf,fieldnames=names);w.writeheader();w.writerows(rows);expected=buf.getvalue().encode('utf8');existing=O/'delivery_v1/measured_policy_results.csv';assert existing.read_bytes()==expected
   with Path(path).open('xb') as f:f.write(expected);f.flush();os.fsync(f.fileno())
   save(D/'successful_existing_CSV_exact_reuse.json',dict(source=pin(existing),copy=pin(path),original_unchanged=True,zero_scientific_calls=True))
  else:old_csv(path,rows)
 ns['csv_union']=csv_route
 # Exercise the transformed real cell-expression, including both physical
 # code length and trial count. This directly regresses the second defect.
 comp=next(n for n in ast.walk(tree) if isinstance(n,ast.ListComp) and isinstance(n.elt,ast.Call) and any(k.arg=='n_coded_bits' for k in n.elt.keywords));expr=ast.Expression(body=comp)
 fake=eval(compile(ast.fix_missing_locations(expr),str(path),'eval'),dict(table={('g',7):dict(n=128,ci=[.1,.9],n_correct=64)},layouts={'g':dict(N=64,q=4,nominal_rate='1/2',k=128,n=256)}))
 assert fake[0]['n_coded_bits']==256 and fake[0]['n']==128
 fake=eval(compile(ast.fix_missing_locations(expr),str(path),'eval'),dict(table={('header',7):dict(n=512,ci=[.1,.9],n_correct=500)},layouts={}))
 assert fake[0]['n_coded_bits']==136 and fake[0]['n']==512
 return ns['run']
def validate():
 w=read(T/'cpu_delivery_control_v1/actual_wait.json');assert w['returncode']==1 and gone(w['process']) and gone(w['parent_process'])
 assert "TypeError: dict() got multiple values for keyword argument 'n'" in read(T/'cpu_delivery_failure.json')['error']
 assert sha(E/'cpu_delivery_v1.py')==OLD_SHA;assembly()
def authorize():
 assert not AUTH.exists() and not C.exists() and not D.exists();validate()
 save(AUTH,dict(status='CPU_METADATA_ONLY_TWO_COLUMN_MEANINGS',reviewer='/root',source=pin(E/'cpu_delivery_metadata_revision_v2.py'),original_CPU_source=pin(E/'cpu_delivery_v1.py'),original_CPU_authorization=pin(O/'cpu_delivery_authorization.json'),failed_CPU_wait=pin(T/'cpu_delivery_control_v1/actual_wait.json'),failed_CPU_evidence=pin(T/'cpu_delivery_failure.json'),successful_existing_CSV=pin(O/'delivery_v1/measured_policy_results.csv'),changes=['physical n column renamed n_coded_bits; preserved trial count n','new output/launch receipt paths only'],real_transformed_cell_expression_synthetic_tests=2,source_math_and_probabilities_unchanged=True,encode_decode_channel_model_score_calls=0,successful_existing_CSV_reused_byte_exact=True,automatic_retry=False));print(canonical(pin(AUTH)))
def worker(expected):
 assert sha(AUTH)==expected;auth=read(AUTH);checked(auth['source']);validate();assembly()(sha(O/'cpu_delivery_authorization.json'))
 save(D/'metadata_revision.json',dict(authorization=pin(AUTH),actual_wrapper=pin(E/'cpu_delivery_metadata_revision_v2.py'),original_CPU_source=pin(E/'cpu_delivery_v1.py'),original_CPU_returncode=1,packet_original_returncode=1,zero_new_scientific_calls=True,physical_column='n_coded_bits',sample_count_column='n'))
def control(expected):
 assert sha(AUTH)==expected;C.mkdir();argv=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'cpu_delivery_metadata_revision_v2.py'),'worker',expected];start=time.monotonic();parent=process(os.getpid());timed=terminated=killed=False
 with (C/'stdout.txt').open('xb') as out,(C/'stderr.txt').open('xb') as err:
  child=subprocess.Popen(argv,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=err,start_new_session=True);identity=process(child.pid);assert identity;save(C/'launch.json',dict(argv=argv,process=identity,parent_process=parent,authorization=pin(AUTH)))
  try:rc=child.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   timed=terminated=True;os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=20)
 save(C/'actual_wait.json',dict(actual_wait=True,returncode=rc,elapsed_seconds=time.monotonic()-start,process=identity,parent_process=parent,argv=argv,timed_out=timed,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(C/'stdout.txt'),stderr=pin(C/'stderr.txt')));return rc
if __name__=='__main__':
 if sys.argv[1]=='authorize':authorize()
 elif sys.argv[1]=='control':raise SystemExit(control(sys.argv[2]))
 else:
  try:worker(sys.argv[2])
  except BaseException:
   save(T/'cpu_delivery_metadata_revision_failure.json',dict(error=traceback.format_exc(),automatic_retry=False));raise
