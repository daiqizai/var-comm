"""Explicit seven-file storage migration and verified resolution for future runs.

Historical registrations are immutable. Consumers opt into the relocation map;
there are no symlinks, silent byte replacements, or GPU calls here.
"""
from pathlib import Path
import ast,copy,hashlib,json,os,time
import shutil

CODE_ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu')
DEST=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/code_assets_20261028')
RECEIPT=DEST/'migration_complete.json'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def pin(path):
 p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def save(path,value):
 with Path(path).open('x',encoding='utf8')as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def migrate(plan_path):
 plan=json.loads(Path(plan_path).read_bytes());assert len(plan['files'])==7
 DEST.mkdir(parents=True,exist_ok=False)
 rows=[]
 # Validate the complete set before changing the first pathname.
 for item in plan['files']:
  p=Path(item['path']);assert p.is_relative_to(CODE_ROOT)and p.resolve()==p and p.is_file()and not p.is_symlink()
  s=p.stat();assert (s.st_size,s.st_ino,s.st_mtime_ns)==(item['bytes'],item['inode'],item['mtime_ns'])
  assert s.st_dev==DEST.stat().st_dev and s.st_nlink==1
  digest=sha(p);assert digest==item['expected_sha256'],p
  target=DEST/p.relative_to(CODE_ROOT);assert target.is_relative_to(DEST)and not target.exists()
  rows.append(dict(old_path=str(p),path=str(target),bytes=s.st_size,sha256=digest,inode=s.st_ino,mtime_ns=s.st_mtime_ns))
 save(DEST/'migration_plan.json',dict(source_plan=pin(plan_path),files=rows,created_unix=time.time()))
 with(DEST/'migration_journal.jsonl').open('x',encoding='utf8')as journal:
  for row in rows:
   src=Path(row['old_path']);dst=Path(row['path']);dst.parent.mkdir(parents=True,exist_ok=True)
   s=src.stat();assert(s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns'])and not dst.exists()
   os.rename(src,dst)
   journal.write(json.dumps(dict(event='RENAMED',**row))+'\n');journal.flush();os.fsync(journal.fileno())
   assert not src.exists()and not dst.is_symlink()and dst.stat().st_ino==row['inode']and sha(dst)==row['sha256']
   journal.write(json.dumps(dict(event='VERIFIED',old_path=str(src),path=str(dst),time=time.time()))+'\n');journal.flush();os.fsync(journal.fileno())
 save(RECEIPT,dict(status='COMPLETE',files=rows,count=len(rows),bytes=sum(r['bytes']for r in rows),old_paths_absent=True,symlinks_created=0,checkpoint_bytes_changed=False,training_or_inference_calls=0,finished_unix=time.time()))
 return json.loads(RECEIPT.read_bytes())

def mapping():
 d=json.loads(RECEIPT.read_bytes());assert d['status']=='COMPLETE'and d['count']==7
 return {r['old_path']:r for r in d['files']}

def copy_move_after_exdev(plan_path):
 """PFS exposes one device ID but refuses rename across its storage classes.

 Only proceeds from the recorded zero-move EXDEV attempt. Copy to a fresh file,
 flush, verify SHA, and only then unlink that exact unchanged source file.
 """
 p=json.loads((DEST/'migration_plan.json').read_bytes())
 assert p['source_plan']==pin(plan_path)and not RECEIPT.exists()
 assert (DEST/'migration_journal.jsonl').read_bytes()==b''
 rows=p['files']
 assert len(rows)==7
 for row in rows:
  src=Path(row['old_path']);dst=Path(row['path'])
  assert src.is_relative_to(CODE_ROOT)and dst.is_relative_to(DEST)and not dst.exists()and not src.is_symlink()
  s=src.stat();assert(s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns'])and sha(src)==row['sha256']
 with(DEST/'copy_move_journal.jsonl').open('x',encoding='utf8')as journal:
  for row in rows:
   src=Path(row['old_path']);dst=Path(row['path']);dst.parent.mkdir(parents=True,exist_ok=True)
   with src.open('rb')as f,dst.open('xb')as g:
    shutil.copyfileobj(f,g,8<<20);g.flush();os.fsync(g.fileno())
   shutil.copystat(src,dst,follow_symlinks=False)
   assert sha(dst)==row['sha256']and dst.stat().st_size==row['bytes']
   s=src.stat();assert(s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns'])
   journal.write(json.dumps(dict(event='DESTINATION_VERIFIED_BEFORE_SOURCE_REMOVAL',**row))+'\n');journal.flush();os.fsync(journal.fileno())
   src.unlink()
   assert not src.exists()and dst.is_file()
   journal.write(json.dumps(dict(event='MOVED',**row))+'\n');journal.flush();os.fsync(journal.fileno())
 save(RECEIPT,dict(status='COMPLETE',files=rows,count=len(rows),bytes=sum(r['bytes']for r in rows),old_paths_absent=True,symlinks_created=0,checkpoint_bytes_changed=False,training_or_inference_calls=0,method='COPY_FSYNC_SHA256_VERIFY_THEN_UNLINK_EXACT_SOURCE',initial_rename_EXDEV_preserved=True,finished_unix=time.time()))
 return json.loads(RECEIPT.read_bytes())

def resolve(old):
 value=str(old);row=mapping().get(value)
 if row is None:return Path(value)
 p=Path(row['path']);assert p.is_relative_to(DEST)and not p.is_symlink()and p.stat().st_size==row['bytes']and sha(p)==row['sha256']
 return p

def relocate_value(value):
 """Only known weight paths move. Never changes source IDs, model hashes or historical files."""
 mp=mapping()
 def visit(x):
  if isinstance(x,str):return mp[x]['path']if x in mp else x
  if isinstance(x,list):return [visit(y)for y in x]
  if isinstance(x,dict):return {k:visit(v)for k,v in x.items()}
  return x
 return visit(value)

def metric_assets(original_module,config):
 """Original bound_assets with exactly its weight pathname lookup relocated.

 Original metadata/constants, source pins, numeric flags and all byte checks stay
 intact; original modules and historical metadata are never modified.
 """
 p=Path(original_module.__file__)
 assert sha(p)=='6bde19df620a7dbf391f5d406daa3b4c1de2a66d051c990c26ae9a2474c1ec8d'
 node=copy.deepcopy(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='bound_assets'))
 matches=[n for n in ast.walk(node)if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='path']
 assert len(matches)==1 and ast.unparse(matches[0].value)=="Path(row['actual_path'])"
 matches[0].value.func=ast.Name(id='_checkpoint_path',ctx=ast.Load())
 ns=dict(original_module.__dict__);ns['_checkpoint_path']=resolve
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),'exec'),ns)
 return ns['bound_assets'](config)

if __name__=='__main__':
 import sys
 if len(sys.argv)==3 and sys.argv[1]=='migrate':print(json.dumps(migrate(sys.argv[2])))
 else:raise SystemExit('Usage: checkpoint_relocation_20261028.py migrate EXACT_REVIEWED_PLAN.json')
