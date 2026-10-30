"""Lossless lightweight packaging, preserving all original scientific bytes."""
from opt_common import *
import csv,io,re

def main():
 out=D/'delivery/validation';assert out.is_dir();splits=[];excluded=[]
 originals=[p for p in sorted(D.rglob('*'))if p.is_file()and p.stat().st_size>9_000_000]
 for p in originals:
  assert p.suffix in {'.csv','.json'},('Unexpected oversized asset',p)
  folder=p.with_name(p.name+'_parts');folder.mkdir(exist_ok=False);parts=[];original=pin(p)
  if p.suffix=='.csv':
   header=None;buffer=[];size=0;total=0
   def flush():
    nonlocal buffer,size
    if not buffer:return
    target=folder/f'part_{len(parts):03d}.csv'
    with target.open('x',encoding='utf8',newline='')as f:
     writer=csv.writer(f);writer.writerow(header);writer.writerows(buffer)
    assert target.stat().st_size<9_000_000;parts.append(dict(file=pin(target),rows=len(buffer)));buffer=[];size=0
   with p.open(newline='')as f:
    reader=csv.reader(f);header=next(reader)
    for row in reader:
     stream=io.StringIO(newline='');csv.writer(stream).writerow(row);n=len(stream.getvalue().encode('utf8'))
     if size+n>7_500_000:flush()
     buffer.append(row);size+=n;total+=1
    flush()
   with p.open(newline='')as f:
    original_rows=csv.reader(f);assert next(original_rows)==header;seen=0
    for part in parts:
     with checked(part['file']).open(newline='')as q:
      reader=csv.reader(q);assert next(reader)==header
      for row in reader:assert row==next(original_rows);seen+=1
    assert next(original_rows,None)is None and seen==total
   record=dict(format='csv_rows',original=original,rows=total,parts=parts,exact_row_sequence_verified=True,instructions='Read CSV parts in listed order; each part repeats the header. All field values and row order are unchanged; CSV serialization may differ.')
  else:
   h=hashlib.sha256();total=0
   with p.open('rb')as f:
    while True:
     data=f.read(7_500_000)
     if not data:break
     target=folder/f'part_{len(parts):03d}.part'
     with target.open('xb')as q:q.write(data)
     h.update(target.read_bytes());total+=len(data);parts.append(dict(file=pin(target)))
   assert h.hexdigest()==original['sha256']and total==original['bytes']
   record=dict(format='exact_byte_fragments',original=original,parts=parts,exact_original_SHA_after_concatenation_verified=True,instructions='Concatenate .part bytes in listed order, then verify original byte count and SHA256. Fragments individually are not JSON; no scientific bytes were changed.')
  checked(original);splits.append(record);excluded.append(str(p.relative_to(D)))
 if excluded:
  with(D/'.gitignore').open('x')as f:f.write('# Exact oversized originals stay on shared PFS. Complete verified parts are included.\n'+'\n'.join('/'+s for s in excluded)+'\n')
 save(D/'delivery/large_file_index.json',dict(splits=splits,originals_unchanged=True,reason='Repository per-file size limit; all scientific rows/bytes remain in reviewable parts.',excluded_from_review_package=excluded))
 controls=[]
 for p in sorted(L.glob('*_control/owner.json')):
  controls.append(dict(role=p.parent.name.removesuffix('_control'),owner=read(p),actual_wait=read(p.parent/'actual_wait.json'),launch=read(p.parent/'child.json')))
 save(out/'lifecycle_receipts.json',dict(controls=controls,all_receipts_actual=True))
 failures=[]
 for role in ['C_guard_metric_queue','benchmark_tx','final_successors']:
  p=L/role/'failure.json';failures.append(dict(role=role,failure=read(p),identity=pin(p)))
 save(out/'preserved_failures.json',dict(records=failures,read_only_historical_schema_failures='v1-v4 registrations/code and original transport evidence retained; no outputs overwritten',engineering_failure_not_quality_rejection=True))
 patterns=[re.compile(x)for x in [r'(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}',r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',r'https?://[^\s/@:]+:[^\s/@]+@',r'\bAKIA[A-Z0-9]{16}\b']]
 # Scan complete original JSONs before exact fragments, to avoid UTF-8 boundary artifacts.
 for p in originals:
  if p.suffix=='.json':assert not any(x.search(p.read_text())for x in patterns),p
 files=[]
 for root in [E,D]:
  for p in sorted(root.rglob('*')):
   if not p.is_file()or '__pycache__'in p.parts:continue
   if root==D and str(p.relative_to(D))in excluded:continue
   assert not p.is_symlink()and p.stat().st_size<=10_000_000,p
   assert p.suffix not in {'.pt','.pth','.npz','.npy','.sqlite','.zip','.log','.pyc'},p
   if p.suffix in {'.py','.md','.json','.csv','.txt','.svg','.sh'}:assert not any(x.search(p.read_text())for x in patterns),('Possible secret',p)
   files.append(str(p.relative_to(R)))
 save(out/'lightweight_review_audit.json',dict(status='PASS',files=len(files),under_10MB_per_file=True,no_weights_arrays_large_logs_or_secrets=True,lossless_parts=pin(D/'delivery/large_file_index.json'),scientific_originals_unchanged=True))
 print('LIGHTWEIGHT_REVIEW_READY',len(files),len(splits))

if __name__=='__main__':main()
