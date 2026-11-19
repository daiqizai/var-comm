"""Verify the old Pilot50 by content. Does not run or substitute a model."""
import argparse,csv,hashlib,io,json,tarfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from profiles import OFF, QUARTER

PILOT='results/layered_entropy_feasibility_20261014_v1/assets/pilot50.json'
CAL='results/layered_mismatch_raw_20261013/r0_reuse/cal200_sources.json'
LE='results/layered_entropy_feasibility_20261014_v1/'
CAND='results/nestcom_unified_calibration_20261103_v1/pilot_calibration/candidate_scores.csv'
BLOBS={LE+'clean_quality.csv':'fc373ad826c477ff03b63739df7ee377f72132a6',LE+'source_lengths.csv':'821d50b391be7c2643e7472105e669cdb249b707',CAND:'7ee2710c063f6b402e01c6fc64c88054834f9df3'}
sha=lambda b:hashlib.sha256(b).hexdigest()
def check(ok,text):
 if not ok:raise ValueError(text)
def blob(o):return hashlib.sha1(f'blob {len(o)}\0'.encode()+o).hexdigest()
def tok(a):return sha(f'int64:{len(a)}\0'.encode()+np.asarray(a,dtype='<i8').tobytes())
def rows(root,key):
 b=(root/key).read_bytes();match=blob(b)==BLOBS[key]
 if not match:b=b.replace(b'\r\n',b'\n')
 check(blob(b)==BLOBS[key],key);return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def writecsv(path,rs):
 with path.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--csv-root',type=Path,required=True);p.add_argument('--asset-tar',type=Path,action='append',default=[]);p.add_argument('--t6',type=Path,action='append',default=[]);p.add_argument('--original-dir',type=Path,action='append',default=[]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 def immutable(path,expected):
  b=(a.metadata/path).read_bytes();check(sha(b)==expected,str(path));return json.loads(b)
 pilots=immutable(PILOT,'b89a964f24127d9fcd2bddd6b7a88d8b366f53764632a71d93dfcf3ffcd94853');parent=immutable(CAL,pilots['cal200']['sha256']);reg=pilots['records'];check(len(reg)==50 and len(parent['records'])==200 and not pilots['selection_uses_quality'],'population')
 domain=pilots['domain'];rank=lambda r:sha(json.dumps([domain,r['source_id']],separators=(',',':')).encode());ordered=sorted(parent['records'],key=lambda r:(rank(r),r['source_id']))[:50]
 clean=rows(a.csv_root,LE+'clean_quality.csv');lens=rows(a.csv_root,LE+'source_lengths.csv');cand=rows(a.csv_root,CAND)
 check(len(clean)==1200 and len(lens)==3400,'LE rows');ident=sorted(set(r['model_identity'] for r in clean));check(len(ident)==1,'mixed model');report=[]
 with ExitStack() as stack:
  tars=[stack.enter_context(tarfile.open(t)) for t in a.asset_tar];assets={}
  for t in tars:
   for m in t.getmembers():
    if '/full1000_assets/' in m.name and m.isfile():
     key=m.name.split('/full1000_assets/')[-1];check(key not in assets,'duplicate lookup');assets[key]=(t,m)
  def locate(k):
   if k in assets:
    t,m=assets[k];return t.extractfile(m).read()
   return next(((d/k).read_bytes() for d in a.original_dir if (d/k).is_file()),None)
  for i,(r,par) in enumerate(zip(reg,ordered)):
   check(r['pilot_slot']==i and r['pilot_selection_sha256']==rank(par) and all(r[k]==par[k] for k in ('source_id','source_index','preprocessing_id','tokens_sha256','archive','checkpoint')),'selection identity');sid=r['source_id'];n=f"{r['source_index']:04d}";cp=locate('source_checkpoints/'+n+'.json');raw=locate('sources/'+n+'.npz');pixel=False;token_ok=False
   if cp is not None:check(sha(cp)==r['checkpoint']['sha256'],'original checkpoint '+n)
   if raw is not None:
    check(sha(raw)==r['archive']['sha256'],'original NPZ '+n)
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
     px=z['pixels'];q=z['tokens'];check(px.shape==(3,256,256) and px.dtype==np.uint8 and sha(px.tobytes())==r['preprocessing_id'] and q.dtype==np.int64 and q.shape==(680,) and tok(q)==r['tokens_sha256'],'original arrays '+n);pixel=token_ok=True
   qr=[v for v in clean if v['source_id']==sid];lr=[v for v in lens if v['source_id']==sid];check(len(qr)==24 and {(int(v['m']),int(v['K'])) for v in qr}==set(QUARTER[:-1]) and all(int(v['pilot_slot'])==i and np.isfinite(float(v['psnr_db'])) and np.isfinite(float(v['dinov2_vitl14_cosine'])) for v in qr),'clean source '+n);edges={(j,m,k) for j in (4,5,6,7) for m,k in QUARTER[:-1] if OFF[m]+k>OFF[j]};check(len(lr)==68 and edges=={(int(v['j']),int(v['m']),int(v['K'])) for v in lr} and all(int(v['pilot_slot'])==i for v in lr),'length identities '+n)
   matches={};found=[d for d in a.t6 if (d/'sources'/(n+'.npz')).exists()];whole=0;full=None
   for d in found:
    rr=json.loads((d/'source_checkpoints'/(n+'.json')).read_bytes());check(rr['source_id']==sid and rr['source_index']==r['source_index'] and rr['source_assets_checkpoint']['sha256']==r['checkpoint']['sha256'] and rr['preprocessing_id']==r['preprocessing_id'] and rr['source_tokens_sha256']==r['tokens_sha256'] and rr['m10_independent_roundtrip'],'T6 identity '+n);b=(d/'sources'/(n+'.npz')).read_bytes();check(sha(b)==rr['archive']['sha256'],'T6 file '+n)
    with np.load(io.BytesIO(b),allow_pickle=False) as z:
     tq=z['tokens'];check(tok(tq)==r['tokens_sha256'],'T6 source tokens');token_ok=True;wcount=0
     for scale,rec in rr['streams']['EC_VAR_WHOLE'].items():
      s=int(scale);bits=z[rec['bits_key']];decoded=z[rec['received_tokens_key']];check(bits.dtype==np.uint8 and (bits<=1).all() and bits.size==rec['payload_bits'] and sha(b'uint8_bits\0'+bits.tobytes())==rec['payload_sha256'] and tok(decoded)==rec['received_tokens_sha256'] and np.array_equal(decoded,tq[:OFF[s]]) and rec['receiver_cache_proof']['canonical'] and rec['receiver_cache_proof']['independent_roundtrip'],'T6 stream '+n);wcount+=1
      if s==10:full=len(bits)
      if s in (6,7):matches[s]=int(all(int(v['b_base'])==len(bits) for v in lr if int(v['j'])==s))
     whole=max(whole,wcount)
   report.append(dict(pilot_slot=i,source_id=sid,source_index=n,source_pixel_sha256=r['preprocessing_id'],source_tokens_sha256=r['tokens_sha256'],original_npz_verified=int(pixel),original_checkpoint_verified=int(cp is not None),tokens_verified=int(token_ok),verified_T6_whole_streams=whole,T6_m6_length_matches_LE=matches.get(6),T6_m7_length_matches_LE=matches.get(7),T6_whole_m10_bits=full,old_split_length_rows=len(lr),old_clean_dino_psnr_rows=len(qr)))
 check(len(set(r['source_id'] for r in report))==50,'duplicate source');scopes={s:dict(Counter(r['method'] for r in cand if r['snr_db']==s)) for s in sorted({r['snr_db'] for r in cand},key=int)}
 out=dict(baseline='564c18059dbe6c80ada4c111df01bdd8320fc52b',manifest_sha256=sha((a.metadata/PILOT).read_bytes()),cal200_sha256=pilots['cal200']['sha256'],domain=domain,git_input_blobs=BLOBS,source_count=len(reg),verified_native_pixel_sources=sum(r['original_npz_verified'] for r in report),verified_original_checkpoints=sum(r['original_checkpoint_verified'] for r in report),verified_token_sources=sum(r['tokens_verified'] for r in report),verified_T6_whole_streams=sum(r['verified_T6_whole_streams'] for r in report),T6_m6_length_matches_LE=sum(r['T6_m6_length_matches_LE'] or 0 for r in report),T6_m7_length_matches_LE=sum(r['T6_m7_length_matches_LE'] or 0 for r in report),T6_payloads_admitted_as_SP_enhancement=0,old_split_lengths=len(lens),old_clean_dino_psnr=len(clean),old_clean_model_identity=ident[0],old_clean_dists_dreamsim_rows=0,old_split_to_m10_rows=0,split_to_m10_planned=150,split_to_m10_including_fallback_j4=200,archived_catalog_counts_by_snr=scopes,sp1_qualified=False)
 writecsv(a.out/'pilot50_asset_inventory.csv',report);(a.out/'pilot50_audit.json').open('x',encoding='utf8').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
