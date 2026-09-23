#!/usr/bin/env python3
"""Read-only historical reference audit; never rewrites old evidence or uses CUDA."""
import argparse,csv,hashlib,json,math,os,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXTERNAL=ROOT/'outputs/EXTERNAL-BASELINE-POSITIONING-20260916'

def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_text(json.dumps(data,indent=2,allow_nan=False));temporary.replace(p)

def resolve_archive(value):
    p=Path(value)
    legacy=Path('/workspace/projects/VAR_COMM')
    if p.is_relative_to(legacy):p=ROOT/p.relative_to(legacy)
    if not p.is_absolute():p=ROOT/p
    resolved=p.resolve()
    if not resolved.is_relative_to((ROOT/'outputs').resolve()) or resolved.suffix!='.npz' or not resolved.is_file():raise ValueError('reference archive outside existing project outputs or missing')
    return resolved

def validate_rows(data,sources):
    seen=set();groups=defaultdict(list)
    for r in data:
        method=r['method'];protocol=r.get('protocol') or 'legacy_paid_information';source=r['image_id'];snr=float(r['snr_db']);seed=int(r['seed'])
        key=(method,protocol,source,snr,seed)
        if key in seen:raise ValueError('duplicate source/SNR/noise/method/protocol key')
        seen.add(key)
        if source not in sources or r['source_pixels_sha256']!=sources[source] or snr not in (1,4,7,13,19) or seed not in (2001,2002,2003):raise ValueError('source preprocessing or registered support mismatch')
        values=[float(r[k]) for k in ('psnr_db','lpips','dino')]
        if not all(map(math.isfinite,values)):raise ValueError('nonfinite historical quality')
        N=int(r['complex_uses']);E=float(r.get('total_energy') or r['actual_total_energy'])
        if N<=0 or not math.isfinite(E) or abs(E-2*N)>.05:raise ValueError('actual historical N/E mismatch')
        if r.get('data_complex_uses') and int(r['data_complex_uses'])+int(r['metadata_complex_uses'])!=N:raise ValueError('paid metadata not in total N')
        groups[(method,protocol)].append(r)
    expected={(s,snr,n) for s in sources for snr in (1,4,7,13,19) for n in (2001,2002,2003)}
    for key,data in groups.items():
        if {(r['image_id'],float(r['snr_db']),int(r['seed'])) for r in data}!=expected:raise ValueError('incomplete historical grid')
        if len({int(r['complex_uses']) for r in data})!=1:raise ValueError('mixed resource workpoint')
    return groups

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('new local audit output only')
    out.mkdir(parents=True,exist_ok=True);os.nice(10)
    from token_efficiency.digital_grid import population
    from var_comm.study import seeded_noise
    records,_=population('development');sources={r['image_id']:r['preprocessing_id'] for r in records};pixels={r['image_id']:r['pixels'] for r in records}
    bindings={str(Path(__file__).resolve()):digest(__file__)};data=[];scope=[]
    for directory,key in [(EXTERNAL/'inputs_001','reference_csv_sha256'),(EXTERNAL/'fast_author_metrics_001','per_frame_sha256')]:
        receipt=directory/'completion.json';done=read(receipt);csvfile=directory/('reference_per_frame.csv' if directory.name=='inputs_001' else 'per_frame.csv')
        if digest(csvfile)!=done[key]:raise RuntimeError('original reference completion hash mismatch')
        bindings[str(receipt)]=digest(receipt);bindings[str(csvfile)]=digest(csvfile)
        selected=rows(csvfile)
        if directory.name!='inputs_001':selected=[r for r in selected if r['protocol']=='common_paid_information']
        for r in selected:r['origin_csv']=str(csvfile);r['origin_csv_sha256']=done[key]
        data.extend(selected)
    groups=validate_rows(data,sources)
    # Tie the original source manifest to actual current CPU pixels.
    manifest=EXTERNAL/'inputs_001/development_inputs.json';old=read(manifest)
    if len(old)!=100 or {r['image_id']:r['source_pixels_sha256'] for r in old}!=sources:raise RuntimeError('historical/current source image mismatch')
    bindings[str(manifest)]=digest(manifest)
    archive_rows=defaultdict(list)
    for r in data:archive_rows[resolve_archive(r['image_archive'])].append(r)
    checked=[];maximum=0.;archive_hashes={};identity={'bindings':bindings,'sources':sources,'rows':len(data),'archive_count':len(archive_rows),'CUDA':False,'new_holdout':False}
    if (out/'registration.json').exists() and read(out/'registration.json')!=identity:raise RuntimeError('audit resume identity changed')
    write(out/'registration.json',identity)
    for ordinal,(archive,items) in enumerate(sorted(archive_rows.items())):
        cell=out/'archives'/f'{ordinal:05d}.json';sha=digest(archive);archive_hashes[str(archive)]=sha
        if cell.exists():
            saved=read(cell)
            if saved['archive']!=str(archive) or saved['sha256']!=sha or saved['row_count']!=len(items):raise RuntimeError('changed audit archive on resume')
            checked.extend(saved['checks']);maximum=max(maximum,saved['max_psnr_error']);continue
        checks=[];max_error=0.
        with np.load(archive,allow_pickle=False) as z:images=z['images']
        for r in items:
            image=images[int(r['image_slot'])]
            if image.shape!=(3,256,256) or not np.isfinite(image).all() or image.min()<0 or image.max()>1 or array_sha(image)!=r['image_sha256']:raise RuntimeError('historical reconstructed pixels changed')
            reference=pixels[r['image_id']].astype(np.float32)/255;mse=float(np.square(image.astype(np.float32)-reference,dtype=np.float64).mean());psnr=-10*math.log10(mse);error=abs(psnr-float(r['psnr_db']))
            if error>1e-4:raise RuntimeError('stored PSNR no longer matches exact source/reconstruction')
            if r.get('standard_noise_sha256') and array_sha(seeded_noise(r['image_id'],int(r['seed']),(3060,2)))!=r['standard_noise_sha256']:raise RuntimeError('legacy full-wave standard noise changed')
            max_error=max(max_error,error)
            checks.append({'method':r['method'],'protocol':r.get('protocol') or 'legacy_paid_information','source_id':r['image_id'],'snr_db':float(r['snr_db']),'noise_seed':int(r['seed']),'mse':mse,'psnr_abs_error':error,'image_sha256':r['image_sha256']})
        del images
        write(cell,{'archive':str(archive),'sha256':sha,'row_count':len(items),'checks':checks,'max_psnr_error':max_error});checked.extend(checks);maximum=max(maximum,max_error)
        write(out/'status.json',{'status':'CPU_REAL_ARCHIVE_AUDIT','archives_checked':ordinal+1,'archives_total':len(archive_rows),'rows_checked':len(checked),'time':time.time()})
        if ordinal%100==0:print('archives',ordinal+1,'/',len(archive_rows),'rows',len(checked),flush=True)
    for (method,protocol),rr in groups.items():
        N=int(rr[0]['complex_uses']);scope.append({'method':method,'protocol':protocol,'N':N,'rows':len(rr),'source_images':100,'snrs':[1,4,7,13,19],'seeds':[2001,2002,2003],'current_source_preprocessing_match':True,'archived_pixels_and_PSNR_pass':True,'quality_metric_requalification':'PENDING_COMMON_METRIC_RESCORING','strict_N3060_reference':N==3060,'natural_resource_point':N!=3060,'same_Dc_causal_comparison':False,'timing':'HISTORICAL_ONLY_REQUIRES_SCOPE_DISCLOSURE'})
    # Keep complete input identity for an actual metric-only rerun without generating new RX samples.
    write(out/'rescore_inputs.json',{'sources':sources,'bindings':bindings,'archive_sha256':archive_hashes,'rows':data,'execution':'reuse verified real received RGB; recompute only common metrics; no new noise or method forward','new_holdout':False})
    write(out/'scope.json',scope)
    write(out/'completion.json',{'status':'REAL_HISTORICAL_SOURCE_RESOURCE_PIXEL_AUDIT_PASS_COMMON_METRICS_PENDING','synthetic':False,'GPU_used':False,'new_holdout':False,'sources':100,'rows':len(data),'methods':len(groups),'archive_count':len(archive_hashes),'max_PSNR_abs_error':maximum,'registration_sha256':digest(out/'registration.json'),'rescore_inputs_sha256':digest(out/'rescore_inputs.json'),'scope_sha256':digest(out/'scope.json'),'no_old_artifact_modified':True})
    print(json.dumps(read(out/'completion.json')))
if __name__=='__main__':main()
