"""Four frozen metrics on native arrays, with exact P0 reuse only."""
from pathlib import Path
import argparse,csv,hashlib,json,time
import numpy as np
import torch
from metrics_v2 import MetricSuite
from codec_adapter import numerical_route

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args();r=json.loads(Path(a.request).read_text());O=Path(r['output'])
    rows=json.loads(Path(r['manifest']).read_text())['sources'];numerical_route();suite=MetricSuite(json.loads(Path(r['metric_assets']).read_text()));new=0;reuse=0;start=time.time()
    old={}
    if r.get('reuse_scores'):
        for row in csv.DictReader(Path(r['reuse_scores']).open()):old[(row['source'],int(row['slices']))]=row
    fields=['source','split','slices','packet_bits','reference_sha256','reconstruction_sha256','psnr_db','lpips_alex','dists','dreamsim','origin']
    with (O/'scores.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,row in enumerate(rows):
            pixels=np.load(row['source_rgb_path']);assert pixels.dtype==np.uint8;reference=np.ascontiguousarray(pixels.transpose(2,0,1).astype(np.float32)/255)
            lengths=json.loads(Path(row['lengths_path']).read_text());bits={0:72,**{v['slices']:v['packet_bits'] for v in lengths}}
            for t in row.get('endpoints',range(29)):
                rec=np.load(Path(row['rx_directory'])/f's{t:02d}_rgb.npy');rh=hashlib.sha256(reference.tobytes()).hexdigest();yh=hashlib.sha256(rec.tobytes()).hexdigest();key=(row['source_id'],t)
                if row.get('reuse_p0'):
                    v=old[key]
                    assert v['reference_sha256']==rh and v['rgb_sha256']==yh,(key,'metric reuse pixels differ')
                    scores={m:float(v[m]) for m in ['psnr_db','lpips_alex','dists','dreamsim']};origin='P0_EXACT_NATIVE_PAIR_REUSE';reuse+=1
                else:
                    receipt=O/(row['source_id']+f'_s{t:02d}_started.json');receipt.write_text(json.dumps(dict(reference_sha256=rh,reconstruction_sha256=yh,time=time.time())))
                    scores=suite.score(reference,rec);origin='NEW_FOUR_METRIC_PAIR';new+=1
                w.writerow(dict(source=row['source_id'],split=row['split'],slices=t,packet_bits=bits[t],reference_sha256=rh,reconstruction_sha256=yh,origin=origin,**scores));f.flush()
            (O/'progress.json').write_text(json.dumps(dict(sources=i+1,total=len(rows),new_pairs=new,reused_pairs=reuse)))
    (O/'complete.json').write_text(json.dumps(dict(passed=True,sources=len(rows),new_pairs=new,reused_pairs=reuse,total_pairs=new+reuse,metrics=['psnr_db','lpips_alex','dists','dreamsim'],seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
