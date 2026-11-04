"""Single shared initialization audit using existing R0 indices and actual bytes.

No source encoding, RX/render, optimizer step, or test image is used here.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import torch
from training_core import PrefixModel

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args();r=json.loads(Path(a.request).read_text());O=Path(r['output'])
    rows=json.loads(Path(r['manifest']).read_text())['sources'];assert len(rows)==128
    model=PrefixModel(r['checkpoint']['path']);model.eval();diff=[];records=[];start=time.time()
    for row in rows:
        idx=torch.from_numpy(np.load(row['indices_path'])).cuda();rates=model.rate_estimates(model.codebook[idx],idx)[0].cpu().numpy()
        lengths=json.loads(Path(row['lengths_path']).read_text());actual=np.array([72]+[v['packet_bits'] for v in lengths],np.float64)
        assert len(actual)==29;diff.append(actual-rates);records.append(dict(source_id=row['source_id'],estimated_packet_bits=rates.tolist(),actual_packet_bits=actual.tolist()))
    correction=np.median(np.stack(diff),axis=0);correction[0]=0
    result=dict(passed=True,checkpoint_sha256=r['checkpoint']['sha256'],source_ids=[v['source_id'] for v in rows],median_packet_minus_proxy_bits=correction.tolist(),records=records,
        source_encoder_calls=0,receiver_calls=0,optimizer_steps=0,teacher_proxy_calls=128,seconds=time.time()-start)
    (O/'complete.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
