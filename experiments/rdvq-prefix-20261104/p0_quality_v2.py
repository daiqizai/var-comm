from pathlib import Path
import argparse,json,csv,time,hashlib
import numpy as np
import torch
from PIL import Image
from codec_adapter import numerical_route
from metrics_v2 import MetricSuite

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();L=Path(a.root);O=L/'p0_quality_v2'
    numerical_route();torch.backends.cudnn.deterministic=True
    spec=json.loads((L/'metric_assets.json').read_text());suite=MetricSuite(spec)
    props=torch.cuda.get_device_properties(0)
    (O/'identity.json').write_text(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,gpu_uuid=str(props.uuid),fp32=True,tf32=False,metrics=spec),indent=2))
    sources=json.loads((L/'p0_tx_v2/complete.json').read_text())['sources'];rows=[]
    for sid in sources:
        x=np.load(L/'p0_tx_v2'/sid/'source_rgb.npy').transpose(2,0,1).astype(np.float32)/np.float32(255)
        for t in range(29):
            path=L/'p0_rx_v2'/sid/f's{t:02d}_rgb.npy';y=np.load(path)
            row=dict(source=sid,slices=t,reference_sha256=hashlib.sha256(x.tobytes()).hexdigest(),rgb_sha256=hashlib.sha256(y.tobytes()).hexdigest(),**suite.score(x,y))
            rows.append(row)
            if t in [0,4,12,20,28]:
                p=O/'display';p.mkdir(exist_ok=True)
                Image.fromarray(np.rint(y.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(p/f'{sid}_s{t:02d}.png')
        (O/'progress.json').write_text(json.dumps(dict(sources=len(rows)//29,pairs=len(rows))))
    with (O/'scores.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (O/'complete.json').write_text(json.dumps(dict(passed=True,sources=len(sources),pairs=len(rows),new_rx=0,new_tx=0,new_phy=0,interpretation='P0 development qualification only'),indent=2))
if __name__=='__main__':main()
