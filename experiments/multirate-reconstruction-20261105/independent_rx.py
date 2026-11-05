"""Separate-process decoding; only checkpoint and on-wire bytes are inputs."""
import argparse,json
from pathlib import Path
from runtime import setup,OLD,sha,write
setup()
import torch
from model import MultiRateCodec
from codec import receive_packet
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.input);out=Path(a.output);out.mkdir(exist_ok=False)
    spec=json.loads((source/'checkpoint.json').read_text());assert sha(spec['path'])==spec['sha256']
    init=json.loads((OLD/'checkpoint.json').read_text());model=MultiRateCodec(init['path'])
    state=torch.load(spec['path'],map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);del state
    raw,ids,meta=receive_packet(model,(source/'packet.bin').read_bytes())
    torch.save(raw.cpu(),out/'rx.pt');torch.save(ids,out/'received_ids.pt')
    write(out/'complete.json',dict(passed=True,metadata=meta,checkpoint_sha256=spec['sha256'],inputs=['checkpoint','packet.bin'],source_or_TX_state_access=False))
if __name__=='__main__':main()
