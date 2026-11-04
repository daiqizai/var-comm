from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from rdvq_phy import PHY,noise,MCS
from packet import SourcePacket

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args();r=json.loads(Path(a.request).read_text());O=Path(r['output']);phy=PHY(r['repository']);rows=[];layouts=[];start=time.time()
    public=Path(r['public_packets']);sid=r['source_id'];packets=[(public/sid/f's{t:02d}.bin').read_bytes() for t in range(29)]
    for N in [512,1024,2048]:
        for m in range(12):
            p=phy.layout(N,m);layouts.append(p)
            feasible=[w for w in packets if len(w)*8<=p['k']];wire=max(feasible,key=lambda w:SourcePacket.from_bytes(w).slices)
            for label,w in [('nonnull',wire),('NULL',packets[0])]:
                assert label=='NULL' or SourcePacket.from_bytes(w).slices>0
                key=f'N{N}_m{m:02d}_{label}';(O/(key+'_started.json')).write_text(json.dumps(dict(new_frame=True,time=time.time())))
                tx,meta=phy.transmit(w,N,m,800000+len(rows));got=phy.receive(tx,30,800000+len(rows));assert got['status']=='ACCEPTED_PACKET' and got['wire']==w and got['received_mcs']==m
                record=dict(case=key,actual_header_decode=1,actual_body_decode=1,received_exact=True,source_rx_exact_cache_reuse=True,wire_sha256=hashlib.sha256(w).hexdigest(),frame_sha256=hashlib.sha256(tx.tobytes()).hexdigest(),transmission=meta)
                rows.append(record);(O/(key+'_complete.json')).write_text(json.dumps(record,indent=2))
    assert len(rows)==72 and len(layouts)==36
    (O/'layouts.json').write_text(json.dumps(layouts,indent=2));(O/'complete.json').write_text(json.dumps(dict(passed=True,frames=72,header_decodes=72,body_decodes=72,layouts=36,source_model_calls=0,clean_link_only=True,seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
