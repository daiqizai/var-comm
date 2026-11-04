"""Actual dynamic-header and body decodes; source validity audited separately."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from rdvq_phy import PHY,noise
from packet import SourcePacket

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args();r=json.loads(Path(a.request).read_text());O=Path(r['output']);phy=PHY(r['repository']);jobs=json.loads(Path(r['jobs']).read_text());start=time.time()
    gate=json.loads(Path(r['qualification']).read_text());assert gate['passed'] and gate['admitted_N']==[512,1024]
    out=O/'events';out.mkdir(exist_ok=False)
    with (O/'events.jsonl').open('x') as journal:
        for i,j in enumerate(jobs):
            key=j['event_id'];event=out/key;event.mkdir(exist_ok=False);(event/'started.json').write_text(json.dumps(j,indent=2))
            public=Path(j['public_directory']);p=phy.layout(j['N'],j['mcs']);packets=[(public/f's{t:02d}.bin').read_bytes() for t in range(j['target']+1)]
            eligible=[w for w in packets if len(w)*8<=p['k']];wire=max(eligible,key=lambda w:SourcePacket.from_bytes(w).slices)
            tx,meta=phy.transmit(wire,j['N'],j['mcs'],j['counter']);standard=noise(j['N'],j['source_id'],j['noise_seed']);observed=tx+standard*10**(-j['snr_db']/20)
            rx=phy.receive(observed,j['snr_db'],j['counter']);received=rx.pop('wire')
            if received is not None:(event/'received_packet.bin').write_bytes(received)
            np.save(event/'observed.npy',observed,allow_pickle=False)
            record=dict(**j,transmission=meta,transmitted_slices=SourcePacket.from_bytes(wire).slices,tx_packet_sha256=hashlib.sha256(wire).hexdigest(),received=rx,
                rx_packet_sha256=hashlib.sha256(received).hexdigest() if received is not None else None,noise_sha256=hashlib.sha256(standard.tobytes()).hexdigest(),
                receiver_given_tx_mcs=False,receiver_given_tx_prefix=False,source_validity='PENDING_EXACT_INDEPENDENT_RX_CACHE' if received is not None else 'REJECTED',event_directory=str(event))
            (event/'complete.json').write_text(json.dumps(record,indent=2));journal.write(json.dumps(record)+'\n');journal.flush()
            (O/'progress.json').write_text(json.dumps(dict(completed=i+1,total=len(jobs),seconds=time.time()-start)))
    (O/'complete.json').write_text(json.dumps(dict(passed=True,actual_frames=len(jobs),header_decodes=len(jobs),new_gpu_calls=0,source_acceptance_audit_pending=True,seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
