"""TX boundary between the two independent source bitstreams and paid 1024-symbol frame."""
import numpy as np
from profiles import capacity


def emit(radio, selection, counter):
    if selection is None:
        # ID 0 has one public filler layout independent of the image and policy.
        br, qe, er, alpha = '1/2', 2, '2/3', .8
        ib = radio.packet.pack_group(np.zeros(2,np.uint8),capacity(2,br)+29)
        ie = radio.packet.pack_group(np.empty(0,np.uint8),capacity(qe,er)+29,True)
        b=radio.layer_tx(ib,2,counter,'B'); e=radio.layer_tx(ie,qe,counter,'E')
        body=np.sqrt(alpha)*b+np.sqrt(1-alpha)*e
        wave=np.vstack([radio.header.transmit(0),body]); assert wave.shape==(1024,2)
        return wave, dict(B=float((b*b).sum(1).mean()),E=float((e*e).sum(1).mean()),body=float((body*body).sum(1).mean()),frame=float((wave*wave).sum(1).mean()),cross=float((b*e).sum(1).mean()))
    pid,bb,eb=selection
    if pid==0 or pid not in radio.directory: raise ValueError('unknown active profile')
    j,m,k,br,qe,er,_=radio.directory[pid]
    ib=radio.packet.pack_group(bb,capacity(2,br)+29)
    ie=radio.packet.pack_group(eb,capacity(qe,er)+29,(j,0)==(m,k))
    return radio.transmit(pid,ib,ie,counter)
