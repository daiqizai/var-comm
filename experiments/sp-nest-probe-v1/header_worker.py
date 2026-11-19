"""Use the original Linux trellis for a Windows qualification host; waveform in, header out."""
import json, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'experiments/prior-aware-grouped-mcs-v1'))
from uep_phy import Header
h=Header(root)
for line in sys.stdin:
    d=json.loads(line); print(json.dumps(h.receive(np.asarray(d['y'],dtype=float),d['snr'])),flush=True)
