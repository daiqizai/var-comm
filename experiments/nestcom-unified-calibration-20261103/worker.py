"""New output/budget namespaces; original algorithms are imported unchanged."""
from study_common import *
kind,request,expected=sys.argv[1:]
if kind=='phy':
 import extended_frame_worker_v3 as engine
elif kind=='rx':
 import original_receiver as engine
elif kind=='metrics':
 import metrics as engine
elif kind=='dists':
 sys.path.insert(0,str(R/'experiments/nestcom-perceptual-followup-20261030'))
 import dists_worker as engine
else:raise ValueError(kind)
engine.L=L
engine.main(request,expected)
