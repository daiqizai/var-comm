"""Original real frame engine with new task-local budget and STOP namespace."""
from followup_common import *
import extended_frame_worker_v3 as engine
engine.L=L
if __name__=='__main__':engine.main(*sys.argv[1:])
