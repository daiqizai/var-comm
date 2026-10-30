"""Restore the previously qualified ARPC package route before its first model call."""
import sys,json
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_bytes());cfg=json.loads(Path(r['timing_config']['path']).read_bytes())
sys.path[:0]=[cfg['arpc_public']['environment']]
from rx_arpc import main
if __name__=='__main__':main(*sys.argv[1:])
