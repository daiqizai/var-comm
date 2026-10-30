import sys,ast
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
f=cr(read(L/'fit_phy_1024/frames.json')[0]['frame']);print('RXPROFILE',f['actual_RX']['rx_profile']);print('TX',f['transmission']);print('B', (L/'fit_B_control/worker.log').read_text()[-500:])
p=R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py';s=p.read_text();t=ast.parse(s)
for n in t.body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in ['Runtime','transmit_frame','receive_frame','decode_header']:print(ast.get_source_segment(s,n))
