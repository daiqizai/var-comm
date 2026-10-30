import sys,hashlib,sqlite3
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for s in read(L/'inputs/fit100.json')[:2]:
 print('SOURCE',s)
 with np.load(checked(s['archive']),allow_pickle=False)as z:
  p=z['pixels'];print('ARRAY',p.dtype,p.shape,'rawsha',hashlib.sha256(p.tobytes()).hexdigest(),'asha',asha(p))
print('TXREG',read(D/'benchmark_tx_registration.json'))
print('TX_LEDGER',sqlite3.connect(L/'benchmark_tx/calls.sqlite').execute('SELECT *FROM calls').fetchall())