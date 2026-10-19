"""Prequalified import-only RX correction for the not-yet-started formal grid."""
import sys,hashlib
from pathlib import Path
E=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-n2048-validation-20261019')
def adapt(data):
 assert hashlib.sha256(data).hexdigest()=='c0668308223d619a3ad735439ccdef6e43935824209220eee95376534feea670'
 original=data.decode();text=original
 edits=[("sys.path.insert(0,str(OLD_E));from a1_native import install_rx_boundary","sys.path.insert(0,str(OLD_E));sys.path.insert(0,str(OLD_E/'upstream'));import wire;from a1_native import install_rx_boundary"),
 ("outputs=[];sourcejobs={};wires=target/'source_inputs';wires.mkdir()","outputs=[];sourcejobs={};wires=target/'source_inputs';wires.mkdir();checkpoints=target/'physical_checkpoints';checkpoints.mkdir()"),
 ("  result=dict(key=j['key'],received=rs,observed_sha256=ep.array_sha(y)","  put(checkpoints/(j['key']+'.json'),dict(key=j['key'],received=rs,observed_sha256=ep.array_sha(y)))\n  result=dict(key=j['key'],received=rs,observed_sha256=ep.array_sha(y)")]
 for a,b in edits:assert text.count(a)==1;text=text.replace(a,b)
 back=text
 for a,b in reversed(edits):assert back.count(b)==1;back=back.replace(b,a)
 assert back==original
 return text
if __name__=='__main__':
 assert sys.argv[1:3]==['rx','formal']
 p=E/'phy_worker.py';exec(compile(adapt(p.read_bytes()),str(p)+'::import_and_checkpoint_only','exec'),dict(__name__='__main__',__file__=str(p)))
