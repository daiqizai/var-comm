"""Actual formal stage owner with exact new receiver entry binding."""
from pathlib import Path
import sys,hashlib
E=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-n2048-validation-20261019');sys.path.insert(0,str(E))
from shared import *
def main():
 auth=read(O/'formal_entry_authorization.json');assert auth['status']=='AUTHORIZED_AFTER_ACTUAL_QUALIFICATION'
 for d in auth['bindings']:verify(d)
 assert pin(__file__) in auth['bindings'] and pin(E/'formal_receiver.py') in auth['bindings']
 assert sys.argv[1] in ['formal_tx','formal_rx','formal_audit','formal_nest','formal_arpc']
 path=E/'control.py';assert sha(path)=='f97c5e54ae68d5234c6e257a08f0f56bdb919f0fa4ee8712670bf963c2a2776d'
 old=path.read_text();a="  argv=[str(CPY),'-B',str(E/'phy_worker.py'),role,phase]";b=a+"\n  if role=='rx':argv[2]=str(E/'formal_receiver.py')";assert old.count(a)==1
 text=old.replace(a,b);assert text.replace(b,a)==old
 exec(compile(text,str(path)+'::formal_rx_import_entry','exec'),dict(__name__='__main__',__file__=str(path)))
if __name__=='__main__':main()
