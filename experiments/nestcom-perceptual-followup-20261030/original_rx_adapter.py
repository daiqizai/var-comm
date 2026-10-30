"""Unchanged frozen receiver, isolated new task output and stop namespace."""
from followup_common import *
import original_receiver as engine
engine.L=L
if __name__=='__main__':engine.main(*sys.argv[1:])
