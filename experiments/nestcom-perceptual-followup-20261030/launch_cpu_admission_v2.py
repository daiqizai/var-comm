"""Permit CPU-only stages while two GPU workers run; GPU limits unchanged.

Independent revision: the first launcher rejected E3 before registration/Popen.
No code bound to running scientific tasks is modified.
"""
from followup_common import *
p=E/'launch.py';text=p.read_text()
old="assert len(active)<2 and(not timing or not active)"
new="assert (not gpu or len(active)<2) and(not timing or not active)"
assert text.count(old)==1
exec(compile(text.replace(old,new),str(p)+'::CPU_ONLY_ADMISSION_V2','exec'),globals())
