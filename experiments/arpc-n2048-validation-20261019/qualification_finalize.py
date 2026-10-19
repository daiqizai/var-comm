"""Close the authorized import recovery and route independent native inputs."""
from shared import *
import types
PINS={'phy_worker.py':'c0668308223d619a3ad735439ccdef6e43935824209220eee95376534feea670','prepare_native.py':'e5dd133bcd73609feceaf511e96fc396bba2fb7db23667d16fb0cfbc2b3a959e'}
def load(name,edits):
 path=E/name;assert sha(path)==PINS[name];original=path.read_text();text=original
 for a,b in edits:assert a in text;text=text.replace(a,b)
 back=text
 for a,b in reversed(edits):back=back.replace(b,a)
 assert back==original
 ns=dict(__name__='_closed_route_adapter',__file__=str(path));exec(compile(text,str(path)+'::closed_recovery_route','exec'),ns);return ns
def main(expected):
 auth=O/'rx_import_continuation_authorization.json';assert sha(auth)=='e7bb3590424f2fdc64e319abf0a5bca7c6ac483e0a30f9350e6c6d9abc890e7a'
 c=L/'controls/qualification_rx_import_recovery_v1';w=read(c/'actual_wait.json');assert type(w['returncode'])is int and w['returncode']==0 and gone(w['process']) and gone(read(c/'supervisor.json')['process'])
 a=load('phy_worker.py',[("rxc=read(target/'rx/completion.json')","rxc=read(target/'rx_import_recovery_v1/completion.json')"),("pin(target/'rx/source_jobs.json')","pin(target/'rx_import_recovery_v1/source_jobs.json')")]);a['audit']('qualification',expected)
 b=load('prepare_native.py',[("(phase+'_rx')","(phase+'_rx_import_recovery_v1')"),("L/phase/'rx/source_jobs.json'","L/phase/'rx_import_recovery_v1/source_jobs.json'")]);b['main']('qualification',expected)
 put(O/'qualification/route_revision.json',dict(status='CLOSED_AND_NATIVE_PREPARED',source=pin(__file__),authorization=pin(auth),actual_wait=pin(c/'actual_wait.json'),originals=[pin(E/x)for x in PINS],native_calls=0,qualified_subframes=48,decoder_calls=100,repeated_decoder_calls=4))
if __name__=='__main__':main(sys.argv[1])
