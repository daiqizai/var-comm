"""Correct only the registered protocol wrapper path; old metadata attempt is kept."""
from followup_common import *
def main():
 p=E/'prepare_e6.py';assert sha(p)=='38230cba0a284f1a1fa2e46667d8021444458e29bd416be0e597bad19abb5f28'
 assert not list((L/'E6_plan').iterdir())
 save(D/'E6_metadata_correction.json',dict(original=pin(p),revision=pin(Path(__file__)),reason='Design is nested under protocol.design; original failed on metadata lookup before any donor read, TX, RX or model calls.',old_output=str(L/'E6_plan'),scientific_calls_before_failure=0,original_output_preserved=True))
 src=p.read_text();a="read(D/'protocol.json')['E6_threshold']";b="read(D/'protocol.json')['design']['E6_threshold']";assert src.count(a)==1
 src=src.replace(a,b).replace("out=L/'E6_plan'","out=L/'E6_plan_metadata_v2'")
 ns={'__name__':'_e6_protocol_metadata_v2','__file__':str(p)};exec(compile(src,str(p),'exec'),ns);ns['main']()
if __name__=='__main__':main()
