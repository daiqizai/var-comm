from pathlib import Path
import ast,hashlib,json,numpy as np
import argparse
ap=argparse.ArgumentParser(description='Audit existing no-fit behavior without PHY/model/metric calls')
ap.add_argument('--repository',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
P=Path(__file__).resolve().parents[1]
records=[]
def extract(rel,functions,constants):
 p=args.repository/rel;raw=p.read_bytes()
 expected=next(x['sha256'] for x in json.loads((P/'validation/implementation_status.json').read_bytes())['sources'] if x['path']==rel)
 assert hashlib.sha256(raw).hexdigest()==expected,rel
 tree=ast.parse(raw)
 selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)]
 ns={'np':np};exec(compile(ast.Module(body=selected,type_ignores=[]),rel,'exec'),ns)
 records.append(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),extracted_original_nodes=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno) for n in selected if isinstance(n,ast.FunctionDef)]))
 return ns
n=extract('experiments/nestcom-n512-20261028/n512_phy.py',{'nt','choose','received_projection'},{'SIZES','ENDS','NEST_W','NEST_P','ADA','NULL'})
profiles=[dict(family=n['NEST_W'],m=4,K=0,q=2,nominal_rate='1/2',k=40),dict(family=n['NULL'],q=2,nominal_rate='1/2',k=40)]
candidate=dict(method='WHOLE',q=2,rate='1/2',target=[4,0]);streams={'m4_K0':np.zeros(12,np.uint8)}
profile,payload,no_fit=n['choose'](streams,profiles,candidate)
assert no_fit and profile['family']==n['NULL'] and payload.tolist()==[0,0]
rx=dict(header=dict(header_ok=True),rx_profile=profile,body=dict(crc_accepted=True,parser_accepted=True,payload=payload))
assert n['received_projection'](rx)==dict(status='FIXED_GRAY',reason='EXPLICIT_NULL_PROFILE')
streams['m4_K0']=np.zeros(11,np.uint8)
p,b,flag=n['choose'](streams,profiles,candidate);assert not flag and p['m']==4 and np.array_equal(b,streams['m4_K0'])
old=extract('experiments/entropy-partial-mismatch-20261010/scripts/ep_plan.py',{'require','ks','token_count','fallback_endpoints','select_stream'},{'SIZES'})
lengths={(m,0):12 for m in range(4,8)};c=dict(target_m=7,target_K=0,source_capacity_bits=11)
try:old['select_stream'](lengths,c)
except ValueError as e:assert 'cannot fabricate channel failure' in str(e)
else:raise AssertionError('N1024 must preserve error')
c['source_capacity_bits']=12;assert old['select_stream'](lengths,c)['m']==7
full=extract('experiments/t1b-m10-calibration-20261016/tb_plan.py',{'nt','fallback','choose'},{'SIZES','ENDS'})
lengths={p:12 for p in full['fallback']('PARTIAL')}
try:full['choose'](lengths,'PARTIAL',11)
except ValueError as e:assert 'preserve and stop' in str(e)
else:raise AssertionError('N1024 m10 must preserve error')
assert full['choose'](lengths,'PARTIAL',12)==(10,0)
record=dict(status='VERIFIED_EXISTING_DIFFERENCE_NO_PROTOCOL_CHANGE',base_commit='2c303b615f73edd0a7a971308c0df8bd5bb885d0',sources=records,checks=['N512 no-fit selects paid NULL and two zero payload bits','N512 accepted NULL projects to fixed gray','N512 fitting stream payload preserved','N1024 original no-fit raises ValueError','N1024 original fitting stream selects longest legal endpoint','N1024 m10 no-fit raises ValueError','N1024 m10 fitting stream selects full endpoint'],checks_passed=7,scope='Synthetic CPU source-selection and RX-projection checks of unmodified AST-extracted functions; no model, PHY or metric calls. Not new wireless qualification.',code_edits=0,protocol_unified=False,rationale='The request requires audit and only conditionally a unified repair. Frozen historical protocols remain unchanged; the manuscript states the difference accurately. All evaluated N512 TX_NO_FIT counts are zero; prior N1024 evidence reported no no-fit source.')
with args.output.open('x',encoding='utf8') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
