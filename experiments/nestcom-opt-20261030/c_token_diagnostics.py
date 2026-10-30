"""Targets join only after receiver exit, for non-selecting token diagnostics."""
from opt_common import *
from route_statistics import csvout

def token_scores(logits,target,K):
 z=np.asarray(logits[K:],np.float64);t=target[K:];maximum=z.max(1)
 lse=maximum+np.log(np.exp(z-maximum[:,None]).sum(1));ce=lse-z[np.arange(len(t)),t]
 return dict(missing_ce=float(ce.mean()),missing_accuracy=float((z.argmax(1)==t).mean()),missing_positions=len(t))

def main(role,logical_path):
 w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
 rows=read(L/role/'outputs.json');bykey={}
 for r in rows:
  if'boundary_logits'in r:bykey.setdefault(r['key'],[]).append(r)
 sources={s['source_id']:s for s in read(L/'inputs/fit100.json')};result=[]
 for f in read(logical_path):
  if f['received_key']not in bykey:continue
  source=sources[f['source_id']];assert f['source_float_sha256']==source['reference_sha256']
  with np.load(checked(source['archive']),allow_pickle=False)as z:tokens=z['tokens'].copy()
  for row in bykey[f['received_key']]:
   d=row['boundary_logits'];assert not d['target_supplied_to_RX'];m,K=d['m'],d['K'];target=tokens[OFFSETS[m]:OFFSETS[m+1]].astype(np.int64)
   for name in ('base','corrected'):
    with checked(d[name]).open('rb')as fp:logits=np.load(fp,allow_pickle=False)
    result.append(dict(source_id=source['source_id'],method=f['method'],N=f['N'],snr_db=f['snr_db'],variant=row['variant'],logits=name,m=m,K=K,**token_scores(logits,target,K)))
 out=D/(role+'_token_diagnostics');out.mkdir(exist_ok=False);csvout(out/'missing_token_diagnostics.csv',result)
 save(out/'missing_token_diagnostics_identity.json',dict(status='COMPLETE',receiver_wait=pin(L/(role+'_control')/'actual_wait.json'),target_access='CPU postprocessing only after receiver exit; no logits or labels fed back',rows=len(result),population='Eligible supported partial endpoints only; four image metrics separately keep all sources and gray failures',selection_uses='Four downstream image metrics, never these token diagnostics'))
if __name__=='__main__':main(*sys.argv[1:])
