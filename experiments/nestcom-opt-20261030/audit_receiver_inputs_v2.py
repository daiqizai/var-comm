"""Post-exit TX/RX diagnostics never repair or relabel received outcomes."""
from opt_common import *
import collections

def main(role,plan,sources_path=None):
 w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
 rows=read(L/role/'outputs.json');outputs={r['key']:r for r in rows if r['variant']=='original'};jobs={r['key']:r['projection']for r in read(L/plan/'inputs.json')}
 sources={s['source_id']:s for s in read(sources_path or L/'inputs/fit100.json')};counts=collections.Counter();differences=[];checked_pairs={}
 for f in read(L/plan/'logical.json'):
  key=f['received_key'];out=outputs[key];counts[(f['method'],f.get('variant','original'),f['N'],f['snr_db'],out['status'])]+=1
  if out['status']!='ACCEPTED_SOURCE':continue
  s=sources[f['source_id']];assert s['preprocessing_id']==f['source_pixels_sha256']and s['reference_sha256']==f['source_float_sha256']
  sid_key=(s['source_id'],key)
  if sid_key in checked_pairs:continue
  p=jobs[key]
  with np.load(checked(s['archive']),allow_pickle=False)as z:truth=z['tokens'].copy()
  with np.load(checked(out['image']),allow_pickle=False)as z:received=z['received_tokens'].copy();image=z['image'].copy()
  assert asha(received)==out['received_tokens_sha256']and asha(image)==out['pixel_sha256']
  n=OFFSETS[p['m']]+p['K'];assert received.shape==(n,);errors=int(np.count_nonzero(received!=truth[:n]));checked_pairs[sid_key]=errors
  if errors:differences.append(dict(source_id=s['source_id'],received_key=key,token_errors=errors))
 result=dict(status='COMPLETE',receiver_wait=pin(L/(role+'_control')/'actual_wait.json'),unique_successful_source_inputs=len(checked_pairs),posthoc_token_mismatches=differences,logical_outcomes=[dict(method=k[0],variant=k[1],N=k[2],snr_db=k[3],receiver_status=k[4],count=v)for k,v in sorted(counts.items())],targets_only_in_posthoc_audit=True,outputs_repaired=False,rejections_retried=False)
 save(D/(role+'_posthoc_input_audit.json'),result);print(result)
if __name__=='__main__':main(*sys.argv[1:])
