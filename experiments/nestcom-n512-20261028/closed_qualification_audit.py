"""Post-output checks only; preserves failed control exits without rerunning models."""
from n512_common import *
from PIL import Image

def main():
 root=L/'qualification_metadata_audit';root.mkdir(exist_ok=False);frames=read(L/'phy_qualification/frames.json');assets=read(L/'assets/cal1000.json');src={s['source_id']:s for s in assets};report={}
 for method in ('nest','ada'):
  role='qualification_rx_'+method;out=L/role;w=read(L/'controls'/role/'actual_wait.json')
  assert w['returncode']==1 and gone(w['process']) and gone(w['owner']) and not w['timed_out']
  f=read(out/'failure.json');assert f['ledger']['unresolved']==0
  assert "PermissionError: RX truth/storage boundary:"in f['traceback'] and f'{role}_registration.json'in f['traceback']
  rows=read(out/'outputs.json');assert len(rows)==15;aud=[]
  for row in rows:
   with np.load(checked(row['image']),allow_pickle=False)as z:im=z['image'].copy();tokens=z['received_tokens'].copy()
   assert asha(im)==row['pixel_sha256'] and asha(tokens)==row['received_tokens_sha256']
   matches=[v for v in frames if v['received_key']==row['key']];assert matches
   if method=='ada':
    assert row['status']=='ACCEPTED_SOURCE'
    for v in matches:
     s=src[v['source_id']];p=v['actual_RX']['rx_profile'];body=v['actual_RX']['body'];prefix=p['m']
     truth=np.load(checked(s['ada_tokens']),allow_pickle=False);assert np.array_equal(tokens,truth[:prefix])
     cdf=cr(s['ada_cdf_hashes']);assert row['cdf_hashes']==cdf[:prefix]
    aud.append(dict(key=row['key'],matching_frames=len(matches),received_tokens_exact=True,cdf_hashes_exact=True,image=row['image']))
   else:assert row['status']=='FIXED_GRAY_SOURCE_REJECT' and np.all(im==.5)
  if method=='ada':
   for i,row in enumerate(rows[:3]):
    with np.load(checked(row['image']),allow_pickle=False)as z:im=z['image']
    Image.fromarray(np.rint(np.transpose(im,(1,2,0))*255).clip(0,255).astype(np.uint8)).save(root/f'ada_{i}.png')
  report[method]=dict(actual_wait=pin(L/'controls'/role/'actual_wait.json'),failure=pin(out/'failure.json'),outputs=pin(out/'outputs.json'),scientific_calls=15,post_output_audit=aud,source_codec_pass=method=='ada',original_exit_preserved=1)
 save(root/'audit.json',report);save(O/'qualification_metadata_audit.json',dict(audit=pin(root/'audit.json'),status='ADA_OUTPUTS_VERIFIED_NEST_ROUTE_BLOCKED',model_calls=0))
 print('AUDITED',json.dumps({k:v['source_codec_pass']for k,v in report.items()}))
if __name__=='__main__':main()
