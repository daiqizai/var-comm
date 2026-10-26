"""Actual received nested-grid prefixes only; original predict/cover/advance render."""
from common import *
from a_codec import receiver_class
import argparse,traceback,numpy as np
def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 inputs=cr(r['inputs']);routes=cr(r['model_routes']);static=r['static'];out=L/role;out.mkdir(exist_ok=False);runtime_cache(out);os.sched_setaffinity(0,{0,1});beg=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
 try:
  sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'));sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'))
  from kn_native import load
  from runtime_boundary import firewall
  allowed=[v['path']for v in routes['files']+routes['author_files']]+[routes['dist']['path'],static['completion']['path'],static['table']['path']]
  firewall(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  def boundary():assert time.monotonic()-beg<r['max_seconds']and not(L/'STOP').exists()
  n,render=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));C,proof=receiver_class();codec=C(R,native=n);save(out/'codec_proof.json',proof)
  from ep_source_codec import InvalidSourceStream
  from h800_ep48_link_core_v1 import bind_static
  bind_static(codec,static);(out/'images').mkdir();rows=[]
  for job in inputs:
   boundary();key=job['key'];rx=job['actual_RX'];p=rx['rx_profile'];b=rx['body'];image=np.full((3,256,256),.5,np.float32);tokens=None;status=rx['status'];origin='FIXED_GRAY'
   if rx['header']['header_ok']and b and b['crc_accepted']and b['parser_accepted']and p['family']!='EXPLICIT_NULL_FIXEDFRAME':
    if job.get('cached_source_decode'):
     cached=job['cached_source_decode'];tokens=np.array(cached['received_tokens'],np.int64);assert asha(tokens)==cached['token_sha256'];origin='EXACT_PRIOR_INDEPENDENT_SOURCE_RX_REUSE'
    else:
     def decode():
      try:v=codec.decode(p['family'],np.array(b['payload'],np.uint8),p['m'],p['K'])
      except InvalidSourceStream as e:v=dict(parse_error=str(e))
      if 'received_tokens'in v:v['received_tokens']=v['received_tokens'].tolist()
      save(out/'images'/(key+'_decoded.json'),v);return v
     v=ledger.call('source_rx',key,decode)
     if 'parse_error'in v:status='SOURCE_PARSE_REJECT'
     else:tokens=np.array(v['received_tokens'],np.int64);origin='NEW_SOURCE_RX'
    if tokens is not None:
     assert len(tokens)==OFFSETS[p['m']]+p['K']
     def reconstruct():
      with n.torch.no_grad():im=render(n,tokens.copy(),p['m'],p['K'])
      ip=out/'images'/(key+'.npz')
      with ip.open('xb')as f:np.savez(f,image=im,received_tokens=tokens);f.flush();os.fsync(f.fileno())
      return im
     image=ledger.call('var_render',key,reconstruct);status='ARITHMETIC_SOURCE_DECODED'
   ip=out/'images'/(key+'.npz')
   if not ip.exists():
    with ip.open('xb')as f:np.savez(f,image=image);f.flush();os.fsync(f.fileno())
   assert image.dtype==np.float32 and image.shape==(3,256,256)and np.isfinite(image).all()and 0<=image.min()<=image.max()<=1
   item=dict(key=key,image=pin(ip),pixel_sha256=asha(image),source_status=status,origin=origin,received_tokens_sha256=None if tokens is None else asha(tokens),source_truth_used=False,TX_CDF_used=False)
   save(ip.with_suffix('.json'),item);rows.append(item)
   if len(rows)%25==0:print('A_RX_COMPLETE',len(rows),len(inputs),flush=True)
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='A_RX_COMPLETE',registration_sha256=expected,outputs=pin(out/'outputs.json'),count=len(rows),ledger=ledger.summary(),source_reads=0,elapsed_seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
