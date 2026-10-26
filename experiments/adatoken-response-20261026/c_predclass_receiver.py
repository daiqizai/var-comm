"""Independent RX: actual arithmetic bits/public prefix/shared models only."""
from common import *
from c_arithmetic import load_codec,generation_seed,canonical
from c_predclass_native import load_native
from c_predclass_protocol import received_class,generation_seed
import argparse,traceback,numpy as np
def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 inputs=cr(r['inputs']);out=L/role;out.mkdir(parents=True,exist_ok=False);runtime_cache(out);os.sched_setaffinity(0,set(range(6)));beg=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
 try:
  sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'))
  # Same existing file-read boundary, with only these shared weight files allowed.
  from runtime_boundary import firewall
  firewall(out,[d['path']for d in r['weights'].values()],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  Encoder,Decoder,cdf,proof=load_codec();save(out/'arithmetic_proof.json',proof);n=load_native(r,out,ledger);rows=[]
  (out/'images').mkdir()
  with n.torch.inference_mode():
   for job in inputs:
    assert time.monotonic()-beg<r['max_seconds']and n.torch.cuda.mem_get_info()[0]>=2<<30
    bits=np.array(job['bits'],np.uint8);prefix=job['prefix'];assert 1<=prefix<=256 and asha(bits)==job['payload_sha256'];key=job['key'];predicted_class=received_class(np.array(job['received_class_bits'],np.uint8));seed=generation_seed(bits,prefix,predicted_class)
    def receive():
     n.start(predicted_class);dec=Decoder(bits);enc=Encoder();tokens=[];traces=[]
     for i in range(prefix):
      logits=n.logits(i);table=n.cdf(logits,cdf);traces.append(asha(table));v=dec.decode(table);enc.encode(v,table);tokens.append(int(v[0]));n.previous=tokens[-1]
     canonical(dec,enc,bits);received=np.array(tokens,np.int64)
     # RNG depends only on actual received bitstream and paid prefix, never source identity.
     with n.torch.random.fork_rng(devices=[0]):
      n.torch.manual_seed(seed)
      for i in range(prefix,256):
       logits=n.logits(i);v=n.tail(logits);tokens.append(int(v.item()));n.previous=tokens[-1]
     tokens=np.array(tokens,np.int64);image=n.render(tokens);p=out/'images'/(key+'.npz')
     with p.open('xb')as f:np.savez(f,image=image,received_tokens=received,completed_tokens=tokens);f.flush();os.fsync(f.fileno())
     record=dict(key=key,prefix=prefix,received_class=predicted_class,payload_sha256=asha(bits),generation_seed=seed,image=pin(p),pixel_sha256=asha(image),received_tokens_sha256=asha(received),cdf_hashes=traces,prior_forwards=256,prior_branch_forwards=512,decoder_forwards=1,source_truth_used=False,TX_CDF_used=False)
     save(p.with_suffix('.json'),record);return record
    rows.append(ledger.call('receive_and_render',key,receive));print('LLAMAGEN_RX_COMMITTED',len(rows),len(inputs),flush=True)
  save(out/'outputs.json',rows);save(out/'completion.json',dict(status='COMPLETE',registration_sha256=expected,outputs=pin(out/'outputs.json'),prior_forwards=256*len(rows),prior_branch_forwards=512*len(rows),decoder_forwards=len(rows),ledger=ledger.summary(),source_reads=0,elapsed_seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
