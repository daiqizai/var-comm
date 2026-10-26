from common import *
from c_arithmetic import load_codec
from c_predclass_native import load_native
from c_predclass_classifier import load_classifier
import argparse,traceback,numpy as np
def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 out=L/role;out.mkdir(parents=True,exist_ok=False);runtime_cache(out);os.sched_setaffinity(0,set(range(6)));beg=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps']);save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
 try:
  sources=cr(r['sources']);Encoder,_,cdf,proof=load_codec();save(out/'arithmetic_proof.json',proof);n=load_native(r,out,ledger);classifier=load_classifier(r,out,ledger);rows=[]
  with n.torch.inference_mode():
   for s in sources:
    assert time.monotonic()-beg<r['max_seconds']and n.torch.cuda.mem_get_info()[0]>=2<<30
    sd=out/('source_%04d'%s['source_index']);sd.mkdir()
    with np.load(checked(s['archive']),allow_pickle=False)as z:raw=z['pixels'].copy()
    assert raw.dtype==np.uint8 and raw.shape==(3,256,256)and hashlib.sha256(raw.tobytes()).hexdigest()==s['preprocessing_id'];x=raw.astype(np.float32)/np.float32(255)
    def classify():
     value=classifier.predict(x);save(sd/'class_commit.json',dict(predicted_class=value,reference_sha256=asha(x),classifier=pin(out/'classifier_identity.json'),source_truth_label_used=False));return value
    predicted_class=ledger.call('classifier_forward',s['source_id'],classify)
    def tokenizer():
     tokens=n.tokenize(x)
     with(sd/'tokens.npy').open('xb')as f:np.save(f,tokens,allow_pickle=False);f.flush();os.fsync(f.fileno())
     save(sd/'tokenizer_commit.json',dict(source_id=s['source_id'],tokens=s['reuse_tokens']if 'reuse_tokens'in s else pin(sd/'tokens.npy'),token_sha256=asha(tokens),predicted_class=predicted_class,paid_class_bits=10,class_commit=pin(sd/'class_commit.json'),reference_sha256=asha(x)));return tokens
    if 'reuse_tokens' in s:
     with checked(s['reuse_tokens']).open('rb')as f:tokens=np.load(f,allow_pickle=False)
     assert asha(tokens)==s['token_sha256'] and tokens.dtype==np.int64 and tokens.shape==(256,)
     save(sd/'tokenizer_reuse.json',dict(tokens=s['reuse_tokens'],token_sha256=s['token_sha256'],source_identity=s['archive'],reference_sha256=asha(x),reuse_reason='Exact source and unchanged VQ tokenizer; prior null-class qualification closed'))
    else:tokens=ledger.call('tokenizer',s['source_id'],tokenizer)
    def encode():
     n.start(predicted_class);enc=Encoder();streams={};trace=[];lengths=[]
     for i,token in enumerate(tokens):
      logits=n.logits(i);table=n.cdf(logits,cdf);trace.append(asha(table));enc.encode(np.array([token],np.int64),table);n.previous=int(token);bits=enc.finish();streams['p%03d'%(i+1)]=bits;lengths.append(dict(prefix=i+1,bits=len(bits),finish_bits=len(bits)-len(enc.bits)))
     with(sd/'streams.npz').open('xb')as f:np.savez(f,**streams);f.flush();os.fsync(f.fileno())
     save(sd/'cdf_hashes.json',trace);info=dict(source_id=s['source_id'],source_index=s['source_index'],archive=s['archive'],preprocessing_id=s['preprocessing_id'],tokens=s['reuse_tokens']if 'reuse_tokens'in s else pin(sd/'tokens.npy'),token_sha256=asha(tokens),predicted_class=predicted_class,paid_class_bits=10,class_commit=pin(sd/'class_commit.json'),streams=pin(sd/'streams.npz'),cdf_hashes=pin(sd/'cdf_hashes.json'),lengths=lengths,prior_forward=256,prior_branch_forwards=512,source_traversals=1)
     save(sd/'source_commit.json',info);return info
    rows.append(ledger.call('source_probability_pass',s['source_id'],encode));print('LLAMAGEN_SOURCE_COMMITTED',s['source_index'],len(rows),flush=True)
  save(out/'sources.json',rows);save(out/'completion.json',dict(status='COMPLETE',registration_sha256=expected,sources=pin(out/'sources.json'),source_count=len(rows),terminated_streams=256*len(rows),prior_forward=256*len(rows),prior_branch_forwards=512*len(rows),ledger=ledger.summary(),elapsed_seconds=time.monotonic()-beg))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
