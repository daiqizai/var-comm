"""One probability pass per fixed source; real independent C/A/B finish. No rendering."""
from block_common import *
from protocol import CONFIGS
import tempfile,traceback

def main(path,expected):
 assert sha(path)==expected;r=read(path);rp=pin(path)
 for d in r['code']:checked(d)
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
 os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);beg=time.monotonic()
 save(out/'launch.json',dict(process=ident(os.getpid()),owner=ident(os.getppid()),request=rp))
 try:
  sources=cr(r['inputs']);routes=cr(r['model_routes']);rows=[]
  sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'))
  from kn_native import load
  from same_scale_codec import SameScaleCodec
  def boundary():assert time.monotonic()-beg<r['max_seconds'] and not(L/'STOP').exists()
  n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));codec=SameScaleCodec(R,native=n)
  for s in sources:
   boundary();assert n.torch.cuda.mem_get_info()[0]>=2<<30
   dest=out/('%04d'%s['source_index']);dest.mkdir()
   with np.load(checked(s['archive']),allow_pickle=False)as z:tokens=z['tokens'].copy();pixels=z['pixels'].copy()
   assert tokens.dtype==np.int64 and tokens.shape==(680,)and pixels.dtype==np.uint8
   assert hashlib.sha256(pixels.tobytes()).hexdigest()==s['preprocessing_id']
   def encode():
    streams,meta,trace=codec.encode_boundary(tokens)
    ip=dest/'streams.npz'
    with ip.open('xb')as f:np.savez(f,**streams);f.flush();os.fsync(f.fileno())
    save(dest/'trace.json',trace)
    # This is TX consistency only, not an RX round-trip claim.
    with np.load(checked(s['nest_streams']),allow_pickle=False)as z:
     for m in(4,5):assert np.array_equal(streams['C'+str(m)],z[f'm{m}_K0'])
    info=dict(source_id=s['source_id'],source_index=s['source_index'],streams=pin(ip),trace=pin(dest/'trace.json'),tokens_sha256=asha(tokens),lengths=meta,probability_passes=1,tokenizer_calls=0,render_calls=0)
    save(dest/'encoded_commit.json',info);return info
   rows.append(ledger.call('source_probability_pass',s['source_id'],encode))
   print('CAB_SOURCE_COMMITTED',s['source_index'],len(rows),len(sources),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'sources.json',rows);save(out/'completion.json',dict(status='COMPLETE_SOURCE',request=rp,sources=pin(out/'sources.json'),count=len(rows),ledger=ledger.summary(),seconds=time.monotonic()-beg,tokenizer_calls=0,render_calls=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
