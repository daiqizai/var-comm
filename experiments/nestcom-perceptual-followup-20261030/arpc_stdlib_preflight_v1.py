"""Import-only qualification in the original ARPC runtime; no model/PHY calls."""
from stdlib_statistics_v1 import bind_stdlib_statistics
standard_statistics=bind_stdlib_statistics()
from followup_common import *
import traceback

def main(request,expected):
 assert sha(request)==expected
 r=read(request);[checked(v) for v in r['code']]
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']))
 try:
  cfg=cr(r['config']);sys.path.insert(0,cfg['arpc_public']['environment'])
  import torch,statistics
  assert torch.__version__=='2.10.0+cu130',torch.__version__
  before=str(Path(statistics.__file__).resolve())
  sys.path.insert(0,str(R/'experiments/five-method-timing-20261020'))
  import native
  native.paths()
  import torch._inductor.compile_fx
  from torch._inductor.runtime.benchmarking import benchmarker
  import statistics as after
  assert after is statistics and str(Path(after.__file__).resolve())==before==standard_statistics['path']
  assert after.median([3,1,2])==2 and bind_stdlib_statistics()==standard_statistics
  assert (R/'experiments/baseline-delivery-20261012/statistics.py').exists()
  save(out/'completion.json',dict(status='COMPLETE',original_ARPC_python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
   statistics_stdlib=standard_statistics,checks=['stdlib_prebound','historical_paths_applied','actual_inductor_compile_fx_import','benchmarking_import','stdlib_median_identity'],
   model_calls=0,PHY_calls=0,timing_calls=0,GPU_calls=0,automatic_retry=False))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),model_calls=0,PHY_calls=0,timing_calls=0,GPU_calls=0,automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
