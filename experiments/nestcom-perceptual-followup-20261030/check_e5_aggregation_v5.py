from pathlib import Path
import ast,json
import numpy as np
from collections import defaultdict
p=Path(__file__).with_name('summarize_e5_lineage_v5.py')
tree=ast.parse(p.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='aggregate')
ns={'np':np,'defaultdict':defaultdict};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
ids=['first','second'];pairs={}
for i,sid in enumerate(ids):
 for rep in range(5):
  phase='warmup'if rep<2 else'measured';tx=1000 if rep<2 else (i+1)*10+rep
  pairs[sid,rep]={'tx':dict(phase=phase,seconds=tx),'rx':dict(phase=phase,seconds=2*tx,success=i==0)}
sources,stats=ns['aggregate'](dict(name='test',method='test',N=1024,snr_db=10),ids,pairs)
assert [v['TX']for v in sources]==[13.,23.]
assert [v['equal_source_mean_seconds']for v in stats]==[18.,36.,54.]
assert stats[1]['success_conditional_RX_mean_seconds']==26.
assert stats[1]['failed_RX_calls']==3 and stats[1]['measured']==6
assert stats[0]['call_p95_seconds']==23.75
bad=dict(pairs);bad.pop(('first',4))
try:ns['aggregate'](dict(name='test',method='test',N=1024,snr_db=10),ids,bad)
except AssertionError:pass
else:raise AssertionError('Missing repetition accepted')
print(json.dumps(dict(status='PASS',checks=['warmups_excluded','source_averaging','paired_sum','failed_RX_retained','conditional_RX','linear_p95','missing_repetition_rejected'],scientific_calls=0)))
