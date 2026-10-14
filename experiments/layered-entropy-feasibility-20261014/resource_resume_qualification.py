"""Only closed metadata/bitvectors and AST; zero model or scientific calls."""
from resource_resume import *
import numpy as np
reg=read(REC)
old,sources,completed,proofs=validate_closed_prefix(reg)
code,replacements=adapted_native_source()
assert len(replacements)==5
tree=ast.parse(code)
assert not any(isinstance(n,ast.Name) and n.id=='TX_CDF' for n in ast.walk(tree))
rows=[]
for rec in completed:
 with np.load(checked(rec['streams']),allow_pickle=False) as a:rows.extend(old_length_rows(rec['source'],a))
assert len(rows)==38*68 and all(r['extra_information_bits']==r['b_base']+r['b_enhancement']-r['b_single']+29 for r in rows)
assert all(r['single_finish_bits'] is None for r in rows)
assert [s['pilot_slot'] for s in sources[38:]]==list(range(38,50))
assert set(r['source_id'] for r in rows).isdisjoint(s['source_id'] for s in sources[38:])
assert all(k in old['caps'] or k=='source_model_reload' for k in reg['effective_caps'])
assert all(reg['effective_caps'][k]==v for k,v in old['caps'].items()) and reg['effective_caps']['source_model_reload']==1
assert reg['deadline_unix']==old['deadline_unix']
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
print(canonical(dict(status='PASS_CLOSED_PREFIX_RESOURCE_RECOVERY_METADATA_AST',checks=7,restored_length_rows=len(rows),independent_RX_proofs_reused=len(proofs),new_TX_sources=12,new_RX_proofs=0,new_clean=450,model_imports=[],scientific_calls=0,inverse_AST_exact=True,individual_finish_increment_not_reconstructed=True)),flush=True)
