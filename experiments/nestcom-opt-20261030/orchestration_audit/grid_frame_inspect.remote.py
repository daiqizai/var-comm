from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
for rel in ['experiments/nestcom-n512-20261028/n512_phy.py','experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py','experiments/adatoken-response-20261026/c_predclass_frame.py']:
 p=R/rel;s=p.read_text();print('FILE',rel)
 if 'n512_phy.py'in rel:print(s)
 else:
  import ast
  for n in ast.parse(s).body:
   if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in ['pack_body','unpack_body','parse_body','receive','make_catalogue','isolated_engine']:
    print(ast.get_source_segment(s,n))
