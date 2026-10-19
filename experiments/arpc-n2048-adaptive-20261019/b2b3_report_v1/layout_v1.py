"""Two deterministic text-layout corrections, using only closed saved report CSVs."""
from pathlib import Path
import ast, csv, hashlib, json, math, os

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
O=R/'results/arpc_n2048_adaptive_20261019_v1/b2b3_report_v1'
L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_n2048_adaptive_20261019_v1')
SOURCE=R/'experiments/arpc-n2048-adaptive-20261019/b2b3_report_v1/report.py'
OUT=O/'figures_layout_v1'

def require(ok,reason):
    if not ok:raise ValueError(reason)

def pin(p):
    p=Path(p);require(p.is_absolute()and '..'not in p.parts and all(not q.is_symlink()for q in(p,*p.parents)),'unsafe path')
    a=p.stat();b=p.read_bytes();z=p.stat();require((a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns),'file changed')
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def save(p,v):
    with Path(p).open('x',encoding='utf8',newline='\n')as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def rows(name):
    with(O/name).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))

require(pin(SOURCE)==dict(path=str(SOURCE),bytes=44204,sha256='6e5e90513c77c5fc28fc9b933dd176153920f27f45193b30dee00517dd86fe5f'),'original report drift')
md=pin(O/'manifest.json');require(md['bytes']==11429 and md['sha256']=='4f8e3ad4d4f476dea26dc8b318e4a603637bd7cbc1f1651063b3f6d1a1511977','manifest drift')
manifest=json.loads((O/'manifest.json').read_bytes());registered={d['path']:d for d in manifest['files']}
names=['b3_direct_paired_14.csv','b2_all_candidates.csv','b2_four_selections.csv']
for name in names:require(pin(O/name)==registered[str(O/name)],'saved CSV drift')
require(not OUT.exists(),'layout already attempted')
source=SOURCE.read_text(encoding='utf8');tree=ast.parse(source);plots=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='plots')
original=ast.get_source_segment(source,plots)
changes=[("paired_label='Agreement (pp)'", "paired_label='Agree. (pp)'"),
 ("for paired_flag,name in ((False,'adaptive_quality'),(True,'direct_paired')):", "for paired_flag,name in ((True,'direct_paired'),):"),
 ("fig.text(.5,.97,'Nine registered calibration cells; filled circle = saved selected policy'", "fig.text(.5,.94,'Nine calibration candidates; filled circle = saved policy'")]
updated=original
for old,new in changes:
    require(updated.count(old)==1,'ambiguous literal edit');updated=updated.replace(old,new,1)
inverse=updated
for old,new in reversed(changes):inverse=inverse.replace(new,old,1)
require(inverse==original,'inverse changed plot math/data/axes')
nodes=[]
for node in tree.body:
    if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in {'METRICS','METHODS','LABELS','STYLE'}for t in node.targets):nodes.append(node)
    if isinstance(node,ast.FunctionDef)and node.name=='number':nodes.append(node)
require(len(nodes)==5,'unexpected extracted scope')
nodes.append(ast.parse(updated).body[0]);module=ast.Module(body=nodes,type_ignores=[])
namespace=dict(Path=Path,os=os,math=math,L=L,pin=pin,save=save,require=require)
exec(compile(module,str(SOURCE),'exec'),namespace)
OUT.mkdir()
save(OUT/'layout_proof.json',dict(original_source=pin(SOURCE),original_report_manifest=md,
    input_csv=[pin(O/n)for n in names],changes=[dict(old=a,new=b)for a,b in changes],
    inverse_plot_source_exact=True,axes_estimates_intervals_unchanged=True,
    original_report_run_called=False,bootstrap_calls=0,scientific_calls=0))
(OUT/'derived_plots.py').write_text(updated+'\n',encoding='utf8')
paired=rows(names[0]);candidates=rows(names[1]);chosen=rows(names[2])
for row in candidates+chosen:
    row['codec']=int(row['codec']);row['snr_db']=int(row['snr_db']);row['calibration_mean_DINO']=float(row['calibration_mean_DINO'])
require(len(paired)==14 and len(candidates)==9 and len(chosen)==4,'wrong saved row counts')
namespace['plots'](OUT,[],paired,candidates,chosen)
(OUT/'captions_en.txt').write_text('Layout correction only. All saved estimates, intervals, source selection and axes are unchanged. In the direct-paired figure, Agree. means ConvNeXt top-1 prediction agreement with the source, and pp denotes percentage points. The calibration header was shortened and moved downward; all nine candidates and four saved selections are retained.\n',encoding='utf8')
mapping=[]
for stem in ('direct_paired','calibration'):
    for suffix in ('.pdf','.svg','.png','_gray.png'):
        mapping.append(dict(original=pin(O/'figures'/(stem+suffix)),corrected=pin(OUT/(stem+suffix))))
files=[pin(p)for p in sorted(OUT.iterdir())if p.is_file()]
save(OUT/'manifest.json',dict(status='LAYOUT_DERIVATIVES_EXPORTED_VISUAL_REVIEW_PENDING',files=files,
    original_to_corrected=mapping,source=pin(__file__),scientific_calls=0,bootstrap_calls=0,report_run_calls=0))
print(json.dumps(dict(manifest=pin(OUT/'manifest.json'),files=len(files),scientific_calls=0)))
