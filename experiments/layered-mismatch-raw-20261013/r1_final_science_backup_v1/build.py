"""Final recovered R1 scientific increment. Import is inert; no science is run.

Requires actual mixed-stage closure, CPU delivery exit zero, checked final
figures and an explicit root-pinned authorization made after visual review.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, json, os, stat, subprocess, sys, tarfile, time, traceback
from pathlib import Path, PurePosixPath
sys.dont_write_bytecode=True
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/layered-mismatch-raw-20261013';O=R/'results/layered_mismatch_raw_20261013'
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
L=T/'layered_mismatch_raw_20261013';BASE=T/'baseline_strength_20261011'
OLD=L/'r1_execution_v1';RUN=L/'r1_cache_recovery_v1'
CONTROL=L/'r1_cache_recovery_delivery_dispatch_v1'
LIGHT=O/'r1_cache_recovery_delivery_v1';HEAVY=L/'r1_cache_recovery_delivery_v1'
FIGURES=O/'r1_cache_recovery_figures_v1'
REG=O/'r1_cache_recovery_v1/registered/registration.json'
MASTER='ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c'
ENTRY=E/'r1_cache_recovery_delivery_v1/deliver.py'
ENTRY_SHA='f37df9e92520a1f45bdc7954dbbd6eaf5c1a7cab4721d3b5df7ec5df65c6de01'
GRAPH=E/'r1_cache_recovery_figures_v1/figures.py'
GRAPH_SHA='b5e5d8f42a2343930103618d589089c9ffea1c8c8d8753e0e506d1934551077d'
PACKAGE='layered_raw_R1_final_science_20261014_v1'
B=BASE/(PACKAGE+'_delivery');SCHEMA='LAYERED_RAW_R1_FINAL_RECOVERED_SCIENCE_INCREMENT_V1'
AUTH_SCHEMA='R1_FINAL_SCIENCE_BACKUP_AUTHORIZATION_V1'
FLAGS=dict(actual_closed_R1=True,R2_included=False,standalone_full_recovery=False,contains_models=False,
    old_R0_repacked=False,science_calls=0,repeated_scientific_audit=False,contains_active_science=False,
    includes_original_initialization_failure=True,actual_figures_visually_reviewed=True)
PUB_NAMES=('checks.json','commit.json','push_verified.json','staged_files.json')
LOCKS=(OLD/'owner.lock',L/'r1_resource_dispatch_v2/dispatch.lock',RUN/'owner.lock',
       L/'r1_cache_recovery_dispatch_v1/dispatch.lock',CONTROL/'dispatch.lock')
RUN_SCOPES=('r1_execution_v1','r1_resource_dispatch_v2','r1_initial_dispatch_control_v1',
    'r1_cache_recovery_v1','r1_cache_recovery_dispatch_v1','r1_cache_recovery_initial_control_v1',
    'r1_cache_recovery_delivery_dispatch_v1','r1_cache_recovery_delivery_v1')
FIGURE_NAMES=('r1_calibration_quality','r1_saved_gate_comparisons')
FIGURE_ARTIFACTS={name+'.'+ext for name in FIGURE_NAMES for ext in ('pdf','svg','png')}|{
    'plot_data_quality.csv','plot_data_gate.csv','CAPTIONS.txt','caption.tex','input_audit.json'}
CACHE_EXCLUSIONS=('r1_execution_v1/cache','r1_cache_recovery_v1/cache',
    'r1_cache_recovery_v1/visual/runtime_cache')
DEPENDENCIES=[
 dict(package='layered_raw_R0_bler_completed_20261013_v1',archive_sha256='da92b4471f931f67adcbbf1de624e59b6c024cbad9b4349fda09b288b7d10da6',server_receipt_sha256='0c4c7c92ab91474f67412cbf21a214165a53b57b4514fb42ea9351a089f4e8de'),
 dict(package='layered_raw_R0_clean_completed_20261014_v1',archive_sha256='967a3e173b04e81520258c783c429ac292af5d528abb19eeae5460173ae936e2',server_receipt_sha256='e46b74a5a0ced56b9b5ed3f53d7ecccc3701d056dcdaef891c43119b0011a0cb'),
 dict(package='layered_raw_R1_registered_20261014_v1',archive_sha256='2940def799f951767b950a9b37db045407d68a3aee0740645e53a194bdad2495',server_receipt_sha256='7a068212b0a8bd65311ef8bad5674f04973a9015ce9e3846402970469607b0db'),
 dict(package='layered_raw_R1_cache_recovery_entry_20261014_v1',archive_sha256='6b873c5d77661831a21f49568f011fac4f2cee9813561543e9192a5b3d02fe7b',server_receipt_sha256='f89da8edf7107901210e4a630f93372d3bf216d9f6b50d9dfd9c0fd7e7fbe11d'),
 dict(package='layered_raw_R1_cache_delivery_entry_20261014_v1',archive_sha256='376c5f09c5db7fa4aa03126035a788c3d7449edea98674aceb024e1a9636cbf7',server_receipt_sha256='e878a87c72288e16ac793e820741d732d1b171242ac46821cf559813f9b8c089')]


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def canonical(path):
    path = Path(path)
    # Must precede all file access, including stat of a forbidden traversal.
    require(path.is_absolute() and '..' not in path.parts, 'Noncanonical source path')
    return path


def regular(path, within=None):
    path = canonical(path)
    if within is not None: require(path.is_relative_to(within), 'Source escaped fixed scope')
    for ancestor in (*reversed(path.parents), path):
        info = ancestor.lstat()
        require(not stat.S_ISLNK(info.st_mode), 'Symlink source is forbidden')
    require(stat.S_ISREG(info.st_mode) and path.resolve() == path, 'Nonregular source')
    return path


def read(path): return json.loads(regular(path).read_text(encoding='utf-8-sig'))


def pin(path): return dict(path=str(regular(path)), sha256=sha(path), bytes=Path(path).stat().st_size)


def same_pin(a, b): return a.get('path') == b.get('path') and a.get('sha256') == b.get('sha256')


def checked(value, within=None):
    p = regular(value['path'], within)
    require(sha(p) == value['sha256'], 'Bound SHA differs: '+str(p))
    if 'bytes' in value: require(type(value['bytes']) is int and p.stat().st_size == value['bytes'], 'Bound bytes differ')
    return p


def process_identity(pid):
    try:
        text = (Path('/proc')/str(pid)/'stat').read_text()
        parts = text[text.rfind(')')+2:].split()
        return dict(pid=int(pid), start_ticks=int(parts[19]), state=parts[0])
    except FileNotFoundError: return None


def gone(identity, probe=process_identity):
    require(type(identity.get('pid')) is int and identity['pid']>0 and type(identity.get('start_ticks')) is int
        and identity['start_ticks']>0, 'Invalid process identity')
    now = probe(identity['pid'])
    require(now is None or now['start_ticks'] != identity['start_ticks'], 'Original process still exists, including zombie')


def published_file(path, commit):
    p = regular(path,R); rel = p.relative_to(R).as_posix()
    data = subprocess.check_output(['git','show',commit+':'+rel],cwd=R)
    require(hashlib.sha256(data).hexdigest()==sha(p),'Published Git bytes differ: '+rel)


def publication(commit, directory, push_sha):
    require(len(commit)==40 and len(push_sha)==64,'Explicit publication pins required')
    pub = canonical(directory)
    require(pub.is_relative_to(T/'baseline_strength_20261011'),'Unapproved publication directory')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==commit,'Unexpected current Git HEAD')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=R,text=True).strip(),'Tracked tree is dirty')
    require(sha(regular(pub/'push_verified.json'))==push_sha,'Normal-push proof SHA differs')
    p=read(pub/'push_verified.json')
    require(p['head']==p['verified_remote']==commit and p['normal_push'] is True and p['branch']=='main'
            and p['repository']=='https://github.com/daiqizai/var-comm','Missing successful normal push proof')
    checks=read(pub/'checks.json')
    require(all(type(checks[k]['exit_code']) is int and checks[k]['exit_code']==0 for k in ('verify_repository','run_cpu_checks')),'Original checks not passed')
    for row in read(pub/'staged_files.json'):
        require(sha(regular(R/row['path'],R))==row['sha256'],'Staged result changed')
    return dict(directory=str(pub),push=pin(pub/'push_verified.json'),git_commit=commit)


def walk_files(base,excluded=()):
    """Used only after successful actual exits, never for active discovery."""
    base=canonical(base);require(stat.S_ISDIR(base.lstat().st_mode),'Required scope is absent or symlink')
    for directory, dirs, names in os.walk(base,followlinks=False):
        for name in dirs:require(stat.S_ISDIR((Path(directory)/name).lstat().st_mode),'Symlink directory in scope')
        dirs[:]=[name for name in dirs if Path(directory)/name not in excluded]
        for name in names:yield regular(Path(directory)/name,base)


def safe_parts(name):
    p=PurePosixPath(name)
    require(name and '\\' not in name and ':' not in name and '\x00' not in name and not p.is_absolute()
            and str(p)==name and all(x not in ('.','..','') for x in p.parts),'Unsafe archive path')
    for part in p.parts:
        require(not part.endswith(('.', ' ')) and part.split('.')[0].upper() not in
                {'CON','PRN','AUX','NUL',*('COM'+str(i) for i in range(1,10)),*('LPT'+str(i) for i in range(1,10))},'Unsafe Windows name')
    return p.parts


def dependency_receipts():
    receipts=[]
    for dependency in DEPENDENCIES:
        p=T/'baseline_strength_20261011'/(dependency['package']+'_delivery')/'receipt.json'
        checked(dict(path=str(p),sha256=dependency['server_receipt_sha256']))
        require(read(p)['sha256']==dependency['archive_sha256'],'Old dependency receipt differs')
        receipts.append(dict(**dependency,receipt=pin(p),old_payload_rehashed=False))
    return receipts


def load_metadata(path,digest,name):
    checked(dict(path=str(path),sha256=digest))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(not any(k in sys.modules for k in ('torch','numpy','tensorflow','sionna','lpips')),
        'Packing must not import scientific implementations')
    return module


def root_pin(value,roots):
    require(set(value)=={'path','sha256','bytes'} and type(value['bytes']) is int and value['bytes']>=0,
        'Root authorization requires exact path/SHA/byte pins')
    path=canonical(value['path']);require(any(path.is_relative_to(root) for root in roots),'Evidence outside fixed task roots')
    return checked(value)


def identity(value):
    require(all(type(value.get(k)) is int and value[k]>0 for k in ('pid','start_ticks')),'Positive process generation required')
    return value['pid'],value['start_ticks']


def figure_closure(auth,delivery,probe=process_identity):
    """The root descriptor supplies actual receipt paths; no fabricated run schema."""
    require(auth['figure_entry']==pin(GRAPH) and auth['figure_entry']['sha256']==GRAPH_SHA,'Different frozen figure entry')
    execution=auth['figure_execution'];wait_path=root_pin(execution['actual_wait'],(L,BASE));wait=read(wait_path)
    process=execution['process'];identity(process)
    require(wait.get('actual_wait') is True and type(wait.get('returncode')) is int and wait['returncode']==0
        and not any(wait.get(k,False) for k in ('timeout','timed_out','terminated','killed','stopped'))
        and all(wait.get(k) is False for k in ('timed_out','terminated','killed'))
        and 'metadata_error' in wait and wait['metadata_error'] is None,'Actual figure process did not exit successfully')
    if 'child' in wait:require(identity(wait['child'])==identity(process),'Figure child generation differs')
    else:
        require(type(wait.get('pid')) is int and wait['pid']==process['pid'],'Actual figure wait PID differs')
        if 'start_ticks' in wait:require(type(wait['start_ticks']) is int and wait['start_ticks']==process['start_ticks'],'Figure wait generation differs')
    parent=execution['parent_process'];identity(parent);gone(process,probe);gone(parent,probe)
    expected_argv=[read(REG)['runtime']['python_cpu'],'-B',str(GRAPH),'--expected-registration',MASTER]
    require(execution['argv']==expected_argv,'Different figure command')
    # A separate pinned launch binds the exact process to the original argv.
    launch_path=root_pin(execution['launch'],(L,BASE));launch=read(launch_path)
    require(identity(launch['process'])==identity(process) and launch['argv']==expected_argv,
        'Figure launch process/argv differs')
    require(identity(launch['parent_process'])==identity(parent),'Figure parent differs')
    for item in ('stdout','stderr'):root_pin(execution[item],(L,BASE))
    result=read(execution['stdout']['path'])
    require(result['status']=='R1_FIGURE_FILES_READY_VISUAL_REVIEW_REQUIRED' and result['output']==str(FIGURES)
        and type(result['scientific_calls']) is int and result['scientific_calls']==0,'Figure stdout is not successful')
    staging=canonical(result['staging_preserved'])
    require(staging.parent==L and staging.name.startswith('r1_cache_recovery_figures_v1.preparing_')
        and stat.S_ISDIR(staging.lstat().st_mode) and not (staging/'failure.json').exists(),'Figure staging failed or differs')
    require(not (wait_path.parent/'failure.json').exists(),'Figure execution failure receipt present')
    for item,filename in (('figure_manifest','figure_manifest.json'),('figure_completion','completion.json')):
        require(root_pin(auth[item],(FIGURES,))==FIGURES/filename,'Figure receipt path differs')
    manifest=read(FIGURES/'figure_manifest.json');complete=read(FIGURES/'completion.json')
    master=dict(path=str(REG),sha256=MASTER)
    require(same_pin(manifest['registration'],master) and same_pin(complete['registration'],master)
        and same_pin(manifest['input_manifest'],delivery['manifest'])
        and complete['manifest_sha256']==sha(FIGURES/'figure_manifest.json')
        and complete['status']=='COMPLETE_R1_FIGURE_FILES_VISUAL_REVIEW_PENDING'
        and complete['requires_actual_plot_process_exit_zero'] is True
        and manifest['visual_review']=='PENDING_REQUIRED','Figure provenance differs')
    require(manifest['exports']==['vector_PDF','editable_text_SVG','600dpi_PNG']
        and type(complete['scientific_calls']) is int and complete['scientific_calls']==0
        and type(manifest['scientific_calls']) is int and manifest['scientific_calls']==0,'Figure export or science scope differs')
    files={}
    for item in manifest['files']:
        path=root_pin(item,(FIGURES,));require(path.parent==FIGURES and path.name not in files,'Duplicate/nested figure artifact')
        files[path.name]=item
    require(set(files)==FIGURE_ARTIFACTS,'Missing or unexpected fixed final figure artifact')
    expected=set(files)|{'figure_manifest.json','completion.json','completion.pending.json'}
    review_path=root_pin(auth['visual_review'],(O,));review=read(review_path)
    # The future descriptor requires this normalized review record. It is only
    # produced after a human actually opens the two final PNGs; no placeholder.
    require(review['status']=='PASS' and review['actual_opened_both_pngs'] is True
        and same_pin(review['registration'],master) and same_pin(review['figure_manifest'],auth['figure_manifest'])
        and isinstance(review['reviewer'],str) and bool(review['reviewer'].strip()),'Actual final visual review is missing')
    wanted=[files[name+'.png'] for name in FIGURE_NAMES]
    require(len(review['figures'])==2 and {x['path'] for x in review['figures']}=={x['path'] for x in wanted},'Review must name exactly both final PNGs')
    for item in review['figures']:
        require(item in wanted,'Reviewed PNG bytes differ from final manifest');checked(item)
    if review_path.parent==FIGURES:expected.add(review_path.name)
    require({p.name for p in walk_files(FIGURES)}==expected,'Unknown or missing final figure artifact')
    require(sha(regular(FIGURES/'completion.pending.json'))==auth['figure_completion']['sha256'],'Figure pending/completion bytes differ')
    pending=(FIGURES/'completion.pending.json').stat();final=(FIGURES/'completion.json').stat()
    require((pending.st_dev,pending.st_ino)==(final.st_dev,final.st_ino),'Figure completion is not the same-file publication marker')
    return dict(manifest=auth['figure_manifest'],completion=auth['figure_completion'],visual_review=auth['visual_review'],
        execution=execution,processes=[process,parent],staging_excluded=str(staging),all_figure_files_SHA_verified=True)


def closure(args):
    require(os.name=='posix' and Path('/proc').is_dir(),'Original Linux server required')
    ap=dict(path=args.authorization_path,sha256=args.authorization_sha256,bytes=args.authorization_bytes)
    root_pin(ap,(O,BASE));auth=read(ap['path'])
    require(auth['schema']==AUTH_SCHEMA and auth['status']=='AUTHORIZED_AFTER_ACTUAL_FINAL_CLOSURE'
        and auth['scientific_calls']==0 and type(auth['scientific_calls']) is int,'Final root descriptor is not actual closure authorization')
    require(auth['registration']==pin(REG) and auth['registration']['sha256']==MASTER,'Recovery registration differs')
    module=load_metadata(ENTRY,ENTRY_SHA,'_final_backup_recovery_delivery')
    original=module.load_original(R)
    reg,process=module.preflight(original,R,L,dict(path=str(REG),sha256=MASTER))
    graph=load_metadata(GRAPH,GRAPH_SHA,'_final_backup_figure_metadata')
    # Only read-only receipt/manifest guard; never load_tables, render, deep_audit or run.
    _,delivery=graph.guard_delivery(R,L,CONTROL)
    for key,target in (('delivery_actual_wait',CONTROL/'actual_wait.json'),('delivery_completion',LIGHT/'completion.json'),
                       ('delivery_audit',LIGHT/'closure_audit.json')):
        require(root_pin(auth[key],(CONTROL,LIGHT))==target,'Different root-pinned delivery evidence')
    audit=read(LIGHT/'closure_audit.json')
    require(audit['process_closure']==process,'Completed delivery and current mixed-stage closure disagree')
    science=audit['scientific_closure']
    require(science['physical_frames']==32400 and science['logical_frames']==43200 and science['source_count']==200
        and science['noise_count']==3 and all(science[k] is True for k in ('no_reselection','no_inference','no_bootstrap','all_signal_failures_retained')),
        'Completed scientific grid differs')
    for stage in ('visual','score'):
        done=read(RUN/stage/'completion.json');summary=done['ledger'] if stage=='visual' else done['counts']
        require(summary['unresolved']==0 and summary['reserved']==summary['completed'],'Unclosed new model/metric ledger')
        ledger=done['ledger_file'] if stage=='visual' else done['ledger']
        require(checked(ledger,RUN/stage)==RUN/stage/'calls.sqlite','Different actual ledger')
        for suffix in ('-wal','-shm','-journal'):
            side=Path(str(ledger['path'])+suffix)
            require(not side.exists() or side.stat().st_size==0,'Mutable SQLite sidecar remains')
    figure=figure_closure(auth,delivery)
    pub=publication(auth['expected_commit'],auth['publication_directory'],auth['push_verification_sha256'])
    for directory in (LIGHT,FIGURES):
        for path in walk_files(directory):
            if path.name!='completion.pending.json':published_file(path,auth['expected_commit'])
    for path in (ENTRY,GRAPH,REG,Path(auth['visual_review']['path'])):published_file(path,auth['expected_commit'])
    science_processes=[]
    for path in (OLD/'owner_launch.json',L/'r1_resource_dispatch_v2/launch.json',
            L/'r1_initial_dispatch_control_v1/launch.json',RUN/'owner_launch.json',
            L/'r1_cache_recovery_dispatch_v1/launch.json',L/'r1_cache_recovery_initial_control_v1/dispatch_snapshot.json'):
        science_processes.append(read(path)['process'])
    for segment,stages in ((OLD,('phy','projection','visual')),(RUN,('visual','score','selection'))):
        for stage in stages:science_processes.append(read(segment/(stage+'_launch.json'))['process'])
    return dict(authorization=ap,registration=auth['registration'],reg=reg,publication=pub,
        process=process,delivery=delivery,figures=figure,delivery_entry=pin(ENTRY),figure_entry=pin(GRAPH),
        processes=science_processes+figure['processes']+[read(CONTROL/'launch.json')[k] for k in ('owner','child')])


def payload_name(name):
    safe_parts(name);p=PurePosixPath(name)
    prefixes=tuple('runtime/'+n+'/' for n in RUN_SCOPES)+(
        'repo/results/layered_mismatch_raw_20261013/r1_cache_recovery_delivery_v1/',
        'repo/results/layered_mismatch_raw_20261013/r1_cache_recovery_figures_v1/',
        'repo/results/layered_mismatch_raw_20261013/r1_cache_failure_audit_v1/',
        'repo/experiments/layered-mismatch-raw-20261013/', 'evidence/')
    exact={'manifest.json','DEPENDENCIES.md','code_bindings.json','authorization.json',
        'runtime/r1_cache_recovery_delivery_v1.claim.json','runtime/r1_cache_recovery_figures_v1.claim.json',
        'repo/results/layered_mismatch_raw_20261013/r1_cache_recovery_v1/completion.json',
        'repo/results/layered_mismatch_raw_20261013/r1_cache_recovery_v1/recovery_completion.json',
        *('publication/'+n for n in PUB_NAMES)}
    require(name in exact or name.startswith(prefixes),'Unapproved final R1 payload scope')
    require('registered' not in p.parts and p.suffix.lower() not in ('.pth','.pt','.safetensors')
        and not any(k in p.parts for k in ('assets','source_restore','r0_bler_v1','r0_clean_supplement_v1')),
        'Prior model/source/R0 assets must remain dependencies')
    require(not any(name=='runtime/'+x or name.startswith('runtime/'+x+'/') for x in CACHE_EXCLUSIONS),
        'Reproducible private runtime/compiler cache is excluded')
    if name.startswith('runtime/') and name not in exact:
        parts=p.parts;scope=parts[1];relative=parts[2:]
        require(relative,'Missing runtime file')
        if scope in ('r1_execution_v1','r1_cache_recovery_v1'):
            stages={'phy','projection','visual'} if scope=='r1_execution_v1' else {'visual','score','selection'}
            require((len(relative)==1 and (p.suffix=='.json' or p.name=='owner.lock'))
                or relative[0] in stages|{'prelaunch','logs'},'Unknown scientific runtime subtree')
        require(p.suffix.lower() in ('.json','.jsonl','.csv','.txt','.md','.log','.lock','.sqlite','.npz','.npy')
            or p.name in ('calls.sqlite-wal','calls.sqlite-shm','calls.sqlite-journal'),
            'Unknown scientific/runtime payload extension')
    if name.startswith('repo/experiments/'):
        require(p.suffix in ('.py','.md','.json','.txt','.csv'),'Non-light code artifact')
    return name


def payload_sources(verified):
    files=[]
    for name in RUN_SCOPES:
        base=L/name;files.extend((p,'runtime/'+name+'/'+p.relative_to(base).as_posix())
            for p in walk_files(base,tuple(L/x for x in CACHE_EXCLUSIONS)))
    for base in (LIGHT,FIGURES,O/'r1_cache_failure_audit_v1'):
        files.extend((p,'repo/'+p.relative_to(R).as_posix()) for p in walk_files(base))
    for name in ('completion.json','recovery_completion.json'):
        p=O/'r1_cache_recovery_v1'/name;files.append((p,'repo/'+p.relative_to(R).as_posix()))
    for name in ('r1_cache_recovery_delivery_v1','r1_cache_recovery_figures_v1'):
        p=L/(name+'.claim.json');files.append((p,'runtime/'+p.name))
    for path,digest in verified['reg']['code_bindings'].items():
        p=Path(path)
        if p.is_relative_to(E) and (p.relative_to(E).parts[0].startswith('r1_') or p.name in ('contract.py','grouped_wire.py','policy_selection.py')):
            checked(dict(path=str(p),sha256=digest),E);files.append((p,'repo/'+p.relative_to(R).as_posix()))
    for name in ('r1_cache_recovery_delivery_v1','r1_cache_recovery_delivery_dispatch_v1','r1_cache_recovery_figures_v1'):
        for p in walk_files(E/name):
            if p.suffix in ('.py','.md','.json'):
                published_file(p,verified['publication']['git_commit']);files.append((p,'repo/'+p.relative_to(R).as_posix()))
    for key in ('actual_wait','launch','stdout','stderr'):
        p=checked(verified['figures']['execution'][key]);files.append((p,'evidence/figure_'+key+p.suffix))
    p=checked(verified['figures']['visual_review'])
    if not p.is_relative_to(FIGURES):files.append((p,'evidence/visual_review.json'))
    files.append((checked(verified['authorization']),'authorization.json'))
    for name in PUB_NAMES:files.append((regular(Path(verified['publication']['directory'])/name),'publication/'+name))
    for p,name in files:
        regular(p);payload_name(name)
        if p.name in ('calls.sqlite-wal','calls.sqlite-shm','calls.sqlite-journal'):
            require(p.stat().st_size==0,'Nonempty SQLite sidecar cannot be archived as closed')
    return files


def empty_runtime_directories():
    """Preserve evidence such as the original failed renderer's empty images/ directory."""
    result=[];excluded={L/x for x in CACHE_EXCLUSIONS}
    for name in RUN_SCOPES:
        base=L/name
        for directory,dirs,names in os.walk(base,followlinks=False):
            path=Path(directory)
            require(stat.S_ISDIR(path.lstat().st_mode),'Nonregular runtime directory')
            for child in dirs:require(stat.S_ISDIR((path/child).lstat().st_mode),'Runtime directory symlink')
            original_dirs=list(dirs);dirs[:]=[child for child in dirs if path/child not in excluded]
            if not original_dirs and not names:
                target='runtime/'+name+'/'+path.relative_to(base).as_posix()
                payload_name(target+'/.empty_directory_metadata.json');result.append(target)
    return sorted(result)


def check_archived_references(records,aliases):
    by_path={r['source_path']:r for r in records+aliases};scopes=tuple(L/name for name in RUN_SCOPES)+(LIGHT,FIGURES)
    excluded_references=[]
    for row in records:
        path=Path(row['source_path'])
        if path.suffix!='.json':continue
        for value in inner_pins(read(path)):
            p=canonical(value['path'])
            if any(p.is_relative_to(L/x) for x in CACHE_EXCLUSIONS):
                excluded_references.append(dict(referencing_file=row['path'],original_pin=value,
                    reason='EXPLICIT_REPRODUCIBLE_PRIVATE_CACHE_DEPENDENCY'));continue
            if any(p.is_relative_to(base) for base in scopes):
                require(str(p) in by_path,'Internal fixed payload reference omitted: '+str(p))
                saved=by_path[str(p)]
                require(saved['sha256']==value['sha256'] and ('bytes' not in value or saved['bytes']==value['bytes']),
                    'Saved payload/reference SHA differs')
    return excluded_references


def build(args):
    require(not B.exists() and not B.is_symlink(),'Existing attempt must be inspected, never rebuilt automatically')
    verified=closure(args)
    import fcntl
    with contextlib.ExitStack() as stack:
        for path in LOCKS:
            stream=stack.enter_context(regular(path).open('rb'));fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=payload_sources(verified);B.mkdir()
        try:
            dependency=B/'DEPENDENCIES.md'
            dependency.write_text('''# Final recovered R1 scientific increment

This is the complete closed 32400 physical / 43200 logical calibration200
increment: original PHY and RX projection, the diagnosed original failed visual
initialization and owner exit1, new completed visual/score/selection, their actual
waits/arrays/ledgers, CPU delivery, and the two final visually reviewed figures.
The one failed initialization did not complete any model load or forward.
No failure is relabelled as success; no inference, metric, ranking, bootstrap or
R2 call is repeated during backup. This package does not include R2.

Keep the existing R0 BLER/clean, R1 registered, cache-recovery-entry and
cache-delivery-entry increments, prior required_static/source_restore and original
model/data/environment recovery packages, baseline final_training/formal/
diagnostic/timing_entry/input_restore and EMAIN/timing_recovery dependencies.
Models, original source images, the full prior model_bundle and old R0 large
payloads are not duplicated. Dependency receipts are checked; their large
archives are not rehashed. This is not an independent full recovery package.

Complete canonical files are included, with source path/SHA and same-inode aliases.
Aliases restore as verified ordinary files, never new PFS symlinks. Duplicate
publication staging, OLD/cache, RUN/cache, RUN/visual/runtime_cache and
Matplotlib caches outside the final runtimes are omitted. Scientific image/metric
reuse arrays and original failure resource history remain complete.
Never resume completed science from the existence of this backup.
''',encoding='utf8')
            code=B/'code_bindings.json';save(code,dict(registration=verified['registration'],
                code_bindings=verified['reg']['code_bindings'],delivery_entry=verified['delivery_entry'],figure_entry=verified['figure_entry']))
            files.extend(((dependency,'DEPENDENCIES.md'),(code,'code_bindings.json')))
            records,aliases=deduplicate(files);excluded_refs=check_archived_references(records,aliases)
            manifest=dict(schema=SCHEMA,git_commit=verified['publication']['git_commit'],**FLAGS,
                verification={k:v for k,v in verified.items() if k!='reg'},dependencies=dependency_receipts(),
                files=records,aliases=aliases,empty_directories=empty_runtime_directories(),
                alias_restore='Verified ordinary files; no symlinks',old_dependency_archives_reverified=False,
                excluded_reproducible_cache_roots=[str(L/x) for x in CACHE_EXCLUSIONS],excluded_cache_references=excluded_refs,
                size_before_compression=dict(canonical_payload_bytes=sum(r['bytes'] for r in records),
                    alias_restore_bytes=sum(r['bytes'] for r in aliases),canonical_payload_files=len(records),
                    alias_restore_files=len(aliases),compression_ratio_not_assumed=True))
            mp=B/'manifest.json';save(mp,manifest);archive=B/(PACKAGE+'.tar.gz');partial=B/(PACKAGE+'.tar.gz.partial')
            with tarfile.open(partial,'w:gz',compresslevel=1,dereference=True) as tar:
                for row in records:tar.add(row['source_path'],arcname=row['path'],recursive=False)
                tar.add(mp,arcname='manifest.json',recursive=False)
            verify_archive(partial,mp,records)
            for row in records+aliases:stable_record(row)
            for process in verified['processes']:gone(process)
            for value in inner_pins({k:v for k,v in verified.items() if k!='reg'}):checked(value)
            partial.rename(archive)
            receipt=dict(schema=SCHEMA,archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
                manifest_sha256=sha(mp),payload_files=len(records),payload_bytes=sum(r['bytes'] for r in records),
                alias_files=len(aliases),alias_bytes=sum(r['bytes'] for r in aliases),restored_files=len(records)+len(aliases),
                git_commit=verified['publication']['git_commit'],**FLAGS,archive_payload_SHA_PASS=True,created_unix=time.time())
            save(B/'receipt.json',receipt);return dict(receipt=receipt,receipt_sha256=sha(B/'receipt.json'))
        except BaseException:
            save(B/'build_failure.json',dict(status='FAILED_PRESERVE_NO_REBUILD',traceback=traceback.format_exc(),science_calls=0));raise


def inner_pins(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):yield value
        for item in value.values():yield from inner_pins(item)
    elif isinstance(value,list):
        for item in value:yield from inner_pins(item)


def deduplicate(files):
    """One canonical path/inode payload. Different aliases are restored by fetch."""
    records=[];aliases=[];paths={};inodes={};names={}
    for path,name in sorted(files,key=lambda x:x[1]):
        path=regular(path);payload_name(name)
        folded=name.casefold()
        if path in paths and paths[path]==name:continue
        require(folded not in names,'Case-colliding/different duplicate archive name')
        names[folded]=path;info=path.stat();signature=stat_signature(path);inode=(info.st_dev,info.st_ino)
        old=inodes.get(inode) if info.st_ino else None
        if old is None:
            record=dict(path=name,source_path=str(path),sha256=sha(path),bytes=info.st_size,source_stat=signature)
            records.append(record)
            if info.st_ino:inodes[inode]=record
        else:
            require(info.st_size==old['bytes'] and sha(path)==old['sha256'],'Hardlinked source changed')
            aliases.append(dict(path=name,source_path=str(path),target=old['path'],sha256=old['sha256'],bytes=old['bytes'],reason='SAME_INODE',source_stat=signature))
        require(stat_signature(path)==signature,'Source changed while hashing')
        paths[path]=name
    return records,aliases


def stat_signature(path):
    info=regular(path).stat()
    return dict(device=info.st_dev,inode=info.st_ino,bytes=info.st_size,mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns)


def stable_record(row):
    path=regular(row['source_path'])
    require(stat_signature(path)==row['source_stat'] and sha(path)==row['sha256']
        and stat_signature(path)==row['source_stat'],'Source changed during packing')


def save(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())


def verify_archive(path,manifest_path,records):
    expected={x['path']:x for x in records};expected['manifest.json']=pin(manifest_path);seen=set()
    with tarfile.open(path,'r:gz') as tar:
        for member in tar:
            payload_name(member.name)
            require(member.isfile() and not member.issym() and not member.islnk() and member.name in expected and member.name not in seen,'Invalid archive member')
            seen.add(member.name);h=hashlib.sha256();n=0
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(8<<20),b''):h.update(block);n+=len(block)
            wanted=expected[member.name]
            require(n==wanted['bytes'] and h.hexdigest()==wanted['sha256'],'Archive payload differs')
    require(seen==set(expected),'Incomplete archive')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization-path',required=True);parser.add_argument('--authorization-sha256',required=True)
    parser.add_argument('--authorization-bytes',required=True,type=int)
    print(json.dumps(build(parser.parse_args())),flush=True)
