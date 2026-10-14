"""Closed six-method timing increment. No experiment module is imported or run."""
from __future__ import annotations
import argparse,base64,contextlib,hashlib,json,os,stat,subprocess,sys,tarfile,time,traceback
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/baseline-delivery-20261012';O=R/'results/baseline_strength_20261011/timing'
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm');BASE=T/'baseline_strength_20261011'
RUN=BASE/'six_method_timing_v6';OUT=O/'six_method_v6';LIGHT=O/'six_method_v6_delivery_v1'
CONTROL=BASE/'six_method_timing_v6_resource_v1';D=BASE/'six_method_timing_v6_resource_dispatch_v1'
I=BASE/'six_method_timing_v6_resource_initial_control_v1';P=BASE/'six_method_timing_v6_resource_postprocess_dispatch_v1'
REG=O/'six_method_v6_resource_v1/registration.json';ENTRY=E/'timing_resource_revision_v1/timing_resource_owner.py'
MASTER='9532c982e0fd349fb7301cfc43eba0755f0276bb699e4754bd7eec8e34507392'
ORIGINAL='54aafaca38b177266ef2c21cac23fce4dee2a0658e9bd9a0cfc5d2836baa269a'
PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
PACKAGE='six_method_timing_final_20261014_v1';B=BASE/(PACKAGE+'_delivery')
SOURCE=E/'timing_final_backup_v1'
SCHEMA='SIX_METHOD_TIMING_FINAL_INCREMENT_V1';AUTH_SCHEMA='SIX_METHOD_TIMING_FINAL_BACKUP_AUTHORIZATION_V1'
FLAGS=dict(complete_six_method_result=True,scientific_calls=0,bootstrap_calls=0,repeated_grid_audit=False,
 contains_active_science=False,standalone_full_recovery=False,contains_models=False,contains_R0=False,contains_R1_scientific_payload=False,
 inherited_complete_calls=576,new_complete_calls=1152,total_complete_calls=1728,measurement_calls=864,
 packet_decodes=2304,known_attempts=1729,historical_failed_partial_preserved_by_dependency=True,shared_environment=True)
PUB_NAMES=('checks.json','commit.json','push_verified.json','staged_files.json')
RUN_SCOPES=('six_method_timing_v6','six_method_timing_v6_resource_v1','six_method_timing_v6_resource_dispatch_v1',
 'six_method_timing_v6_resource_initial_control_v1','six_method_timing_v6_resource_postprocess_dispatch_v1',
 'timing_resource_revision_registration_20261014_v1','timing_resource_revision_qualification_20261014_v1')
REPO_SCOPES=(OUT,LIGHT,REG.parent,O/'six_method_v6_resource_preparation_v1',E/'timing_revision_v6',
 E/'timing_resource_revision_v1',E/'timing_postprocess_v1',SOURCE)
CACHE_EXCLUSIONS=(RUN/'torchinductor_root',)
LOCKS=tuple(O/('six_method_v%d'%n)/'owner.lock' for n in range(1,7))+tuple(T/'layered_mismatch_raw_20261013'/n for n in
 ('r1_execution_v1/owner.lock','r1_resource_dispatch_v2/dispatch.lock','r1_cache_recovery_v1/owner.lock',
 'r1_cache_recovery_dispatch_v1/dispatch.lock','r1_cache_recovery_delivery_dispatch_v1/dispatch.lock'))
DEPENDENCIES=[dict(package='timing_recovery_20261013_v1',archive_sha256='165d4a5b8dca4530bdfc3d886e267c66f490ad146796a3afb429fafffe3eaa41',
 server_receipt_sha256='9847ff9f1c0e33eea49a0b3e32e6ec2072708ad7967fab3ed6669685ab9d2b50',
 local_verification_sha256='9f029c8e8acd10df478f95fab7176aa49fd51886337587900ce542e18c2d93b6')]
SOURCE_NAMES=('build.py','fetch.py','closed_evidence.json','test_backup.py','README.md')
DOC_NAMES={'README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md','release_manifest.json'}


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


def identity(value):
    require(all(type(value.get(k)) is int and value[k]>0 for k in ('pid','start_ticks')),'Positive process generation required')
    return value['pid'],value['start_ticks']


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


def payload_name(name):
    parts=safe_parts(name);p=PurePosixPath(name)
    exact={'manifest.json','DEPENDENCIES.md','authorization.json','closed_evidence.json'}
    allowed=name in exact or name.startswith('publication/')
    if name.startswith('runtime/'):
        allowed=len(parts)>=3 and parts[1] in RUN_SCOPES
        require(not (parts[1:3]==('six_method_timing_v6','torchinductor_root')),'Exact compiler cache excluded')
    if name.startswith('repo/'):
        rel=p.relative_to('repo');r=rel.parts
        allowed=(str(rel) in DOC_NAMES or (len(r)>=4 and r[:3]==('results','baseline_strength_20261011','timing')
                  and (r[3].startswith('six_method_v6') or r[3]=='six_method_timing_final_backup_preparation_v1')) or
                 (len(r)>=3 and r[:2]==('experiments','baseline-delivery-20261012') and r[2].startswith('timing_')) or
                 str(rel) in {'results/baseline_strength_20261011/README.md','results/layered_mismatch_raw_20261013/README.md'})
    require(allowed,'Unapproved final timing payload scope')
    require(p.suffix.lower() in ('.json','.jsonl','.md','.txt','.log','.lock','.py','.csv','.tex'),
            'Only fixed timing metadata/code/log payloads; no model/data/environment')
    return name


def validate_summary(done):
    require(done['status']=='COMPLETE_FIXED16_SIX_METHOD_TIMING','Timing is incomplete')
    expected=dict(frames=1728,measured_frames=864,packet_calls=2304,inherited_complete_frames=576,
        newly_completed_frames=1152,completed_frames_across_attempts=1728,failed_partial_visual_attempts=1,
        known_fresh_visual_attempts_across_attempts=1729,physical_endpoint_entries_across_attempts=1729,
        original_intervals_recomputed=0,training_updates=0,prior_failure_preserved=True,
        prior_failed_slot_not_counted_as_warmup=True,shared_environment=True,exclusive_or_uncontended_claimed=False)
    for key,value in expected.items():
        require(type(done[key]) is type(value) and done[key]==value,'Unexpected completed timing scope: '+key)
    return expected


def strict_wait(wait,launch,child_key,expected_argv,probe=process_identity):
    require(wait.get('actual_wait') is True and type(wait.get('returncode')) is int and wait['returncode']==0,
            'Missing actual successful wait')
    require(all(wait.get(k) is False for k in ('timed_out','terminated','killed')) and
            'metadata_error' in wait and wait['metadata_error'] is None,'Failed or uncertain wait')
    require(wait['argv']==launch['argv']==expected_argv,'Actual wait/launch command differs')
    process=wait[child_key];parent=wait['parent_process']
    require(identity(process)==identity(launch['process']) and identity(parent)==identity(launch['parent_process']),
            'Actual wait generation differs from launch')
    gone(process,probe);gone(parent,probe)
    return [process,parent]


def evidence_pins():
    value=read(SOURCE/'closed_evidence.json')
    require(value['schema']=='TIMING_FINAL_KNOWN_CLOSED_EVIDENCE_V1','Different closed evidence schema')
    rows=value['files'];require(len({r['path'] for r in rows})==len(rows),'Duplicate evidence path')
    for row in rows:
        path=canonical(row['path'])
        require(any(path.is_relative_to(x) for x in (*REPO_SCOPES,*(BASE/n for n in RUN_SCOPES))),
                'Unexpected fixed evidence scope')
        require(set(row)=={'path','bytes','sha256'} and type(row['bytes']) is int,'Non-exact evidence pin')
        checked(row)
    return rows


def wrapper_identity(value):
    """Only frozen original wrapper receipts used decimal-string start_ticks."""
    require(type(value.get('pid')) is int and value['pid']>0,'Invalid original wrapper PID')
    ticks=value.get('start_ticks')
    require(type(ticks) is str and ticks.isascii() and ticks.isdecimal() and not ticks.startswith('0'),
            'Original wrapper generation must be its exact positive decimal string')
    result=dict(value,start_ticks=int(ticks));identity(result);return result


def receipt_closure(probe=process_identity):
    """Read the already-passed audit receipts, never parse/recompute the call grid."""
    rows=evidence_pins();known={r['path']:r for r in rows}
    require(known[str(REG)]['sha256']==MASTER and known[str(OUT/'registration.json')]['sha256']==ORIGINAL,
            'Original or supplemental registration differs')
    resource=read(REG)
    for path,digest in resource['original_source_sha256'].items():
        require(Path(path).is_relative_to(E),'Original source outside timing repository scope')
        checked(dict(path=path,sha256=digest))
    done=read(OUT/'completion.json');summary=validate_summary(done)
    original_wait=read(OUT/'actual_wait.json')
    require(set(original_wait)=={'actual_child_waited','exit_code','timed_out'} and
            original_wait['actual_child_waited'] is True and type(original_wait['exit_code']) is int and
            original_wait['exit_code']==0 and original_wait['timed_out'] is False,'Original wait is not actual exit zero')
    processes=[]
    owner_argv=[PY,'-B',str(ENTRY),'run','--expected-resource-registration',MASTER]
    processes+=strict_wait(read(D/'actual_wait.json'),read(D/'launch.json'),'child',owner_argv,probe)
    owner=read(CONTROL/'owner_resource_execution.json');worker=read(CONTROL/'worker_resource_execution.json')
    for role,item,argv in (('owner',owner,owner_argv),('worker',worker,[PY,str(ENTRY),'_worker'])):
        require(item['role']==('run' if role=='owner' else '_worker') and item['process']['argv']==argv,
                'Original wrapper execution role/argv differs')
        require(item['resource_registration']==pin(REG) and item['original_registration']==pin(OUT/'registration.json'),
                'Executed registration differs')
        returned=read(CONTROL/(role+'_returned.json'))
        require(returned['status']=='ORIGINAL_V6_RETURNED_SUCCESS' and returned['resource_registration']==pin(REG)
                and returned['original_completion']==pin(OUT/'completion.json') and
                returned['process']==item['process'],'Returned wrapper binding differs')
        for value in (item['process'],item['parent_process']):
            process=wrapper_identity(value);gone(process,probe);processes.append(process)
    require(identity(wrapper_identity(owner['process']))==identity(read(D/'launch.json')['process']) and
            identity(wrapper_identity(worker['parent_process']))==identity(wrapper_identity(owner['process'])),'Owner/worker ancestry differs')
    require(identity(read(I/'launch.json')['process'])==identity(wrapper_identity(owner['parent_process'])),'Supervisor generation differs')
    pp_argv=[PY,'-B',str(ENTRY),'postprocess','--output',str(LIGHT),'--expected-resource-registration',MASTER]
    processes+=strict_wait(read(P/'actual_wait.json'),read(P/'launch.json'),'process',pp_argv,probe)
    post=read(LIGHT/'completion.json')
    require(post['status']=='COMPLETE_CPU_TIMING_AUDIT_AND_TABLES' and post['input_registration_sha256']==ORIGINAL
            and type(post['measured_calls']) is int and post['measured_calls']==864
            and post['timing_scope']=='shared-environment latency'
            and type(post['new_scientific_calls']) is int and post['new_scientific_calls']==0
            and type(post['bootstrap_calls']) is int and post['bootstrap_calls']==0,'Postprocess scope differs')
    expected=set(post['output_sha256'])|{'completion.json','resource_revision.json'}
    require(len(expected)==13 and {p.name for p in walk_files(LIGHT)}==expected,'Final output set differs')
    for name,digest in post['output_sha256'].items():
        require(Path(name).name==name,'Unexpected final output path');checked(dict(path=str(LIGHT/name),sha256=digest))
    returned=read(CONTROL/'postprocess_returned.json');side=read(LIGHT/'resource_revision.json')
    require(returned['status']=='ORIGINAL_POSTPROCESS_AND_RESOURCE_SIDECAR_RETURNED_SUCCESS'
        and returned['binding']==pin(CONTROL/'postprocess_binding.json')
        and returned['resource_revision']==pin(LIGHT/'resource_revision.json')
        and returned['original_completion']==pin(LIGHT/'completion.json'),'Postprocess return differs')
    require(side['binding']['resource_registration']==pin(REG) and side['binding']['original_registration']==pin(OUT/'registration.json')
        and side['startup_free_MiB']==8192 and side['prior_startup_free_MiB']==18432
        and side['allocator_cap_GiB']==16 and side['runtime_free_floor_MiB']==2048
        and side['shared_environment'] is True and side['full_resource_conditions_identical'] is False
        and side['original_statistics_functions_changed'] is False,'Resource sidecar differs')
    for key in ('case_journal','raw_wall_and_audit','per_boundary_resource_history','pure_provenance_exclusions'):
        checked(done[key])  # SHA only; no repeated grid/metric computation.
    return dict(fixed_evidence=rows,summary=summary,processes=processes,original_completion=pin(OUT/'completion.json'),
        original_wait=pin(OUT/'actual_wait.json'),postprocess_completion=pin(LIGHT/'completion.json'),
        postprocess_actual_wait=pin(P/'actual_wait.json'),resource_sidecar=pin(LIGHT/'resource_revision.json'))


def dependency_receipts():
    result=[]
    for expected in DEPENDENCIES:
        p=BASE/(expected['package']+'_delivery')/'backup_archive_receipt.json'
        checked(dict(path=str(p),sha256=expected['server_receipt_sha256']))
        require(read(p)['sha256']==expected['archive_sha256'],'Historical timing dependency differs')
        result.append(dict(**expected,receipt=pin(p),old_payload_rehashed=False))
    return result


def no_current_failure():
    for root in (OUT,CONTROL,D,I,P,RUN):
        for path in walk_files(root,CACHE_EXCLUSIONS):
            require(not any('failure' in part.lower() for part in path.relative_to(root).parts),
                    'New timing failure evidence exists: '+str(path))


def publication_receipts(directory,commit):
    directory=canonical(directory)
    require(directory==BASE/'six_method_timing_final_publication_20261014_v1','Different final timing publication scope')
    pins={name:pin(directory/name) for name in PUB_NAMES}
    record=read(directory/'commit.json')
    require(record['head']==commit and record['normal_commit'] is True and
        record['checks_sha256']==pins['checks.json']['sha256'] and
        record['staged_files_sha256']==pins['staged_files.json']['sha256'],'Normal commit receipt differs')
    for filename in ('commit_actual_wait.json','push_actual_wait.json'):
        value=read(directory/filename)
        require(value['actual_wait'] is True and type(value['returncode']) is int and value['returncode']==0,
                'Publication command did not actually exit zero')
        pins[filename]=pin(directory/filename)
    return pins


def closure(args):
    require(os.name=='posix' and Path('/proc').is_dir(),'Original Linux server required')
    ap=dict(path=args.authorization_path,sha256=args.authorization_sha256,bytes=args.authorization_bytes)
    require(set(ap)=={'path','sha256','bytes'} and type(ap['bytes']) is int,'Exact root authorization required')
    p=checked(ap,BASE);require(p.parent==BASE/(PACKAGE+'_root_authorization_v1') and p.name=='authorization.json',
                             'Unexpected backup authorization path')
    auth=read(p)
    require(auth['schema']==AUTH_SCHEMA and auth['status']=='AUTHORIZED_AFTER_ACTUAL_TIMING_PUBLICATION'
        and auth['package']==PACKAGE and auth['resource_registration']==pin(REG)
        and type(auth['scientific_calls']) is int and auth['scientific_calls']==0,'Missing actual final authorization')
    require(set(auth['source_pins'])==set(SOURCE_NAMES),'All five helper source pins required')
    require(Path(__file__).resolve()==SOURCE/'build.py','Run the exact deployed helper')
    for name,value in auth['source_pins'].items():require(checked(value,SOURCE)==SOURCE/name,'Backup source binding differs')
    closed=receipt_closure();no_current_failure()
    pub=publication(auth['expected_commit'],auth['publication_directory'],auth['push_verification_sha256'])
    publication_pins=publication_receipts(auth['publication_directory'],auth['expected_commit'])
    for name in SOURCE_NAMES:published_file(SOURCE/name,auth['expected_commit'])
    for base in (LIGHT,REG.parent,E/'timing_resource_revision_v1'):
        for item in walk_files(base):published_file(item,auth['expected_commit'])
    support=auth['support_evidence'];require(isinstance(support,list) and bool(support),'Qualification/review support required')
    for value in support:
        q=checked(value)
        require(q.is_relative_to(O/'six_method_timing_final_backup_preparation_v1') or
                q.is_relative_to(BASE/'timing_final_backup_qualification_20261014_v1'),
                'Unexpected new helper review/qualification support')
    for key in ('qualification_actual_wait','independent_review'):
        require(auth[key] in support,'Required qualified support absent from payload')
        q=checked(auth[key],O/'six_method_timing_final_backup_preparation_v1');published_file(q,auth['expected_commit'])
    qualification=read(auth['qualification_actual_wait']['path']);code=qualification.get('returncode',qualification.get('exit_code'))
    require(type(code) is int and code==0 and (qualification.get('actual_wait') is True or
        qualification.get('actual_child_waited') is True) and not any(qualification.get(k,False) for k in
        ('timed_out','terminated','killed')),'Helper synthetic qualification did not actually pass')
    review=read(auth['independent_review']['path'])
    require(isinstance(review.get('status'),str) and review['status'].startswith('PASS') and review['findings']==[],
            'Independent helper review has not passed')
    return dict(authorization=ap,source_pins=auth['source_pins'],support_evidence=auth.get('support_evidence',[]),
        publication=pub,publication_receipts=publication_pins,registration=pin(REG),dependencies=dependency_receipts(),**closed)


def build_control_closure(expected_wait_sha):
    """Read only after the external build child and its supervisor have exited."""
    directory=BASE/(PACKAGE+'_build_dispatch_v1')
    require(len(expected_wait_sha)==64 and all(c in '0123456789abcdef' for c in expected_wait_sha),'Explicit build wait SHA required')
    require(sha(regular(directory/'actual_wait.json'))==expected_wait_sha,'Build actual wait differs')
    require(not (directory/'failure.json').exists() and not (B/'build_failure.json').exists(),'Failed build retained')
    launch=read(directory/'launch.json');wait=read(directory/'actual_wait.json');argv=launch['argv']
    cpu='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
    require(len(argv)==9 and argv[:3] in ([cpu,'-B',str(SOURCE/'build.py')],[PY,'-B',str(SOURCE/'build.py')])
        and argv[3]=='--authorization-path' and argv[5]=='--authorization-sha256' and argv[7]=='--authorization-bytes',
        'Build command differs')
    strict_wait(wait,launch,'child' if 'child' in wait else 'process',argv)
    ap=dict(path=argv[4],sha256=argv[6],bytes=int(argv[8]));checked(ap)
    require(ap['path']==str(BASE/(PACKAGE+'_root_authorization_v1')/'authorization.json'),'Wrong build authorization')
    paths=[directory/'launch.json',directory/'actual_wait.json']
    for name in ('stdout','stderr'):
        value=wait[name];p=checked(value,directory)
        require(p.parent==directory and p.name in (name+'.txt',name+'.log'),'Unexpected build log path');paths.append(p)
    return dict(schema='TIMING_FINAL_BUILD_CONTROL_CLOSED_V1',receipt=pin(B/'receipt.json'),authorization=ap,
        files=[dict(**pin(p),data_base64=base64.b64encode(p.read_bytes()).decode('ascii')) for p in paths],
        archived_inside_its_own_tar=False,actual_build_wait_zero=True,all_recorded_processes_gone=True)


def payload_sources(verified):
    files=[]
    for name in RUN_SCOPES:
        base=BASE/name
        files.extend((p,'runtime/'+name+'/'+p.relative_to(base).as_posix()) for p in walk_files(base,CACHE_EXCLUSIONS))
    for base in REPO_SCOPES:
        files.extend((p,'repo/'+p.relative_to(R).as_posix()) for p in walk_files(base))
    for path,digest in read(REG)['original_source_sha256'].items():
        p=checked(dict(path=path,sha256=digest),E);files.append((p,'repo/'+p.relative_to(R).as_posix()))
    pub=Path(verified['publication']['directory'])
    for row in read(pub/'staged_files.json'):
        p=regular(R/row['path'],R);payload_name('repo/'+p.relative_to(R).as_posix())
        require(sha(p)==row['sha256'],'Published staged content changed');published_file(p,verified['publication']['git_commit'])
        files.append((p,'repo/'+p.relative_to(R).as_posix()))
    for p in walk_files(pub):files.append((p,'publication/'+p.relative_to(pub).as_posix()))
    for value in verified['support_evidence']:
        p=checked(value)
        if p.is_relative_to(R):files.append((p,'repo/'+p.relative_to(R).as_posix()))
        else:files.append((p,'publication/backup_qualification/'+p.relative_to(BASE/'timing_final_backup_qualification_20261014_v1').as_posix()))
    files.append((checked(verified['authorization']),'authorization.json'))
    files.append((regular(SOURCE/'closed_evidence.json'),'closed_evidence.json'))
    for p,name in files:regular(p);payload_name(name)
    return files


def empty_runtime_directories():
    result=[]
    for name in RUN_SCOPES:
        base=BASE/name
        for directory,dirs,files in os.walk(base,followlinks=False):
            p=Path(directory);require(stat.S_ISDIR(p.lstat().st_mode),'Invalid directory')
            for child in dirs:require(stat.S_ISDIR((p/child).lstat().st_mode),'Symlink directory')
            original=list(dirs);dirs[:]=[d for d in dirs if p/d not in CACHE_EXCLUSIONS]
            if not original and not files and p!=base:
                target='runtime/'+name+'/'+p.relative_to(base).as_posix();payload_name(target+'/.empty_directory_metadata.json');result.append(target)
    return sorted(result)


def verify_internal_references(records,aliases):
    by_path={r['source_path']:r for r in records+aliases}
    scopes=(*REPO_SCOPES,*(BASE/n for n in RUN_SCOPES));excluded=[]
    for row in records:
        if Path(row['source_path']).suffix!='.json':continue
        for value in inner_pins(read(row['source_path'])):
            p=Path(value['path'])
            # Independent reviews may preserve Windows-author fixture paths. They
            # are opaque provenance, never external file access or runtime scope.
            if not p.is_absolute():continue
            p=canonical(p)
            if any(p.is_relative_to(c) for c in CACHE_EXCLUSIONS):
                excluded.append(value);continue
            if any(p.is_relative_to(base) for base in scopes):
                require(str(p) in by_path,'Internal scoped evidence omitted: '+str(p))
                saved=by_path[str(p)];require(saved['sha256']==value['sha256'] and
                    ('bytes' not in value or saved['bytes']==value['bytes']),'Archived internal reference differs')
    return excluded


def build(args):
    require(not B.exists() and not B.is_symlink(),'Existing build attempt: preserve and diagnose, no rebuild')
    import fcntl
    with contextlib.ExitStack() as stack:
        for lock in LOCKS:
            stream=stack.enter_context(regular(lock).open('rb'));fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        verified=closure(args);files=payload_sources(verified);B.mkdir(mode=0o700)
        try:
            dep=B/'DEPENDENCIES.md'
            dep.write_text('# Final six-method timing increment\n\n1728 completed calls: 576 inherited plus 1152 new; 864 measurements and 2304 packet decodes. '
                'The 1729 known endpoint entries include the preserved historical failed partial warmup. '
                'This package contains the closed v6 runtime, actual exits, resource revision, postprocess outputs and final publication. '
                'Only the precise v6 torchinductor_root reproducible cache is excluded.\n\n'
                'Keep timing_recovery_20261013_v1 with its 576-call v5 journal and prior failed partial, original v2/v3 failures and '
                'unlaunched lineage. Keep every dependency named by that verified package, including original models, source data, '
                'environments, native components and timing_entry. They are not repacked or rehashed here. '
                'This increment is not an independent recovery bundle and does not duplicate R0/R1 or training payloads.\n\n'
                'Startup admission changed from 18 to 8 GiB; allocator 16 GiB and free floor 2 GiB stayed unchanged. '
                'Results are shared-environment latency across two segments. No new timing, model, audit, statistics or bootstrap runs here. '
                'Lock files are inert evidence bytes; their presence does not mean a lock is held. Aliases restore as ordinary files.\n\n'
                'The final build actual wait cannot be embedded in the archive it describes. Keep separately verified build-control '
                'receipts beside this package; fetch requires their closure before download.\n',encoding='utf8')
            files.append((dep,'DEPENDENCIES.md'));records,aliases=deduplicate(files)
            omitted=verify_internal_references(records,aliases)
            manifest=dict(schema=SCHEMA,git_commit=verified['publication']['git_commit'],**FLAGS,
                verification=verified,dependencies=verified['dependencies'],files=records,aliases=aliases,
                empty_directories=empty_runtime_directories(),alias_restore='Verified ordinary files; no symlinks',
                excluded_reproducible_cache_roots=[str(p) for p in CACHE_EXCLUSIONS],excluded_cache_references=omitted,
                old_dependency_archives_reverified=False)
            mp=B/'manifest.json';save(mp,manifest);archive=B/(PACKAGE+'.tar.gz');partial=B/(PACKAGE+'.tar.gz.partial')
            with tarfile.open(partial,'w:gz',compresslevel=1,dereference=True) as tar:
                for row in records:tar.add(row['source_path'],arcname=row['path'],recursive=False)
                tar.add(mp,arcname='manifest.json',recursive=False)
            verify_archive(partial,mp,records)
            for row in records+aliases:stable_record(row)
            for process in verified['processes']:gone(process)
            no_current_failure()
            require(publication(verified['publication']['git_commit'],verified['publication']['directory'],
                verified['publication']['push']['sha256'])==verified['publication'],'Publication changed while packing')
            require(not archive.exists() and not archive.is_symlink(),'Archive target appeared unexpectedly')
            partial.rename(archive)  # Original proven same-directory PFS operation; exclusive fresh package directory.
            receipt=dict(schema=SCHEMA,archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
                manifest_sha256=sha(mp),payload_files=len(records),payload_bytes=sum(r['bytes'] for r in records),
                alias_files=len(aliases),alias_bytes=sum(r['bytes'] for r in aliases),restored_files=len(records)+len(aliases),
                git_commit=verified['publication']['git_commit'],**FLAGS,archive_payload_SHA_PASS=True,created_unix=time.time())
            save(B/'receipt.json',receipt);return dict(receipt=receipt,receipt_sha256=sha(B/'receipt.json'))
        except BaseException:
            save(B/'build_failure.json',dict(status='FAILED_PRESERVE_NO_REBUILD',traceback=traceback.format_exc(),science_calls=0));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization-path',required=True);parser.add_argument('--authorization-sha256',required=True)
    parser.add_argument('--authorization-bytes',required=True,type=int)
    print(json.dumps(build(parser.parse_args())),flush=True)
