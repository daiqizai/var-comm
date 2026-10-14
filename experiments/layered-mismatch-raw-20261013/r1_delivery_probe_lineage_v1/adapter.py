"""Explicit CPU-only metadata revision; original scientific/CPU sources stay pinned."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

NAME = 'r1_delivery_probe_lineage_v1'
R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
L = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013')
E = R/'experiments/layered-mismatch-raw-20261013'
O = R/'results/layered_mismatch_raw_20261013'
CPY = '/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
MASTER = 'ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c'
SOURCES = {
    'delivery': ('r1_cache_recovery_delivery_v1/deliver.py', 'f37df9e92520a1f45bdc7954dbbd6eaf5c1a7cab4721d3b5df7ec5df65c6de01'),
    'dispatcher': ('r1_cache_recovery_delivery_dispatch_v1/dispatch_once.py', '38a7a17dc2d6473972472b05a1fccd9c6954bba0269d013625ee25c28ce7207d'),
    'figures': ('r1_cache_recovery_figures_v1/figures.py', 'b5e5d8f42a2343930103618d589089c9ffea1c8c8d8753e0e506d1934551077d'),
    'backup': ('r1_final_science_backup_v1/build.py', 'eb97b1d96081ef808c56dbd0ea4aab6566c05dd111ef8913f65c7f6f2b035979'),
}
REPORT = dict(path=str(O/'r1_execution_preparation_v1/receiver_import_compatibility_v1.json'), bytes=9715,
              sha256='68d8bfb1812eb3b7bb4ccdf973c7911b468820d3321ed1714e00db8c137e2b4b')
BINDINGS = dict(path=str(O/'r1_execution_preparation_v1/final_frozen_bindings.json'), bytes=5066,
                sha256='0a299af06edf3791aa8fbd8c43c73c3aa60f17bb3fbf513c04d67c82e157cc1b')
REVIEW = dict(path=str(O/'r1_execution_preparation_v1/coordinator_review.json'), bytes=3704,
              sha256='9b6a45548a7f61a180c80433f1c47f7e804cb7fc78f0279a65dbba6e34547f1b')
PROBES = (
    (1, 'failure.json', 186468, 'e8cfe1a26edc37a2b57e51df965cfe6278a31878b67f105d41363f6e7f8a41d4'),
    (2, 'failure.json', 275578, '46f54364d6770c1626b0a32ffc99da9d77f3991b1c9f42fbddad8081d669dc56'),
    (3, 'completion.json', 276657, '52b4996c531b42e78d45277ff6f5491f7bdda05b30dff998d6a3f1489ee669d5'),
    (4, 'completion.json', 276904, 'd90110708059fd2a6905497d3d1e73ad2c1841d76218c93cfac049071e751131'),
    (5, 'failure.json', 276877, '1d868e20a0669f21427d689ede177334b10c2c6361877fac9106592d8b1d7476'),
    (6, 'completion.json', 273602, '530b5d6780a452afa938cdf4abe47ed1fc37361c02f947f9001342c6b50fe0ee'),
)
ZERO = ('channel_calls', 'model_constructors', 'qualification_frames', 'scientific_calls',
        'source_archive_reads', 'source_token_reads', 'tensor_operations_requested')
# Filled with the independently inventoried, closed metadata list before freezing.
INVENTORY_SHA = 'fd7e626188e95e18b361fe4ff788f607a13b4a8e471065253705ba2a33fd102f'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Absolute canonical path required')
    for parent in (path, *path.parents):
        st = parent.lstat()
        require(not stat.S_ISLNK(st.st_mode) and not
                (getattr(st,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024)),
                'Symlink/junction in metadata path')
    require(stat.S_ISREG(path.lstat().st_mode), 'Regular metadata file required')
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(pin):
    require(isinstance(pin, dict) and type(pin.get('bytes')) is int and pin['bytes'] >= 0,
            'Strict metadata byte count required')
    path = regular(pin['path'])
    before = path.stat()
    with path.open('rb') as stream:
        data = stream.read()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Metadata changed while reading')
    require(len(data) == pin['bytes'] and digest(data) == pin['sha256'], 'Metadata pin changed: '+str(path))
    return data


def pin(path):
    path = regular(path)
    data = path.read_bytes()
    result = dict(path=str(path), bytes=len(data), sha256=digest(data))
    checked(result)
    return result


def readpin(value):
    return json.loads(checked(value).decode('utf-8-sig'))


def strict_zero(value, keys):
    require(all(type(value.get(k)) is int and value[k] == 0 for k in keys),
            'Historical preparation requested scientific work or has noninteger counters')


def validate_probe(value, attempt, failure):
    require(value.get('schema') == 'R1_RX_ZERO_MODEL_IMPORT_PROBE_V1', 'Foreign probe schema')
    require(value.get('status') == ('FAILED_METADATA_ONLY' if failure else 'PASS_ZERO_MODEL_IMPORTS'),
            'Historical probe disposition changed')
    strict_zero(value, ZERO)
    require(value.get('CUDA_VISIBLE_DEVICES') == '' and value.get('actual_R1_registration_created') is False,
            'Historical probe was not isolated zero-model preparation')
    if attempt == 6:
        require(value.get('CUDA_initialized') is False and value.get('blocked') == [],
                'Final import probe must remain CUDA-uninitialized and have no blocked reads')


def authenticate(report_pin=REPORT, bindings_pin=BINDINGS, review_pin=REVIEW, probe_specs=None):
    """All six exact records authenticate lineage; only v1/v2/v5 may be allowed."""
    report = readpin(report_pin);bindings = readpin(bindings_pin);review = readpin(review_pin)
    require(review.get('approved') is True and review.get('independent_review_bundle') == bindings_pin,
            'Root review no longer binds original entry review')
    require(report_pin in bindings.get('review_receipts', []), 'Entry review does not bind probe lineage')
    require(report.get('schema') == 'R1_RX_IMPORT_COMPATIBILITY_REVIEW_V1'
            and report.get('status') == 'PASS_ZERO_MODEL_IMPORTS', 'Different compatibility lineage')
    strict_zero(report, ('channel_calls', 'qualification_frames', 'real_model_constructors', 'scientific_calls',
                         'source_archive_reads', 'source_token_reads', 'tensor_operations_requested'))
    require(report.get('CUDA_initialized') is False and report.get('CUDA_visible_devices') == ''
            and report.get('actual_R1_registration_created') is False, 'Compatibility was not zero-model')
    specs = probe_specs if probe_specs is not None else [
        (n, filename == 'failure.json', dict(path=str(L/'r1_execution_v1'/
            ('preparation_zero_model_import_probe_v'+str(n))/filename), bytes=size, sha256=sha))
        for n, filename, size, sha in PROBES]
    require([x[0] for x in specs] == [1,2,3,4,5,6], 'Exact six historical attempts required')
    prior = report['prior_attempts_preserved']
    require([x['attempt'] for x in prior] == [1,2,3,4,5], 'Prior attempt list changed')
    allowed = []
    for n, failure, evidence in specs:
        require(failure is (n in (1,2,5)), 'Failure exemption list changed')
        if n < 6:
            row = prior[n-1]
            require(row['record'] == evidence and row['historical_only'] is True,
                    'Probe record not bound to historical published disposition')
        else:
            require(report['final_probe'] == evidence, 'Different final import probe')
        value = readpin(evidence);validate_probe(value,n,failure)
        if failure:
            allowed.append(evidence)
    return dict(compatibility_review=report_pin,entry_frozen_review=bindings_pin,root_review=review_pin,
                exact_historical_failure_allowances=allowed,validated_probes=6,
                failed_metadata_probes=3,scientific_calls=0,not_scientific_numerical_qualification=True)


def patch_function(module, name, replacements, helpers):
    """Authenticated source, exact one-time substitutions, inverse AST equality."""
    source = regular(module.__file__).read_text(encoding='utf-8-sig')
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, 'Unique metadata function required')
    old = ast.get_source_segment(source,nodes[0]);new = old
    for before, after in replacements:
        require(new.count(before) == 1, 'Frozen metadata shape changed: '+name)
        new = new.replace(before,after)
    restored = new
    for before, after in reversed(replacements):
        require(restored.count(after) == 1, 'Metadata inverse shape changed')
        restored = restored.replace(after,before)
    require(ast.dump(ast.parse(old)) == ast.dump(ast.parse(restored)), 'Inverse metadata AST differs')
    module.__dict__.update(helpers)
    exec(compile(new,module.__file__,'exec'),module.__dict__)
    return dict(function=name,substitutions=len(replacements),inverse_AST=True,
                original_AST_sha256=digest(ast.dump(ast.parse(old)).encode()))


class Context:
    def __init__(self, adapter_sha, runner_sha, authorization):
        self.directory = E/NAME
        self.adapter = pin(self.directory/'adapter.py');self.runner = pin(self.directory/'runner.py')
        require(self.adapter['sha256'] == adapter_sha and self.runner['sha256'] == runner_sha,
                'Explicit execution adapter/runner pins required')
        self.inventory_pin = pin(self.directory/'runtime_inventory.json')
        require(self.inventory_pin['sha256'] == INVENTORY_SHA, 'Historical metadata inventory changed')
        self.inventory = readpin(self.inventory_pin)
        self.lineage = authenticate()
        require(Path(authorization['path'])==O/'r1_delivery_probe_lineage_preparation_v1/authorization.json',
                'Exact metadata revision authorization path required')
        self.authorization_pin=authorization;self.authorization=readpin(authorization)
        auth=self.authorization
        require(auth.get('schema')=='R1_CPU_PROBE_LINEAGE_AUTHORIZATION_V1'
                and auth.get('status')=='AUTHORIZED_METADATA_ONLY'
                and type(auth.get('scientific_calls')) is int and auth['scientific_calls']==0
                and auth.get('registration_sha256')==MASTER,'Invalid CPU metadata-only authorization')
        required={self.directory/n for n in ('adapter.py','runner.py','runtime_inventory.json','README.md','test_adapter.py')}
        required|={E/'r1_final_science_backup_v1'/n for n in ('build.py','fetch.py','README.md','test_backup.py','test_cli_review.py')}
        supplied=set()
        for item in auth['files']:
            path=Path(item['path'])
            require(path not in supplied and (path in required or path.is_relative_to(O/'r1_delivery_probe_lineage_preparation_v1')),
                    'Foreign or duplicate metadata authorization evidence')
            require(path.suffix in ('.py','.json','.md','.txt','.log'),'Nonmetadata authorization evidence')
            checked(item)
            supplied.add(path)
        require(required<=supplied,'Required exact revision/backup source evidence omitted')
        require(all(x in auth['files'] for x in (self.adapter,self.runner,self.inventory_pin)),
                'Authorization does not bind this execution revision')
        self.proofs = []

    def revision(self):
        return dict(schema='R1_CPU_PREPARATION_PROBE_LINEAGE_REVISION_V1',name=NAME,
                    adapter=self.adapter,runner=self.runner,historical_inventory=self.inventory_pin,
                    authorization=self.authorization_pin,preserved_revision_evidence=self.authorization['files'],
                    historical_preparation=self.lineage,scientific_calls=0,
                    original_source_bytes_unchanged=True,scientific_math_CSV_selection_unchanged=True)

    def command(self, role):
        require(role in SOURCES, 'Unknown CPU metadata role')
        return [CPY,'-B',self.runner['path'],role,'--expected-adapter',self.adapter['sha256'],
                '--expected-runner',self.runner['sha256'],'--expected-registration',MASTER,
                '--revision-authorization',self.authorization_pin['path'],
                '--revision-authorization-sha256',self.authorization_pin['sha256'],
                '--revision-authorization-bytes',str(self.authorization_pin['bytes'])]

    def check_revision(self, receipt):
        require(receipt.get('metadata_revision') == self.revision(), 'Missing/different explicit metadata execution revision')

    def load(self, role):
        relative,expected = SOURCES[role];path = E/relative
        source = pin(path);require(source['sha256'] == expected, 'Frozen '+role+' source changed')
        spec = importlib.util.spec_from_file_location('_'+NAME+'_'+role,path)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        getattr(self,'install_'+role)(module)
        return module

    def install_delivery(self,module):
        original_guard = module.no_failure_files;original_preflight = module.preflight
        def no_failure_files(base,allowed=()):
            if Path(base) == L/'r1_execution_v1':
                current = authenticate()
                require(current == self.lineage,'Historical probe lineage changed after installation')
                allowed = (*allowed,*(Path(p['path']) for p in current['exact_historical_failure_allowances']))
            return original_guard(base,allowed)
        def preflight(*args,**kwargs):
            reg,closure = original_preflight(*args,**kwargs)
            require('metadata_revision' not in closure,'Existing metadata revision collision')
            closure['metadata_revision'] = self.revision()
            return reg,closure
        module.no_failure_files = no_failure_files;module.preflight = preflight

    def install_dispatcher(self,module):
        original_loader = module.load_entry
        def load_entry(path):
            require(Path(path) == E/SOURCES['delivery'][0], 'Foreign dispatcher CPU entry')
            delivery = original_loader(path);self.install_delivery(delivery);return delivery
        module.load_entry = load_entry
        self.proofs.append(patch_function(module,'run',[
            ("argv=[python,'-B',str(entry),'--expected-registration',MASTER]", "argv=_metadata_context.command('delivery')"),
            ("common=dict(schema=SCHEMA,registration=registration,delivery_entry=pin(entry),argv=argv,",
             "common=dict(schema=SCHEMA,registration=registration,delivery_entry=pin(entry),argv=argv,metadata_revision=_metadata_context.revision(),")
        ],dict(_metadata_context=self)))

    def install_figures(self,module):
        self.proofs.append(patch_function(module,'guard_delivery',[
            ("expected_argv=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(entry),'--expected-registration',MASTER_SHA]",
             "expected_argv=_metadata_context.command('delivery')"),
            ("for receipt in (prepared,launch,wait):\n        require(receipt['schema']",
             "for receipt in (prepared,launch,wait):\n        _metadata_context.check_revision(receipt)\n        require(receipt['schema']"),
            ("recovery_provenance(audit,root,expected_master)",
             "recovery_provenance(audit,root,expected_master)\n    _metadata_context.check_revision(audit['process_closure'])")
        ],dict(_metadata_context=self)))

    def inventory_files(self):
        require(self.inventory.get('schema') == 'R1_DELIVERY_PRIOR_PREPARATION_INVENTORY_V1', 'Historical inventory schema')
        rows = self.inventory['files'];seen = set()
        for item in rows:
            path = Path(item['path'])
            require(path.is_relative_to(L/'r1_execution_v1') and path not in seen, 'Foreign/duplicate preparation metadata path')
            checked(item);seen.add(path)
        require(len(rows)==16 and len(self.inventory['empty_directories'])==6,'Historical metadata inventory count changed')
        for value in self.inventory['empty_directories']:
            path=Path(value)
            require(path.is_relative_to(L/'r1_execution_v1') and path.is_dir() and not path.is_symlink()
                    and not any(path.iterdir()),'Exact historical empty preparation directory is missing or nonempty')
        # Enumerate only these nine exact closed preparation roots, never the
        # surrounding scientific arrays, to reject unregistered added evidence.
        roots={L/'r1_execution_v1'/Path(p['path']).relative_to(L/'r1_execution_v1').parts[0] for p in rows}
        observed=set()
        for root in roots:
            for directory,dirs,files in os.walk(root,followlinks=False):
                base=Path(directory)
                require(not base.is_symlink(),'Preparation metadata directory became a link')
                for name in dirs:require(not (base/name).is_symlink(),'Preparation metadata child became a link')
                for name in files:observed.add(regular(base/name))
        require(observed==seen,'Unexpected or omitted file inside exact historical preparation roots')
        return rows

    def install_backup(self,module):
        original_loader = module.load_metadata;original_closure = module.closure
        def load_metadata(path,expected,name):
            loaded = original_loader(path,expected,name)
            if Path(path) == E/SOURCES['delivery'][0]:
                require(expected == SOURCES['delivery'][1],'Delivery pin mismatch');self.install_delivery(loaded)
            elif Path(path) == E/SOURCES['figures'][0]:
                require(expected == SOURCES['figures'][1],'Figure pin mismatch');self.install_figures(loaded)
            else:
                raise RuntimeError('Unrecognized final backup metadata loader target')
            return loaded
        module.load_metadata = load_metadata
        self.proofs.append(patch_function(module,'figure_closure',[
            ("expected_argv=[read(REG)['runtime']['python_cpu'],'-B',str(GRAPH),'--expected-registration',MASTER]",
             "expected_argv=_metadata_context.command('figures')\n    _metadata_context.check_revision(execution)"),
            ("launch_path=root_pin(execution['launch'],(L,BASE));launch=read(launch_path)",
             "launch_path=root_pin(execution['launch'],(L,BASE));launch=read(launch_path)\n    _metadata_context.check_revision(launch)")
        ],dict(_metadata_context=self)))
        def closure(args):
            value = original_closure(args)
            auth = module.read(value['authorization']['path']);self.check_revision(auth)
            value['metadata_revision'] = self.revision()
            return value
        module.closure = closure
        original_name = module.payload_name
        historical = { 'runtime/r1_execution_v1/'+Path(p['path']).relative_to(L/'r1_execution_v1').as_posix():p
                      for p in self.inventory_files() }
        empty = {'runtime/r1_execution_v1/'+Path(p).relative_to(L/'r1_execution_v1').as_posix()+
                 '/.empty_directory_metadata.json':Path(p) for p in self.inventory['empty_directories']}
        def payload_name(name):
            if name in historical:
                module.safe_parts(name);checked(historical[name]);return name
            if name in empty:
                module.safe_parts(name);path=empty[name]
                require(path.is_dir() and not path.is_symlink() and not any(path.iterdir()),
                        'Exact historical preparation cache is no longer empty')
                return name
            return original_name(name)
        module.payload_name = payload_name
        original_sources = module.payload_sources
        def payload_sources(verified):
            files = original_sources(verified)
            extra = [self.authorization_pin,*self.authorization['files'],*self.inventory['published_receipts'].values()]
            for index,item in enumerate(extra):
                checked(item);module.published_file(Path(item['path']),verified['publication']['git_commit'])
                name = ('repo/'+Path(item['path']).relative_to(R).as_posix()
                        if Path(item['path']).is_relative_to(E) else 'evidence/probe_lineage_'+str(index)+'.json')
                module.payload_name(name);files.append((Path(item['path']),name))
            return files
        module.payload_sources = payload_sources


def install_fetch_scope(fetch,build,inventory,expected_sources=None):
    """Local names are allowed only by frozen inventory; manifest SHA/bytes also checked."""
    require(inventory.get('schema')=='R1_DELIVERY_PRIOR_PREPARATION_INVENTORY_V1','Fetch historical inventory schema')
    prefix=(L/'r1_execution_v1').as_posix()+'/'
    files={}
    for item in inventory['files']:
        require(item['path'].startswith(prefix),'Foreign historical fetch path')
        name='runtime/r1_execution_v1/'+item['path'][len(prefix):]
        build.safe_parts(name);require(name not in files,'Duplicate historical fetch name');files[name]=item
    empty=set()
    for path in inventory['empty_directories']:
        require(path.startswith(prefix),'Foreign historical empty directory')
        name='runtime/r1_execution_v1/'+path[len(prefix):]
        build.safe_parts(name);require(name not in empty,'Duplicate historical empty directory');empty.add(name)
    require(len(files)==16 and len(empty)==6,'Exact historical preparation inventory cardinality changed')
    original_name=fetch.payload_name
    def payload_name(name):
        if name in files or name in {n+'/.empty_directory_metadata.json' for n in empty}:
            build.safe_parts(name);return name
        return original_name(name)
    fetch.payload_name=payload_name
    original_validate=fetch.validate_manifest
    def validate_manifest(manifest,receipt,names):
        result=original_validate(manifest,receipt,names)
        rows={row['path']:row for row in manifest['files']+manifest['aliases']}
        require(files.keys()<=rows.keys(),'Historical preparation evidence omitted from final archive')
        for name,item in files.items():
            row=rows[name]
            require(row['source_path']==item['path'] and row['sha256']==item['sha256']
                    and type(row['bytes']) is int and row['bytes']==item['bytes'],
                    'Archived historical preparation evidence differs')
        require(empty<=set(manifest['empty_directories']),'Historical empty evidence directories omitted')
        revision=manifest['verification']['metadata_revision']
        require(revision['schema']=='R1_CPU_PREPARATION_PROBE_LINEAGE_REVISION_V1'
                and revision['historical_inventory']['sha256']==INVENTORY_SHA,
                'Final archive omitted the explicit execution revision')
        if expected_sources is not None:
            require(revision['adapter']['sha256']==expected_sources['adapter']
                    and revision['runner']['sha256']==expected_sources['runner'],
                    'Archive execution revision differs from reviewed local fetch adapter')
        return result
    fetch.validate_manifest=validate_manifest


def load_local_fetch(directory,adapter_sha,runner_sha):
    directory=Path(directory)
    require(pin(directory/'adapter.py')['sha256']==adapter_sha and pin(directory/'runner.py')['sha256']==runner_sha,
            'Local execution revision changed')
    ip=pin(directory/'runtime_inventory.json')
    require(ip['sha256']==INVENTORY_SHA,'Local historical inventory changed')
    original=directory.parent/'r1_final_science_backup_v1'
    def load(name,path,expected):
        require(pin(path)['sha256']==expected,'Frozen local backup/fetch source differs')
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    build=load('_probe_lineage_local_backup',original/'build.py',SOURCES['backup'][1])
    old=sys.modules.get('build')
    try:
        sys.modules['build']=build
        fetch=load('_probe_lineage_local_fetch',original/'fetch.py',
                   'c425f44edb6188404c592901f3efe674edc42195659da43a796d5358788e372d')
    finally:
        if old is None:sys.modules.pop('build',None)
        else:sys.modules['build']=old
    install_fetch_scope(fetch,build,readpin(ip),dict(adapter=adapter_sha,runner=runner_sha))
    return fetch

