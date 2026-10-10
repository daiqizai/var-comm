"""Finite two-old-source migration qualification. No default scientific action.

prepare is metadata-only; run explicitly owns one GPU child and actually waits.
Original JSON/source bytes are never rewritten. This is not a production owner.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
import traceback

BASE = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu')
UUID = 'GPU-bf8340be-bd73-7413-62d5-0537be7f30cb'
CUDA_UUID = 'GPU-16eaab81-e5da-5138-3c04-192515cada82'
PCI_BUS_ID = '0000:a2:00.0'
OLD = '/home/liulu/projects/VAR_COMM/'
B = OLD+'outputs/WCL-EVIDENCE-CLOSURE-20261009/T1_entropy_whole/'
H = OLD+'outputs/CONTENT-REAL-64QAM-20261006/H/'
SCHEMA = 'LEO_EXISTING2_WHOLE_COMPATIBILITY_GATE_V1'
IDS = ['n01440764/ILSVRC2012_val_00045866_n01440764',
       'n01443537/ILSVRC2012_val_00044095_n01443537']
CAPS = dict(model_load=1, encoder=2, source_tx=2, source_rx=12, var_render=6,
            prior_scale=156, decoder_forward=6)
SIZES = (1,2,3,4,5,6,8,10,13,16)
OFFSETS = tuple(sum(x*x for x in SIZES[:i]) for i in range(11))
ANCHORS = {
 H+'full1000_source/completion.json':'e4bd19e04b74db88e91baec4539d7a305db4e07ad0d456b1bf399ad0d99605b5',
 H+'full1000_assets/manifest.json':'49bb6b8dfb45ac9fec5968481fe7f1691959a80964972bad1c0bde4e468e3cae',
 B+'source_gpu_owner_v1/source32_gate/completion.json':'601478aec601cbc051239bc40b191f7d0c88b800645556ebaee66fc966884d79',
 B+'source_gpu_owner_v1/source1000_sealed/completion.json':'2859ba7426b6a53f1644965f442175cc12a706dfa4acb92badd47c133b91a1ea',
 OLD+'outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v2_repaired_20260921/decoder_gate.json':'b892e539dd7f2584a9ba0b04151cc36b03864087ce9d6ddddfa152bda67e99f4',
}
CODE = {
 'src/var_comm/next_scale_prior.py':'568217086c94ee7607abc256415d705a11702ffcaaed9c79ca04d5e9a62da562',
 'experiments/scale-causal-partial-residual-20261002/partial_receiver.py':'46b676720283e3d7e90ef38bab4f4ea102ae51f38f414913d613ccb4b7e01cdc',
 'experiments/content-real-64qam-20261006/h_source_driver.py':'b38f8451ca2d94a8ceb4381df3647e327dba38e3d4cf969aa15597bc47c342ce',
 'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py':'d556732b86cb83499c8912a2412a8858f599e1ddd58f28418a0f1889d2b705a1',
 'src/var_comm/entropy.py':'f3adf956fba05d60e687472fa153e0bf684119ef6c8885a75fe4ea21962b2b2b',
 'src/var_comm/whole_entropy.py':'5d08d2f8c218f0bdf5b6d2a8ca7b3aa729baa75e274531afbfd8f31c95b24a5b',
 'experiments/wcl-evidence-closure-20261009/scripts/t1_entropy_core.py':'7cefad21134ac64c5eab5541afe4e188fdeaf343f374d8ea3eac966ff51901c9',
 'experiments/wcl-evidence-closure-20261009/scripts/t1_codec_runtime.py':'245f0bc9a67cffbcc08e36d7942c5e555b144f400a860940f04da62a012b1d13',
}
MODELS = dict(vae='6830fc533a34d52c8f765a7b212782a5b5ef6138992099e8a817ce920fe6fe2b',
 var='b82cb0855b24e1d0d8d0aacebbb2d2d6728c2dcc6509b667dfa697c9330347d5',
 decoder='bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1')
WEIGHTS = {
 'vae':'/home/liulu/projects/VAR-MAP-GATE0/checkpoints/vae_ch160v4096z32.pth',
 'var':'/home/liulu/projects/VAR-MAP-GATE0/checkpoints/var_d16.pth'}
WEIGHT_SHA = dict(vae='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c',
 var='4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb')
AUTHOR = '/home/liulu/projects/VAR-MAP-GATE0/third_party/VAR/models/'
UPSTREAM_COMMIT = '78b95394fc5896192e3a003e4b295f8ea743c48f'
UPSTREAM_AUDIT_SHA = '6e4c9db361b48b803ae91efc824391b8ee40e5ee3a16bb1b75d1a31bdef0e299'
UPSTREAM_COMMIT_JSON_SHA = '0b8c5c3cba188fe8a574741ce3a08f2021887e56f29d58809a8e9b1fccd2c602'
UPSTREAM_DIST_SHA = 'b55322b4ae4c8c2b600b98870dc98b26435a813b72b3468a4315d7d74771e370'
EXPECTED_RUNTIME = dict(python='3.10.12', torch='2.11.0+cu128', numpy='2.2.6',
 cuda='12.8', cudnn=91900, threads=6, interop_threads=2)
SHA_RE = re.compile('[0-9a-f]{64}')
STOP = False

class GateMismatch(RuntimeError):
    def __init__(self, layer, message):
        self.layer=layer;super().__init__(message)

def require(ok, message):
    if not ok: raise RuntimeError(message)

def normalize_pci_bus_id(value):
    require(isinstance(value,str),'PCI identity must be text')
    match=re.fullmatch(r'([0-9a-f]{4}|[0-9a-f]{8}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])',value.strip(),re.IGNORECASE)
    require(match is not None,'Invalid PCI bus identity')
    domain,bus,device,function=(int(x,16) for x in match.groups())
    require(device<=31,'Invalid PCI device number')
    return f'{domain:04x}:{bus:02x}:{device:02x}.{function}'

def normalize_gpu_uuid(value):
    require(isinstance(value,str),'GPU UUID must be text')
    match=re.fullmatch(r'(?:GPU-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',value.strip(),re.IGNORECASE)
    require(match is not None,'Invalid GPU UUID')
    return 'GPU-'+match.group(1).lower()

def cuda_environment_binding():
    # CUDA and NVML expose different UUID namespaces on this shared host.
    # The fixed ordinal works in both APIs; require both UUIDs and the same
    # physical PCI identity before any model load instead of patching PyTorch.
    return dict(CUDA_VISIBLE_DEVICES='5',CUDA_DEVICE_ORDER='PCI_BUS_ID')

def check_cuda_environment(environment=None):
    environment=os.environ if environment is None else environment
    require(all(environment.get(k)==v for k,v in cuda_environment_binding().items()),
            'Explicit admitted ordinal5 and PCI_BUS_ID order required; both hardware identities remain checked')

def device_identity_binding():
    return dict(nvml_gpu_index=5,nvml_uuid=UUID,cuda_uuid=CUDA_UUID,pci_bus_id=PCI_BUS_ID,
                NVML_and_CUDA_UUID_namespaces_differ=True,**cuda_environment_binding())

def validate_cuda_device_identity(actual_uuid,actual_pci):
    require(normalize_gpu_uuid(actual_uuid)==CUDA_UUID,'Logical CUDA0 UUID differs from admitted CUDA identity')
    require(normalize_pci_bus_id(actual_pci)==PCI_BUS_ID,'Logical CUDA0 PCI differs from selected NVML GPU5')
    return dict(actual_cuda_uuid=normalize_gpu_uuid(actual_uuid),actual_pci_bus_id=normalize_pci_bus_id(actual_pci),
                configured_nvml_uuid=UUID,configured_nvml_index=5,same_physical_PCI=True)

def query_nvml_device_identity(runner=None):
    runner=subprocess.run if runner is None else runner
    argv=['nvidia-smi','--id='+UUID,'--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits']
    result=runner(argv,check=True,capture_output=True,text=True,timeout=10)
    lines=result.stdout.strip().splitlines()
    require(len(lines)==1,'Exactly one selected NVML identity required')
    fields=[x.strip() for x in lines[0].split(',')]
    require(len(fields)==3 and fields[0]=='5','Selected NVML device index differs')
    require(normalize_gpu_uuid(fields[1])==UUID,'Selected NVML UUID differs')
    require(normalize_pci_bus_id(fields[2])==PCI_BUS_ID,'Selected NVML PCI differs')
    return dict(status='NVML_SELECTED_DEVICE_IDENTITY_OBSERVED',argv=argv,actual_nvml_index=5,
                actual_nvml_uuid=normalize_gpu_uuid(fields[1]),observed_pci_bus_id=fields[2],
                normalized_pci_bus_id=normalize_pci_bus_id(fields[2]),
                expected_cuda_uuid=CUDA_UUID,cuda_identity_not_yet_observed=True,
                model_calls=0,device_setting_changes=0)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def write(p, obj):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def checked_json(path, digest):
    require(isinstance(digest,str) and SHA_RE.fullmatch(digest),'Exact JSON SHA required')
    b=Path(path).read_bytes()
    require(hashlib.sha256(b).hexdigest()==digest,'JSON pin changed: '+str(path))
    return json.loads(b)

def inside(path, base=BASE):
    p=Path(path);base=Path(base).absolute()
    require(p.is_absolute() and '..' not in p.parts and p.is_relative_to(base),'Personal absolute path required')
    require(p.resolve().is_relative_to(base.resolve()),'Path escapes personal directory')
    # Refuse link-mediated file outputs/inputs, including intermediate aliases.
    cursor=p
    while cursor!=base:
        require(not cursor.is_symlink(),'Symlink in personal input/output path: '+str(cursor));cursor=cursor.parent
    return p

class Resolver:
    def __init__(self, rows, base=BASE):
        require(isinstance(rows,list) and 1<=len(rows)<=50000,'Finite explicit file map required')
        self.base=Path(base);self.rows={};self.used={};destinations=set()
        for row in rows:
            require(set(row)=={'original_path','path','sha256','bytes'},'Exact mapped-file fields required')
            old=row['original_path'];p=PurePosixPath(old)
            require(p.is_absolute() and str(p)==old and '..' not in p.parts and '\\' not in old,'Unsafe original identity')
            require(old not in self.rows and row['path'] not in destinations,'Duplicate mapping')
            inside(row['path'],self.base)
            require(SHA_RE.fullmatch(row['sha256']) and type(row['bytes']) is int and row['bytes']>=0,'Invalid mapped pin')
            self.rows[old]=dict(row);destinations.add(row['path'])
    def path(self, old, expected=None):
        require(old in self.rows,'Unmapped original path: '+old);r=self.rows[old]
        require(expected is None or r['sha256']==expected,'Original descriptor disagrees with map: '+old)
        p=inside(r['path'],self.base)
        require(p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'],'Mapped regular file missing/size changed')
        require(sha(p)==r['sha256'],'Mapped bytes changed: '+old)
        self.used[old]=dict(r);return p
    def read(self, old, expected=None):
        p=self.path(old,expected);return checked_json(p,self.rows[old]['sha256'])
    def reverify(self):
        for old,row in list(self.used.items()):self.path(old,row['sha256'])

def import_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def helper():
    p=Path(__file__).with_name('leo_shared_preflight.py')
    require(sha(p)=='0f7473aa003991edca43a4587bbd03dfef0955e13f85cf23ffaff1dae1cf66ab','Shared guard source changed')
    return import_file(p,'_leo_whole_shared_guard')

def source_metadata(resolver):
    h=resolver.read(H+'full1000_source/completion.json',ANCHORS[H+'full1000_source/completion.json'])
    a=resolver.read(H+'full1000_assets/manifest.json',ANCHORS[H+'full1000_assets/manifest.json'])
    gate=resolver.read(B+'source_gpu_owner_v1/source32_gate/completion.json',ANCHORS[B+'source_gpu_owner_v1/source32_gate/completion.json'])
    sealed=resolver.read(B+'source_gpu_owner_v1/source1000_sealed/completion.json',ANCHORS[B+'source_gpu_owner_v1/source1000_sealed/completion.json'])
    mp=B+'source_gpu_owner_v1/source1000_sealed/manifest.json'
    manifest=resolver.read(mp,sealed['outputs'][mp])
    require(h['status']=='H_FULL1000_SOURCE_CODEC_COMPLETE' and h['source_count']==1000,'Old H completion differs')
    require(h['source_ids'][:2]==a['source_ids'][:2]==manifest['source_ids'][:2]==IDS,'Original first-two order differs')
    require(gate['status']=='T1_SOURCE32_REAL_RECOVERY_GATE_COMPLETE' and gate['exact_equal_image_pairs']==96,'Old rendered witness gate absent')
    require(h['numerical_runtime']['threads']==6 and h['numerical_runtime']['interop_threads']==2,'Old numerical thread contract differs')
    records=[]
    for i in range(2):
        hr=h['records'][i];hp=resolver.read(hr['checkpoint'],hr['sha256'])
        require(h['outputs'][hr['checkpoint']]==hr['sha256'],'Old source checkpoint not sealed')
        sr=manifest['records'][i];sp=resolver.read(sr['checkpoint'],sr['checkpoint_sha256'])
        require(sealed['outputs'][sr['checkpoint']]==sr['checkpoint_sha256'],'Sealed source checkpoint not sealed')
        ap=hp['source_assets_checkpoint'];asset=resolver.read(ap['path'],ap['sha256'])
        require(hp['source_index']==sp['source_index']==asset['source_index']==i and hp['source_id']==sp['source_id']==asset['source_id']==IDS[i],'Source identity mismatch')
        require(sp['source_role']=='calibration' and sp['schema']=='T1_SOURCE_ASSET_V1','Wrong source role')
        require(hp['tokens_sha256']==sp['source_tokens_sha256']==asset['tokens_sha256'],'Original token pin mismatch')
        require(hp['preprocessing_id']==sp['preprocessing_id']==asset['preprocessing_id'],'Original pixel pin mismatch')
        require(sp['source_assets_checkpoint']==ap,'Source asset provenance differs')
        archive=asset['archive'];resolver.path(archive,asset['outputs'][archive])
        old_archive,old_sha=next(iter(hp['outputs'].items()));resolver.path(old_archive,old_sha)
        require(h['outputs'][old_archive]==old_sha,'Original bits archive unsealed')
        resolver.path(sp['archive']['path'],sp['archive']['sha256'])
        require(sealed['outputs'][sp['archive']['path']]==sp['archive']['sha256'],'Sealed streams archive unsealed')
        image=B+f'source_gpu_owner_v1/source32_gate/sources/{i:04d}.npz';resolver.path(image,gate['outputs'][image])
        for m in range(4,10):
            proof=sp['streams']['EC_VAR_WHOLE'][str(m)]['receiver_cache_proof']
            require(proof['independent_roundtrip'] and proof['canonical'] and proof['zero_extension_reads']==30,'Old stream proof incomplete')
        records.append(dict(index=i,source_id=IDS[i],asset=asset,sealed=sp,original_stream_archive=old_archive,image_archive=image))
    return records,h

class RuntimeFiles:
    """One original byte identity may have several admitted regular copies."""
    def __init__(self,env,base=BASE):
        self.base=base;self.rows={};self.used={};old_pins={};self.generated=env['projection_generated_files']
        require(1<=len(env['files'])<=200000,'Finite original runtime map required')
        for row in env['files']:
            require(set(row)=={'original_path','path','sha256','bytes'},'Unexpected runtime map fields')
            old=row['original_path'];p=PurePosixPath(old)
            require(p.is_absolute() and str(p)==old and '..' not in p.parts and '\\' not in old,'Invalid original runtime identity')
            require(row['path'] not in self.rows,'Duplicate runtime destination')
            # Full no-link/opened-file checks run in reverify before execution.
            # Keep construction metadata-only instead of resolving every shared
            # ancestor tens of thousands of times on the shared filesystem.
            target=Path(row['path']);root=Path(base).absolute()
            require(target.is_absolute() and '..' not in target.parts and target!=root and
                    target.is_relative_to(root) and str(target)==row['path'],'Noncanonical personal runtime path')
            require(SHA_RE.fullmatch(row['sha256']) and type(row['bytes']) is int and row['bytes']>=0,'Invalid runtime pin')
            key=(row['sha256'],row['bytes']);require(old not in old_pins or old_pins[old]==key,'Conflicting original runtime bytes')
            old_pins[old]=key;self.rows[row['path']]=dict(row)
        require(len(self.generated)==4,'Exactly two new private venv configurations and .pth files required')
        for row in self.generated:
            inside(row['path'],base);require(row['path'] not in self.rows and row['generated_new_bytes'] is True,'Derived runtime bytes must be separate')
            data=row['content'].encode('utf-8');require(len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],'Derived runtime content pin differs')
            p=Path(row['path']);require(p.name in ('pyvenv.cfg','frozen_package_roots.pth'),'Unapproved generated runtime file')
            if p.suffix=='.pth':
                for line in row['content'].splitlines():require(line and inside(line,base).is_dir(),'Only personal path lines permitted in .pth')
            else:
                fields=dict(x.split(' = ',1) for x in row['content'].splitlines())
                require(fields['include-system-site-packages']=='false' and inside(fields['home'],base).is_dir(),'Private venv configuration differs')
        self.host_native=env['host_native_library_facts']
        require(len(self.host_native)<=1000,'Finite observed native dependency list required')
    def path_for_new(self,path):
        key=str(path);require(key in self.rows,'Imported runtime file not pinned: '+key);row=self.rows[key]
        p=inside(key,self.base);require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Frozen runtime bytes changed: '+key)
        self.used[key]=row;return p
    def reverify(self):
        started=time.monotonic()
        if sys.platform.startswith('linux'):
            verifier=import_file(Path(__file__).with_name('leo_runtime_verify_v1.py'),'_leo_runtime_byte_verifier')
            self.batch_verification=verifier.verify_runtime_rows(list(self.rows.values()),self.base)
            self.batch_verification['elapsed_seconds']=time.monotonic()-started
            self.used.update(self.rows)
        else:
            # Portable tiny CPU fixtures retain the original path validation.
            # Real gate ownership is Linux-only.
            for p in self.rows:self.path_for_new(p)
        for row in self.generated:
            p=inside(row['path'],self.base)
            require(p.is_file() and p.read_bytes()==row['content'].encode('utf-8'),'Derived runtime setup changed')
        for row in self.host_native:
            p=Path(row['path'])
            require(p.is_absolute() and '..' not in p.parts and p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Observed host native dependency changed')

def runtime_from_receipt(pin,base=BASE):
    """Metadata-only normalization of the actual relocation receipt; no import."""
    receipt=checked_json(inside(pin['path'],base),pin['sha256'])
    require(receipt['schema']=='FROZEN_RUNTIME_RELOCATION_RECEIPT_V1' and receipt['status']=='FROZEN_RUNTIME_RELOCATION_IMPORT_QUALIFICATION_COMPLETE','Runtime relocation/import not complete')
    require(receipt['actual_children_waited'] is True and receipt['model_calls']==receipt['packet_calls']==0 and receipt['numeric_qualification'] is False,'Runtime preparation boundary differs')
    role=receipt['runtimes']['UM'];selection_pin=receipt['original_environment_selection']
    selection=checked_json(inside(selection_pin['path'],base),selection_pin['sha256'])
    require(selection_pin['sha256']=='95fb384346a6601692977bafd41bd259dc600dfec7972fecb2ebc5e026692818','Original frozen environment selection differs')
    pp=receipt['original_file_projection'];projection=checked_json(inside(pp['path'],base),pp['sha256'])
    require(projection['schema']=='FROZEN_RUNTIME_ORIGINAL_FILE_MAP_V1' and projection['selection']==selection_pin,'Runtime projection provenance differs')
    for stage,status in [('bootstrap','BOOTSTRAP_STDLIB_ADMISSION_PASS'),('imports','FRAMEWORK_IMPORT_ADMISSION_PASS')]:
        outcome=role['outcomes'][stage];waited=outcome['actual_child_wait']
        require(waited['actual_wait'] is True and waited['returncode']==0 and waited['interrupted_or_timeout'] is False,'Runtime child not actually waited successfully')
        facts=checked_json(inside(outcome['result']['path'],base),outcome['result']['sha256'])
        require(facts==outcome['facts'] and facts['status']==status and facts['CUDA_VISIBLE_DEVICES']=='' and facts['model_calls']==0,'Runtime import facts differ')
    # Hash the same bytes that supply the source map; never reread an unbound map.
    mp=projection['seed_mapping'];data=inside(mp['path'],base).read_bytes()
    require(hashlib.sha256(data).hexdigest()==mp['sha256'],'Frozen namespace map changed')
    mapped=[json.loads(line) for line in data.splitlines()];require(len(mapped)==mp['records'],'Frozen namespace map count differs')
    selected={r['source_path']:r for r in selection['members'] if r['kind']=='file'}
    files={}
    def add(row):
        old=row['original_path'];require(old in selected and (row['sha256'],row['bytes'])==(selected[old]['sha256'],selected[old]['bytes']),'Runtime bytes lack exact original selected provider')
        normalized={k:row[k] for k in ('original_path','path','sha256','bytes')}
        require(row['path'] not in files or files[row['path']]==normalized,'Runtime target has conflicting proofs');files[row['path']]=normalized
    for row in mapped:add(dict(original_path=row['original_path'],path=row['actual_path'],sha256=row['sha256'],bytes=row['bytes']))
    aliases={r['path']:r for r in receipt['alias_regular_copies']}
    for row in projection['files']:
        row=dict(row)
        if row.get('alias_regular_copy'):
            require(row['path'] in aliases,'Unproved regular alias copy');row['original_path']=aliases[row['path']]['terminal_original_path']
        add(row)
    require(role['lexical_python'] in files,'Private interpreter absent from original byte map')
    # This gate imports only UM. Preserve every full restoration/projection pin,
    # but avoid repeatedly reading the unused, separately sealed LDPC package tree.
    active,excluded=active_um_files(list(files.values()))
    by_path={r['path']:r for r in active};imports=role['outcomes']['imports']['facts']
    for fact in [*imports['imports'].values(),imports['torch_native'],role['original_binary']]:
        require(fact['path'] in by_path and (fact['sha256'],fact['bytes'])==
                (by_path[fact['path']]['sha256'],by_path[fact['path']]['bytes']),'UM imported dependency is not in active original map')
    native_facts=normalize_native_facts(role['native_library_facts'],by_path,base)
    return dict(schema='LEO_FROZEN_UM_RUNTIME_PROJECTION_V1',status='PROJECTED_ORIGINAL_FILES_REVERIFIED',
        python=role['lexical_python'],expected_runtime=EXPECTED_RUNTIME,files=active,
        original_environment_selection=selection_pin,projection_receipt=pin,
        complete_original_file_projection=pp,complete_restored_namespace_map=mp,
        excluded_unused_runtime_roots=[OLD+'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment/'],
        excluded_unused_runtime_files=len(excluded),excluded_unused_runtime_bytes=sum(r['bytes'] for r in excluded),
        projection_generated_files=receipt['projection_generated_files'],
        host_native_library_facts=native_facts,host_libraries_not_claimed_old_identical=True,
        LD_LIBRARY_PATH=role['runtime_environment_for_import_check_only']['LD_LIBRARY_PATH'],
        runtime_preparation_is_not_numerical_qualification=True)

def normalize_native_facts(facts,by_path,base=BASE):
    """Keep ldd spelling as evidence; admit only the same sealed regular target."""
    base=Path(base).absolute();normalized=[];pins={}
    require(isinstance(facts,list) and len(facts)<=1000,'Finite observed native dependency list required')
    for fact in facts:
        observed=fact['path'];p=Path(observed)
        require(p.is_absolute() and SHA_RE.fullmatch(fact['sha256']) and
                type(fact['bytes']) is int and fact['bytes']>=0,'Invalid observed native dependency pin')
        personal=p.is_relative_to(base)
        if personal:
            # Never collapse a symlink/.. pair lexically: it may name another
            # target. Check every traversed component, including cancelled ones.
            require(not base.is_symlink(),'Symlink personal native base')
            cursor=base
            parts=p.relative_to(base).parts
            for i,part in enumerate(parts):
                if part=='..':
                    require(cursor!=base,'Native path traverses outside personal base')
                    cursor=cursor.parent
                else:
                    cursor=inside(cursor/part,base)
                    require(cursor.exists(),'Observed native path component missing')
                if i<len(parts)-1:require(cursor.is_dir(),'Native path traverses a non-directory')
            canonical=p.resolve(strict=True)
            require(canonical==cursor and canonical.is_relative_to(base),'Native path escapes personal directory')
            key=str(canonical)
            require(key in by_path and (fact['sha256'],fact['bytes'])==
                    (by_path[key]['sha256'],by_path[key]['bytes']),
                    'Personal UM native library absent from active original map or pin differs')
        else:
            canonical=p.resolve(strict=True)
            require(not canonical.is_relative_to(base.resolve()),'Host native alias enters personal directory')
        require(canonical.is_file() and canonical.stat().st_size==fact['bytes'],'Observed native dependency size changed')
        key=str(canonical);identity=(fact['sha256'],fact['bytes'])
        require(key not in pins or pins[key]==identity,'Conflicting native aliases for the same target')
        pins[key]=identity
        normalized.append(dict(fact,path=key,observed_lexical_path=observed))
    return normalized

def active_um_files(rows):
    prefix=OLD+'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment/'
    excluded=[r for r in rows if r['original_path'].startswith(prefix)]
    active=[r for r in rows if not r['original_path'].startswith(prefix)]
    require(active,'No UM runtime files remain');return active,excluded

def attach_host_native_addendum(runtime, pin, projection_sha256, base=BASE):
    """Admit pinned new-host observations without changing the original projection."""
    require(isinstance(pin,dict) and set(pin)=={'path','sha256'},'Exact host-native addendum descriptor required')
    path=inside(pin['path'],base)
    require(path.is_file() and not path.is_symlink(),'Regular personal addendum receipt required')
    receipt=checked_json(path,pin['sha256'])
    require(receipt.get('schema')=='LEO_NEW_HOST_NATIVE_ADDENDUM_V1' and
            receipt.get('status')=='CPU_IMPORT_OBSERVED_NOT_NUMERICAL','Host-native addendum is not a CPU import observation')
    require(type(receipt.get('model_calls')) is int and receipt['model_calls']==0 and
            type(receipt.get('tensor_calls')) is int and receipt['tensor_calls']==0,'Host-native addendum must have zero model/tensor calls')
    require(receipt.get('original_runtime_projection_sha256')==projection_sha256,
            'Host-native addendum refers to a different original runtime projection')
    rows=receipt.get('files')
    require(isinstance(rows,list) and 1<=len(rows)<=32,'Finite nonempty host-native addendum required')
    existing={row['path']:(row['sha256'],row['bytes']) for row in runtime.host_native}
    combined=[dict(row) for row in runtime.host_native];seen=set();root=Path(base).resolve()
    for row in rows:
        require(isinstance(row,dict) and set(row)=={'path','sha256','bytes'},'Exact native file pin required')
        require(isinstance(row['path'],str) and isinstance(row['sha256'],str) and SHA_RE.fullmatch(row['sha256']) and
                type(row['bytes']) is int and row['bytes']>=0,'Invalid native addendum pin')
        p=Path(row['path'])
        require(p.is_absolute() and '..' not in p.parts and str(p)==row['path'] and
                str(p.resolve(strict=True))==row['path'] and not p.is_symlink(),'Canonical host native file required')
        require(not p.is_relative_to(root) and re.fullmatch(r'.+\.so(?:\.[0-9]+)*',p.name),
                'Addendum is restricted to shared libraries outside the personal base')
        require(row['path'] not in seen,'Duplicate host-native addendum path');seen.add(row['path'])
        identity=(row['sha256'],row['bytes'])
        require(row['path'] not in existing or existing[row['path']]==identity,'Host-native addendum conflicts with original facts')
        before=p.stat()
        require(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'] and sha(p)==row['sha256'],
                'Host-native addendum file bytes changed')
        after=p.stat()
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                'Host-native addendum file changed while hashing')
        if row['path'] not in existing:combined.append(dict(row))
    # Replace only the resolver's list after complete validation. The environment
    # projection and its original receipt remain byte-for-byte unchanged.
    runtime.host_native=combined

def native_candidate(candidate,authors,resolver,base=BASE):
    """Explicit new dependency candidate, never a claim of old dist provenance."""
    require(candidate['old_dist_sha_unavailable'] is True,'Missing old dist provenance must remain explicit')
    p=candidate['audit'];require(p['sha256']==UPSTREAM_AUDIT_SHA,'Unreviewed upstream audit')
    audit=checked_json(inside(p['path'],base),p['sha256'])
    p=candidate['commit_record'];require(p['sha256']==UPSTREAM_COMMIT_JSON_SHA,'Unreviewed upstream revision')
    commit=checked_json(inside(p['path'],base),p['sha256'])
    require(commit['sha']==UPSTREAM_COMMIT and audit['commit']==UPSTREAM_COMMIT,'Upstream revision changed')
    require(audit['status']=='UPSTREAM_REFERENCE_ONLY_NOT_ORIGINAL_DIST_PROVENANCE' and
            audit['old_dist_sha_unavailable'] is True,'Upstream provenance boundary changed')
    root=inside(candidate['root'],base);entries={x['path']:x for x in audit['files']}
    expected={'dist.py','LICENSE'}|{'models/'+PurePosixPath(p).name for p in authors}
    require(set(entries)==expected and len(audit['files'])==9,'Upstream dependency file set changed')
    for rel,row in entries.items():
        path=inside(root/rel,base)
        require(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Upstream candidate bytes changed: '+rel)
        require(row['url']==f'https://raw.githubusercontent.com/FoundationVision/VAR/{UPSTREAM_COMMIT}/{rel}','Upstream identity differs')
        if rel.startswith('models/'):
            old=AUTHOR+PurePosixPath(rel).name
            require(row['sha256']==row['original_sha256']==authors[old] and row['exact_original_match'] is True,'Original seven model source comparison failed')
            require(resolver.path(old,authors[old])==path,'Original seven must be mapped into distinct candidate root')
    require(entries['dist.py']['sha256']==UPSTREAM_DIST_SHA,'Unexpected dependency candidate')
    require(set(p.name for p in root.glob('*.py'))=={'dist.py'},'Unregistered upstream root module')
    require(set(p.name for p in (root/'models').glob('*.py'))=={PurePosixPath(p).name for p in authors},'Unregistered model module')
    return dict(root=str(root),dist=str(root/'dist.py'),dist_sha256=UPSTREAM_DIST_SHA,
        fixed_upstream_commit=UPSTREAM_COMMIT,old_dist_sha_unavailable=True,
        provenance='New migration dependency candidate; seven model source files match old bytes, old dist identity unobservable',
        audit=candidate['audit'],commit_record=candidate['commit_record'])

def validate_spec(spec, base=BASE, verify_environment=True):
    require(spec['schema']=='LEO_WHOLE_GATE_INPUT_MAP_V1','Wrong input-map schema')
    require(spec['source_indices']==[0,1] and spec['caps']==CAPS and spec['gpu_uuid']==UUID,'Registered gate scope changed')
    require(type(spec['deadline_unix']) in (int,float) and 0<spec['max_seconds']<=1800,'Finite at-most30-minute window required')
    require(time.time()<spec['deadline_unix'],'Registered deadline expired')
    root=inside(spec['project_root'],base);r=Resolver(spec['files'],base)
    for rel,digest in CODE.items():require(r.path(OLD+rel,digest)==root/rel,'Frozen code must use exact mapped repo path')
    records,h=source_metadata(r)
    authors={p:s for p,s in h['source_bindings'].items() if p.startswith(AUTHOR)}
    require(len(authors)==7,'Exactly seven original VAR author files required')
    dependency=native_candidate(spec['native_dependency_candidate'],authors,r,base)
    author_root=Path(dependency['root'])
    for key,old in WEIGHTS.items():r.path(old,WEIGHT_SHA[key])
    gate_path=next(p for p in ANCHORS if p.endswith('/decoder_gate.json'));dgate=r.read(gate_path,ANCHORS[gate_path])
    selection=dgate['selection'];r.path(selection['checkpoint'],selection['checkpoint_sha256'])
    require(selection['checkpoint_sha256']=='e3d757a5a16206f5f9be345d0e6330b0f1fd6c78bf167670e70378b429d931c4','Original selected Dc checkpoint differs')
    env_pin=spec['environment'];env_path=inside(env_pin['path'],base);env=checked_json(env_path,env_pin['sha256'])
    require(env['schema']=='LEO_FROZEN_UM_RUNTIME_PROJECTION_V1' and env['status']=='PROJECTED_ORIGINAL_FILES_REVERIFIED','Closed original runtime projection required')
    require(env['expected_runtime']==EXPECTED_RUNTIME,'Frozen numerical package/thread contract differs')
    expected_env=runtime_from_receipt(env['projection_receipt'],base)
    require(env==expected_env,'Runtime projection does not match original actual relocation receipt')
    er=RuntimeFiles(env,base)
    if 'host_native_addendum' in spec:
        attach_host_native_addendum(er,spec['host_native_addendum'],env_pin['sha256'],base)
    python=Path(os.path.abspath(spec['python']))
    require(python==Path(env['python']) and inside(python,base).is_file(),'Exact personal interpreter spelling required')
    require(str(python) in er.rows,'Interpreter not in original runtime pins')
    require(env.get('original_environment_selection') and env.get('projection_receipt'),'Runtime source/projection receipts required')
    for pin in (env['original_environment_selection'],env['projection_receipt']):
        checked_json(inside(pin['path'],base),pin['sha256'])
    if verify_environment:er.reverify()
    return r,er,records,h,dict(author_root=str(author_root),decoder=selection['checkpoint'],environment=env,native_dependency_candidate=dependency,
        runtime_byte_verification=getattr(er,'batch_verification',None))

class Ledger:
    def __init__(self,out,guard,caps=None):
        self.out=Path(out);self.out.mkdir();self.guard=guard;self.caps=dict(CAPS if caps is None else caps)
        self.reserved=dict.fromkeys(self.caps,0);self.completed=dict.fromkeys(self.caps,0);self.number=0
    def call(self,kind,operation,**context):
        self.guard();require(kind in self.caps and self.reserved[kind]<self.caps[kind],'Call cap exhausted: '+kind)
        self.number+=1;number=self.number;self.reserved[kind]+=1
        stem=self.out/f'{number:04d}_{kind}'
        write(str(stem)+'.reserved.json',dict(kind=kind,context=context,pid=os.getpid(),started_unix=time.time()))
        value=operation();self.completed[kind]+=1
        write(str(stem)+'.complete.json',dict(kind=kind,context=context,completed_unix=time.time()))
        return value
    def summary(self):return dict(caps=self.caps,reserved=self.reserved,completed=self.completed,unresolved=sum(self.reserved.values())-sum(self.completed.values()))

def image_sha(a):
    import numpy as np
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def exact(actual, expected, name, out):
    import numpy as np
    a,b=np.asarray(actual),np.asarray(expected)
    ok=a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b) and a.tobytes()==b.tobytes()
    record=dict(name=name,exact=ok,actual_sha256=image_sha(a),expected_sha256=image_sha(b),actual_shape=list(a.shape),expected_shape=list(b.shape),actual_dtype=str(a.dtype),expected_dtype=str(b.dtype))
    if a.shape==b.shape and a.size and np.issubdtype(a.dtype,np.number) and np.issubdtype(b.dtype,np.number):record['max_abs_difference']=float(np.max(np.abs(a.astype('float64')-b.astype('float64'))))
    write(Path(out)/(name+'.json'),record)
    if not ok:
        with (Path(out)/(name+'.npz')).open('xb') as f:np.savez(f,actual=a,expected=b)
        layer='tokens' if '_encoder' in name else 'bitstream' if '_tx_' in name else 'received_tokens' if '_rx_' in name else 'VAR_render'
        raise GateMismatch(layer,'Exact witness mismatch: '+name)
    return record

def load_source(record,resolver):
    import numpy as np
    a,s=record['asset'],record['sealed']
    with np.load(resolver.path(a['archive']),allow_pickle=False) as z:tokens=z['tokens'].copy();pixels=z['pixels'].copy()
    require(tokens.dtype==np.int64 and tokens.shape==(680,) and ((tokens>=0)&(tokens<4096)).all(),'Original token array malformed')
    require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256),'Original pixel array malformed')
    require(hashlib.sha256(pixels.tobytes()).hexdigest()==a['preprocessing_id'],'Original pixel hash changed')
    require(hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==a['tokens_sha256'],'Original token hash changed')
    streams={}
    with np.load(resolver.path(s['archive']['path'],s['archive']['sha256']),allow_pickle=False) as z:
        require(np.array_equal(z[s['tokens_key']],tokens),'Sealed source token array differs')
        for m in range(4,10):
            d=s['streams']['EC_VAR_WHOLE'][str(m)];bits=z[d['bits_key']].copy();received=z[d['received_tokens_key']]
            require(bits.dtype==np.uint8 and bits.ndim==1 and np.isin(bits,(0,1)).all(),'Original bits malformed')
            require(len(bits)==d['payload_bits'] and hashlib.sha256(b'uint8_bits\0'+bits.tobytes()).hexdigest()==d['payload_sha256'],'Old bitstream pin differs')
            rx_hash=hashlib.sha256(f'int64:{len(received)}\0'.encode()+received.astype('<i8').tobytes()).hexdigest()
            require(rx_hash==d['received_tokens_sha256'] and np.array_equal(received,tokens[:OFFSETS[m]]),'Old actual RX token proof differs')
            streams[m]=bits
    with np.load(resolver.path(record['original_stream_archive']),allow_pickle=False) as z:
        require(all(np.array_equal(z[f'm{m}_bits'],streams[m]) for m in range(6,10)),'Historical H and sealed streams differ')
    with np.load(resolver.path(record['image_archive']),allow_pickle=False) as z:
        images={m:z[f'm{m}_image'].copy() for m in (7,8,9)}
    require(all(v.dtype==np.float32 and v.shape==(3,256,256) and np.isfinite(v).all() for v in images.values()),'Historical RGB witness malformed')
    return dict(tokens=tokens,pixels=pixels,streams=streams,images=images)

def exercise(sources, backend, ledger, out):
    """Injectable scientific sequence; tests use fake arrays/callbacks only."""
    out=Path(out);out.mkdir();checks=[]
    require(len(sources)==2,'Exactly two source witnesses required')
    for i,source in enumerate(sources):
        tokens=ledger.call('encoder',lambda:backend.encode_pixels(source['pixels']),source=i)
        checks.append(exact(tokens,source['tokens'],f's{i}_encoder',out))
        tx=ledger.call('source_tx',lambda:backend.source_tx(source['tokens']),source=i)
        require(set(tx)==set(range(4,10)),'Missing complete TX endpoints')
        for m in range(4,10):checks.append(exact(tx[m]['bits'],source['streams'][m],f's{i}_tx_m{m}',out))
        for m in range(4,10):
            rx=ledger.call('source_rx',lambda m=m:backend.source_rx(source['streams'][m].copy(),m),source=i,m=m)
            require(rx['canonical'] and rx['zero_extension_reads']==30,'Independent canonical parse failed')
            checks.append(exact(rx['received_tokens'],source['tokens'][:OFFSETS[m]],f's{i}_rx_m{m}',out))
        for m in (7,8,9):
            # Only independently recovered source bits enter the rendering path.
            image=ledger.call('var_render',lambda m=m:backend.render(m),source=i,m=m)
            checks.append(exact(image,source['images'][m],f's{i}_VAR_m{m}',out))
        backend.end_source()
    return checks

def prepared(path,digest):
    r=checked_json(path,digest)
    require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED','Wrong gate registration')
    require(r.get('device_identity_binding')==device_identity_binding(),'Fresh dual-identity registration required; do not retry the old NVML-mask request')
    for n,s in r['tool_bindings'].items():require(sha(Path(__file__).with_name(n))==s,'New gate implementation changed')
    return r

def prepare(a):
    out=inside(a.out);require(not out.exists(),'Fresh registration output required')
    spec=checked_json(inside(a.spec),a.spec_sha256)
    r,er,records,h,extra=validate_spec(spec)
    out.mkdir(parents=True)
    request=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',spec=spec,spec_origin=dict(path=a.spec,sha256=a.spec_sha256),
        caps=CAPS,source_ids=IDS,old_runtime=h['numerical_runtime'],created_unix=time.time(),
        device_identity_binding=device_identity_binding(),
        tool_bindings={n:sha(Path(__file__).with_name(n)) for n in ('leo_whole_gate_v1.py','leo_whole_math_v1.py','leo_shared_preflight.py','leo_runtime_verify_v1.py')},
        budget_scope='Additional migration calls; original EP32 and actual48 budgets unchanged',
        old_VAR_CDF='UNOBSERVABLE_IN_SELECTED_SEED',old_Direct='UNOBSERVABLE_IN_SELECTED_SEED',
        native_dependency_candidate=extra['native_dependency_candidate'],old_dist_sha_unavailable=True,
        preparation_model_calls=0,numerical_qualification=False)
    request['runtime_byte_verification']=extra['runtime_byte_verification']
    write(out/'request.json',request);return dict(path=str(out/'request.json'),sha256=sha(out/'request.json'))

def resource_snapshot(shared,prelaunch=False):
    s=shared.collect_snapshot(BASE,UUID)
    try:
        ad=shared.check_resources(s,2)
        require(s['gpu']['index']==5 and s['gpu']['free_MiB']>=20*1024,'Selected physical GPU5 lacks20GiB fresh margin')
        if prelaunch:require(s['gpu']['utilization_percent']<=50,'Selected shared GPU is busy; no child started, no retry')
    except Exception as error:
        error.resource_snapshot=s;raise
    return s,ad

def worker_identity(out, request_sha256, owner_pid):
    """Refuse a direct worker CLI; only the live admitted parent may own it."""
    require(type(owner_pid) is int and owner_pid>0 and os.getppid()==owner_pid,'Worker has no admitted live parent')
    intent=json.loads((Path(out)/'intent.json').read_text(encoding='utf-8'))
    require(intent['owner_pid']==owner_pid and intent['request_sha256']==request_sha256,'Worker intent differs')
    launch_path=Path(out)/'child_started.json';until=time.monotonic()+2
    while not launch_path.exists() and time.monotonic()<until:
        require(os.getppid()==owner_pid,'Admitted parent exited before launch receipt');time.sleep(.01)
    require(launch_path.is_file(),'Actual parent launch receipt absent')
    launch=json.loads(launch_path.read_text(encoding='utf-8'))
    require(launch['pid']==os.getpid() and launch['owner_pid']==owner_pid and
            launch['request_sha256']==request_sha256,'Actual owned child identity differs')
    require(launch['argv'][1:]==['-B','-u',str(Path(__file__).absolute()),'_worker','--request',
            str(Path(out).parent/'request.json'),'--request-sha256',request_sha256,'--owner-pid',str(owner_pid)],
            'Actual worker command differs')
    return launch

def run(a):
    require(sys.platform.startswith('linux'),'Real owner requires Linux')
    request_path=inside(a.request);r=prepared(request_path,a.request_sha256);spec=r['spec']
    out=request_path.parent/'run';shared=helper();child=None
    with shared.owner_lock(inside(BASE/'controls/leo_whole_gate_v1/GPU5.owner.lock')):
        require(not out.exists(),'Run already claimed; no automatic replay');out.mkdir()
        write(out/'intent.json',dict(request_sha256=a.request_sha256,owner_pid=os.getpid(),started_unix=time.time(),caps=CAPS))
        try:
            _,_,_,_,extra=validate_spec(spec)
            write(out/'runtime_byte_verification_owner.json',extra['runtime_byte_verification'])
            snapshot,ad=resource_snapshot(shared,prelaunch=True);write(out/'prelaunch_resources.json',snapshot)
            write(out/'prelaunch_device_identity.json',query_nvml_device_identity())
            _,controlled=shared.controlled_environment(BASE,out,6)
            # Preserve only the admitted projected library path, not ambient LD_PRELOAD/PYTHONPATH/proxy configuration.
            env=dict(controlled);(out/'home').mkdir()
            libraries=extra['environment']['LD_LIBRARY_PATH']
            for directory in libraries.split(':'):require(inside(directory).is_dir(),'Unadmitted native library directory')
            env.update(PATH=str(Path(spec['python']).parent),LD_LIBRARY_PATH=libraries,HOME=str(out/'home'),LANG='C.UTF-8',LC_ALL='C.UTF-8')
            env.update(**cuda_environment_binding(),CUBLAS_WORKSPACE_CONFIG=':4096:8',VIRTUAL_ENV=str(Path(spec['python']).parent.parent))
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
            # Thread values preserve the original math context; two CPUs is a disclosed scheduling restriction.
            controlled.update(**cuda_environment_binding(),CUBLAS_WORKSPACE_CONFIG=':4096:8',torch_threads=6,torch_interop_threads=2)
            controlled.update({k:env[k] for k in ('PATH','LD_LIBRARY_PATH','HOME','LANG','LC_ALL')})
            write(out/'controlled_environment.json',controlled)
            argv=[spec['python'],'-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(request_path),'--request-sha256',a.request_sha256,'--owner-pid',str(os.getpid())]
            seconds=min(spec['max_seconds'],spec['deadline_unix']-time.time());require(seconds>0,'Deadline expired before launch')
            with (out/'child.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen(argv,env=env,cwd=spec['project_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    preexec_fn=shared.child_limits(ad['cpu_affinity'],seconds,2))
                write(out/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),request_sha256=a.request_sha256,argv=argv,cpu_affinity=ad['cpu_affinity'],started_unix=time.time()))
                waited=shared.wait_owned(child,seconds)
            write(out/'actual_child_wait.json',waited);require(waited['success'],'GPU child failed/OOM/timed out; retain evidence')
            cp=out/'worker_completion.json';done=json.loads(cp.read_text())
            require(done['status']=='PASS_TWO_SOURCE_WHOLE_COMPATIBILITY_ONLY' and done['request_sha256']==a.request_sha256,'Worker completion missing/stale')
            require(done['counts']['completed']==CAPS and done['counts']['unresolved']==0,'Actual call ledger did not close')
            validate_spec(spec)
            result=dict(status='PASS_TWO_SOURCE_WHOLE_COMPATIBILITY_ONLY',schema=SCHEMA,request_sha256=a.request_sha256,
                worker_completion=dict(path=str(cp),sha256=sha(cp)),actual_children_waited=True,worker_exit_codes=[child.returncode],
                old_VAR_CDF='UNOBSERVABLE_IN_SELECTED_SEED',old_Direct='UNOBSERVABLE_IN_SELECTED_SEED',production_admission=False,
                native_dependency_candidate=r['native_dependency_candidate'],old_dist_sha_unavailable=True,
                timing_measurement=False,shared_GPU=True,automatic_successor=False)
            write(out/'completion.json',result);return result
        except BaseException as e:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),
                 resource_snapshot=getattr(e,'resource_snapshot',None),
                 child_pid=None if child is None else child.pid,child_exit_code=None if child is None else child.returncode))
            raise

def worker(a):
    global STOP
    require(sys.platform.startswith('linux'),'Real worker requires Linux')
    check_cuda_environment()
    r=prepared(inside(a.request),a.request_sha256);spec=r['spec'];out=Path(a.request).parent/'run'
    launch=worker_identity(out,a.request_sha256,a.owner_pid)
    require(launch['argv'][0]==spec['python'],'Registered interpreter spelling differs')
    started=time.monotonic();ledger=None;backend=None;shared=helper();last=0.
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def guard():
        require(not STOP and not (out/'STOP').exists() and not (BASE/'STOP').exists(),'Owned stop requested')
        require(time.time()<spec['deadline_unix'] and time.monotonic()-started<spec['max_seconds'],'Finite gate deadline exhausted')
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>=2:
            snapshot,_=resource_snapshot(shared);last=time.monotonic()
            with (out/'resources.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(snapshot)+'\n')
    try:
        write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256,started_unix=time.time()))
        resolver,env_resolver,records,h,extra=validate_spec(spec);boundary()
        write(out/'runtime_byte_verification_worker.json',extra['runtime_byte_verification'])
        # Parse all required old witness arrays before constructing any model.
        sources=[load_source(x,resolver) for x in records]
        ledger=Ledger(out/'calls',guard)
        module=import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_leo_whole_frozen_math')
        backend=ledger.call('model_load',lambda:module.Backend(resolver,env_resolver,spec,extra,ledger,boundary,out))
        checks=exercise(sources,backend,ledger,out/'checks');backend.close();resolver.reverify();env_resolver.reverify();guard()
        counts=ledger.summary();require(counts['completed']==CAPS and counts['unresolved']==0,'Registered scientific counts differ')
        result=dict(status='PASS_TWO_SOURCE_WHOLE_COMPATIBILITY_ONLY',schema=SCHEMA,request_sha256=a.request_sha256,
            counts=counts,source_ids=IDS,checks=len(checks),old_VAR_CDF='UNOBSERVABLE_IN_SELECTED_SEED',old_Direct='UNOBSERVABLE_IN_SELECTED_SEED',
            new_host_TX_RX_CDF_exact=True,production_admission=False,timing_measurement=False,
            native_dependency_candidate=extra['native_dependency_candidate'],old_dist_sha_unavailable=True,
            max_allocated_bytes=backend.t.cuda.max_memory_allocated(),max_reserved_bytes=backend.t.cuda.max_memory_reserved(),
            allocator_limit_bytes=16*(1<<30),allocator_cap_is_not_total_GPU_isolation=True)
        write(out/'worker_completion.json',result)
    except BaseException as e:
        if backend is not None:backend.failure_evidence()
        write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,error=repr(e),
             mismatch_layer=getattr(e,'layer',None),old_VAR_CDF='UNOBSERVABLE_IN_SELECTED_SEED',
             interpretation='A failed exact gate is not a proof that future explicitly separate migration protocols are impossible',
             traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare');s.add_argument('--spec',required=True);s.add_argument('--spec-sha256',required=True);s.add_argument('--out',required=True)
    for n in ('run','_worker'):
        s=sub.add_parser(n);s.add_argument('--request',required=True);s.add_argument('--request-sha256',required=True)
        if n=='_worker':s.add_argument('--owner-pid',required=True,type=int)
    a=p.parse_args();result=prepare(a) if a.command=='prepare' else run(a) if a.command=='run' else worker(a)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
