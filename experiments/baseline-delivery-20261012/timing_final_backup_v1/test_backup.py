"""Synthetic filesystem and metadata fixtures only; no real preflight/grid/model."""
import copy,hashlib,io,json,os,stat,tarfile,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import build as b
import fetch as f


def summary():
    return dict(status='COMPLETE_FIXED16_SIX_METHOD_TIMING',frames=1728,measured_frames=864,packet_calls=2304,
        inherited_complete_frames=576,newly_completed_frames=1152,completed_frames_across_attempts=1728,
        failed_partial_visual_attempts=1,known_fresh_visual_attempts_across_attempts=1729,physical_endpoint_entries_across_attempts=1729,
        original_intervals_recomputed=0,training_updates=0,prior_failure_preserved=True,
        prior_failed_slot_not_counted_as_warmup=True,shared_environment=True,exclusive_or_uncontended_claimed=False)


def proc(pid,ticks):return dict(pid=pid,start_ticks=ticks)

def fixture():
    def pin(p):
        h=b.MASTER if p==b.REG else b.ORIGINAL if p==b.OUT/'registration.json' else 'a'*64
        return dict(path=str(p),sha256=h,bytes=1)
    ownerargv=[b.PY,'-B',str(b.ENTRY),'run','--expected-resource-registration',b.MASTER]
    ppargv=[b.PY,'-B',str(b.ENTRY),'postprocess','--output',str(b.LIGHT),'--expected-resource-registration',b.MASTER]
    dlaunch=dict(process=proc(2,20),parent_process=proc(1,10),argv=ownerargv)
    plaunch=dict(process=proc(4,40),parent_process=proc(3,30),argv=ppargv)
    def wait(launch,key):return dict(actual_wait=True,returncode=0,timed_out=False,terminated=False,killed=False,
        metadata_error=None,argv=launch['argv'],parent_process=launch['parent_process'],**{key:launch['process']})
    owner=dict(role='run',process=dict(proc(2,'20'),argv=ownerargv),parent_process=proc(1,'10'),
        resource_registration=pin(b.REG),original_registration=pin(b.OUT/'registration.json'))
    worker=dict(role='_worker',process=dict(proc(5,'50'),argv=[b.PY,str(b.ENTRY),'_worker']),parent_process=proc(2,'20'),
        resource_registration=pin(b.REG),original_registration=pin(b.OUT/'registration.json'))
    done=summary()
    for key in ('case_journal','raw_wall_and_audit','per_boundary_resource_history','pure_provenance_exclusions'):
        done[key]=pin(b.RUN/(key+'.json'))
    outputs={f'file{i}.csv':'a'*64 for i in range(11)}
    post=dict(status='COMPLETE_CPU_TIMING_AUDIT_AND_TABLES',input_registration_sha256=b.ORIGINAL,
        measured_calls=864,timing_scope='shared-environment latency',new_scientific_calls=0,bootstrap_calls=0,output_sha256=outputs)
    side=dict(binding=dict(resource_registration=pin(b.REG),original_registration=pin(b.OUT/'registration.json')),
        startup_free_MiB=8192,prior_startup_free_MiB=18432,allocator_cap_GiB=16,runtime_free_floor_MiB=2048,
        shared_environment=True,full_resource_conditions_identical=False,original_statistics_functions_changed=False)
    data={b.REG:dict(original_source_sha256={}),b.OUT/'completion.json':done,
        b.OUT/'actual_wait.json':dict(actual_child_waited=True,exit_code=0,timed_out=False),
        b.D/'launch.json':dlaunch,b.D/'actual_wait.json':wait(dlaunch,'child'),b.I/'launch.json':dict(process=proc(1,10)),
        b.CONTROL/'owner_resource_execution.json':owner,b.CONTROL/'worker_resource_execution.json':worker,
        b.P/'launch.json':plaunch,b.P/'actual_wait.json':wait(plaunch,'process'),b.LIGHT/'completion.json':post,
        b.LIGHT/'resource_revision.json':side,
        b.CONTROL/'postprocess_returned.json':dict(status='ORIGINAL_POSTPROCESS_AND_RESOURCE_SIDECAR_RETURNED_SUCCESS',
            binding=pin(b.CONTROL/'postprocess_binding.json'),resource_revision=pin(b.LIGHT/'resource_revision.json'),
            original_completion=pin(b.LIGHT/'completion.json'))}
    for role,item in (('owner',owner),('worker',worker)):
        data[b.CONTROL/(role+'_returned.json')]=dict(status='ORIGINAL_V6_RETURNED_SUCCESS',resource_registration=pin(b.REG),
            original_completion=pin(b.OUT/'completion.json'),process=item['process'])
    rows=[pin(b.REG),pin(b.OUT/'registration.json')]
    return data,pin,rows,[b.LIGHT/n for n in list(outputs)+['completion.json','resource_revision.json']]


class ClosureTests(unittest.TestCase):
    def run_fixture(self,mutate=lambda d:None,probe=lambda pid:None):
        data,pinner,rows,files=fixture();mutate(data)
        with patch.object(b,'evidence_pins',return_value=rows),patch.object(b,'read',side_effect=lambda p:data[p]),\
             patch.object(b,'pin',side_effect=pinner),patch.object(b,'checked',return_value=Path('/synthetic/file')),\
             patch.object(b,'walk_files',return_value=files):return b.receipt_closure(probe)
    def reject(self,path,key,value):
        with self.assertRaises((RuntimeError,KeyError)):self.run_fixture(lambda d:d[path].__setitem__(key,value))
    def test_valid_real_receipt_field_shapes(self):self.assertEqual(self.run_fixture()['summary']['frames'],1728)
    def test_owner_top_level_argv_is_not_accepted(self):
        def bad(d):d[b.CONTROL/'owner_resource_execution.json']['process'].pop('argv')
        with self.assertRaises(KeyError):self.run_fixture(bad)
    def test_returned_wrong_completion(self):self.reject(b.CONTROL/'owner_returned.json','original_completion',{})
    def test_returned_wrong_registration(self):self.reject(b.CONTROL/'worker_returned.json','resource_registration',{})
    def test_actual_wait_bool_zero_rejected(self):self.reject(b.D/'actual_wait.json','returncode',False)
    def test_actual_wait_timeout_rejected(self):self.reject(b.P/'actual_wait.json','timed_out',True)
    def test_actual_wait_metadata_error(self):self.reject(b.P/'actual_wait.json','metadata_error','failed')
    def test_actual_wait_missing_flag(self):
        with self.assertRaises(RuntimeError):self.run_fixture(lambda d:d[b.P/'actual_wait.json'].pop('killed'))
    def test_original_wait_false_zero_rejected(self):self.reject(b.OUT/'actual_wait.json','exit_code',False)
    def test_wrong_measured_scope(self):self.reject(b.LIGHT/'completion.json','measured_calls',865)
    def test_recount_attempts_rejected(self):self.reject(b.OUT/'completion.json','known_fresh_visual_attempts_across_attempts',1728)
    def test_new_science_rejected(self):self.reject(b.LIGHT/'completion.json','new_scientific_calls',1)
    def test_resource_sidecar_rejected(self):self.reject(b.LIGHT/'resource_revision.json','startup_free_MiB',18432)
    def test_live_owner_rejected(self):
        with self.assertRaises(RuntimeError):self.run_fixture(probe=lambda pid:proc(pid,20) if pid==2 else None)
    def test_live_postprocess_parent_rejected(self):
        with self.assertRaises(RuntimeError):self.run_fixture(probe=lambda pid:proc(pid,30) if pid==3 else None)
    def test_pid_reuse_is_gone(self):self.assertEqual(self.run_fixture(probe=lambda pid:proc(pid,999))['summary']['frames'],1728)
    def test_wrong_owner_command(self):
        def bad(d):d[b.CONTROL/'owner_resource_execution.json']['process']['argv']=['bad']
        with self.assertRaises(RuntimeError):self.run_fixture(bad)
    def test_wrong_worker_ancestry(self):
        def bad(d):d[b.CONTROL/'worker_resource_execution.json']['parent_process']=proc(20,'20')
        with self.assertRaises(RuntimeError):self.run_fixture(bad)
    def test_postprocess_fork_empty_process_argv_allowed(self):
        def good(d):d[b.P/'actual_wait.json']['process']['argv']=[]
        self.assertEqual(self.run_fixture(good)['summary']['frames'],1728)
    def test_missing_shared_disclosure_rejected(self):self.reject(b.OUT/'completion.json','shared_environment',False)
    def test_wrapper_string_ticks_only(self):
        self.assertEqual(b.wrapper_identity(proc(1,'42'))['start_ticks'],42)
        for value in (42,True,4.2,'0','01','-1','1.0','',None,'１２'):
            with self.subTest(value=value),self.assertRaises(RuntimeError):b.wrapper_identity(proc(1,value))
    def test_supervisor_ticks_remain_strict_integer(self):
        self.reject(b.P/'actual_wait.json','process',proc(4,'40'))


class FileTests(unittest.TestCase):
    def test_bad_archive_names(self):
        for name in ('../a','/a','a\\b','a//b','a/./b','a/../b','a:N','CON.txt','a/foo.','a/NUL','a/LPT3.json'):
            with self.subTest(name=name),self.assertRaises(RuntimeError):b.safe_parts(name)
    def test_exact_cache_only(self):
        with self.assertRaises(RuntimeError):b.payload_name('runtime/six_method_timing_v6/torchinductor_root/a.json')
        self.assertEqual(b.payload_name('runtime/six_method_timing_v6/cache_provenance.json'),'runtime/six_method_timing_v6/cache_provenance.json')
    def test_models_and_R1_arrays_rejected(self):
        for name in ('repo/results/layered_mismatch_raw_20261013/a.npy','runtime/six_method_timing_v6/model.pt','repo/data/model.json'):
            with self.subTest(name=name),self.assertRaises(RuntimeError):b.payload_name(name)
    def test_known_doc_and_log_lock_allowed(self):
        for name in ('repo/results/layered_mismatch_raw_20261013/README.md','runtime/six_method_timing_v6/worker.log',
                'repo/results/baseline_strength_20261011/timing/six_method_v6/owner.lock'):
            self.assertEqual(b.payload_name(name),name)
    def test_stable_hash_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'a.txt';p.write_text('a');rows,_=b.deduplicate([(p,'DEPENDENCIES.md')]);p.write_text('b')
            with self.assertRaises(RuntimeError):b.stable_record(rows[0])
    def test_same_inode_alias_and_safe_ordinary_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'a.txt';p.write_text('a');q=root/'b.txt';os.link(p,q)
            rows,aliases=b.deduplicate([(p,'runtime/six_method_timing_v6/a.txt'),(q,'runtime/six_method_timing_v6/b.txt')])
            self.assertEqual((len(rows),len(aliases)),(1,1));dest=root/'out';target=f.destination(dest,rows[0]['path'])
            target.parent.mkdir(parents=True);target.write_text('a');f.materialize_aliases(dest,aliases)
            restored=f.destination(dest,aliases[0]['path']);self.assertEqual(restored.read_text(),'a');self.assertFalse(restored.is_symlink())
    def test_case_collision_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'a';p.write_text('a')
            with self.assertRaises(RuntimeError):b.deduplicate([(p,'DEPENDENCIES.md'),(p,'dependencies.md')])
    def test_existing_receipt_conflict_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'receipt.json';f.save(p,{'a':1})
            with self.assertRaises(RuntimeError):f.save(p,{'a':2})
            self.assertEqual(json.loads(p.read_text()),{'a':1})
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);link=root/'link';original=Path.lstat
            def fake_lstat(path):
                return SimpleNamespace(st_mode=stat.S_IFLNK|0o777) if path==link else original(path)
            with patch.object(Path,'lstat',autospec=True,side_effect=fake_lstat):
                for path in (link,link/'child.json'):
                    with self.subTest(path=str(path)),self.assertRaises(RuntimeError):b.regular(path)
    def test_tar_roundtrip_and_alias_and_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'data';source.write_bytes(b'synthetic')
            rows,aliases=b.deduplicate([(source,'runtime/six_method_timing_v6/a.txt')])
            aliases=[dict(rows[0],path='runtime/six_method_timing_v6/b.txt',target=rows[0]['path'],reason='SAME_INODE')]
            manifest=dict(schema=b.SCHEMA,git_commit='c'*40,files=rows,aliases=aliases,
                empty_directories=['runtime/six_method_timing_v6/empty'],**b.FLAGS)
            mp=root/'manifest.json';b.save(mp,manifest);archive=root/'a.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                tar.add(source,arcname=rows[0]['path']);tar.add(mp,arcname='manifest.json')
            b.verify_archive(archive,mp,rows)
            receipt=dict(schema=b.SCHEMA,git_commit='c'*40,manifest_sha256=b.sha(mp),payload_files=1,alias_files=1,
                restored_files=2,payload_bytes=9,alias_bytes=9,**b.FLAGS)
            with patch.object(f,'D',root/'restore'):result=f.extract(archive,receipt)
            self.assertEqual(result,manifest);self.assertTrue((root/'restore/runtime/six_method_timing_v6/empty').is_dir())
    def test_manifest_path_conflicts(self):
        def check(paths,empty=[]):
            rows=[dict(path=n,bytes=1,sha256='a'*64) for n in paths]
            manifest=dict(schema=b.SCHEMA,git_commit='c'*40,files=rows,aliases=[],empty_directories=empty,**b.FLAGS)
            receipt=dict(git_commit='c'*40,payload_files=len(rows),alias_files=0,restored_files=len(rows),payload_bytes=len(rows),alias_bytes=0)
            return f.validate_manifest(manifest,receipt,paths+['manifest.json'])
        base='runtime/six_method_timing_v6/'
        for paths,empty in [([base+'a.json',base+'A.json'],[]),([base+'a.json',base+'a.json/b.json'],[]),
                ([base+'a.json'],[base+'a.json/empty']),([base+'a.json'],[base+'EMPTY',base+'empty/sub'])]:
            with self.subTest(paths=paths,empty=empty),self.assertRaises(RuntimeError):check(paths,empty)
    def test_archive_link_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'a.tar.gz';mp=root/'m';mp.write_text('{}')
            with tarfile.open(archive,'w:gz') as tar:
                info=tarfile.TarInfo('DEPENDENCIES.md');info.type=tarfile.SYMTYPE;info.linkname='bad';tar.addfile(info)
            with self.assertRaises(RuntimeError):b.verify_archive(archive,mp,[])
    def test_authorization_cli_needs_real_inputs(self):
        self.assertEqual(b.SOURCE_NAMES,('build.py','fetch.py','closed_evidence.json','test_backup.py','README.md'))
        self.assertEqual(b.CACHE_EXCLUSIONS,(b.RUN/'torchinductor_root',))
    def test_unknown_nested_failure_rejected(self):
        with patch.object(b,'walk_files',side_effect=lambda root,excluded:[root/'sub/failure.json']):
            with self.assertRaises(RuntimeError):b.no_current_failure()
    def test_missing_commit_receipt_rejected(self):
        def pin(p):
            if p.name=='commit.json':raise FileNotFoundError('fixture absent commit')
            return dict(path=str(p),bytes=1,sha256='a'*64)
        with patch.object(b,'canonical',side_effect=lambda p:p),patch.object(b,'pin',side_effect=pin):
            with self.assertRaises(FileNotFoundError):b.publication_receipts(b.BASE/'six_method_timing_final_publication_20261014_v1','c'*40)


if __name__=='__main__':unittest.main(verbosity=2)
