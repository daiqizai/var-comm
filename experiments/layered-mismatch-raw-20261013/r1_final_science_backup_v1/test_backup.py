"""Synthetic paths, manifests and mocked closed-process metadata only.

No SSH, repository result, model, data source, active quality, or production
build is accessed. The subprocess fixture archives only its own tiny text.
"""
from __future__ import annotations
import ast, copy, hashlib, importlib.util, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE))
import build as b
import fetch as f


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')
    return b.pin(path)


class FigureFixture:
    def __init__(self,base):
        self.R=base/'repo';self.O=self.R/'results';self.L=base/'runtime';self.BASE=base/'publication'
        self.FIGURES=self.O/'r1_cache_recovery_figures_v1';self.REG=self.O/'registration.json';self.GRAPH=self.R/'figures.py'
        self.FIGURES.mkdir(parents=True);self.L.mkdir();self.BASE.mkdir()
        put(self.REG,{'runtime':{'python_cpu':'synthetic_python'}})
        self.GRAPH.write_text('# synthetic figure entry; never executed\n',encoding='utf8')
        self.patches=mock.patch.multiple(b,R=self.R,O=self.O,L=self.L,BASE=self.BASE,FIGURES=self.FIGURES,
            REG=self.REG,GRAPH=self.GRAPH,GRAPH_SHA=b.sha(self.GRAPH))
        self.patches.start()
        self.process={'pid':987654,'start_ticks':222};self.parent={'pid':987655,'start_ticks':223}
        control=self.L/'synthetic_figure_control';control.mkdir()
        staging=self.L/'r1_cache_recovery_figures_v1.preparing_SYNTHETIC';staging.mkdir()
        argv=['synthetic_python','-B',str(self.GRAPH),'--expected-registration',b.MASTER]
        wait=put(control/'actual_wait.json',dict(actual_wait=True,returncode=0,child=self.process,
            terminated=False,killed=False,timed_out=False,metadata_error=None))
        launch=put(control/'launch.json',dict(process=self.process,parent_process=self.parent,argv=argv))
        stdout=put(control/'stdout.json',dict(status='R1_FIGURE_FILES_READY_VISUAL_REVIEW_REQUIRED',
            output=str(self.FIGURES),staging_preserved=str(staging),scientific_calls=0))
        stderr=control/'stderr.txt';stderr.write_text('',encoding='utf8')
        master={'path':str(self.REG),'sha256':b.MASTER}
        self.delivery={'manifest':{'path':str(self.O/'delivery_manifest.json'),'sha256':'1'*64}}
        files=[]
        for name in sorted(b.FIGURE_ARTIFACTS):
            path=self.FIGURES/name;path.write_text('SYNTHETIC ONLY: '+name,encoding='utf8');files.append(b.pin(path))
        manifest=put(self.FIGURES/'figure_manifest.json',dict(registration=master,files=files,
            input_manifest=self.delivery['manifest'],exports=['vector_PDF','editable_text_SVG','600dpi_PNG'],
            visual_review='PENDING_REQUIRED',scientific_calls=0))
        completion=put(self.FIGURES/'completion.json',dict(registration=master,manifest_sha256=manifest['sha256'],
            status='COMPLETE_R1_FIGURE_FILES_VISUAL_REVIEW_PENDING',requires_actual_plot_process_exit_zero=True,scientific_calls=0))
        os.link(self.FIGURES/'completion.json',self.FIGURES/'completion.pending.json')
        review=put(self.O/'synthetic_review.json',dict(status='PASS',actual_opened_both_pngs=True,registration=master,
            figure_manifest=manifest,reviewer='SYNTHETIC FIXTURE, no actual human review',
            figures=[x for x in files if x['path'].endswith('.png')]))
        self.auth=dict(figure_entry=b.pin(self.GRAPH),figure_manifest=manifest,figure_completion=completion,
            visual_review=review,figure_execution=dict(actual_wait=wait,launch=launch,stdout=stdout,stderr=b.pin(stderr),
            process=self.process,parent_process=self.parent,argv=argv))

    def close(self):self.patches.stop()
    def run(self,probe=lambda pid:None):return b.figure_closure(self.auth,self.delivery,probe)
    def rewrite_wait(self,**change):
        path=Path(self.auth['figure_execution']['actual_wait']['path']);value=b.read(path);value.update(change)
        self.auth['figure_execution']['actual_wait']=put(path,value)
    def rewrite_review(self,**change):
        path=Path(self.auth['visual_review']['path']);value=b.read(path);value.update(change)
        self.auth['visual_review']=put(path,value)


class FigureGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='R1_SYNTHETIC_BACKUP_')
        self.fx=FigureFixture(Path(self.temp.name));self.addCleanup(self.temp.cleanup);self.addCleanup(self.fx.close)

    def test_complete_synthetic_metadata_passes(self):
        result=self.fx.run();self.assertTrue(result['all_figure_files_SHA_verified'])

    def test_wait_codes_and_flags_fail_before_manifest_read(self):
        for change in ({'returncode':1},{'returncode':False},{'actual_wait':False},{'timed_out':True},{'metadata_error':'failure'}):
            with self.subTest(change=change):
                original=copy.deepcopy(self.fx.auth);path=Path(original['figure_execution']['actual_wait']['path']);raw=path.read_bytes()
                self.fx.rewrite_wait(**change)
                with mock.patch.object(b,'walk_files',side_effect=AssertionError('premature figure walk')):
                    with self.assertRaisesRegex(RuntimeError,'Actual figure'):self.fx.run()
                path.write_bytes(raw);self.fx.auth=original

    def test_live_child_or_parent_refused(self):
        for live in (self.fx.process,self.fx.parent):
            with self.subTest(live=live):
                with self.assertRaisesRegex(RuntimeError,'still exists'):
                    self.fx.run(lambda pid:live if pid==live['pid'] else None)

    def test_reused_pid_with_other_generation_is_gone(self):
        self.fx.run(lambda pid:dict(pid=pid,start_ticks=99999))

    def test_wrong_child_generation_refused(self):
        self.fx.rewrite_wait(child=dict(pid=self.fx.process['pid'],start_ticks=333))
        with self.assertRaisesRegex(RuntimeError,'generation'):self.fx.run()

    def test_boolean_process_identity_refused(self):
        self.fx.auth['figure_execution']['process']['start_ticks']=True
        with self.assertRaisesRegex(RuntimeError,'generation'):self.fx.run()

    def test_wrong_graph_identity_refused(self):
        self.fx.auth['figure_entry']['sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'figure entry'):self.fx.run()

    def test_changed_artifact_refused(self):
        (self.fx.FIGURES/'r1_calibration_quality.png').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'SHA differs'):self.fx.run()

    def test_only_one_reviewed_png_refused(self):
        value=b.read(self.fx.auth['visual_review']['path']);self.fx.rewrite_review(figures=value['figures'][:1])
        with self.assertRaisesRegex(RuntimeError,'exactly both'):self.fx.run()

    def test_visual_review_outside_published_result_scope_refused(self):
        old=b.read(self.fx.auth['visual_review']['path']);self.fx.auth['visual_review']=put(self.fx.L/'review.json',old)
        with self.assertRaisesRegex(RuntimeError,'fixed task roots'):self.fx.run()

    def test_pending_copy_instead_of_same_file_refused(self):
        pending=self.fx.FIGURES/'completion.pending.json';pending.unlink();pending.write_bytes((self.fx.FIGURES/'completion.json').read_bytes())
        with self.assertRaisesRegex(RuntimeError,'same-file'):self.fx.run()

    def test_actual_failure_receipt_refused(self):
        put(Path(self.fx.auth['figure_execution']['actual_wait']['path']).parent/'failure.json',{})
        with self.assertRaisesRegex(RuntimeError,'failure receipt'):self.fx.run()

    def test_extra_unmanifested_file_refused(self):
        (self.fx.FIGURES/'extra.txt').write_text('extra')
        with self.assertRaisesRegex(RuntimeError,'Unknown or missing'):self.fx.run()


class ScopeTests(unittest.TestCase):
    def test_old_phy_projection_new_images_and_failure_preserved(self):
        for name in ('runtime/r1_execution_v1/phy/journal/packets/x.complete.json',
                'runtime/r1_execution_v1/projection/model_inputs.json',
                'runtime/r1_execution_v1/visual/resource_history_at_failure.json',
                'runtime/r1_cache_recovery_v1/visual/images/x.npz',
                'runtime/r1_cache_recovery_v1/score/calls.sqlite',
                'runtime/r1_cache_recovery_v1/selection/source_means.json'):
            self.assertEqual(name,b.payload_name(name))

    def test_exact_compiler_caches_and_outside_payload_rejected(self):
        for name in ('runtime/r1_execution_v1/cache/x.json','runtime/r1_cache_recovery_v1/cache/x.json',
                'runtime/r1_cache_recovery_v1/visual/runtime_cache/tmp/x.json','runtime/r1_cache_recovery_v1/visual/model.pth',
                'runtime/r0_bler_v1/completion.json','runtime/r1_cache_recovery_v1/unknown/x.json','../outside','C:/outside',
                'runtime/r1_cache_recovery_v1/visual/images/../../checkpoint.pt'):
            with self.subTest(name=name),self.assertRaises(RuntimeError):b.payload_name(name)

    def test_scientific_cache_name_is_not_generically_omitted(self):
        b.payload_name('runtime/r1_cache_recovery_v1/score/rows/cache_identity.json')
        b.payload_name('runtime/r1_cache_recovery_v1/visual/images/cache_key.npz')

    def test_regular_rejects_symlink_metadata_without_creating_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'regular.json';path.write_text('{}')
            original=Path.lstat
            def fake(self):
                if self==path:return type('Info',(),{'st_mode':0o120777})()
                return original(self)
            with mock.patch.object(Path,'lstat',fake),self.assertRaisesRegex(RuntimeError,'Symlink'):b.regular(path)

    def test_walk_prunes_only_exact_named_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'cache').mkdir();(root/'cache'/'rebuild.txt').write_text('x')
            (root/'score_cache').mkdir();(root/'score_cache'/'scientific.json').write_text('{}')
            got=list(b.walk_files(root,(root/'cache',)))
            self.assertEqual(got,[root/'score_cache'/'scientific.json'])

    def test_unclosed_build_never_creates_output_or_enumerates_payload(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(b,'B',Path(tmp)/'new'):
            with mock.patch.object(b,'closure',side_effect=RuntimeError('not closed')),mock.patch.object(b,'payload_sources') as scan:
                with self.assertRaisesRegex(RuntimeError,'not closed'):b.build(None)
            scan.assert_not_called();self.assertFalse(b.B.exists())

    def test_existing_attempt_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(b,'B',Path(tmp)),mock.patch.object(b,'closure') as gate:
            with self.assertRaisesRegex(RuntimeError,'Existing attempt'):b.build(None)
            gate.assert_not_called()

    def test_only_same_inode_alias_deduplication_and_stat_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.txt';a.write_text('same');link=root/'alias.txt';os.link(a,link)
            c=root/'separate.txt';c.write_text('same')
            records,aliases=b.deduplicate([(a,'evidence/a.txt'),(link,'evidence/alias.txt'),(c,'evidence/separate.txt')])
            self.assertEqual((len(records),len(aliases)),(2,1));self.assertEqual(aliases[0]['reason'],'SAME_INODE')
            b.stable_record(records[0]);a.write_text('new!')
            with self.assertRaisesRegex(RuntimeError,'Source changed'):b.stable_record(records[0])

    def test_missing_scientific_pin_rejected_but_exact_cache_reference_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'record.json';l=root/'L'
            with mock.patch.object(b,'L',l):
                put(path,{'file':{'path':str(l/'r1_cache_recovery_v1/visual/images/a.npz'),'sha256':'f'*64}})
                records=[dict(source_path=str(path),path='evidence/record.json')]
                with self.assertRaisesRegex(RuntimeError,'omitted'):b.check_archived_references(records,[])
                put(path,{'file':{'path':str(l/'r1_cache_recovery_v1/visual/runtime_cache/tmp/a.json'),'sha256':'f'*64}})
                result=b.check_archived_references(records,[])
                self.assertEqual(len(result),1)


class ManifestTests(unittest.TestCase):
    def fixture(self):
        row=dict(path='evidence/a.txt',source_path='/synthetic/a',bytes=2,sha256=hashlib.sha256(b'aa').hexdigest())
        alias=dict(path='evidence/b.txt',source_path='/synthetic/b',target=row['path'],bytes=2,sha256=row['sha256'],reason='SAME_INODE')
        manifest=dict(schema=b.SCHEMA,**b.FLAGS,git_commit='1'*40,files=[row],aliases=[alias],
            empty_directories=['runtime/r1_execution_v1/visual/images'])
        receipt=dict(git_commit='1'*40,payload_files=1,alias_files=1,restored_files=2,payload_bytes=2,alias_bytes=2)
        return manifest,receipt,[row['path'],'manifest.json']

    def test_valid_small_manifest(self):
        m,r,n=self.fixture();self.assertEqual(len(f.validate_manifest(m,r,n)),1)

    def test_scope_bool_is_not_integer_science_zero(self):
        m,r,n=self.fixture();m['science_calls']=False
        with self.assertRaisesRegex(RuntimeError,'scope'):f.validate_manifest(m,r,n)

    def test_bad_alias_missing_file_case_collision_or_directory_rejected(self):
        for change in ('alias','missing','case','empty'):
            m,r,n=self.fixture()
            if change=='alias':m['aliases'][0]['sha256']='0'*64
            elif change=='missing':n=[]
            elif change=='case':m['aliases'][0]['path']='EVIDENCE/A.TXT'
            else:m['empty_directories']=['runtime/r1_cache_recovery_v1/visual/runtime_cache']
            with self.subTest(change=change),self.assertRaises(RuntimeError):f.validate_manifest(m,r,n)

    def test_alias_restores_as_ordinary_file(self):
        m,_,_=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'evidence').mkdir();(root/'evidence/a.txt').write_bytes(b'aa')
            f.materialize_aliases(root,m['aliases'])
            dest=root/'evidence/b.txt';self.assertTrue(dest.is_file());self.assertFalse(dest.is_symlink())
            self.assertEqual(dest.read_bytes(),b'aa')

    def test_windows_long_path_helper_is_absolute(self):
        if os.name=='nt':
            value=f.extended_local_path('D:/Lu_Data/test/'+'x'*200+'/file.txt')
            self.assertTrue(str(value).startswith('\\\\?\\'));self.assertTrue(value.is_absolute())
        else:self.assertEqual(f.extended_local_path('/synthetic/a'),Path('/synthetic/a'))

    def test_module_import_has_no_network_or_scientific_imports(self):
        self.assertFalse(any(name in sys.modules for name in ('torch','numpy','tensorflow','sionna','lpips')))


class CLIOrderTests(unittest.TestCase):
    def test_subprocess_mocked_closure_reaches_real_build_helpers(self):
        source=(HERE/'build.py').read_text(encoding='utf8');tree=ast.parse(source)
        main=[node for node in tree.body if isinstance(node,ast.If) and "__name__" in ast.get_source_segment(source,node.test)][-1]
        self.assertEqual(tree.body[-1],main,'Main must follow every helper definition')
        # Inject only synthetic metadata before the actual CLI block. If a helper
        # remained below __main__, this subprocess would fail like production.
        before,main_text=source[:main.col_offset+sum(len(x) for x in source.splitlines(keepends=True)[:main.lineno-1])],ast.get_source_segment(source,main)
        with tempfile.TemporaryDirectory(prefix='R1_SYNTHETIC_CLI_') as tmp:
            fixture=Path(tmp);script=fixture/'synthetic_cli_ONLY.py'
            injected='''
from types import SimpleNamespace
F=Path(os.environ['R1_SYNTHETIC_ROOT']);B=F/'package';LOCKS=()
sys.modules['fcntl']=SimpleNamespace()
payload=F/'only_synthetic.txt';payload.write_text('SYNTHETIC, NOT SCIENTIFIC',encoding='utf8')
dummy=pin(payload)
verified=dict(registration=dummy,publication={'git_commit':'1'*40},reg={'code_bindings':{}},
    delivery_entry=dummy,figure_entry=dummy,processes=[])
closure=lambda args:verified
payload_sources=lambda state:[(payload,'evidence/synthetic.txt')]
dependency_receipts=lambda:[]
empty_runtime_directories=lambda:[]
'''
            script.write_text(before+injected+'\n'+main_text+'\n',encoding='utf8')
            completed=subprocess.run([sys.executable,'-B',str(script),'--authorization-path','SYNTHETIC',
                '--authorization-sha256','0'*64,'--authorization-bytes','0'],env=dict(os.environ,R1_SYNTHETIC_ROOT=str(fixture)),
                capture_output=True,text=True,timeout=30)
            self.assertEqual(completed.returncode,0,completed.stderr)
            result=json.loads(completed.stdout)
            self.assertTrue(result['receipt']['archive_payload_SHA_PASS'])
            self.assertTrue((fixture/'package'/'receipt.json').is_file())


if __name__=='__main__':unittest.main(verbosity=2)
