"""Pure metadata tests for the actual source100 graph-schema recovery."""
import copy
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import register_development_render_r1 as r

HERE=Path(__file__).absolute().parent
ACTUAL=HERE.parent/'current/h18_render_registration_failure/H/development100_source_execution_v1/visual_source_closure.json'
ACTUAL_SHA='981e1ccd3315ab902cbe997adf1db2a6b0e40c73b247825d2bb433714057ccd7'


def fixture(base):
    e,o,a=base/'old_execution',base/'old_operations',base/'archive'
    e.mkdir();o.mkdir();a.mkdir();out=base/'old_render'
    failure=dict(status='FAILED_PRESERVE_NO_LAUNCH',traceback="register_development_render.py line130\nKeyError: 'inputs'")
    r.d.save(e/'registration_failure.json',failure);r.d.save(o/'registration_failure.json',failure)
    r.d.save(e/'predecessor_closure.json',{'unchanged':True})
    r.d.save(o/'request.json',{'execution_dir':str(e),'render_out':str(out)})
    graph=base/'source_graph.json'
    r.d.save(graph,dict(status='EXACT_SOURCE_CLOSURE_MATCH',source_bindings={'/old.py':'a'*64},
                       newly_bound_sources={},original_registered_files_changed=False))
    actual={str(p):r.d.sha(p) for root in (e,o) for p in root.rglob('*') if p.is_file()}
    copies={}
    for i,(p,s) in enumerate(actual.items()):
        dest=a/f'{i}.original';shutil.copyfile(p,dest);copies[p]=dict(path=str(dest),sha256=s)
    pin=lambda p:dict(path=str(p),sha256=r.d.sha(p))
    value=dict(status=r.RECOVERY,original_execution_dir=str(e),original_operations_dir=str(o),
        original_render_out=str(out),original_files=actual,preserved_files=copies,
        original_request=pin(o/'request.json'),original_registration_failure=pin(e/'registration_failure.json'),
        original_private_failure=pin(o/'registration_failure.json'),original_graph=pin(graph))
    rp=base/'receipt.json';r.d.save(rp,value)
    return pin(rp),value


class RecoveryTests(unittest.TestCase):
    def test_real_downloaded_graph_without_inputs_preserves_all_metadata(self):
        p=Path(os.environ.get('H_ACTUAL_DEVELOPMENT_SOURCE_GRAPH',ACTUAL))
        self.assertEqual(r.d.sha(p),ACTUAL_SHA)
        source=r.d.read(p);self.assertNotIn('inputs',source)
        before=copy.deepcopy(source);actual=dict(source['source_bindings']);actual['/new_r1.py']='b'*64
        value=r.graph_record(source,actual,p)
        self.assertEqual(source,before)
        self.assertEqual(value['previous_graph_metadata'],{k:v for k,v in source.items() if k!='source_bindings'})
        self.assertEqual(value['previous_graph']['sha256'],ACTUAL_SHA)
        self.assertNotIn('inputs',value)
        self.assertEqual(value['newly_bound_sources'],{'/new_r1.py':'b'*64})

    def test_existing_visual_source_sha_cannot_change(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'graph.json';old={'status':'EXACT_SOURCE_CLOSURE_MATCH','source_bindings':{'/a':'a'*64}}
            r.d.save(p,old)
            with self.assertRaisesRegex(RuntimeError,'visual source changed'):
                r.graph_record(old,{'/a':'b'*64},p)

    def test_exact_unlaunched_failure_and_copies_are_bound(self):
        with tempfile.TemporaryDirectory() as td:
            pin,receipt=fixture(Path(td));checked,bindings=r.recovery_evidence(pin)
            self.assertEqual(checked,receipt)
            self.assertTrue(set(receipt['original_files'])<=set(bindings))
            self.assertTrue(all(x['path'] in bindings for x in receipt['preserved_files'].values()))

    def test_changed_or_added_original_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            pin,receipt=fixture(Path(td))
            (Path(receipt['original_execution_dir'])/'unexpected').write_text('new')
            with self.assertRaisesRegex(RuntimeError,'inventory changed'):r.recovery_evidence(pin)

    def test_preserved_copy_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            pin,receipt=fixture(Path(td));saved=next(iter(receipt['preserved_files'].values()))
            Path(saved['path']).write_text('altered')
            with self.assertRaisesRegex(RuntimeError,'Pinned input'):r.recovery_evidence(pin)

    def test_existing_original_science_output_blocks_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            pin,receipt=fixture(Path(td));Path(receipt['original_render_out']).mkdir()
            with self.assertRaisesRegex(RuntimeError,'absent science output'):r.recovery_evidence(pin)


if __name__=='__main__':unittest.main()
