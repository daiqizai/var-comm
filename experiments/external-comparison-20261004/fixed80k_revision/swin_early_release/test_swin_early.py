"""CPU tests for honest one-method scope and exact physical cache admission."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import swin_early_common as c

class Scope(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.rgb=np.zeros((3,256,256),dtype=np.float32)
        self.spec=c.frame_specs(0)[0];self.source=dict(image_id='source0',rgb=self.rgb)
        self.signal=np.ones((1024,2),dtype=np.float64)
        self.archive=self.root/'frame.npz';digest=c.atomic_npz(self.archive,images=self.rgb[None],observed=self.signal,transmitted=self.signal)
        row=dict(**self.spec,method=c.METHODS[0],replay_row_id=c.row_id(self.spec,c.METHODS[0]),
            image_sha256=c.rgb_sha(self.rgb),observed_sha256=c.array_sha(self.signal),reference_sha256=c.rgb_sha(self.rgb),
            header_accepted=True,audit_used_to_control_receiver=False,TX_seconds=.01,RX_seconds=.02,
            selected_step=80000,selected_checkpoint_sha256=c.CHECKPOINT_SHA,NFE=0,complete_posterior_schedule=False)
        self.value=dict(binding='binding',frame=self.spec,source_id='source0',reference_sha256=c.rgb_sha(self.rgb),
            rows=[row],archive=str(self.archive),archive_sha256=digest,observed_sha256=c.array_sha(self.signal),
            transmitted_sha256=c.array_sha(self.signal),hifi_evaluated=False)
        self.seal_value()
    def seal_value(self):self.value['payload_sha256']=c.identity({k:v for k,v in self.value.items() if k!='payload_sha256'})
    def test_grid_is1800_single_method(self):
        ids=[rid for i in range(100) for rid in c.expected_ids(i)]
        self.assertEqual(len(ids),1800);self.assertEqual(len(set(ids)),1800)
        self.assertEqual(c.METHODS,('SwinJSCC_new_shared',))
        self.assertEqual({(s['N'],s['snr_db']) for s in c.frame_specs(0)}, {(n,s) for n in (1024,2048) for s in (1,7,13)})
    def test_preserves_original_swin_row_ids(self):
        for spec in c.frame_specs(1):self.assertEqual(c.row_id(spec,c.METHODS[0]),c.paired.row_id(spec,c.paired.METHODS[0]))
    def test_admits_one_actual_frame(self):self.assertEqual(c.validate_frame_receipt(self.value,'binding',self.spec,self.source),self.value)
    def test_rejects_second_invented_method(self):
        self.value['rows'].append(copy.deepcopy(self.value['rows'][0]));self.seal_value()
        with self.assertRaises(RuntimeError):c.validate_frame_receipt(self.value,'binding',self.spec,self.source)
    def test_rejects_other_checkpoint(self):
        self.value['rows'][0]['selected_step']=72500;self.seal_value()
        with self.assertRaises(RuntimeError):c.validate_frame_receipt(self.value,'binding',self.spec,self.source)
    def test_rejects_claimed_generation(self):
        self.value['rows'][0]['NFE']=255;self.seal_value()
        with self.assertRaises(RuntimeError):c.validate_frame_receipt(self.value,'binding',self.spec,self.source)
    def test_rejects_mutated_archive(self):
        self.archive.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):c.validate_frame_receipt(self.value,'binding',self.spec,self.source)
    def test_rejects_failed_header_non_gray(self):
        self.value['rows'][0]['header_accepted']=False;self.seal_value()
        with self.assertRaises(RuntimeError):c.validate_frame_receipt(self.value,'binding',self.spec,self.source)
    def test_no_empty_fake_paired_table(self):
        text=(c.HERE/'swin_early_score.py').read_text()
        self.assertNotIn('("metrics_paired_intervals.csv", paired)',text)
        self.assertNotIn('METHODS[1]',text);self.assertIn('metric_groups=6',text)
        self.assertIn('SWIN_EARLY_EVALUATION_COMPLETE',text)
    def test_no_diffusion_model_load_in_early_reconstruction(self):
        tree=ast.parse((c.HERE/'swin_early.py').read_text())
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='FrozenHiFiSwinReceiver' for n in ast.walk(tree)))
    def test_all_science_files_parse(self):
        for name in c.FILES:ast.parse((c.HERE/name).read_text())
    def test_18row_source_archive(self):
        rows=[]
        for spec in c.frame_specs(0):rows.append(dict(self.value['rows'][0],**spec,replay_row_id=c.row_id(spec,c.METHODS[0])))
        path=self.root/'source.npz';digest=c.atomic_npz(path,images=self.rgb[None],source_rgb=self.rgb,
            row_ids=np.asarray(c.expected_ids(0)),image_slots=np.zeros(18,dtype=np.int64))
        source=dict(binding='binding',source_index=0,rows=rows,frame_bindings={},float_reconstructions=dict(path=str(path),sha256=digest,image_slots=[0]*18))
        source['payload_sha256']=c.identity(source)
        self.assertEqual(c.validate_source(source,'binding',0),source)

if __name__=='__main__':unittest.main(verbosity=2)
